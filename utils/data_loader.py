import SimpleITK as sitk
import numpy as np
import tensorflow as tf
import glob
import scipy
from tqdm import tqdm


class NiiDataLoader:
    def __init__(self, path):
        self.files_volume = glob.glob(path + "\\volume*.nii")
        self.files_segmenation = glob.glob(path + "\\segmentation*.nii")

        print(f"initalised with path {path}")
        print(f"files: {len(self.files_volume)},{len(self.files_segmenation)}")

    def preprocessing_3d(self, t1) -> np.ndarray:
        min = 50
        max = 200
        t1 = np.where(t1 < min, min, t1)
        t1 = np.where(t1 > max, max, t1)
        t1 -= min
        return t1 / (max - min)

    def label_seperator_liver(self, img):
        img = np.where(img == 2, 1, img)
        return img

    def label_seperator_lesion(self, img):
        img = np.where(img == 1, 0, img)
        img = np.where(img == 2, 1, img)
        return img

    def reading_data(self, path: str) -> np.ndarray:
        sitk_t1 = sitk.ReadImage(path)
        t1 = sitk.GetArrayFromImage(sitk_t1)
        t1 = t1.reshape(t1.shape[0], t1.shape[1], t1.shape[2], 1)

        t1 = tf.dtypes.cast(t1, tf.float32)

        return t1

    def generator_data_len(self):
        return len(self.files_volume)

    def z_transform(self, img_volume, img_seg, param_z=256):
        return_img_volume = np.zeros(
            (param_z, img_volume.shape[1], img_volume.shape[2], img_volume.shape[3])
        )
        return_img_seg = np.zeros(
            (param_z, img_seg.shape[1], img_seg.shape[2], img_seg.shape[3])
        )
        for i in range(img_seg.shape[1]):
            return_img_volume[:, i, :, 0:1] = tf.image.resize(
                img_volume[:, i, :, 0:1], [param_z, 512], method="nearest"
            )
            return_img_seg[:, i, :, 0:2] = tf.image.resize(
                img_seg[:, i, :, 0:2],
                [param_z, 512],
                method="nearest",
            )
        return return_img_volume, return_img_seg

    def data_generator_2d_liver(self):
        for i, file_volume in enumerate(self.files_volume):
            img_3d_volume = self.reading_data(file_volume)
            img_3d_volume = self.preprocessing_3d(img_3d_volume)

            img_3d_segmentation = self.reading_data(self.files_segmenation[i])
            img_3d_segmentation = self.label_seperator_liver(img_3d_segmentation)
            img_3d_segmentation = tf.keras.utils.to_categorical(img_3d_segmentation, 2)
            for j in range(len(img_3d_segmentation)):
                if "Training_Batch_2" in self.files_volume[0]:
                    if np.sum(img_3d_segmentation[j, :, :, 1]) > 10:
                        yield (
                            tf.image.resize(img_3d_volume[j, :, :, 0:1], [256, 256]),
                            tf.image.resize(
                                img_3d_segmentation[j, :, :, 0:2],
                                [256, 256],
                                method="nearest",
                            ),
                        )
                    elif np.sum(img_3d_segmentation[j, :, :, 1]) < 10:
                        success = np.random.uniform(0,1)
                        if success > 0.99:
                            yield (
                                tf.image.resize(img_3d_volume[j, :, :, 0:1], [256, 256]),
                                tf.image.resize(
                                    img_3d_segmentation[j, :, :, 0:2],
                                    [256, 256],
                                    method="nearest",
                                ),
                            )
                        else:
                            continue
                elif (
                        "Training_Batch_1" in self.files_volume[0]
                    ):
                    yield (
                        tf.image.resize(img_3d_volume[j, :, :, 0:1], [256, 256]),
                        tf.image.resize(
                            img_3d_segmentation[j, :, :, 0:2],
                            [256, 256],
                            method="nearest",
                        ),
                    )

    def data_generator_2d_liver_classifier(self):
        for i, file_volume in enumerate(self.files_volume):
            img_3d_volume = self.reading_data(file_volume)
            img_3d_volume = self.preprocessing_3d(img_3d_volume)

            img_3d_segmentation = self.reading_data(self.files_segmenation[i])
            img_3d_segmentation = self.label_seperator_liver(img_3d_segmentation)
            img_3d_segmentation = tf.keras.utils.to_categorical(img_3d_segmentation, 2)
            for j in range(len(img_3d_segmentation)):
                if np.sum(img_3d_segmentation[j, :, :, 1]) > 0:
                    y_tensor = tf.constant([0,1])
                else:
                    y_tensor = tf.constant([1,0])

                tf.shape(y_tensor)
                yield (
                    tf.image.resize(img_3d_volume[j, :, :, 0:1], [256, 256]),
                    y_tensor,
                )

    def data_generator_2d_lesion(self):
        for i, file_volume in enumerate(self.files_volume):
            img_3d_volume = self.reading_data(file_volume)
            img_3d_volume = self.preprocessing_3d(img_3d_volume)

            img_3d_segmentation = self.reading_data(self.files_segmenation[i])
            img_3d_segmentation_liver = self.label_seperator_liver(img_3d_segmentation)
            img_3d_segmentation_liver = tf.keras.utils.to_categorical(
                img_3d_segmentation_liver, 2
            )

            img_3d_segmentation_lesion = self.label_seperator_lesion(
                img_3d_segmentation
            )
            img_3d_segmentation_lesion = tf.keras.utils.to_categorical(
                img_3d_segmentation_lesion, 2
            )

            img_3d_volume = np.where(
                img_3d_segmentation_liver[:, :, :, 1:2] == 1, img_3d_volume, 0
            )
            for j in range(len(img_3d_segmentation_lesion)):
                # in training
                if (
                    "Training_Batch_2" in self.files_volume[0]
                    and np.sum(img_3d_segmentation_liver[j, :, :, 1]) > 10
                ):
                    yield (
                        tf.image.resize(
                            img_3d_volume[j, :, :, 0:1], [128, 128], method="nearest"
                        ),
                        tf.image.resize(
                            img_3d_segmentation_lesion[j, :, :, 0:2],
                            [128, 128],
                            method="nearest",
                        ),
                    )
                # in validation
                if "Training_Batch_1" in self.files_volume[0]:
                    yield (
                        tf.image.resize(
                            img_3d_volume[j, :, :, 0:1], [128, 128], method="nearest"
                        ),
                        tf.image.resize(
                            img_3d_segmentation_lesion[j, :, :, 0:2],
                            [128, 128],
                            method="nearest",
                        ),
                    )

    def data_generator_3d_lesion_chunks(self, chunk_size=32):
        for i, file_volume in enumerate(self.files_volume):
            print(f"processed files: {np.round(i/len(self.files_volume),2)}")
            img_3d_volume = self.reading_data(file_volume)
            img_3d_volume = self.preprocessing_3d(img_3d_volume)
            img_3d_segmentation = self.reading_data(self.files_segmenation[i])
            img_3d_segmentation_liver = self.label_seperator_liver(img_3d_segmentation)

            img_3d_segmentation_lesion = self.label_seperator_lesion(
                img_3d_segmentation
            )
            #img_3d_segmentation_lesion = tf.keras.utils.to_categorical(
            #    img_3d_segmentation_lesion, 2
            #)

            #img_3d_segmentation_liver = tf.keras.utils.to_categorical(
            #    img_3d_segmentation_liver, 2
            #)

            img_3d_volume = np.where(
            img_3d_segmentation_liver == 1, img_3d_volume, 0
            )


            # z transform
            img_3d_volume, img_3d_segmentation_lesion = self.z_transform(
                img_3d_volume, img_3d_segmentation_lesion,param_z = 640
            )
            img_shape = img_3d_volume.shape
            cords_shift = chunk_size  // 4
            
            for z in range(img_shape[0] // cords_shift):  # z direction
                for y in range(img_shape[1] // cords_shift):
                    for x in range(img_shape[2] // cords_shift):
                        tmp_image = img_3d_volume[
                            z * cords_shift : z * cords_shift + chunk_size,
                            y * cords_shift : y * cords_shift + chunk_size,
                            x * cords_shift : x * cords_shift + chunk_size,
                            0:1,
                        ]
                        tmp_seg = img_3d_segmentation_lesion[
                            z * cords_shift : z * cords_shift + chunk_size,
                            y * cords_shift : y * cords_shift + chunk_size,
                            x * cords_shift : x * cords_shift + chunk_size,
                            0:1,
                        ]

                        if np.sum(tmp_seg[4:chunk_size-4:,4:chunk_size-4,4:chunk_size-4,0]) > 1500:
                            yield (tmp_image, tmp_seg)

class NPYLoader:
    def __init__(self, path):
        self.files_volume = glob.glob(path + "\\*_vol.npy")
        self.files_segmenation = glob.glob(path + "\\*_seg.npy")

        print(f"initalised with path {path}")
        print(f"files: {len(self.files_volume)},{len(self.files_segmenation)}")

    def data_generator_2d(self):
        for i, file_volume in enumerate(self.files_volume):
            img_2d_vol = np.load(file_volume)
            img_2d_seg = np.load(self.files_segmenation[i])

            yield (img_2d_vol, img_2d_seg)
