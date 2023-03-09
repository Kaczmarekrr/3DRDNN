import os
import SimpleITK as sitk
import numpy as np
import scipy
import tensorflow as tf
import glob

class NiiDataLoader():
    def __init__(self,path):
        self.files_volume = glob.glob(path + '\\volume*.nii')
        self.files_segmenation =  glob.glob(path + '\\segmentation*.nii')

        print(f"initalised with path {path}")
        print(f"files: {len(self.files_volume)},{len(self.files_segmenation)}")

    def preprossing_3d(self,t1)->np.ndarray:
        min = 10
        max = 200
        t1= np.where(t1<min,min,t1)
        t1= np.where(t1>max,max,t1)
        t1 -= min
        return t1/max

    def label_seperator(self,img):
        img = np.where(img==2,1,img)
        return img

    def reading_data(self,path:str)->np.ndarray:
            sitk_t1 = sitk.ReadImage(path)
            t1 = sitk.GetArrayFromImage(sitk_t1)
            t1 = t1.reshape(t1.shape[0],t1.shape[1],t1.shape[2],1)

            t1 = tf.dtypes.cast(t1,tf.float32)

            return t1

    def generator_data_len(self):
         return len(self.files_volume)
    
    def data_generator_2d(self):

        for i,file_volume in enumerate(self.files_volume):
            img_3d_volume = self.reading_data(file_volume)
            img_3d_volume = self.preprossing_3d(img_3d_volume)

            img_3d_segmentation = self.reading_data(self.files_segmenation[i])
            img_3d_segmentation = self.label_seperator(img_3d_segmentation)
            img_3d_segmentation = tf.keras.utils.to_categorical(img_3d_segmentation,2)
            for j in range(len(img_3d_segmentation)):
                    yield (tf.image.resize(img_3d_volume[j,:,:,0:1],[256,256]),tf.image.resize(img_3d_segmentation[j,:,:,0:2],[256,256],method="nearest"))


class NPYLoader():
    def __init__(self,path):
        self.files_volume = glob.glob(path + '\\*_vol.npy')
        self.files_segmenation =  glob.glob(path + '\\*_seg.npy')

        print(f"initalised with path {path}")
        print(f"files: {len(self.files_volume)},{len(self.files_segmenation)}")

    def data_generator_2d(self):

        for i,file_volume in enumerate(self.files_volume):
            img_2d_vol = np.load(file_volume)
            img_2d_seg = np.load(self.files_segmenation[i])

            yield (img_2d_vol,img_2d_seg)

         

        