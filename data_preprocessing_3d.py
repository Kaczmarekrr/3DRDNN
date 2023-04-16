## preprocesing data to .npy to be used in data loader functions
import numpy as np
import tensorflow as tf
from utils import data_loader
import glob
from pathlib import Path

# 1) read Nii file
# 2) preprocess data with desire preprocessing (all data, only with liver, only with lession etc)
# 3) save to tfrecords


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):  # if value ist tensor
        value = value.numpy()  # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a floast_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array


# tfrecords tutorial from tensorflow wiki
def parse_tfr_element(element):
    # use the same structure as above; it's kinda an outline of the structure we now want to create
    data = {
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
        "depth": tf.io.FixedLenFeature([], tf.int64),
        "channel": tf.io.FixedLenFeature([], tf.int64),
        "raw_image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.string),
    }

    content = tf.io.parse_single_example(element, data)

    depth = content["depth"]
    height = content["height"]
    width = content["width"]
    channel = content["channel"]
    raw_image = content["raw_image"]
    gt = content["label"]

    # get our 'feature'-- our image -- and reshape it appropriately
    image_0 = tf.io.parse_tensor(raw_image, out_type=tf.float32)
    image_0 = tf.reshape(image_0, shape=[depth, height, width, channel])

    image_1 = tf.io.parse_tensor(gt, out_type=tf.float32)
    image_1 = tf.reshape(image_1, shape=[depth, height, width, channel * 2])
    return (image_0, image_1)


def parse_single_image(image, label):
    # define the dictionary -- the structure -- of our single example
    data = {
        "depth": _int64_feature(image.shape[0]),
        "height": _int64_feature(image.shape[1]),
        "width": _int64_feature(image.shape[2]),
        "channel": _int64_feature(image.shape[3]),
        "raw_image": _bytes_feature(serialize_array(image)),
        "label": _bytes_feature(serialize_array(label)),
    }
    # create an Example, wrapping the single features
    out = tf.train.Example(features=tf.train.Features(feature=data))
    return out


def write_images_to_tfr_short(images, filename: str = "images"):
    filename = filename + ".tfrecords"
    writer = tf.io.TFRecordWriter(
        filename
    )  # create a writer that'll store our data to disk
    count = 0

    # get the data we want to write
    for index in range(len(images[0])):
        # get the data we want to write
        current_image = images[0][index, :, :, :, 0:1]
        current_labels = images[1][index, :, :, :, 0:2]
        out = parse_single_image(image=current_image, label=current_labels)
        writer.write(out.SerializeToString())
        count += 1

    writer.close()
    print(f"Wrote {count} elements to TFRecord")
    return count


def main():
    path_train = "data\LITS_Challenge\Training_Batch_2"
    path_valid = "data\LITS_Challenge\Training_Batch_1"

    loader_train = data_loader.NiiDataLoader(path_train)
    loader_valid = data_loader.NiiDataLoader(path_valid)

    train_generator = loader_train.data_generator_3d_lesion_chunks()
    valid_generator = loader_valid.data_generator_3d_lesion_chunks()

    batch_in_file = 500
    data_image = (
        np.zeros((batch_in_file, 32, 64, 64, 1), dtype=np.float32),
        np.zeros((batch_in_file, 32, 64, 64, 2), dtype=np.float32),
    )

    i = 0
    j = 0
    gen_list = [train_generator]#, valid_generator]
    #gen_list = [valid_generator]#
    #output_path = [r"D:\\tfrecords_valid\\images"] #[r"D:\\tfrecords_train\\images", r"D:\\tfrecords_valid\\images"]
    output_path= ["C:\\Users\\kaczm\\programming\\3DRDNN\\data\\LITS_TFRecords\\train\\images"]#,"C:\\Users\\kaczm\\programming\\3DRDNN\\data\\LITS_TFRecords\\valid\\images"]
    for gen_id, gen in enumerate(gen_list):
        for x in gen:
            if i < batch_in_file:
                try:
                    data_image[0][i, :, :, :, 0:1] = x[0]
                    data_image[1][i, :, :, :, 0:2] = x[1]
                    i += 1
                except ValueError:
                    print("wrong input shape from generator", x[0].shape, x[1].shape)
            else:
                count = write_images_to_tfr_short(
                    data_image, filename=output_path[gen_id] + str(j)
                )
                i = 0
                j += 1
                data_image = (
                    np.zeros((batch_in_file, 32, 64, 64, 1), dtype=np.float32),
                    np.zeros((batch_in_file, 32, 64, 64, 2), dtype=np.float32),
                )
        print("done")
        print(count)


def get_dataset_large(tfr_dir: str = "/data/LITS_TFRecords/"):
    glob_path = Path(tfr_dir)
    file_list = [str(pp) for pp in glob_path.glob("**/*.tfrecords")]
    print(file_list)
    # create the dataset
    dataset = tf.data.TFRecordDataset(file_list)

    # pass every single feature through our mapping function
    dataset = dataset.map(parse_tfr_element)

    return dataset


def print_dataset(tfr_dir: str):
    dataset_large = get_dataset_large(tfr_dir)

    print("testing data output")
    count = 0
    for sample in dataset_large.take(-1):
        count+=1
    print('ok')
    print(f"there are {count} examples")

if __name__ == "__main__":
    main()
    print_dataset("C:\\Users\\kaczm\\programming\\3DRDNN\\data\\LITS_TFRecords\\train")
