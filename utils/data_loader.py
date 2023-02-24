import os
import SimpleITK as sitk
import numpy as np
import scipy
import tensorflow as tf
import glob

class NiiDataLoader():
    def __init__(self):
        self.files_volume = glob.glob('data\\LITS_Challenge\\Training_Batch 1\\volume*.nii')
        self.files_segmenation =  glob.glob('data\\LITS_Challenge\\Training_Batch 1\\segmentation*.nii')

    def preprossing_3d(self,t1)->np.ndarray:
        min = -100
        max = 300
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

    def data_generator_2d(self,valid:int):
        if valid == 0:
            files_volume = self.files_volume[0:22]
            files_segmenation = self.files_segmenation[0:22]
        elif valid == 1:
            files_volume = self.files_volume[22:]
            files_segmenation = self.files_segmenation[22:]

        for i,file_volume in enumerate(files_volume):
            img_3d_volume = self.reading_data(file_volume)
            img_3d_volume = self.preprossing_3d(img_3d_volume)

            img_3d_segmentation = self.reading_data(files_segmenation[i])
            img_3d_segmentation = self.label_seperator(img_3d_segmentation)
            img_3d_segmentation = tf.keras.utils.to_categorical(img_3d_segmentation,2)
            for j in range(len(img_3d_segmentation)):
                if np.sum(img_3d_segmentation[j,:,:,1]) > 10:
                    yield (tf.image.resize(img_3d_volume[j,:,:,0:1],[256,256]),tf.image.resize(img_3d_segmentation[j,:,:,0:2],[256,256],method="nearest"))