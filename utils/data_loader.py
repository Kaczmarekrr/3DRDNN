import os
import SimpleITK as sitk
import numpy as np

class NiiDataLoader():
    def __init__(self):
        self.path = "data/LITS_Challenge/Training_Batch 1/"

    def preprossing_2d(self,img)->np.ndarray:
        print("processing...")
        return img

    def reading_data(self,path:str)->np.ndarray:
            sitk_t1 = sitk.ReadImage(path)
            t1 = sitk.GetArrayFromImage(sitk_t1)
            t1 = self.preprossing_2d(t1)

            return t1

    def data_generator_2d(self,file_list:list):
        for file in file_list:
            file_path = self.path+file
            img_3d = self.reading_data(file_path)
            # Read the .nii image containing the volume with SimpleITK:
            for slice in img_3d:
                yield slice