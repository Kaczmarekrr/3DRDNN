import os
import sys
from glob import glob
from pathlib import Path
from tqdm import tqdm
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

class NiiDataLoader():
    def __init__(self):
        self.path = "data/LITS_Challenge/Training_Batch 1/"
    def basic_data_generator(self,file_list:list)->np.ndarray:
        for file in file_list:
            # Read the .nii image containing the volume with SimpleITK:
            file_path = self.path+file
            print(file_path)
            sitk_t1 = sitk.ReadImage(file_path)
            t1 = sitk.GetArrayFromImage(sitk_t1)
            for slide_2d in t1:
                yield slide_2d
