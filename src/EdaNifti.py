import argparse
import json
import pandas as pd
import glob
import re
import argparse
import glob
import os, sys
import nibabel as nib 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Config:
    parser = argparse.ArgumentParser(description='configuration for nifti analysis')

    parser.add_argument('--task1_data_path', default='C:/Users/parkm/Desktop/github/miccai_stroke_2022/01.data/task1.stroke_segmentation/dataset-ISLES22^release1 unzipped version')
    parser.add_argument('--task2_data_path', default='C:/Users/parkm/Desktop/github/miccai_stroke_2022/01.data/task2.ATLAS_2.0/ATLAS_2')

    parser.add_argument('--save_path', default='./')
    parse = parser.parse_args()
    params = {
        'DATA1_PATH': parse.task1_data_path,
        'DATA2_PATH': parse.task2_data_path,
        'SAVE_PATH': parse.save_path
    }


class NiftiAnalysis:
    def __init__(self, config):
        self.config = config

    def save_nifti_images(self, input_nifti_path, axis, layer_numb_list, interval, save_path):
        file_name = input_nifti_path.split('\\')[-1].rstrip('.nii.gz')
        img = nib.load(input_nifti_path)
        data = img.get_fdata()
        fig, axs = plt.subplots(3, 3, figsize=(10,10))
	    
        if layer_numb_list==None:
            layer_numb_list = [int((data.shape[axis]/2+(interval*4))-interval*i) for i in range(9)]
            for ax, layer_numb in zip(axs.ravel(), layer_numb_list):
                if axis==2:
                    sample = data[:,:,layer_numb]

                elif axis==1:
                    sample = data[:,layer_numb,:]
	            
                elif axis==0:
                    sample = data[layer_numb,:,:]
                
                ax.imshow(sample, cmap='gray')
	            
        else:
            for ax, layer_numb in zip(axs.ravel(), layer_numb_list):
                if axis==2:
                    sample = data[:,:,layer_numb]
                elif axis==1:
                    sample = data[:,layer_numb,:]
                elif axis==0:
                    sample = data[layer_numb,:,:]
                ax.imshow(sample, cmap='gray')
	    
        plt.savefig(os.path.join(save_path,file_name)+'.png', dpi=300)


    def reculsive_glob_list(self, data_path):
        whole_file_list = glob.glob(repr(data_path), recursive=True)
        
        return whole_file_list
        
        
if __name__ == "__main__":
	config = Config.params

	task1_derivateive_file_path = str(config['DATA1_PATH']) + '/derivatives/**/*.nii.gz'
	task1_rawdata_file_path = str(config['DATA1_PATH']) + + '/rawdata/**/*.nii.gz'

	task2_test_path = str(config['DATA2_PATH']) + '/Testing/**/**/**/**/*.nii.gz'
	task2_train_path = str(config['DATA2_PATH']) + '/Training/**/**/**/**/*.nii.gz'

	eda = NiftiAnalysis()
	task1_de = eda.reculsive_glob_list(task1_derivateive_file_path)
	task1_raw = eda.reculsive_glob_list(task1_rawdata_file_path)

	task2_test = eda.reculsive_glob_list(task2_test_path)
	task2_train = eda.reculsive_glob_list(task2_train_path)