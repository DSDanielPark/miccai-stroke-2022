import argparse
import pandas as pd
import glob
import os
import nibabel as nib 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Config:
    parser = argparse.ArgumentParser(description='configuration for nifti analysis')
    parser.add_argument('--task1_data_path', default='C:/Users/parkm/Desktop/github/miccai_stroke_2022/data/task1.stroke_segmentation/dataset-ISLES22^release1 unzipped version')
    parser.add_argument('--task2_data_path', default='C:/Users/parkm/Desktop/github/miccai_stroke_2022/data/task2.ATLAS_2.0/ATLAS_2')
    parser.add_argument('--save_path', default='./')
    parse = parser.parse_args()
    params = {
        'DATA1_PATH': parse.task1_data_path,
        'DATA2_PATH': parse.task2_data_path,
        'SAVE_PATH': parse.save_path
    }


class NiftiAnalysis:
    def __init__(self, Config):
        self.config = Config

    def save_nifti_images(self, input_nifti_path, axis, layer_numb_list, interval, save_path):
        '''
        axis: 0,1,2 축 결정
        layer_numb_list: 9장의 시각화할 단면 번호, None으로 입력시 전체 이미지의 중간 단면 수를 
                         기준으로 interval=5만큼 이동하면서 총 9개의 단면을 결정함
        interval: 중간 단면 수로부터 선택할 슬라이드 간격
        save_path: 저장할 경로 입력

        ex) save_nifti_images('input_path', 0, None, 5, 'output_path')
        '''

        if not os.path.exists(save_path):
            os.makedirs(save_path)

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
        plt.close(fig)

        return None


    def reculsive_glob_list(self, data_path):
        whole_file_list = glob.glob(r"{}".format(data_path), recursive=True)
        
        return whole_file_list
        
        
if __name__ == "__main__":
    config = Config().params

    task1_derivateive_file_path = config['DATA1_PATH'] + '/derivatives/**/*.nii.gz'
    task1_rawdata_file_path = config['DATA1_PATH'] +  '/rawdata/**/*.nii.gz'
    task2_test_path = config['DATA2_PATH'] + '/Testing/**/**/**/**/*.nii.gz'
    task2_train_path = config['DATA2_PATH'] + '/Training/**/**/**/**/*.nii.gz'



    eda = NiftiAnalysis(config)
    task1_de = eda.reculsive_glob_list(task1_derivateive_file_path)
    task1_raw = eda.reculsive_glob_list(task1_rawdata_file_path)
    
    task2_test = eda.reculsive_glob_list(task2_test_path)
    task2_train = eda.reculsive_glob_list(task2_train_path)


    interval = 4

    for k in range(len(3)):
        [eda.save_nifti_images(task1_de[i], k, None, interval, '../result/task1_mask/') for i in range(len(task1_de))]
        [eda.save_nifti_images(task1_raw[i], k, None, interval, '../result/task1_raw/') for i in range(len(task1_raw))]
        [eda.save_nifti_images(task2_test[i], k, None, interval, '../result/task2_test/') for i in range(len(task2_test))]
        [eda.save_nifti_images(task2_train[i], k, None, interval, '../result/task2_train/') for i in range(len(task2_train))]

