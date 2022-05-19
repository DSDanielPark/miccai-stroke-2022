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


    def recursive_glob_list(self, data_path):
        whole_file_list = glob.glob(r"{}".format(data_path), recursive=True)
        
        return whole_file_list
        
    def save_summary_table(self, globbed_nifti_file_paths, save_full_path_with_file_name):
    	total_dict = dict()

    	for i, nifti_path in enumerate(globed_nifti_file_paths):
		#print(i)
        	#print(nifti_path)
        	temp_dict = dict()
        	img = nib.load(nifti_path)
		
		# 0, 1이 아닌 값들이 segmentation binary nifti file에 존재하는지 확인
		img_flatten = np.ravel(img)
		abnormal_mask = img_flatten[(img_flatten != 0) & (img_flatten != 1)] 
		 

        	file_name = nifti_path.split('\\')[-1].rstrip('.nii.gz')

        	hdr = img.header
        	hdr_info = save_print_instance(hdr)
        	raw = hdr.structarr		
        	temp_dict['img_affine_shape'] = img.affine.shape
        	temp_dict['img_affine_metrix'] = np.round(img.affine)
        	temp_dict['file_name'] = file_name
        	temp_dict['data_dtype'] = img.get_data_dtype()
        	temp_dict['nifti_img_shape'] = img.shape
        	temp_dict['hdr_info'] = hdr_info.strip("<class 'nibabel.nifti1.Nifti1Header'> object, ")
        	temp_dict['hdr.get_xyzt_units'] = hdr.get_xyzt_units()
        	temp_dict['hdr_raw'] = raw
        	temp_dict['3d_array_mean'] = img_array.mean()
        	temp_dict['3d_array_std'] = img_array.std()
        	temp_dict['3d_array_min'] = img_array.min()
        	temp_dict['3d_array_max'] = img_array.min()
		temp_dict['unique_values'] = np.unique(img_flatten, return_counts=True)[0]
		temp_dict['exception_in_binary_file'] = abnormal_mask.size
		

    		total_dict[i] = temp_dict
        
    	df_summary = pd.DataFrame(total_dict).T
    	df_summary.to_csv(save_full_path_with_file_name)
        
    return None
	
	
if __name__ == "__main__":
    config = Config().params

    task1_derivateive_file_path = config['DATA1_PATH'] + '/derivatives/**/*.nii.gz'
    task1_rawdata_file_path = config['DATA1_PATH'] +  '/rawdata/**/*.nii.gz'
    task2_test_path = config['DATA2_PATH'] + '/Testing/**/**/**/**/*.nii.gz'
    task2_train_path = config['DATA2_PATH'] + '/Training/**/**/**/**/*.nii.gz'

    eda = NiftiAnalysis(config)
    task1_de = eda.recursive_glob_list(task1_derivateive_file_path)
    task1_raw = eda.recursive_glob_list(task1_rawdata_file_path)
    
    task2_test = eda.recursive_glob_list(task2_test_path)
    task2_train = eda.recursive_glob_list(task2_train_path)

    interval = 4

    for k in range(len(3)):
        [eda.save_nifti_images(task1_de[i], k, None, interval, '../result/task1_mask'+str(k)+'/') for i in range(len(task1_de))]
        [eda.save_nifti_images(task1_raw[i], k, None, interval, '../result/task1_raw'+str(k)+'/') for i in range(len(task1_raw))]
        [eda.save_nifti_images(task2_test[i], k, None, interval, '../result/task2_test'+str(k)+'/') for i in range(len(task2_test))]
        [eda.save_nifti_images(task2_train[i], k, None, interval, '../result/task2_train'+str(k)+'/') for i in range(len(task2_train))]
	
    eda.save_summary_table(task1_de, './task1_derivatives_nii.csv')
    eda.save_summary_table(task1_raw, './task1_rawdata_nii.csv')
    eda.save_summary_table(task2_test, './task2_test_data.csv')
    eda.save_summary_table(task2_train, './task2_train_data.csv')
	
