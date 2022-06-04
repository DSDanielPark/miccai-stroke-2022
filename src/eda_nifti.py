import argparse
from hamcrest import ends_with
import pandas as pd
import glob
import os
import nibabel as nib 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO


parser = argparse.ArgumentParser(description='configuration for nifti analysis')
parser.add_argument('--nifti_folder', default='./data')
parser.add_argument('--save_path', default='./')



class NiftiAnalysis:
    def __init__(self):
        pass

    def save_nifti_images(self, input_nifti_path:str, axis:str, layer_numb_list:list, interval:int, save_path:str) -> None:
        '''
        input: 단일 nifti file path (*.nii.gz)
        output: nifti 파일의 2d 단면 
	
	
        axis: 0,1,2 축 결정
        layer_numb_list: 9장의 시각화할 단면 번호, None으로 입력시 전체 이미지의 중간 단면 수를 
                         기준으로 interval=5만큼 이동하면서 총 9개의 단면을 결정함
			 커스텀하려면 [0,1,2,3,4,5,6,7,8]과 같이 9개의 리스트 입력
			 그러면 단면 0, 단면 1, 단면3... 의 이미지들이 저장됨
        interval: 중간 단면 수로부터 선택할 슬라이드 간격
        save_path: 저장할 경로 입력

        ex) save_nifti_images('input_path', 0, None, 5, 'output_path')
        '''

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        file_name = input_nifti_path.split('\\')[-1].rstrip('.nii.gz')
        print
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
        print('...saved img of {file_name}')
        plt.close()


        return None


    def recursive_find_all_files(self, top_folder_path, file_format):
        gathered_file_pathes = []

        for (root, directories, files) in os.walk(top_folder_path):
            for file in files:
                if file_format in file:
                    detect_file_path = os.path.join(root, file)
                    gathered_file_pathes.append(detect_file_path)

        return gathered_file_pathes
        

    def save_print_instance(*message):
        io = StringIO()
        print(*message, file=io, end="")
        return io.getvalue()

        
    def save_summary_table(self, globbed_nifti_file_paths, save_full_path_with_file_name):
        '''
        input: glob으로 리스트로 만든 k개의 nifti 파일들의 경로들 
            ex) ['./1.nii.gz', './2.nii.gz' .... './k.nii.gz']
            
        output: save_full_path_with_file_name에 입력한 csv 파일 
            ex) './result.csv' 
            
        칼럼으로 저장되는 내용들은 nibabel 공식 도큐멘테이션 참조 

        '''
        total_dict = dict()
        
        for i, nifti_path in enumerate(globbed_nifti_file_paths):
		    #print(i)
        	#print(nifti_path)
            temp_dict = dict()
            img = nib.load(nifti_path)
		
            # 0, 1이 아닌 값들이 segmentation binary nifti file에 존재하는지 확인
            img_array = img.get_fdata()
            img_flatten_array = np.ravel(img_array)
            abnormal_mask = img_flatten_array[(img_flatten_array != 0) & (img_flatten_array != 1)] 
            file_name = nifti_path.split('\\')[-1].rstrip('.nii.gz')
            hdr = img.header
            hdr_info = self.save_print_instance(hdr)
            raw = hdr.structarr		
            temp_dict['img_affine_shape'] = img.affine.shape
            temp_dict['img_affine_metrix'] = np.round(img.affine)
            temp_dict['img_affine_metrix(raw value)'] = img.affine
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
            temp_dict['unique_values'] = np.unique(img_flatten_array, return_counts=True)[0]
            temp_dict['exception_val_count_in_binary_file'] = abnormal_mask.size
		
            total_dict[i] = temp_dict
        
        df_summary = pd.DataFrame(total_dict).T
        df_summary.to_csv(save_full_path_with_file_name)
        print('...summary table saved')

        return None
	

if __name__ == "__main__":
    
    niftieda = NiftiAnalysis()
    args = parser.parse_args()

    nifit_folder = args.nifti_folder
    all_nifti_files_under_nifti_folder = niftieda.recursive_find_all_files(nifit_folder, '.nii.gz')
	
    #01. nifti_folder 안에 있는 모든 '.nii.gz' 파일을 대상으로 메타 정보 엑셀 파일 생성
    ## 'nii'파일 존재시 list join 해주세요.
    niftieda.save_summary_table(all_nifti_files_under_nifti_folder, './nifti_eda_result.csv')
	
