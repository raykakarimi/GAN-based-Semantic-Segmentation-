import os
from os.path import splitext
from os import listdir
from glob import glob
import pathlib
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom    
def data_prep(src_path, trgt_path, src_img_suffix = 'image', src_lbl_suffix = 'label',
              trgt_img_prefix = 'img_p', trgt_lbl_prefix = 'msk_p', trgt_hdr_prefix = 'hd_p'):
    # Create target directory if not exist
    pathlib.Path(trgt_path).mkdir(parents=True, exist_ok=True) 

    # Read file names from source directory
    file_names = [file for file in listdir(src_path) if not file.startswith('.')]
    for index, file_name in enumerate(file_names):
        # Read the .nii file
        header={}
        if file_name[3:7] == 'test':
            index= 2 * index
        file = sitk.ReadImage(os.path.join(src_path, file_name))
        
        # Extract the the numpy array:
        np_array = sitk.GetArrayFromImage(file)
                
        ################################################
        #saving resized images for increasing the speed. 
        #It will be optional if pairRedsized are utilized from transforms. 
        ################################################
        # img_resized = [32, 96, 96]
       #np_array = resize(img_resized,  np_array1)
        ################################################
        #Some mask files have uint16 type (instead of int16), which causes problem during converting to torch tensors.
        #Convert their datatype to int16
        if np_array.dtype == 'uint16':
            np_array = np.int16(np_array)
        
        # Save into processed dir as an .np file
        with open(trgt_path + '/' + (trgt_img_prefix if src_img_suffix in file_name else trgt_lbl_prefix) 
                  + str(int(np.floor(index/2)+1)).zfill(3) + '.np', 'wb') as f:
            np.save(f, np_array, allow_pickle=True)
            
        if src_img_suffix in file_name:
            # SAVE THE HEADERS INTO A SEPARATE FILE
            
            for k in file.GetMetaDataKeys():
                header[k] = file.GetMetaData(k)
            header['ITK_FileNotes'] =  file_name + str(np_array.shape)
                    # Save into processed dir as an .np file
            with open(trgt_path + '/' + trgt_hdr_prefix  
                  + str(int(np.floor(index/2)+1)).zfill(3) + '.np', 'wb') as f:
                np.save(f, header, allow_pickle=True)
            
            
        print("{}.{} processed successfully!".format(str(index+1).zfill(3), file_name))
    print("Data preparation completed!")
    
def resize(new_sizes,image):

    resized_image = zoom(image, ( new_sizes[0]/image.shape[0],
        new_sizes[1]/image.shape[1], new_sizes[2]/image.shape[2]), mode = 'nearest')
    return resized_image