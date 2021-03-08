import os
import random
import numpy as np
from os import listdir

def make_dataset(data_path, train_dir, test_dir, hdr_prefix = 'hd_p', img_prefix = 'img_p',
                 msk_prefix = 'msk_p', val_ratio=0.2, dataset_file_name='CT_dataset.np'):
    #load train and validation sets
    file_paths = [file for file in listdir(os.path.join(data_path, train_dir))
                   if (not file.startswith('.') and img_prefix in file)]
    for index, file_path in enumerate(file_paths):
        file_paths[index] = os.path.abspath(os.path.join(data_path, train_dir, file_paths[index]))
        
    # Shuffle paths
    random.shuffle(file_paths)
    
    train_paths = []
    for path in file_paths[:int(np.round(len(file_paths)*(1-val_ratio)))]:
        train_paths.append({'header':path.replace(img_prefix, hdr_prefix),
                            'image': path, 'mask': path.replace(img_prefix, msk_prefix)})
        
    val_paths = []
    for path in file_paths[int(np.round(len(file_paths)*(1-val_ratio))):]:
        val_paths.append({'header':path.replace(img_prefix, hdr_prefix),
                            'image': path, 'mask': path.replace(img_prefix, msk_prefix)})
        
    test_paths = [file for file in listdir(os.path.join(data_path, test_dir))
                   if (not file.startswith('.') and img_prefix in file)]
    for index, test_path in enumerate(test_paths):
        test_paths[index] = {'header':os.path.abspath(os.path.join(data_path, test_dir,
                            test_paths[index].replace(img_prefix, hdr_prefix))),
                            'image': os.path.abspath(os.path.join(data_path, test_dir, test_paths[index]))}
    
    # Save paths
    with open(os.path.join(data_path, dataset_file_name), 'wb') as f:
        np.save(f, {'train':train_paths, 'val': val_paths, 'test': test_paths}, allow_pickle=True)
        
    print("{} is created successfully!".format(dataset_file_name))