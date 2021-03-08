import os
import torch
from torch.utils.data import Dataset
import numpy as np

class Data_Handler(Dataset):
    def __init__(self, dataset_path, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
        # Load data absolute paths from the dataset file
        with open(dataset_path, 'rb') as f:
            self.absolute_paths = np.load(f, allow_pickle=True).item()
            
    def __len__(self):
        for subset_key in self.absolute_paths:
            if self.subset in subset_key:
                return len(self.absolute_paths[subset_key])
        return 0

    def __getitem__(self, i):
        header = image = mask = []
        for subset_key in self.absolute_paths:
            if self.subset in subset_key:
                subset_file_paths = self.absolute_paths[subset_key]
                sample_file_paths = list(subset_file_paths[i].values())
                # Load header
                file_path = sample_file_paths[0]
                if (os.path.exists(file_path)):
                    with open(file_path, 'rb') as f:
                        header = np.load(f, allow_pickle=True)

                file_path = sample_file_paths[1]
                
                if (os.path.exists(file_path)):
                    with open(file_path, 'rb') as f:
                        image = np.load(f, allow_pickle=True)
                
                # Load mask
                if not self.subset == 'test':
                    file_path = sample_file_paths[2]
                    if (os.path.exists(file_path)):
                        with open(file_path, 'rb') as f:
                            mask = np.load(f, allow_pickle=True)

        sample = {
            'header': header.item(),
            'image': np.expand_dims(image, axis=0),
            'mask': np.expand_dims(mask, axis=0)
        }
        
        if self.transform is not None:
            sample = self.transform(sample)
            
        return sample