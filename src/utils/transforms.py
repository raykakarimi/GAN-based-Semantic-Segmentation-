import torch
import numpy as np
from scipy.ndimage import zoom

class PairResize(torch.nn.Module):

    def __init__(self, new_sizes, mode = 'train'):
        self.new_sizes = new_sizes
        self.mode = mode 

    def __call__(self, sample):
        image = sample['image']

        resized_image = zoom(image, (1, self.new_sizes[0]/image.shape[1],
            self.new_sizes[1]/image.shape[2], self.new_sizes[2]/image.shape[3]))
        if self.mode == 'train':
            mask = sample['mask']
            resized_mask = zoom(mask, (1, self.new_sizes[0]/image.shape[1],
                self.new_sizes[1]/image.shape[2], self.new_sizes[2]/image.shape[3]))
            return {'image': resized_image, 'mask': resized_mask}
        return {'image': resized_image,'header':sample['header']}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, mode = 'train'):
        self.mode = mode 

    def __call__(self, sample):
        image = np.array(sample['image'])
        if self.mode == 'train':
            mask =  np.array(sample['mask'])

            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            # image = image.transpose((3, 0, 1, 2))
            # mask = mask.transpose((3, 0, 1, 2))
            return {'image': torch.from_numpy(image),'mask': torch.from_numpy(mask)}
        return {'image': torch.from_numpy(image),'header':sample['header']}