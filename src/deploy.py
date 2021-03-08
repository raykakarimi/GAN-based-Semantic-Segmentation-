### deploy
import os
from os.path import splitext
from os import listdir
from glob import glob
import pathlib
import numpy as np
from scipy.ndimage import zoom
import SimpleITK as sitk
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.optim as optim
from barbar import Bar
from data_handler import Data_Handler
import torchvision
from utils.transforms import PairResize, ToTensor
from experiment import train_network

#saving  images
def save_image(image,header, target_test_mask_dir):
    
    #transforming images into initial size of images(before downsampling)
    shape = header['ITK_FileNotes'][0][-14:-1]
    file_name = header['ITK_FileNotes'][0][0:-27]
    image_real_size = [int(shape[0:3]),int(shape[5:8]),int(shape[10:13])]
    plt.imshow(image[1,1,100,:,:].squeeze())
    plt.show()
    resized_image = zoom(image, (1,1, image_real_size[0]/image.shape[2],
            image_real_size[1]/image.shape[3], image_real_size[2]/image.shape[4]))
    
# ######## masking with 7 values
#     myList = [0, 420, 550, 205, 850, 500, 820, 600]
#     resized_image = resized_image.squeeze()
#     data1 = resized_image.reshape((image_real_size[0]*image_real_size[1]*image_real_size[2]))
#     mask_func = lambda myList,myNumber: min(myList, key=lambda x:abs(x-myNumber))
#     data2 = np.array([mask_func(myList,i) for i in data1 ])
#     resized_image = data2.reshape(image_real_size)

#saving with dicom format in the output directory
    img = sitk.GetImageFromArray(resized_image)
    for key in header:
        img.SetMetaData(key,header[key][0])
    sitk.WriteImage(img, os.path.join(target_test_mask_dir, file_name +'mask.nii.gz'))
    print("{}' mask saved successfully!".format( file_name))

##################deploy
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(device)

target_dataset_path = '../data/'
dataset_file_name = 'CT_dataset.np'
experiment_path = '../data/models/experiment{}/'.format(experiment_number)
target_test_mask_dir = target_dataset_path + 'deploy'

img_resized = [32, 96, 96]

batch_size = 1
lr_rate = 0.0001
lr_gamma = 1
lr_milestones = [1,1]
amsgrad = False
weight_decay = 0.01

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(device)
# torch.device("cpu")
Path(experiment_path).mkdir(parents=True, exist_ok=True)

transforms1 = torchvision.transforms.Compose([PairResize(img_resized, mode = 'test'),
    
    ToTensor(mode = 'test')
])
dataset_test = Data_Handler(target_dataset_path + dataset_file_name, 'test', transforms1)
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

# instantiating model and experiment class
train_step = train_network(1,1,device,  lr_rate,  amsgrad,  weight_decay)

# load model
training_losses = []
validation_losses = []
dice_losses = []
min_val_loss = float('inf')
    
start_epoch = 0
training_time = 0
state_filename = experiment_path + "state_last.pt"
if os.path.isfile(state_filename):
    checkpoint = torch.load(state_filename)
    start_epoch = checkpoint['epoch']
    train_step.netD.load_state_dict(checkpoint['state_dict_D'])
    train_step.netG.load_state_dict(checkpoint['state_dict_G'])
    train_step.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
    train_step.optimizer_D.load_state_dict(checkpoint['optimizer_D'])
    training_losses = checkpoint['training_losses']
    validation_losses = checkpoint['validation_losses']
    dice_losses = checkpoint['dice_losses']
    min_val_loss = checkpoint['min_val_loss']
    print("=> loaded checkpoint '{}' (epoch {})"
              .format(state_filename, checkpoint['epoch']))
else:
    print("=> no checkpoint found at '{}'".format(state_filename))
        
########################
### masking test set ###
########################

model.eval()
with torch.no_grad():
    for batch in Bar(dataloader_test):
        imgs = batch['image'].to(device=device, dtype=torch.float32)
        yhat = train_step.netG(imgs)
        save_image(yhat,batch['header'],target_test_mask_dir)

    
