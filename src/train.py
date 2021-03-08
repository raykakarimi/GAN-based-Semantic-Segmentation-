%load_ext autoreload
%autoreload 2

import os
import numpy as np
from pathlib import Path
from data_prep import data_prep
from make_dataset import make_dataset
import matplotlib.pyplot as plt
from utils.transforms import PairResize, ToTensor
import torchvision
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.optim as optim
from torchsummary import summary
from barbar import Bar
from data_handler import Data_Handler
from experiment import experiment
from model.unet3d import UNet3D
from experiment import train_network

source_dataset_path = '../'
source_dataset_train_dir = 'ct_train'
source_dataset_test_dir = 'ct_test'
source_image_suffix = 'image'
source_label_suffix = 'label'

target_dataset_path = '../data/'
target_dataset_train_dir = 'train'
target_dataset_test_dir = 'test'
target_header_prefix = 'hd_p'
target_image_prefix = 'img_p'
target_mask_prefix = 'msk_p'

validation_set_ratio = 0.2
dataset_file_name = 'CT_dataset.np'

experiment_number = 1
experiment_path = '../data/models/experiment{}/'.format(experiment_number)
# experiment_path = '../data/models/experiment{}/'.format(experiment_number)

img_resized = [32, 96, 96]

n_epochs = 1000
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

################
#data prepration
################

# Preparing train data
data_prep(source_dataset_path + source_dataset_train_dir, target_dataset_path + target_dataset_train_dir,
          source_image_suffix, source_label_suffix, target_image_prefix, target_mask_prefix, target_header_prefix)

# Preparing test data
data_prep(source_dataset_path + source_dataset_test_dir, target_dataset_path + target_dataset_test_dir,
          source_image_suffix, source_label_suffix, target_image_prefix, target_mask_prefix, target_header_prefix)

# Save data absolute paths in CT_dataset.np
make_dataset(target_dataset_path, target_dataset_train_dir, target_dataset_test_dir, target_header_prefix,
             target_image_prefix, target_mask_prefix, validation_set_ratio, dataset_file_name)

#data handler and data loader call

transforms = torchvision.transforms.Compose([PairResize(img_resized),
    
    ToTensor()
])

transforms1 = torchvision.transforms.Compose([PairResize(img_resized, mode = 'test'),
    
    ToTensor(mode = 'test')
])


dataset_train = Data_Handler(target_dataset_path + dataset_file_name, 'train', transforms)
dataset_val = Data_Handler(target_dataset_path + dataset_file_name, 'val', transforms)
dataset_test = Data_Handler(target_dataset_path + dataset_file_name, 'test', transforms1)

dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

# instantiating model and experiment class
train_step = train_network(1,1,device,  lr_rate,  amsgrad,  weight_decay)

##############################loading and training model (move this section to experiment class)
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
    
###training phase
n_epochs -= start_epoch

for epoch in range(n_epochs):  
    batch_losses = []

    for batch in Bar(dataloader_train):
        train_step.set_input(batch)
        loss, score = train_step.optimize_parameters()
        batch_losses.append(loss)
    training_loss = np.mean(batch_losses)
    training_losses.append(training_loss)

    with torch.no_grad():
        val_losses = []
        for batch in Bar(dataloader_val):
            imgs = batch['image'].to(device=device, dtype=torch.float32)
            true_masks = batch['mask'].to(device=device, dtype=torch.float32)
            train_step.netG.eval()
            yhat = train_step.netG(imgs)
            # val_loss = train_step.calculate_loss(true_masks, yhat, imgs)
            val_loss=0
            validation_losses.append(val_loss)
            dice_loss = dice_loss_func(yhat,true_masks).item()
            dice_losses.append(dice_loss)
        dice_loss = np.mean(dice_losses)
        dice_losses.append(dice_loss)
        val_loss = np.mean(validation_losses)
        validation_losses.append(val_loss)
    
    if (dice_loss < min_val_loss):
        min_val_loss = dice_loss
        
        best_state = {'epoch': start_epoch + epoch + 1, 'state_dict_G': train_step.netG.state_dict(),'state_dict_D': train_step.netD.state_dict(), 'optimizer_G': train_step.optimizer_G.state_dict(),'optimizer_D': train_step.optimizer_D.state_dict(),
                      'validation_losses': validation_losses,'dice_losses': dice_losses, 'training_losses': training_losses, 'min_val_loss': min_val_loss}
        
        torch.save(best_state, experiment_path + "state_best.pt")
        
        print(f"[{start_epoch + epoch + 1}] Learning Rate: {train_step.optimizer_G.param_groups[0]['lr']} Training loss: {training_loss:.5f}\t Validation loss: {val_loss:.5f}\t dice loss: {dice_loss:.5f}\t Saved!")
    else:
        print(f"[{start_epoch + epoch + 1}] Learning Rate: {train_step.optimizer_G.param_groups[0]['lr']} Training loss: {training_loss:.5f}\t Validation loss: {val_loss:.5f}\t dice loss: {dice_loss:.5f}")
    
    if (epoch%5 == 0):
        last_state = {'epoch': start_epoch + epoch + 1, 'state_dict_G': train_step.netG.state_dict(),'state_dict_D': train_step.netD.state_dict(), 'optimizer_G': train_step.optimizer_G.state_dict(),'optimizer_D': train_step.optimizer_D.state_dict(),
                      'validation_losses': validation_losses,'dice_losses': dice_losses, 'training_losses': training_losses, 'min_val_loss': min_val_loss}
        torch.save(last_state, experiment_path + "state_last.pt")
        


if(n_epochs > 0):
      best_state = {'epoch': start_epoch + epoch + 1, 'state_dict_G': train_step.netG.state_dict(),'state_dict_D': train_step.netD.state_dict(), 'optimizer_G': train_step.optimizer_G.state_dict(),'optimizer_D': train_step.optimizer_D.state_dict(),
                      'validation_losses': validation_losses,'dice_losses': dice_losses, 'training_losses': training_losses, 'min_val_loss': min_val_loss}
      torch.save(last_state, experiment_path + "state_last.pt")
    
      print('Training Time: {}s'.format(np.round(training_time, 2)))
      if os.path.isfile(experiment_path + "report.txt"):
        text_file = open(experiment_path + "report.txt", "a")
        report_text = "training time: {}s\n".format(np.round(training_time, 2))
        text_file.write(report_text)
        text_file.close()
##############################

#sketching figure of validation loss and training loss across  #epochs
val_loss=list(np.subtract([1] * len(experiment1.validation_losses),(.experiment1validation_losses)))
train_loss=list(np.subtract([1] * len(experiment1.training_losses),(experiment1.training_losses)))
fig = plt.figure()
plt.plot(train_loss)
plt.plot(val_loss)
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
fig.savefig(experiment_path + 'train-history.png')