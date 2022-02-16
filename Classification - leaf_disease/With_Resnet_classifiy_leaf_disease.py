# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 12:01:14 2022

@author: BMCL
"""
# Make folders

# Preparation in learning model
import os
import torch

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

BATCH_SIZE = 256
EPOCH = 30

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize([64,64]), 
        transforms.RandomHorizontalFlip(), 
        transforms.RandomVerticalFlip(), 
        transforms.RandomCrop(52),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
        
        ]), 
    'val': transforms.Compose([
        transforms.Resize([64,64]), 
        transforms.RandomCrop(52),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
        ])
    }

data_dir = './splitted'
image_datasets = {x: ImageFolder(root = os.path.join(data_dir, x), 
                                 transform = data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], 
                                              batch_size = BATCH_SIZE, shuffle=True, num_workers==4)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

class_names = image_datasets['train'].classes

from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

resnet = models.resnet50(pretrained=True)
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 33)
resnet = resnet.to(DEVICE)

crterion = nn.CrossEntropyLoss()

optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, resnet.parameters()), lr=0.001)
# Only parameter which require gradient will be optimized

from torch.optim import lr_scheduler

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size = 7, gamma = 0.1)

ct = 0
for child in resnet.children():
    ct += 1
    if ct <6:
        for param in child.parameters():
            param.requires_grad = False
            



    
    
    
    
    
    
    
    
    
    
    
    
    
    