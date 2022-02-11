# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 12:01:14 2022

@author: BMCL
"""
# Make folders

import os
import shutil

original_dataset_dir = './dataset'
classes_list = os.listdir(original_dataset_dir)

base_dir = './splitted'
os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'val')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

for clss in classes_list:
    os.mkdir(os.path.join(train_dir, clss))
    os.mkdir(os.path.join(validation_dir, clss))
    os.mkdir(os.path.join(test_dir,clss))
    
# Assign all data

import math

for clss in classes_list:
    path = os.path.join(original_dataset_dir, clss)
    fnames = os.listdir(path)
    
    train_size = math.floor(len(fnames) * 0.6)
    validation_size = math.floor(len(fnames) *0.2)
    test_size = math.floor(len(fnames)*0.2)
    
    train_fnames = fnames[:train_size]
    print('train size(',clss, '): ', len(train_fnames))
    for fname in train_fnames:
        src = os.path.join(path, fname)
        dst = os.path.join(os.path.join(os.path.join(train_dir,clss), fname))
        shutil.copyfile(src, dst)
    
    validation_fnames = fnames[train_size:(validation_size + train_size)]
    print('Validation size(', clss, '): ', len(validation_fnames))
    for fname in validation_fnames:
        src = os.path.join(path, fname)
        dst = os.path.join(os.path.join(validation_dir, clss), fname)
        shutil.copyfile(src, dst)
        
    test_fnames = fnames[(train_size + validation_size):
                         (validation_size + train_size + test_size)]
    
    print('Test size(', clss, '): ', len(test_fnames))
    for fname in test_fnames:
        src = os.path.join(path, fname)
        dst = os.path.join(os.path.join(test_dir, clss), fname)
        shutil.copyfile(src,dst)
        
# Preparation in learning model
import torch

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

BATCH_SIZE = 256
EPOCH = 30

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

transform_base = transforms.Compose([transforms.Resize((64,64,)), transforms.ToTensor()])
train_dataset = ImageFolder(root = './splitted/train', transform = transform_base)
val_dataset = ImageFolder(root = './splitted/val', transform = transform_base)

from torch.utils.data import DataLoader

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers =4)
val_loader =torch.utils.data.DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle=True, num_workers = 4)


# Make base
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(3,32,3,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32,64,3, padding=1)
        self.conv3 = nn.Conv2d(64,64,3, padding=1)
        
        self.fc1 = nn.Linear(4096, 512)
        self.fc2 =nn.Linear(512,33)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = F.dropout(x, p=0.25, training=self.training)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = F.dropout(x, p=0.25, training=self.training)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)
        x = F.dropout(x, p=0.25, training=self.training)
        
        x = x.view(-1, 4096)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training = self.training)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)
    
model_base = Net().to(DEVICE)
optimizer = optim.Adam(model_base.parameters(), lr=0.001)


# Train model
def train(model, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        if data.shape[0] != BATCH_SIZE:
            continue
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
# Evaluate model
def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100.*correct/len(test_loader.dataset)
    return test_loss, test_accuracy


    
import time
import copy

def train_baseline(model, train_loader, val_loader, optimizer, num_epochs = 30):
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(1, num_epochs +1):
        since = time.time()
        train(model, train_loader, optimizer)
        train_loss, train_acc = evaluate(model, train_loader)
        val_loss ,val_acc = evaluate(model, val_loader)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            
        time_elapsed = time.time() - since
        print('------------epoch {} ---------------'.format(epoch))
        print('train Loss : {:.4f}, Accuracy: {:.2f}%'.format(train_loss, train_acc))
        print('val Loss : {:.4f}, Accuracy: {:.2f}%'.format(val_loss, val_acc))
        print('Completed in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
    model.load_state_dict(best_model_wts)
    return model
base = train_baseline(model_base, train_loader, val_loader, optimizer, EPOCH)
torch.save(base, 'bseline.pt')
    
