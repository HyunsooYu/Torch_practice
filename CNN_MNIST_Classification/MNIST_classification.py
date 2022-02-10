# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 17:16:40 2022

@author: BMCL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from matplotlib import pyplot as plt

is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')

print('Current cuda device is ', device)

# set Hyperparameter

batch_size = 50
epoch_num = 15
learning_rate = 0.0001

# Load MNIST data
train_data = datasets.MNIST(root = './data', train = True, download = True, transform = transforms.ToTensor())
test_data = datasets.MNIST(root = './data', train = False, download = True, transform = transforms.ToTensor())

print('number of training data: ', len(train_data))
print('number of test data :', len(test_data))

