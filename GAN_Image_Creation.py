# Deep Convolutional Generative Adversarial Network - input by Ryan L Buchanan 25AUG20

# Importing the libraries
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

# Setting some hyperparameters
batchSize = 64
imageSize = 64

# Creating the transformations
transform = transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

# Loading the dataset
dataset = dset.CIFAR10(root = './data', download = True, transform = transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle = True, num_workers = 2) 

# Defining the weights_init function that takes as an input a neural network m and that will initialize all of its weights
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal__(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal__(1.0, 0.02)
        m.bias.data.fill__(0)
    
# Defining the generator
