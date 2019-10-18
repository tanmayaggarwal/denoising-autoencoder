import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
# % matplotlib inline
import torch.nn as nn
import torch.nn.functional as F

# import the datasets

from import_visualize_datasets import import_datasets
train_loader, test_loader, batch_size = import_datasets()

# visualize the data

from import_visualize_datasets import visualize_data
visualize_data(train_loader)

# build the denoising autoencoder network

from conv_denoiser import ConvDenoiser
model = ConvDenoiser()

# train the network

from train import train_model
train_model(train_loader, model)

# checking the results
check_results(test_loader, batch_size)
