import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
# % matplotlib inline
import torch.nn as nn
import torch.nn.functional as F

# training the denoising autoencoder

def train_model(train_loader, model):
    # specify the loss function
    criterion = nn.MSELoss()  # using MSELoss given we are comparing input vs. output pixels (i.e., regression) as compared to probabilities
    # specify the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    # number of epochs to train the model
    n_epochs = 20

    # for adding noise to images
    noise_factor = 0.5

    for epoch in range(1, n_epochs+1):
        # monitor training loss
        train_loss = 0.0

        ###################
        # train the model #
        ###################
        for data in train_loader:
            # _ stands in for labels, here
            # no need to flatten images
            images, _ = data

            ## add random noise to the input images
            noisy_imgs = images + noise_factor * torch.randn(*images.shape)
            # Clip the images to be between 0 and 1
            noisy_imgs = np.clip(noisy_imgs, 0., 1.)

            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            ## forward pass: compute predicted outputs by passing *noisy* images to the model
            outputs = model(noisy_imgs)
            # calculate the loss
            # the "target" is still the original, not-noisy images
            loss = criterion(outputs, images)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item()*images.size(0)

        # print avg training statistics
        train_loss = train_loss/len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch,
            train_loss
            ))

    return
