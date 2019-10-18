# denoising-autoencoder
Autoencoder for denoising images

This autoencoder helps de-noise images using convolutional layers.

For the training process, we use noisy images as input and the original, clean images as targets.

The model architecture is as follows:

class ConvDenoiser(nn.Module):
    def __init__(self):
        super(ConvDenoiser, self).__init__()
        ## encoder layers ##
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 16, 3, 1, 1)
        self.conv3 = nn.Conv2d(16, 8, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)

        ## decoder layers ##
        ## kernel of 2 and stride of 2 increases the spatial dimensions by 2
        self.t_conv0 = nn.ConvTranspose2d(8, 8, 3, stride = 2)
        self.t_conv1 = nn.ConvTranspose2d(8, 16, 2, stride = 2)
        self.t_conv2 = nn.ConvTranspose2d(16, 32, 2, stride = 2)
        self.conv_out = nn.Conv2d(32, 1, 3, padding = 1)

    def forward(self, x):
        ## encode ##
        x = self.pool(F.relu(self.conv1(x)))  # size 14 x 14 x 32
        x = self.pool(F.relu(self.conv2(x)))  # size 7 x 7 x 16
        x = self.pool(F.relu(self.conv3(x)))

        ## decode ##
        x = F.relu(self.t_conv0(x))           
        x = F.relu(self.t_conv1(x))           # size 14 x 14 x 16
        x = F.relu(self.t_conv2(x))           # size 28 x 28 x 32
        x = F.sigmoid(self.conv_out(x))        # size 28 x 28 x 1
        return x
        
We use the MSE Loss function and the Adam optimizer.

Standard dependencies exist:
Torch, Numpy, Torchvision, Transforms, Matplotlib, Torch.nn, Torch.nn.functional
