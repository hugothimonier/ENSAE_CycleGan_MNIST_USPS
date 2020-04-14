
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

import solver
import utilities

def sample_noise(batch_size, dim):
    """
    Generate a PyTorch Tensor of uniform random noise.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - dim: Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Tensor of shape (batch_size, dim, 1, 1) containing uniform
      random noise in the range (-1, 1).
    """
    return to_var(torch.rand(batch_size, dim) * 2 - 1).unsqueeze(2).unsqueeze(3)

def upconv(in_channels, out_channels, kernel_size, stride=2, padding=2, batch_norm=True):
    """Creates a upsample-and-convolution layer, with optional batch normalization.
    """
    layers = []
    if stride>1:
      layers.append(nn.Upsample(scale_factor=stride))
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
    layers.append(conv_layer)
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


def conv(in_channels, out_channels, kernel_size, stride=2, padding=2, batch_norm=True, init_zero_weights=False):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    if init_zero_weights:
        conv_layer.weight.data = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.001
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)
  
class ResnetBlock(nn.Module):
    def __init__(self, conv_dim):
        super(ResnetBlock, self).__init__()
        self.conv_layer = conv(in_channels=conv_dim, out_channels=conv_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = x + self.conv_layer(x)
        return out
  
class DCDiscriminator(nn.Module):
    """Defines the architecture of the discriminator network.
       Note: Both discriminators D_X and D_Y have the same architecture in this assignment.
    """
    def __init__(self, conv_dim=64):
        super(DCDiscriminator, self).__init__()
            # input is (nc) x 32 x 32 (in our case nc = 1 because no colors)
        self.conv1 = conv(nc, conv_dim, kernel_size = 5, stride = 2 , padding = 2)
            # state size. (conv_dim) x 16 x 16
        self.conv2 = conv(conv_dim, conv_dim * 2, kernel_size = 5, stride = 2 , padding = 2)
            # state size. (conv_dim*2) x 8 x 8
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, kernel_size = 5, stride = 2 , padding = 2)
            # state size. (conv_dim*4) x 4 x 4
        self.conv4 = conv(conv_dim * 4, 1,  kernel_size = 4, stride = 1 , padding = 0)


    def forward(self, x):
        batch_size = x.size(0)

        out = F.relu(self.conv1(x))    # BS x 64 x 16 x 16
        out = F.relu(self.conv2(out))    # BS x 64 x 8 x 8
        out = F.relu(self.conv3(out))    # BS x 64 x 4 x 4

        out = self.conv4(out).squeeze()
        out_size = out.size()
        if out_size != torch.Size([batch_size,]):
          raise ValueError("expect {} x 1, but get {}".format(batch_size, out_size))
        return out


class CycleGenerator(nn.Module):
    """Defines the architecture of the generator network.
       Note: Both generators G_XtoY and G_YtoX have the same architecture in this assignment.
    """
    def __init__(self, conv_dim=64, init_zero_weights=False):
        super(CycleGenerator, self).__init__()

        ###########################################
        ##   FILL THIS IN: CREATE ARCHITECTURE   ##
        ###########################################

        # 1. Define the encoder part of the generator (that extracts features from the input image)
        # self.conv1 = conv(...)
        self.conv1 = conv(1, conv_dim, kernel_size = 5, stride = 2 , padding = 2)
        self.conv2 = conv(conv_dim, conv_dim * 2, kernel_size = 5, stride = 2 , padding = 2)

        # 2. Define the transformation part of the generator
        self.resnet_block = ResnetBlock(conv_dim * 2)

        # 3. Define the decoder part of the generator (that builds up the output image from features)
        self.upconv1 = upconv(conv_dim * 2, conv_dim,kernel_size= 5, stride = 2, padding = 3)
        self.upconv2 = upconv(conv_dim, 1, kernel_size= 5, stride = 2, padding = 0, batch_norm=False)

    def forward(self, x):
        """Generates an image conditioned on an input image.

            Input
            -----
                x: BS x 3 x 32 x 32

            Output
            ------
                out: BS x 3 x 32 x 32
        """
        batch_size = x.size(0)
        
        out = F.relu(self.conv1(x))            # BS x 32 x 16 x 16
        out = F.relu(self.conv2(out))          # BS x 64 x 8 x 8
        
        out = F.relu(self.resnet_block(out))   # BS x 64 x 8 x 8

        out = F.relu(self.upconv1(out))        # BS x 32 x 16 x 16
        out = torch.tanh(self.upconv2(out))        # BS x 1 x 32 x 32
        
        out_size = out.size()
        if out_size != torch.Size([batch_size, 1, 32, 32]):
          raise ValueError("expect {} x 1 x 32 x 32, but get {}".format(batch_size, out_size))


        return out