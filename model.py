import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

"""

PyTorch implementation of:
[1] [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)

With a few modifications as suggested by:
[2] [Deconvolution and Checkerboard Artifacts](http://distill.pub/2016/deconv-checkerboard/)

"""


def create_loss_model(vgg, end_layer, device, use_maxpool=True):
    """
        [1] uses the output of vgg16 relu2_2 layer as a loss function (layer8 on PyTorch default vgg16 model).
        This function expects a vgg16 model from PyTorch and will return a custom version up until layer = end_layer
        that will be used as our loss function.
    """

    vgg = copy.deepcopy(vgg)

    model = nn.Sequential()
    model.to(device)

    i = 0
    for layer in list(vgg):

        if i > end_layer:
            break

        if isinstance(layer, nn.Conv2d):
            name = "conv_" + str(i)
            model.add_module(name, layer) # 传的是引用!!!

        if isinstance(layer, nn.ReLU):
            name = "relu_" + str(i)
            model.add_module(name, layer)

        if isinstance(layer, nn.MaxPool2d):
            name = "pool_" + str(i)
            if use_maxpool:
                model.add_module(name, layer)
            else:
                avgpool = nn.AvgPool2d(kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding)
                model.add_module(name, avgpool)
        i += 1
    return model


class ResidualBlock(nn.Module):
    """ Residual blocks as implemented in [1] """
    def __init__(self, num, device):
        super(ResidualBlock, self).__init__()
        self.c1 = nn.Conv2d(num, num, kernel_size=3, stride=1, padding=1).to(device)
        self.c2 = nn.Conv2d(num, num, kernel_size=3, stride=1, padding=1).to(device)
        self.b1 = nn.BatchNorm2d(num).to(device)
        self.b2 = nn.BatchNorm2d(num).to(device)

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return h + x


class UpsampleBlock(nn.Module):
    """ Upsample block suggested by [2] to remove checkerboard pattern from images """
    def __init__(self, num, device):
        super(UpsampleBlock, self).__init__()
        self.up1 = nn.UpsamplingNearest2d(scale_factor=2).to(device) # in favor of interpolate
        self.c2 = nn.Conv2d(num, num, kernel_size=3, stride=1, padding=0).to(device)
        self.b3 = nn.BatchNorm2d(num).to(device)


    def forward(self, x):
        h = self.up1(x)
        h = F.pad(h, (1, 1, 1, 1), mode='reflect')
        h = self.b3(self.c2(h))
        return F.relu(h)


class SuperRes4x(nn.Module):
    def __init__(self, device, use_UpBlock=True):

        super(SuperRes4x, self).__init__()
        # To-do: Retrain with self.uplock and self.use_cuda as parameters

        # self.upblock = use_UpBlock
        # self.use_cuda = use_cuda
        upblock = True

        # Downsizing layer
        self.c1 = nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4).to(device)
        self.b2 = nn.BatchNorm2d(64).to(device)

        if upblock:
            # Loop for residual blocks
            self.rs = nn.ModuleList([ResidualBlock(64, device=device) for i in range(4)])
            # Loop for upsampling
            self.up = nn.ModuleList([UpsampleBlock(64, device=device) for i in range(2)])
        else:
            # Loop for residual blocks
            self.rs = nn.ModuleList([ResidualBlock(64, device=device) for i in range(4)])
            # Transposed convolution blocks
            self.dc2 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1).to(device)
            self.bc2 = nn.BatchNorm2d(64).to(device)
            self.dc3 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1).to(device)
            self.bc3 = nn.BatchNorm2d(64).to(device)

        # Last convolutional layer
        self.c3 = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4).to(device)


    def forward(self, x):
        upblock = True
        # Downsizing layer - Large Kernel ensures large receptive field on the residual blocks
        h = F.relu(self.b2(self.c1(x)))

        # Residual Layers
        for r in self.rs:
            h = r(h)  # will go through all residual blocks in this loop

        if upblock:
            # Upsampling Layers - improvement suggested by [2] to remove "checkerboard pattern"
            for u in self.up:
                h = u(h)  # will go through all upsampling blocks in this loop
        else:
            # As recommended by [1]
            h = F.relu(self.bc2(self.dc2(h)))
            h = F.relu(self.bc3(self.dc3(h)))

        # Last layer and scaled tanh activation - Scaled from 0 to 1 instead of 0 - 255
        h = torch.tanh(self.c3(h))
        h = torch.add(h, 1.)
        h = torch.mul(h, 0.5)
        return h
