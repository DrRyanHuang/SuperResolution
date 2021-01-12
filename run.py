# 该程序只适合单卡

import argparse
import os
import sys
import numpy as np

from PIL import Image

import torch
import torch.optim as optim
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import models
from torch.autograd import Variable

import model_utils
import model as model_
# from model import ResidualBlock, SuperRes4x, UpsampleBlock


def train_SuperRes(model, optimizer, loader, device, epochs=1, use_vgg_loss=True, save_best=True):

    model.train()

    loss_fn = nn.MSELoss(reduction='sum')

    best_loss = 1e6

    epoch = [1]
    while epoch[0] <= epochs:

        epoch_loss = 0

        for iteration, batch in enumerate(loader, 1):
            inp, tgt = Variable(batch[0]).to(device), Variable(batch[1]).to(device)

            # Forward
            pred = model(inp)

            # Compute loss
            if use_vgg_loss:
                vgg_loss_inp = vgg_loss(pred)
                vgg_loss_tgt = vgg_loss(tgt)
                loss = loss_fn(vgg_loss_inp, vgg_loss_tgt)
            else:
                loss = loss_fn(pred, tgt)

            # Update loss
            epoch_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iteration % 100 == 0:
                print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(loader), loss.item()))

        if epoch[0] % 1 == 0:
            print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(loader)))

        if save_best and epoch_loss < best_loss:
            print("Saving model at epoch: {}".format(epoch))
            torch.save(model.state_dict(), "best_model.pth")
            best_loss = epoch_loss

        epoch[0] += 1


def arg_parse():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode',
        type=str,
        default='upsample',
        help='Flag for Training or Upsampling. Valid entries are \'train\' and \'upsample\' '
    )
    parser.add_argument(
        '--image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images will be read for training or saved for upsampling.'
    )
    parser.add_argument(
        '--hi_res_size',
        type=int,
        nargs='?',
        default=256,
        help='Size of hi_res image for training mode. Default is 256.'
    )
    parser.add_argument(
        '--trn_epochs',
        type=int,
        nargs='?',
        default=5,
        help='Number of training epochs. Default is 5.'
    )
    parser.add_argument(
        '--l_rate',
        type=float,
        nargs='?',
        default=1e-3,
        help='Learning rate for training. Default is 1e-3.'
    )
    parser.add_argument(
        '--target_image',
        type=str,
        nargs='?',
        default='',
        help='Path to target image to upsample.'
    )
    parser.add_argument(
        '--model',
        type=str,
        nargs='?',
        default='',
        help='Path to saved model file.'
    )
    parser.add_argument(
        '--gpu-id',
        type=int,
        nargs='?',
        default=1,
        help='GPU device id for training. Default is 0'
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    args = arg_parse()
    if args.mode not in ['train', 'upsample']:
        print ("Please choose \'train\' or \'upsample\' as mode")
        sys.exit()

    if args.mode == 'train':
        print ("\n *** INITIALIZING *** ")
        if args.image_folder == '':
            print ("Please specify folder with training input images\n")
            sys.exit()
        else:
            print ("running in mode: {}".format(args.mode))
            print ("training folder: {} \n".format(args.image_folder))
        
        # Define if flag to use GPU and dtype for variables
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:{}".format(args.gpu_id) if use_cuda else "cpu")

        # Load training dataset
        train_dataset = model_utils.get_training_set(path=args.image_folder,
                                                     hr_size=args.hi_res_size,
                                                     upscale_factor=4)
        train_data_loader = DataLoader(dataset=train_dataset,
                                       num_workers=0,
                                       batch_size=(8 if use_cuda else 1),
                                       shuffle=True,
                                       drop_last=True)


        # Create VGG model for loss function
        vgg16 = models.vgg16(pretrained=True).features
        vgg16.to(device)

        vgg_loss = model_.create_loss_model(vgg16, 8, device)

        for param in vgg_loss.parameters():
            param.requires_grad = False # 也就是这里的VGG参数不更新

        model = model_.SuperRes4x(device)
        optimizer = optim.Adam(model.parameters(), lr=args.l_rate)
        train_SuperRes(model, optimizer, train_data_loader, device,
                       epochs=args.trn_epochs, use_vgg_loss=True)

    if args.mode == 'upsample':
        print ("\n *** INITIALIZING *** ")
        if args.target_image == '':
            print ("Please specify image to upsample\n")
            sys.exit()
        if args.model == '':
            print ("Please specify path to saved model\n")
            sys.exit()
        else:
            print ("running in mode: {}".format(args.mode))
            print ("upsample image: {}".format(args.target_image))
            print("using model: {} \n".format(args.model))

        use_cuda = torch.cuda.is_available()
        dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

        # SuperRes4x__ = model_.SuperRes4x(use_cuda=use_cuda)
        # from model import *
        # SuperRes4x__.load_state_dict(torch.load(args.model))

        image = model_utils.image_loader(args.target_image).type(dtype)
        upsampled = model(image)

        if use_cuda:
            upsampled = upsampled.cpu().data.numpy().squeeze()
        else:
            upsampled = upsampled.data.numpy().squeeze()

        upsampled = np.swapaxes(upsampled, 0, 2)
        upsampled = np.swapaxes(upsampled, 0, 1)
        upsampled = np.array(upsampled * 255, dtype=np.uint8)
        upsampled = Image.fromarray(upsampled, 'RGB')
        upsampled.save('up_' + args.target_image.split('/')[-1])


'''
# 你在 spyder 里的调试信息
debugfile('/home/ryan/code/SuperResolution/run.py', 
          wdir='/home/ryan/code/SuperResolution', 
          args='--mode train --image_folder /home/ryan/Dataset/VOCdevkit/VOC2007/JPEGImages --trn_epochs 1 --l_rate 0.01')
'''

