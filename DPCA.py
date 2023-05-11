import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils, datasets, transforms
import time
import scipy
from numpy.linalg import inv
import sys
import matplotlib.pyplot as plt
from PIL import Image

from model import CRCA, STEM, generate_featuremaps, CPCA
from utils import plot, view_image
# from utils import plot, plot_faces, plot_faces_2
# from data import load_data, load_faces

def run_cifar(args):
    # load cifar-10 dataset
    kwargs = {}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./datasets/data_cifar10', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=10, shuffle=True, **kwargs)

    # extract one batch
    for idx, (data, target) in enumerate(train_loader):
        if idx >= 1:
            break
    target = F.one_hot(target, 10)

    # data: shape [batch, 3, H, W] where 3 means RGB channels, batch=10, H=W=32
    # target: shape [batch, 10] where 10 means 10 classes, batch=10
    if args.model == 'PCA':
        targets = data.permute(0,2,3,1).reshape(-1, 3).float().t()
        stem, weights = CPCA(data, targets, num_components=args.num_components) #(10240,27), (27,2)
        featuremaps = generate_featuremaps(stem, weights, False)

    elif args.model == 'RCA':
        stem,weights = CRCA(data, target, num_components=args.num_components)
        featuremaps = generate_featuremaps(stem, weights, True)

    plot(data.permute(0,2,3,1))  # plot original images, permute to have shape (batch, H, W, 3)

    plot(featuremaps) # plot feature maps, shape (batch, H, W, 3)

def run_single_img(args):
    # load babyface image
    img = Image.open('results/baby_mini_d3_gaussian.jpg')
    img_data = torch.from_numpy(np.asarray(img).astype(np.float32))
    img_data = img_data.unsqueeze(0)/255
    img_data = img_data*2 - 1 # normalize to [-1, 1]
    img_data = img_data.permute(0, 3, 1, 2) #shape [batch, C, H, W]

    B, C, H, W = img_data.shape # data: shape [batch, 3, H, W] where 3 means RGB channels, batch=10, H=W=32

    compressed_img = torch.nn.functional.interpolate(img_data, scale_factor=0.5, mode='bilinear')
    X_data = torch.nn.functional.interpolate(compressed_img, scale_factor=2, mode='nearest')

    if args.model == 'PCA':
        targets = img_data.permute(0,2,3,1).reshape(-1, C).float().t() # (3, 170*170)
        stem, weights = CPCA(X_data, targets, num_components=args.num_components) #(170*170,27), (27,2)
        featuremaps = torch.matmul(weights, stem.t()).reshape(C, B, H, W).permute(1,2,3,0) # from (C,BHW) to (B,H,W,C)
    else:
        raise NotImplementedError

    view_image(img_data[0])  # plot original images
    view_image(torch.clamp(featuremaps[0].permute(2, 0, 1), min=-1.0, max=1.0)) # visualize the first image in the batch (optional)

def run_faces(img):
    # load babyface image
    # img = Image.open('results/baby_mini_d3_gaussian.jpg')
    # img_data = torch.from_numpy(np.asarray(img).astype(np.float32))
    img_data = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)
    img_data = img_data.unsqueeze(0)/255  #shape [batch, C, H, W]
    img_data = img_data*2 - 1 # normalize to [-1, 1]

    B, C, H, W = img_data.shape # data: shape [batch, 3, H, W] where 3 means RGB channels, batch=10, H=W=32

    compressed_img = torch.nn.functional.interpolate(img_data, scale_factor=0.5, mode='bilinear')
    X_data = torch.nn.functional.interpolate(compressed_img, scale_factor=2, mode='nearest')

    targets = img_data.permute(0,2,3,1).reshape(-1, C).float().t() # (1, 112*92)
    stem, weights = CPCA(X_data, targets, 1, channel=1) #(112*92,9), (1,9)
    featuremaps = torch.matmul(weights, stem.t()).reshape(B, H, W)

    restored_img = torch.clamp(featuremaps[0].squeeze(0), min=-1.0, max=1.0)
    restored_img = restored_img / 2 + 0.5
    return restored_img.cpu().data.numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument('--model', type=str, default='PCA', help='choose between PCA and RCA')
    parser.add_argument('--num_components', type=int, default=2, help='choose between 1,2,3')
    parser.add_argument('--dataset', type=str, default='cifar', help='choose between single_img, cifar and faces')
    args = parser.parse_args()
    run_cifar(args)


