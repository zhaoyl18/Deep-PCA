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

from DPCA import run_single_img, run_cifar, run_faces


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument('--model', type=str, default='PCA', help='choose between PCA and RCA')
    parser.add_argument('--num_components', type=int, default=2, help='choose between 1,2,3')
    parser.add_argument('--dataset', type=str, default='single_img', help='choose between single_img, cifar and faces')
    args = parser.parse_args()

    if args.dataset == 'cifar':
        run_cifar(args)
    elif args.dataset == 'single_img':
        run_single_img(args)
    elif args.dataset == 'faces':
        run_faces(args)