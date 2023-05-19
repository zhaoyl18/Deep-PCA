import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import time
import scipy
from numpy.linalg import inv
import sys
import matplotlib.pyplot as plt

def main(args):
    # load cifar-10 dataset
    kwargs = {}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data_cifar10', train=True, download=True,
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
        stem,components = CPCA(data, number_components=2) 

    elif args.model == 'RCA':
        stem,components = CRCA(data, target, number_components=2)

    # TODO: give proper input arguments yourself
    leading_component = get_leading_component()

    # TODO: give proper input arguments yourself
    featuremaps = generate_featuremaps()

    plot(data.permute(0,2,3,1))  # plot original images, permute to have shape (batch, H, W, 3)

    plot(featuremaps) # plot feature maps, shape (batch, H, W, 1)

def plot(data):
    fig, axs = plt.subplots(3, 3)
    for id in range(3*3):
        if id<data.shape[0]:
            axs[id//3, id%3].imshow(data[id].cpu().data.numpy())
        axs[id // 3, id % 3].set_xticks([])
        axs[id // 3, id % 3].set_yticks([])
    plt.show()

def CPCA(inputs, number_components):
    stem = STEM(inputs, kernel=3)
    PCA_output = PCA_eig(stem, k=number_components) # k is number of components
    components = PCA_output['components']
    return stem,components.float().T


def PCA_eig(X, k=1, center=True, scale=False):
    n,p = X.size()
    ones = torch.ones(n).view([n,1])
    h = ((1/n) * torch.mm(ones, ones.t())) if center else torch.zeros(n*n).view([n,n])
    H = torch.eye(n) - h
    X_center =  torch.mm(H.double(), X.double())
    covariance = 1/(n-1) * torch.mm(X_center.t(), X_center).view(p,p)
    scaling =  torch.sqrt(1/torch.diag(covariance)).double() if scale else torch.ones(p).double()
    scaled_covariance = torch.mm(torch.diag(scaling).view(p,p), covariance)
    eigenvalues, eigenvectors = torch.eig(scaled_covariance, True)
    components = (eigenvectors[:, :k]).t()
    explained_variance = eigenvalues[:k, 0]
    return { 'X':X, 'k':k, 'components':components, 'explained_variance':explained_variance }


def CRCA(inputs, targets, number_components):
    stem = STEM(inputs, kernel=3)
    targets = torch.repeat_interleave(targets, inputs.shape[-1]*inputs.shape[-2], dim=0).float()
    rm = torch.matmul(torch.inverse(torch.matmul(stem.t(), stem)+1e-3*torch.eye(stem.shape[-1])),
                    torch.matmul(torch.matmul(stem.t(), targets), torch.matmul(targets.t(), stem)))
    components, _ = torch.eig(rm)
    return stem,components[:,:number_components]


def STEM(inputs, kernel=3):
    '''
    return: stem features with shape [batchxHxW, 3xkernelxkernel] where batch=10, H=W=32, kernel=3
    '''
    B, C, H, W = inputs.shape
    X = []
    inputs_with_padding = torch.cat((inputs,torch.zeros((B,C,H,kernel))),-1)
    inputs_with_padding = torch.cat((inputs_with_padding,torch.zeros(B,C,kernel,W+kernel)),-2)
    for i in range(inputs.shape[2]):
        row = []
        for j in range(inputs.shape[3]):
            row.append(inputs_with_padding[:,:,i:i+kernel,j:j+kernel])
        X.append(row)
    X = torch.stack([torch.stack(x,2) for x in X],2).permute(0,2,3,1,4,5)
    return X.reshape(-1,3*kernel*kernel)


# TODO: write your own function to extract the leading component, with shape:  [3xkernelxkernel, 1]
def get_leading_component():
    pass

# TODO: write your own function to generate the feature-maps using the leading component, the return feature maps has shape [batch, H, W]
def generate_featuremaps():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument('--model', type=str, default='PCA', help='choose between PCA and RCA')
    args = parser.parse_args()
    main(args)



