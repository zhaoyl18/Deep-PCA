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
        datasets.CIFAR10('./data_cifar10', train=True, download=False,
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
        stem,weights = CPCA(data, number_components=2) #(10240,27), (27,2)
        featuremaps = generate_featuremaps(stem, weights, False)

    elif args.model == 'RCA':
        stem,weights = CRCA(data, target, number_components=2)
        featuremaps = generate_featuremaps(stem, weights, True)

    plot(data.permute(0,2,3,1))  # plot original images, permute to have shape (batch, H, W, 3)

    plot(featuremaps) # plot feature maps, shape (batch, H, W, 3)

def plot(data):
    fig, axs = plt.subplots(3, 3)
    for id in range(3*3):
        if id<data.shape[0]:
            axs[id//3, id%3].imshow(data[id].cpu().data.numpy())
        axs[id // 3, id % 3].set_xticks([])
        axs[id // 3, id % 3].set_yticks([])
    plt.show()

def CPCA(inputs, number_components):
    stem = STEM(inputs, kernel=3)  #(10240,27)
    targets = inputs.permute(0,2,3,1).reshape(-1, 3).float().t() #(3,10240)
    PCA_output = DPCA_eig(targets, stem.t(), m=number_components) # k is number of components
    # components = PCA_output['components']
    weights = PCA_output['W']
    return stem, weights


# def PCA_eig(X, k=1, center=True, scale=False):
#     n,p = X.size()
#     ones = torch.ones(n).view([n,1])
#     h = ((1/n) * torch.mm(ones, ones.t())) if center else torch.zeros(n*n).view([n,n])
#     H = torch.eye(n) - h
#     X_center =  torch.mm(H.double(), X.double())
#     covariance = 1/(n-1) * torch.mm(X_center.t(), X_center).view(p,p)
#     scaling =  torch.sqrt(1/torch.diag(covariance)).double() if scale else torch.ones(p).double()
#     scaled_covariance = torch.mm(torch.diag(scaling).view(p,p), covariance)
#     eigenvalues, eigenvectors = torch.linalg.eig(scaled_covariance)
#     components = (eigenvectors[:, :k]).t().real
#     explained_variance = eigenvalues[:k].real
#     return { 'X':X, 'k':k, 'components':components, 'explained_variance':explained_variance }

def DPCA_eig(Y, X, m=1): #here we minimize $\| Y - F U X \|^2$
    # Y: (3,10240), X: (27, 10240)
    n,p = X.size()
    XYt = torch.mm(X, Y.t())
    RM = torch.mm(torch.inverse(torch.mm(X, X.t())+1e-3*torch.eye(X.shape[0])),
			        torch.mm(XYt, XYt.t()))
    eigenvalues, eigenvectors = torch.linalg.eig(RM)
    U = (eigenvectors[:, :m]).t().real  #(2,27)
    explained_variance = eigenvalues[:m].real
    UX = torch.mm(U,X)
    F = torch.mm(torch.mm(Y, UX.t()),
                 torch.inverse(torch.mm(UX, UX.t())+1e-3*torch.eye(UX.shape[0])),
			        )
    W = torch.mm(F,U)   #W = FU is a rank-2 matrix
    return { 'X':X, 'k':m, 'components':U, 'explained_variance':explained_variance,
            'W': W}

def CRCA(inputs, targets, number_components):
    m = number_components
    stem = STEM(inputs, kernel=3)
    targets = torch.repeat_interleave(targets, inputs.shape[-1]*inputs.shape[-2], dim=0).float()
    rm = torch.mm(torch.inverse(torch.mm(stem.t(), stem)+1e-3*torch.eye(stem.shape[-1])),
                torch.mm(torch.mm(stem.t(), targets), torch.mm(targets.t(), stem)))
    eigenvalues, eigenvectors = torch.linalg.eig(rm)
    U = (eigenvectors[:, :m]).t().real  #(2,27)
    explained_variance = eigenvalues[:m].real
    UX = torch.mm(U,stem.t())
    F = torch.mm(torch.mm(targets.t(), UX.t()),
                 torch.inverse(torch.mm(UX, UX.t())+1e-3*torch.eye(UX.shape[0])),
			        )
    W = torch.mm(F,U)
    return stem, W


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
def get_leading_component( components):
    return components[:,0]

# TODO: write your own function to generate the feature-maps using the leading component, the return feature maps has shape [batch, H, W]
def generate_featuremaps(stem, weights, label=False):
    if not label:
        C = weights.shape[0]
        featuremaps = torch.matmul(weights, stem.t()).reshape(C, 10, 32, 32).permute(1,2,3,0) # from (C,BHW) to (B,H,W,C)
    else:
        featuremaps = torch.argmax(torch.matmul(weights, stem.t()),0).reshape(10, 32, 32, 1)
    return torch.clamp(featuremaps, min=0.0, max=1.0)

# def recover_image(stem, components, weights):
#     C = weights.shape[0]
#     featuremaps = torch.matmul(weights, stem.t()).reshape(C, 10, 32, 32).permute(1,2,3,0) # from (C,BHW) to (B,H,W,C)
#     return torch.clamp(featuremaps, min=0.0, max=1.0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument('--model', type=str, default='RCA', help='choose between PCA and RCA')
    args = parser.parse_args()
    main(args)



