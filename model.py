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


def PCA_eig(X, k=1, center=True, scale=False):
    n,p = X.size()
    ones = torch.ones(n).view([n,1])
    h = ((1/n) * torch.mm(ones, ones.t())) if center else torch.zeros(n*n).view([n,n])
    H = torch.eye(n) - h
    X_center =  torch.mm(H.double(), X.double())
    covariance = 1/(n-1) * torch.mm(X_center.t(), X_center).view(p,p)
    scaling =  torch.sqrt(1/torch.diag(covariance)).double() if scale else torch.ones(p).double()
    scaled_covariance = torch.mm(torch.diag(scaling).view(p,p), covariance)
    eigenvalues, eigenvectors = torch.linalg.eig(scaled_covariance)
    components = (eigenvectors[:, :k]).t().real
    explained_variance = eigenvalues[:k].real
    return { 'X':X, 'k':k, 'components':components, 'explained_variance':explained_variance }

def DPCA_eig(Y, X, m): #here we minimize $\| Y - F U X \|^2$
    # Y: (3,BHW), X: (27, BHW)
    n,p = X.size()
    XYt = torch.mm(X, Y.t())
    RM = torch.mm(torch.inverse(torch.mm(X, X.t())+1e-3*torch.eye(X.shape[0])),
			        torch.mm(XYt, XYt.t()))
    eigenvalues, eigenvectors = torch.linalg.eig(RM)
    
    values, indices = torch.sort(eigenvalues.real, descending=True)
    
    U = (eigenvectors[:, indices[:m]]).t().real  #(m,27)
    explained_variance = values[:m]
    UX = torch.mm(U,X)
    F = torch.mm(torch.mm(Y, UX.t()),
                 torch.inverse(torch.mm(UX, UX.t())+1e-3*torch.eye(UX.shape[0])),
			        )
    W = torch.mm(F, U)   #W = FU is a rank-m matrix
    return { 'X':X, 'k':m, 'components':U, 'explained_variance':explained_variance,
            'W': W}

def CRCA(data, targets, num_components, channel=3):
    m = num_components
    stem = STEM(data, channel=channel)
    targets = torch.repeat_interleave(targets, data.shape[-1]*data.shape[-2], dim=0).float()
    rm = torch.mm(torch.inverse(torch.mm(stem.t(), stem)+1e-3*torch.eye(stem.shape[-1])),
                torch.mm(torch.mm(stem.t(), targets), torch.mm(targets.t(), stem)))
    eigenvalues, eigenvectors = torch.linalg.eig(rm)
    U = (eigenvectors[:, :m]).t().real  #(2,27)
    explained_variance = eigenvalues[:m].real
    UX = torch.mm(U,stem.t())
    F = torch.mm(torch.mm(targets.t(), UX.t()),
                 torch.inverse(torch.mm(UX, UX.t())+1e-3*torch.eye(UX.shape[0])))
    W = torch.mm(F,U)
    return stem, W

def CPCA(data, targets, num_components, channel=3):
    stem = STEM(data, channel=channel)  #(BHW,27)
    # Y(target): (3,BHW), X(X_stem): (27, BHW)
    PCA_output = DPCA_eig(targets, stem.t(), m=num_components) # m is number of components
    # components = PCA_output['components']
    weights = PCA_output['W']
    return stem, weights


def STEM(inputs, channel=3):
    '''
    return: stem features with shape [batchxHxW, 3xkernelxkernel] where batch=10, H=W=32, kernel=3
    '''
    kernel = 3
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
    return X.reshape(-1,channel*kernel*kernel)


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