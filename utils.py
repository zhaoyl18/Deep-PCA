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

def plot(data):
    fig, axs = plt.subplots(3, 3)
    for id in range(3*3):
        if id<data.shape[0]:
            axs[id//3, id%3].imshow(data[id].cpu().data.numpy())
        axs[id // 3, id % 3].set_xticks([])
        axs[id // 3, id % 3].set_yticks([])
    plt.show()

def view_image(tensor):  # input: (C, H, W)
	img = utils.make_grid(tensor)
	img = img / 2 + 0.5   # unnormalize
	npimg = img.numpy()   # convert from tensor
	plt.imshow(np.transpose(npimg, (1, 2, 0))) 
	plt.show()