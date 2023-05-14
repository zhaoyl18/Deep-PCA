import argparse
from cmath import log10, sqrt
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
import cv2

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
    
'''
PSNR is most commonly used to estimate the efficiency of compressors, filters, etc.
The larger the value of PSNR, 
the more efficient is a corresponding compression or filter method.
'''    
def PSNR(original_path, compressed_path):
    original = cv2.imread(original_path)
    compressed = cv2.imread(compressed_path, 1)
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

'''
show image size
'''
def show_image_size(image_path):
     # get image
    img = Image.open(image_path)
    
    # get width and height
    width = img.width
    height = img.height
    
    # display width and height
    print("The height of the image is: ", height)
    print("The width of the image is: ", width)