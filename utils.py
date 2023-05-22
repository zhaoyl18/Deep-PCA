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
from skimage.metrics import structural_similarity as ssim
import os

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
def mse(original_path, compressed_path):
    # the MSE between the two images is the sum of the squared difference between the two images
    original = cv2.imread(original_path)
    compressed = cv2.imread(compressed_path, 1)
    mse = np.mean((original - compressed) ** 2)
    return mse
# define function that combines all three image quality metrics
def compare_images(original_path, compressed_path):
    scores = []
    scores.append(PSNR(original_path, compressed_path))
    scores.append(mse(original_path, compressed_path))

    original = cv2.imread(original_path)
    compressed = cv2.imread(compressed_path, 1)
    scores.append(ssim(compressed, original, multichannel =True)) # target, ref (original)
    
    return scores
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

def Display_images_as_subplots(original,compressed,restored):
    plt.rcParams.update({'figure.max_open_warning': 0})

    # display images as subplots
    original = cv2.imread(original)
    compressed = cv2.imread(compressed)
    restored = cv2.imread(restored) # 1 means read as color image

    fig, axs = plt.subplots(1, 3, figsize=(20, 8))
    axs[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Original')
    axs[1].imshow(cv2.cvtColor(compressed, cv2.COLOR_BGR2RGB))
    axs[1].set_title('Compressed')
    axs[2].imshow(cv2.cvtColor(restored, cv2.COLOR_BGR2RGB))
    axs[2].set_title('DeepPCA Restored')

    # remove the x and y ticks
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

def plot_one_image(img):
    # plt.imshow('image window', img)
    # plt.imshow(img.cpu().data.numpy())
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
def show_three_dataset():
    # plt.rcParams.update({'figure.max_open_warning': 0})
    original_dir = os.path.join('datasets', 'att_faces')
    compressed_dir = os.path.join('datasets', 'att_faces_compress')
    restored_dir = os.path.join('datasets', 'att_faces_restore')
    for face_id in range(1, 40 + 1):
            for test_id in range(1, 11):
                # if (test_id in self.training_ids[face_id-1]) == False:          # we skip the image if it is part of the training set
                path_to_img_original = os.path.join(original_dir,
                        's' + str(face_id), str(test_id) + '.pgm')   
                path_to_img_compressed = os.path.join(compressed_dir,
                        's' + str(face_id), str(test_id) + '.pgm') 
                path_to_img_restore = os.path.join(restored_dir,
                        's' + str(face_id), str(test_id) + '.pgm') 
                Display_images_as_subplots(path_to_img_original,path_to_img_compressed,path_to_img_restore)
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
def mse(original_path, compressed_path):
    # the MSE between the two images is the sum of the squared difference between the two images
    original = cv2.imread(original_path)
    compressed = cv2.imread(compressed_path, 1)
    mse = np.mean((original - compressed) ** 2)
    return mse
# define function that combines all three image quality metrics
def compare_images(original_path, compressed_path):
    scores = []
    scores.append(PSNR(original_path, compressed_path))
    scores.append(mse(original_path, compressed_path))

    original = cv2.imread(original_path)
    compressed = cv2.imread(compressed_path, 1)
    scores.append(ssim(compressed, original, multichannel =True)) # target, ref (original)
    
    return scores