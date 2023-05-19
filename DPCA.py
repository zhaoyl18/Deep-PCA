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
import cv2
from model import CRCA, STEM, generate_featuremaps, CPCA
from utils import plot, view_image, plot_one_image
# from utils import plot, plot_faces, plot_faces_2
# from data import load_data, load_faces
import os
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
def prepare_images(img, factor):

    # # loop through the files in the directory
    # for file in os.listdir(path):
    # open the file
    
    # plot_one_image(img)
    print(img.shape) #(112, 92)

    # find old and new image dimensions
    h, w = img.shape
    new_height = int(h / factor)
    new_width = int(w / factor)

    # resize the image - down
    img = cv2.resize(img, (new_width, new_height), interpolation = cv2.INTER_CUBIC)


    # resize the image - up
    # img = cv2.resize(img, (w, h), interpolation = cv2.INTER_CUBIC)
        
    # save the image
    # print('Saving {}'.format(file))

    # cv2.imwrite(path_output, img) #save img to path_output
    return img
def preprocess_img(img): #from (112, 92) to (1, 112X92)
    #preprocess
    convert_tensor = transforms.ToTensor()
    # print(convert_tensor(img))
    tensor_img = convert_tensor(img) #torch.Size([1, 112, 92])
    print(tensor_img.size())

    tensor_img = tensor_img.unsqueeze(0) #[1, 1, 112, 92])
    # data = data*2 - 1 # normalize to [-1, 1]
    # tensor_img = tensor_img.permute(0, 3, 1, 2)
    print(tensor_img.size())
    tensor_img = tensor_img*2 - 1 # normalize to [-1, 1]

    return tensor_img

def restore_via_LSE(path_to_img, path_to_restore):
    img = cv2.imread(path_to_img,0) #orginal 112 X 92
    print("original image: ",img.shape)
    compressed_img = prepare_images(img, 2) #downsample 56 X 46
    print("compressed_img.shape: ", compressed_img.shape)
    # plot_one_image(compressed_img)
    X_data =  prepare_images(compressed_img, 0.5)
    print("Upsample compressed image: ", X_data.shape) #upsample 112 X 92
    # plot_one_image(X_data)

    tensor_img = preprocess_img(img)
    B, C, H, W = tensor_img.shape
    print("tensor_img.shape: ", tensor_img.shape)
    targets = tensor_img.permute(0,2,3,1).reshape(-1, C).float().t() # (1, 112*92) (1,1034)
    print("targets.shape: ", targets.shape)

    X_data = preprocess_img(X_data)
    print("X_data.shape: ", X_data.shape)

    stem, weights = CPCA(X_data, targets, 1, channel=1) #X_data: (B,C,H,W) (112*92,9), (1,9)
    featuremaps = torch.matmul(weights, stem.t()).reshape(B, H, W)

    restored_img = torch.clamp(featuremaps[0].squeeze(0), min=-1.0, max=1.0)
    restored_img = restored_img / 2 + 0.5
    print("restored img shape:",restored_img.shape)
    restored_img = restored_img.cpu().data.numpy()
    # print(restored_img)
    # plot_one_image(restored_img)
    plt.imshow(cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB))
    plt.imsave(path_to_restore,restored_img) 
    # plt.show()
    # plt.imsave(path_to_restore, restored_img)
    # cv2.imwrite(path_to_restore, restored_img) #save img to path_output
    
    test_img = cv2.imread(path_to_restore,0) #orginal 112 X 92
    # plot_one_image(test_img)

    print("new path: ",path_to_restore[:-3]+"pgm")
    cv2.imwrite(path_to_restore[:-3]+"pgm", test_img)

    # im = Image.open(path_to_restore)
    # im = im.convert('RGB')
    # im.save(path_to_restore[:-3]+"pgm")
    

    test_img = cv2.imread(path_to_restore[:-3]+"pgm") #orginal 112 X 92
    # plot_one_image(test_img)

    # return restored_img.cpu().data.numpy()
    
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


