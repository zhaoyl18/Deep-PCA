import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import utils, datasets, transforms
import os
import cv2
import shutil
import random
from utils import plot, view_image, plot_one_image, show_three_dataset
from DPCA import run_single_img, run_cifar, run_faces, restore_via_LSE

# matplotlib.rcParams["figure.dpi"] = 1200


def save_cifar():
	meta_file = r'D:\Datum\ECE571 Deep Learning Networks\Deep-PCA\datasets\data_cifar10\cifar-10-batches-py\batches.meta'
	meta_data = unpickle(meta_file)
	file = r'D:\Datum\ECE571 Deep Learning Networks\Deep-PCA\datasets\data_cifar10\cifar-10-batches-py\data_batch_1'
	data_batch_1 = unpickle(file)
	# label names
	label_name = meta_data['label_names']
	# take first image
	for idx in range(len(data_batch_1['data'])):
		if idx > 5:
			break
		print(idx)
		image = data_batch_1['data'][idx]
		# take first image label index
		label = data_batch_1['labels'][idx]  # [3072, ], min 0, max 255
		# Reshape the image
		image = image.reshape(3,32,32)
		# Transpose the image
		image = image.transpose(1,2,0)
		# Display the image
		plt.imshow(image)
		plt.title(label_name[label])
		plt.savefig('./datasets/data_cifar10/pngs/%s.png'%idx)

def unpickle(file):
	import pickle
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='latin1')
	return dict

def restore_faces():													# *** COMMENTS ***
	faces_count = 40                                                          # directory path to the AT&T faces                                                      # number of faces used for training
	print('> Initializing started')

	faces_dir = './datasets/att_faces'
	restore_dir = './datasets/att_faces_restore'
	training_ids = []                                                  # train image id's for every at&t face


	cur_img = 0
	for face_id in range(1, faces_count + 1):
		print(face_id)
		training_ids = range(1, 11)
		if not os.path.exists(os.path.join(restore_dir,
					's' + str(face_id))):                                           # create a folder where to store the results
			os.makedirs(os.path.join(restore_dir,
					's' + str(face_id)))
		for training_id in training_ids:
			path_to_img = os.path.join(faces_dir,
					's' + str(face_id), str(training_id) + '.pgm')          # relative path
			print('> reading file: ' + path_to_img)
			img = cv2.imread(path_to_img, 0)                                # read a grayscale image
			# img_restore = run_faces(img)
			
			# path_to_restored = os.path.join(restore_dir,
			# 		's' + str(face_id), str(training_id) + '.pgm')
			path_to_restored = os.path.join(restore_dir,
					's' + str(face_id), str(training_id) + '.jpg')
			restore_via_LSE(path_to_img,path_to_restored)
			# os.remove(path_to_restored)
			print('> wrting file: ' + path_to_restored)

			# cv2.imwrite(path_to_restored, img_restore)

def compress_faces():													# *** COMMENTS ***
	faces_count = 40                                                          # directory path to the AT&T faces                                                                 # length of the column vector

	print('> Initializing started')

	faces_dir = './datasets/att_faces'
	compress_dir = './datasets/att_faces_compress'
	training_ids = []                                                  # train image id's for every at&t face


	cur_img = 0
	i = 0
	for face_id in range(1, faces_count + 1):
		# if i>1: #for test
		# 	break
		print(face_id)
		training_ids = range(1, 11)
		if not os.path.exists(os.path.join(compress_dir,
					's' + str(face_id))):                                           # create a folder where to store the results
			os.makedirs(os.path.join(compress_dir,
					's' + str(face_id)))
		# test
		
		for training_id in training_ids:
			i = i+1
			path_to_img = os.path.join(faces_dir,
					's' + str(face_id), str(training_id) + '.pgm') 
			path_to_compressed = os.path.join(compress_dir,
					's' + str(face_id), str(training_id) + '.pgm')         # relative path
			print('> reading file: ' + path_to_img)

			path = path_to_img
			factor = 2
			path_output = path_to_compressed

			prepare_images(path, factor, path_output)
			
			print('> wrting file: ' + path_to_compressed)

def prepare_images(path, factor, path_output):
    
    # # loop through the files in the directory
    # for file in os.listdir(path):
	# open the file
	img = cv2.imread(path,0)
	# plot_one_image(img)
	print(img.shape) #(112, 92)
	
	# find old and new image dimensions
	h, w = img.shape
	new_height = int(h / factor)
	new_width = int(w / factor)
	
	# resize the image - down
	img = cv2.resize(img, (new_width, new_height), interpolation = cv2.INTER_CUBIC)
	
	# # resize the image - up
	# img = cv2.resize(img, (w, h), interpolation = cv2.INTER_CUBIC)
	
	# save the image
	# print('Saving {}'.format(file))
	cv2.imwrite(path_output, img)
	print("After: ",img.shape) #(112, 92)
	# plot_one_image(img)
if __name__ == '__main__':
	# dataset = datasets.CIFAR10('./data_cifar10', train=True, download=True)
	# save_cifar()

	compress_faces()
	# restore_faces()
	# show_three_dataset()


