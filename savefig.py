import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from torchvision import utils, datasets, transforms

matplotlib.rcParams["figure.dpi"] = 1200


def save_the_image():
	meta_file = r'D:\Datum\ECE571 Deep Learning Networks\Deep-LSE\data.cifar10\cifar-10-batches-py\batches.meta'
	meta_data = unpickle(meta_file)
	file = r'D:\Datum\ECE571 Deep Learning Networks\Deep-LSE\data.cifar10\cifar-10-batches-py\data_batch_1'
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
		plt.savefig('pngs/%s.png'%idx)

def unpickle(file):
	import pickle
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='latin1')
	return dict

if __name__ == '__main__':
	dataset = datasets.CIFAR10('./data.cifar10', train=True, download=True)
	save_the_image()