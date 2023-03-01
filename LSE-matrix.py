import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils, datasets, transforms
import matplotlib.pyplot as plt

from PIL import Image
 
 
# load the image and convert into
# numpy array


def main(args):
	kwargs = {}
	img = Image.open('baby_mini_d3_gaussian.jpg')
	img_data = torch.from_numpy(np.asarray(img)).double()
	# view_image(img_data)
	# img = utils.make_grid(img_data)
	# plt.imshow(img_data) 
	# plt.show()

	data = img_data.unsqueeze(0)/255
	data = data*2 - 1 # normalize to [-1, 1]
	data = data.permute(0, 3, 1, 2)

	# data: shape [batch, 3, H, W] where 3 means RGB channels, batch=10, H=W=32

	view_image(data[0]) # visualize the first image in the batch (optional)

	raw_img = data.clone() # save a copy of raw image shape
	compressed_img = torch.nn.functional.interpolate(data, scale_factor=0.5, mode='bilinear')
	X_data = torch.nn.functional.interpolate(compressed_img, scale_factor=2, mode='nearest')	
	
	LSE((X_data, raw_img))



def LSE(inputs):
	X_data, raw_image = inputs
	B, C, H, W = X_data.shape
	stem = STEM(X_data, kernel=3) # [batchxHxW, 3xkernelxkernel]
	targets = raw_image.permute(0,2,3,1).reshape(B*H*W, C).float() # [batch, 3, H, W] -> [batchxHxW, 3]
	kernel_weights = torch.matmul(torch.inverse(torch.matmul(stem.t(), stem)+1e-3*torch.eye(stem.shape[-1])),
			        torch.matmul(stem.t(), targets))
	conved = torch.matmul(stem, kernel_weights).reshape(B, H, W, C).permute(0,3,1,2)
	view_image(torch.clamp(conved[0], min=-1.0, max=1.0))
	# kernel_weights = kernel_weights.reshape(3, 3, 3, 3) # (in_channels, kernel_size, kernel_size, out_channels,)
	# kernel_weights = kernel_weights.permute(0, 3, 1, 2) # (in_channels, out_channels, kernel_size, kernel_size)
	return kernel_weights

def naive_conv(inputs): # not completed yet..
	model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=0),
            torch.nn.Dropout(p=0.5))
	learning_rate = 0.001
	criterion = torch.nn.CrossEntropyLoss()    # Softmax is internally computed.
	optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

	print('Training the kernel ...')
	train_cost = []

	training_epochs = 15

	for epoch in range(training_epochs):
		avg_cost = 0
		X_data, raw_image = inputs    # image is already size of (28x28), no reshape

		optimizer.zero_grad() # <= initialization of the gradients
		
		# forward propagation
		out = model(X_data)
		output = out.view(out.size(0), -1)
		cost = criterion(output, raw_image) # <= compute the loss function
		
		# Backward propagation
		cost.backward() # <= compute the gradient of the loss/cost function     
		optimizer.step() # <= Update the gradients
		
		train_cost.append(cost.item())

		print("[Epoch: {:>4}], cost = {:>.9}".format(epoch + 1, cost.item()))


		print('Learning Finished!')

	# Test model and check accuracy
	model.eval()    # set the model to evaluation mode (dropout=False)

	X_data, raw_image = inputs

	prediction = model(X_data)
	view_image(prediction[0]) # visualize the first image in the batch (optional)

def STEM(inputs, kernel=3):
	'''
	hint: stem should have shape [batchxHxW, 3xkernelxkernel] where batch=10, H=W=32, kernel=3
	'''
	B, C, H, W = inputs.shape
	X_pad = torch.nn.functional.pad(inputs, (1, 1, 1, 1), mode='constant', value=0) # padding it to be [B, C, (H+2), [W+2]]
	outputs = torch.zeros((B*H*W, C*kernel*kernel))
	for b in range(B):
		for i in range(H):
			for j in range(W):
				outputs[b*H*W + i*W + j] = X_pad[b, :, i:i+kernel, j:j+kernel].reshape(-1)
	return outputs

def view_image(tensor):  # input: (C, H, W)
	img = utils.make_grid(tensor)
	img = img / 2 + 0.5   # unnormalize
	npimg = img.numpy()   # convert from tensor
	plt.imshow(np.transpose(npimg, (1, 2, 0))) 
	plt.show()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="PyTorch Training")
	parser.add_argument('--model', type=str, default='LSE', help='choose between LSE and CONV')
	args = parser.parse_args()
	main(args)



