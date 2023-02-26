import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils, datasets, transforms
import matplotlib.pyplot as plt
from numpy.linalg import inv
import sys


def main(args):
	kwargs = {}
	train_loader = torch.utils.data.DataLoader(
		datasets.CIFAR10('./data.cifar10', train=True, download=False,
					   transform=transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
					   ])),
		batch_size=10, shuffle=False, num_workers=1, **kwargs)

	dataiter = iter(train_loader)
	data, lbls = dataiter.next()
	
	target = F.one_hot(lbls, 10)

	# data: shape [batch, 3, H, W] where 3 means RGB channels, batch=10, H=W=32
	# target: shape [batch, 10] where 10 means 10 classes, batch=10
	view_image(utils.make_grid(data[0]))

	LSE(data) # TODO: get whatever information needed by putting them as output in CPCA function


def LSE(inputs):
	stem = STEM(inputs, kernel=3)
	targets = torch.repeat_interleave(targets, inputs.shape[-1]*inputs.shape[-2], dim=0).float()
	rm = torch.matmul(torch.inverse(torch.matmul(stem.t(), stem)+1e-3*torch.eye(stem.shape[-1])), torch.matmul(torch.matmul(stem.t(), targets), torch.matmul(targets.t(), stem)))
	components, _ = torch.eig(rm)
	pass

# def CPCA(inputs):
# 	stem = STEM(inputs, kernel=3)
# 	PCA_output = PCA_eig(stem, k=1) # k is number of components
# 	components = PCA_output['components']
# 	pass


# def PCA_eig(X, k=1, center=True, scale=False):
#     n,p = X.size()
#     ones = torch.ones(n).view([n,1])
#     h = ((1/n) * torch.mm(ones, ones.t())) if center else torch.zeros(n*n).view([n,n])
#     H = torch.eye(n) - h
#     X_center =  torch.mm(H.double(), X.double())
#     covariance = 1/(n-1) * torch.mm(X_center.t(), X_center).view(p,p)
#     scaling =  torch.sqrt(1/torch.diag(covariance)).double() if scale else torch.ones(p).double()
#     scaled_covariance = torch.mm(torch.diag(scaling).view(p,p), covariance)
#     eigenvalues, eigenvectors = torch.eig(scaled_covariance, True)
#     components = (eigenvectors[:, :k]).t()
#     explained_variance = eigenvalues[:k, 0]
#     return { 'X':X, 'k':k, 'components':components, 'explained_variance':explained_variance }


# def CRCA(inputs, targets):
# 	stem = STEM(inputs, kernel=3)
# 	targets = torch.repeat_interleave(targets, inputs.shape[-1]*inputs.shape[-2], dim=0).float()
# 	rm = torch.matmul(torch.inverse(torch.matmul(stem.t(), stem)+1e-3*torch.eye(stem.shape[-1])), torch.matmul(torch.matmul(stem.t(), targets), torch.matmul(targets.t(), stem)))
# 	components, _ = torch.eig(rm)
# 	pass


# TODO: write your STEM function
def STEM(inputs, kernel=3):
	'''
	hint: stem should have shape [batchxHxW, 3xkernelxkernel] where batch=10, H=W=32, kernel=3
	'''
	pass

def view_image(image):
	img = image / 2 + 0.5   # unnormalize
	npimg = img.numpy()   # convert from tensor
	plt.imshow(np.transpose(npimg, (1, 2, 0))) 
	plt.show()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="PyTorch Training")
	parser.add_argument('--model', type=str, default='LSE', help='choose between LSE, PCA and RCA')
	args = parser.parse_args()
	main(args)



