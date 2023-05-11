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
import os
import cv2
import shutil
import random

from DPCA import run_single_img, run_cifar, run_faces, CPCA
from model import DPCA_eig, STEM

class Eigenfaces(object):                                                       # *** COMMENTS ***
    faces_count = 40

    faces_dir = '.'                                                             # directory path to the AT&T faces

    train_faces_count = 6                                                       # number of faces used for training
    test_faces_count = 4                                                       # number of faces used for testing

    l = train_faces_count * faces_count                                         # training images count
    m = 92                                                                      # number of columns of the image
    n = 112                                                                     # number of rows of the image
    mn = m * n                                                                  # length of the column vector

    """
    Initializing the Eigenfaces model.
    """
    def __init__(self, _faces_dir = '.'):
        print('> Initializing started')

        self.faces_dir = _faces_dir
        self.energy = 0.85
        self.training_ids = []                                                  # train image id's for every at&t face

        L = np.empty(shape=(self.mn, self.l), dtype='float64')                  # each row of L represents one train image

        cur_img = 0
        for face_id in range(1, self.faces_count + 1):

            training_ids = random.sample(range(1, 11), self.train_faces_count)  # the id's of the 6 random training images
            self.training_ids.append(training_ids)                              # remembering the training id's for later

            for training_id in training_ids:
                path_to_img = os.path.join(self.faces_dir,
                        's' + str(face_id), str(training_id) + '.pgm')          # relative path
                #print '> reading file: ' + path_to_img

                img = cv2.imread(path_to_img, 0)                                # read a grayscale image
                img_col = np.array(img, dtype='float64').flatten()              # flatten the 2d image into 1d

                L[:, cur_img] = img_col[:]                                      # set the cur_img-th column to the current training image
                cur_img += 1

        self.mean_img_col = np.sum(L, axis=1) / self.l                          # get the mean of all images / over the rows of L

        for j in range(0, self.l):                                             # subtract from all training images
            L[:, j] -= self.mean_img_col[:]

        C = np.matrix(L.transpose()) * np.matrix(L)                             # instead of computing the covariance matrix as
        C /= self.l                                                             # L*L^T, we set C = L^T*L, and end up with way
                                                                                # smaller and computentionally inexpensive one
                                                                                # we also need to divide by the number of training
        # data = torch.from_numpy(np.matrix(L.transpose())).reshape(self.l, 92, 112).unsqueeze(1)   #[320, 1, 92, 112]
        # stem = STEM(data, channel=1)  #[3297280, 9]
        # X = stem.reshape(self.l, 92*112, 9).reshape(self.l,-1).t()  #[HW9, 320]
        # Y = torch.from_numpy(np.matrix(L))                                 #[HW, 320]
        
        # XYt = torch.mm(X, Y.t())
        # # RM = torch.mm(torch.inverse(torch.mm(X, X.t())+1e-3*torch.eye(X.shape[0])),
        # #                 torch.mm(XYt, XYt.t()))
        # RM_ = torch.mm(X.t(),torch.inverse(torch.mm(X, X.t())+1e-3*torch.eye(X.shape[0])),
        #                 torch.mm(XYt, Y))
        # eigenvalues, eigenvectors = torch.linalg.eig(RM_)
        
        # values, indices = torch.sort(eigenvalues.real, descending=True)
        
        # m=50
        # U = (eigenvectors[:, indices[:m]]).t().real  #(m,HW9)
        
        values, eigenvectors = np.linalg.eig(C)                          # eigenvectors/values of the covariance matrix
        indices = values.argsort()[::-1]                             # getting their correct order - decreasing
        self.evalues = values[indices]                               # puttin the evalues in that order
        self.evectors = eigenvectors[:,indices]                             # same for the evectors

        evalues_sum = sum(self.evalues[:])                                      # include only the first k evectors/values so
        evalues_count = 0                                                       # that they include approx. 85% of the energy
        evalues_energy = 0.0
        for evalue in self.evalues:
            evalues_count += 1
            evalues_energy += evalue / evalues_sum

            if evalues_energy >= self.energy:
                break

        self.evalues = self.evalues[0:evalues_count]                            # reduce the number of eigenvectors/values to consider
        self.evectors = self.evectors[:,0:evalues_count]

        self.evectors = L * self.evectors    #(HW,62)
        norms = np.linalg.norm(self.evectors, axis=0)                           # find the norm of each eigenvector
        self.evectors = self.evectors / norms                                   # normalize all eigenvectors

        self.W = self.evectors.transpose() * L       #(62,320)

        print('> Initializing ended')
    """
    Classify an image to one of the eigenfaces.
    """
    def classify(self, path_to_img):
        img = cv2.imread(path_to_img, 0)                                        # read as a grayscale image
        img_col = np.array(img, dtype='float64').flatten()                      # flatten the image
        img_col -= self.mean_img_col                                            # subract the mean column
        img_col = np.reshape(img_col, (self.mn, 1))                             # from row vector to col vector

        S = self.evectors.transpose() * img_col                                 # projecting the normalized probe onto the
                                                                                # Eigenspace, to find out the weights

        diff = self.W - S                                                       # finding the min ||W_j - S||
        norms = np.linalg.norm(diff, axis=0)

        closest_face_id = np.argmin(norms)                                      # the id [0..240) of the minerror face to the sample
        return int(closest_face_id / self.train_faces_count) + 1                   # return the faceid (1..40)

    """
    Evaluate the model using the 4 test faces left
    from every different face in the AT&T set.
    """
    def evaluate(self):
        print('> Evaluating AT&T faces started')
        results_file = os.path.join('results', 'att_restore_results.txt')               # filename for writing the evaluating results in
        f = open(results_file, 'w')                                             # the actual file

        test_count = self.test_faces_count * self.faces_count                   # number of all AT&T test images/faces
        test_correct = 0
        for face_id in range(1, self.faces_count + 1):
            for test_id in range(1, 11):
                if (test_id in self.training_ids[face_id-1]) == False:          # we skip the image if it is part of the training set
                    path_to_img = os.path.join(self.faces_dir,
                            's' + str(face_id), str(test_id) + '.pgm')          # relative path

                    result_id = self.classify(path_to_img)
                    result = (result_id == face_id)

                    if result == True:
                        test_correct += 1
                        f.write('image: %s\nresult: correct\n\n' % path_to_img)
                    else:
                        f.write('image: %s\nresult: wrong, got %2d\n\n' %
                                (path_to_img, result_id))

        print('> Evaluating AT&T faces ended')
        self.accuracy = float(100. * test_correct / test_count)
        print('Correct: ' + str(self.accuracy) + '%')
        f.write('Correct: %.2f\n' % (self.accuracy))
        f.close()                                                               # closing the file

if __name__ == "__main__":
    random.seed(0)
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument('--model', type=str, default='PCA', help='choose between PCA and RCA')
    parser.add_argument('--num_components', type=int, default=2, help='choose between 1,2,3')
    parser.add_argument('--dataset', type=str, default='faces', help='choose between single_img, cifar and faces')
    args = parser.parse_args()

    if args.dataset == 'cifar':
        run_cifar(args)
    elif args.dataset == 'single_img':
        run_single_img(args)
    elif args.dataset == 'faces':
        # aaa = Eigenfaces('./datasets/att_faces_restore')
        # if not os.path.exists('results'):                                           # create a folder where to store the results
        #     os.makedirs('results')
        # aaa.evaluate()
        
        faces = Eigenfaces('./datasets/att_faces_restore')
        faces.evaluate()