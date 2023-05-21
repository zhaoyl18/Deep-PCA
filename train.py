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
from utils import PSNR, compare_images, Display_images_as_subplots, plot_one_image

from DPCA import run_single_img, run_cifar, run_faces, CPCA
from model import DPCA_eig, STEM

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


class Eigenfaces(object):                                                       # *** COMMENTS ***
    faces_count = 40

    faces_dir = '.'                                                             # directory path to the AT&T faces

    train_faces_count = 6                                                       # number of faces used for training
    test_faces_count = 4                                                       # number of faces used for testing

    l = train_faces_count * faces_count                                         # training images count
    m = 92                                                                      # number of columns of the image
    n = 112                                                                     # number of rows of the image
    mn = m * n  
                                                                    # length of the column vector

    """
    Initializing the Eigenfaces model.
    """
    def __init__(self, _faces_dir = '.'):
        print('> Initializing started')

        self.faces_dir = _faces_dir
        print("self.faces_dir: ", self.faces_dir)
        self.energy = 0.85
        self.training_ids = []  
        # train image id's for every at&t face
        # self.L = np.empty(shape=(self.mn, self.l), dtype='float64')                  # each row of L represents one train image
        print('> Initializing ended')
    def ini_pca(self):
        cur_img = 0
        for face_id in range(1, self.faces_count + 1):

            training_ids = random.sample(range(1, 11), self.train_faces_count)  # the id's of the 6 random training images
            self.training_ids.append(training_ids)                              # remembering the training id's for later

            for training_id in training_ids:
                path_to_img = os.path.join(self.faces_dir,
                        's' + str(face_id), str(training_id) + '.pgm')          # relative path
                #print '> reading file: ' + path_to_img

                img = cv2.imread(path_to_img, 0)  
                h, w = img.shape                              # read a grayscale image 112 X 92, or 62 X 47
                print("h and w: ", h, w) # (112, 92)
                self.m = w
                self.n = h
                self.mn = self.m * self.n  
                self.L = np.empty(shape=(self.mn, self.l), dtype='float64')  
                print("self.L shape: ", self.L.shape) # (10304, 320)")

                img_col = np.array(img, dtype='float64').flatten()              # flatten the 2d image into 1d
                print("img_col shape: ", img_col.shape) # (10304,)

                self.L[:, cur_img] = img_col[:]                                      # set the cur_img-th column to the current training image
                print("self.L shape: ", self.L.shape) # (10304, 320)")
                cur_img += 1

        self.mean_img_col = np.sum(self.L, axis=1) / self.l                          # get the mean of all images / over the rows of L
        print("mean_img_col shape: ", self.mean_img_col.shape) # (10304,)

        for j in range(0, self.l):                                             # subtract from all training images
            self.L[:, j] -= self.mean_img_col[:]

        C = np.matrix(self.L.transpose()) * np.matrix(self.L)                             # instead of computing the covariance matrix as
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

        self.evectors = self.L * self.evectors    #(HW,62)
        norms = np.linalg.norm(self.evectors, axis=0)                           # find the norm of each eigenvector
        self.evectors = self.evectors / norms                                   # normalize all eigenvectors

        self.W = self.evectors.transpose() * self.L       #(62,320) 
        print(">> W shape: ", self.W.shape) # (101,240)
    def dataLoader(self):
        print('> Loading Data')
        print("self.faces_dir: ", self.faces_dir)
        self.energy = 0.85
        self.training_ids = []  
        cur_img = 0
        X = []
        y = []
        for face_id in range(1, self.faces_count + 1):
            # training_ids = random.sample(range(1, 11), self.train_faces_count)  # the id's of the 6 random training images
            # self.training_ids.append(training_ids)                              # remembering the training id's for later
            for training_id in range(1, 11):
                path_to_img = os.path.join(self.faces_dir,
                        's' + str(face_id), str(training_id) + '.pgm')          # relative path
                #print '> reading file: ' + path_to_img
                img = cv2.imread(path_to_img, 0)
                # plot_one_image(img)
                img_array = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                X.append(img_array) #image
                y.append(face_id) #label
        print("length of X and y:")
        print(len(X))
        print(len(y))
        X = np.array(X)
        y = np.array(y)
        # Reshape X to 2D array
        print("X shape is:")
        print(X.shape)
        # pass
        n_samples, height, width, channels = X.shape
        X = X.reshape(n_samples, height * width * channels)

        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        path_to_X_npy = os.path.join(self.faces_dir,'X.npy')          # relative path
        path_to_y_npy = os.path.join(self.faces_dir,'Y.npy')          # relative path
        np.save(path_to_X_npy, X_scaled)
        np.save(path_to_y_npy, y)
        print("Done: loading dataset and Saving npy files")
        return X_scaled,y
    def classify_tra_PCA(self):

        X_train, y_train = self.dataLoader()
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X_train)
        print(pca.explained_variance_ratio_)
        print(X_train.shape)
        print(pca_result.shape) 

        #plot - 1
        # plt.scatter(pca_result[:4000, 0], pca_result[:4000, 1], c=y_train[:4000], edgecolor='none', alpha=0.5,
        #    cmap=plt.get_cmap('jet', 10), s=5)
        # plt.colorbar()
        # plt.show()

        #plot -2
        # pca = PCA(200)
        # pca_full = pca.fit(X_train)

        # plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
        # plt.xlabel('# of components')
        # plt.ylabel('Cumulative explained variance')
        # plt.show()

        #pca
        pca = PCA(n_components=50)
        X_train_transformed = pca.fit_transform(X_train)
        # X_submission_transformed = pca.transform(X_submission)

        X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_train_transformed, y_train, test_size=0.4, random_state=13)
        
        # components = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        components = [5, 10, 50, 100, 150, 200, 250,300, 350, 400]
        neighbors = [1, 2, 3, 4, 5, 6, 7]
        neighbors = [1, 15, 20, 25, 30, 35, 40]

        scores = np.zeros( (components[len(components)-1]+1, neighbors[len(neighbors)-1]+1 ) )
        results_file = os.path.join('results', f'{self.faces_dir[-6:]}_traditional_pca.txt')
        f = open(results_file, 'w') 
        for component in components:
            for n in neighbors:
                knn = KNeighborsClassifier(n_neighbors=n)
                knn.fit(X_train_pca[:,:component], y_train_pca)
                score = knn.score(X_test_pca[:,:component], y_test_pca)
                #predict = knn.predict(X_test_pca[:,:component])
                scores[component][n] = score
                print('Components = ', component, ', neighbors = ', n,', Score = ', score)
                f.write(f"Components = {component}, neighbors = {n}, Score = {score}\n")
        f.close()
        scores = np.reshape(scores[scores != 0], (len(components), len(neighbors)))

        x = [0, 1, 2, 3, 4, 5, 6]
        y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        plt.rcParams["axes.grid"] = False

        fig, ax = plt.subplots()
        plt.imshow(scores, cmap='hot', interpolation='none', vmin=.90, vmax=1)
        plt.xlabel('neighbors')
        plt.ylabel('components')
        plt.xticks(x, neighbors)
        plt.yticks(y, components)
        plt.title('KNN score heatmap')

        plt.colorbar()
        plt.show()   

    """
    Classify an image to one of the eigenfaces.
    """
    def get_U():
        pass
    def get_F():
        pass
    def get_W():
        pass
    def classify_rca_dpca(self, path_to_img):
        pass
    def classify(self, path_to_img):
        img = cv2.imread(path_to_img, 0)                                        # read as a grayscale image
        img_col = np.array(img, dtype='float64').flatten()                      # flatten the image
        img_col -= self.mean_img_col 
        
                                                   # subract the mean column
        h, w = img.shape                              # read a grayscale image 112 X 92, or 62 X 47
        self.m = w
        self.n = h
        self.mn = self.m * self.n  

        img_col = np.reshape(img_col, (self.mn, 1))                             # from row vector to col vector

        S = self.evectors.transpose() * img_col                                 # projecting the normalized probe onto the
                                                                                # Eigenspace, to find out the weights

        diff = self.W - S                                                       # finding the min ||W_j - S||
        norms = np.linalg.norm(diff, axis=0)

        closest_face_id = np.argmin(norms)                                      # the id [0..240) of the minerror face to the sample
        return int(closest_face_id / self.train_faces_count) + 1                   # return the faceid (1..40)
    def all_metrics(self):
        # scores = compare_images(target, ref)
        print('> Evaluating metrics: PSNR, MSE, SSIM started')
        original_dir = os.path.join('datasets', 'att_faces')
        compressed_dir = os.path.join('datasets', 'att_faces_compress')
        if not os.path.exists(compressed_dir):   
            assert("commpresed_dis does not exist")  #assert a message "no path error"  
        restored_dir = os.path.join('datasets', 'att_faces_restore')
        if not os.path.exists(restored_dir):                                           # create a folder where to store the results
            assert("restored_dir does not exist")  #assert a message "no path error"")                                 
        results_file = os.path.join('results', 'three_metrics_results.txt')
        f = open(results_file, 'w')                                       # the actual file

        for face_id in range(1, self.faces_count + 1):
            for test_id in range(1, 11):
                # if (test_id in self.training_ids[face_id-1]) == False:          # we skip the image if it is part of the training set
                path_to_img_original = os.path.join(original_dir,
                        's' + str(face_id), str(test_id) + '.pgm')   
                path_to_img_compressed = os.path.join(compressed_dir,
                        's' + str(face_id), str(test_id) + '.pgm') 
                path_to_img_restore = os.path.join(restored_dir,
                        's' + str(face_id), str(test_id) + '.pgm') 
                # scores_ori_comp = compare_images(path_to_img_original, path_to_img_compressed)
                # f.write(f"Image: {face_id}_{test_id} scores_ori_comp\nPSNR: {scores_ori_comp[0]}, MSE: {scores_ori_comp[1]}, SSIM: {scores_ori_comp[2]}\n")
                scores_ori_rest = compare_images(path_to_img_original, path_to_img_restore)
                f.write(f"Image: {face_id}_{test_id} scores_ori_rest\nPSNR: {scores_ori_rest[0]}, MSE: {scores_ori_rest[1]}, SSIM: {scores_ori_rest[2]}\n")
                # scores_rest_comp = compare_images(path_to_img_restore, path_to_img_compressed)
                # f.write(f"Image: {face_id}_{test_id} scores_rest_comp\nPSNR: {scores_rest_comp[0]}, MSE: {scores_rest_comp[1]}, SSIM: {scores_rest_comp[2]}\n")
                # Display_images_as_subplots(path_to_img_original,path_to_img_compressed,path_to_img_restore)
                # scores_rest_ori = compare_images(path_to_img_restore, path_to_img_original)
                # f.write(f"Image: {face_id}_{test_id} scores_rest_ori\nPSNR: {scores_rest_ori[0]}, MSE: {scores_rest_ori[1]}, SSIM: {scores_rest_ori[2]}\n")
        f.close()                                                               # closing the file

    def compute_psnr(self):
        print('> Evaluating PSNR (signal-to-noise ratio) started')
        original_dir = os.path.join('datasets', 'att_faces')
        compressed_dir = os.path.join('datasets', 'att_faces_compress')
        if not os.path.exists(compressed_dir):   
            assert("commpresed_dis does not exist")  #assert a message "no path error"  
        restored_dir = os.path.join('datasets', 'att_faces_restore')
        if not os.path.exists(restored_dir):                                           # create a folder where to store the results
            assert("restored_dir does not exist")  #assert a message "no path error"")                                 
        results_file = os.path.join('results', 'psnr_results.txt')
        f = open(results_file, 'w')                                       # the actual file

        for face_id in range(1, self.faces_count + 1):
            for test_id in range(1, 11):
                # if (test_id in self.training_ids[face_id-1]) == False:          # we skip the image if it is part of the training set
                path_to_img_original = os.path.join(original_dir,
                        's' + str(face_id), str(test_id) + '.pgm')   
                path_to_img_compressed = os.path.join(compressed_dir,
                        's' + str(face_id), str(test_id) + '.pgm') 
                path_to_img_restore = os.path.join(restored_dir,
                        's' + str(face_id), str(test_id) + '.pgm') 
                # psnr_ori_comp = PSNR(path_to_img_original,path_to_img_compressed)
                psnr_ori_rest = PSNR(path_to_img_original,path_to_img_restore)
                # psnr_rest_comp = PSNR(path_to_img_restore,path_to_img_compressed)
                # psnr_rest_ori = PSNR(path_to_img_restore,path_to_img_original)
                # write the result to a csv file
                # f.write(f"Image: {face_id}_{test_id}\npsnr_ori_comp: {psnr_ori_comp}\npsnr_ori_rest: {psnr_ori_rest}\npsnr_rest_comp: {psnr_rest_comp}\npsnr_rest_ori: {psnr_rest_ori}\n\n")
                f.write(f"Image: {face_id}_{test_id}\npsnr_ori_rest: {psnr_ori_rest}\n\n")
                # f.write('image: %s\npsnr_ori_comp: %.2f\npsnr_ori_rest: %.2f\npsnr_rest_comp: %.2f\n\n' %
                                # ("Image: "+str(face_id)+"_"+str(test_id), psnr_ori_comp, psnr_ori_rest, psnr_rest_comp))
                # f.write('image: %s\nresult: correct\n\n' % path_to_img)

        print('> Evaluating AT&T faces ended')
        f.close()                                                               # closing the file

                
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
        print("faces_count: ", self.faces_count)
        for face_id in range(1, self.faces_count + 1):
            for test_id in range(1, 11):
                if (test_id in self.training_ids[face_id-1]) == False:          # we skip the image if it is part of the training set
                    path_to_img = os.path.join(self.faces_dir,
                            's' + str(face_id), str(test_id) + '.pgm')          # relative path

                    result_id = self.classify(path_to_img)
                    # print("result_id: ", result_id, "face_id: ", face_id, "test_id: ", test_id)
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
    parser.add_argument('--model', type=str, default='RCA-DPCA', help='choose between PCA and RCA, and tra_PCA')
    parser.add_argument('--num_components', type=int, default=2, help='choose between 1,2,3')
    parser.add_argument('--dataset', type=str, default='faces', help='choose between single_img, cifar and faces')
    args = parser.parse_args()

    if args.dataset == 'cifar':
        run_cifar(args)
    elif args.dataset == 'single_img':
        run_single_img(args)
    elif args.dataset == 'faces':
        if args.model == 'tra_PCA':
            # faces = Eigenfaces('./datasets/att_faces')
            # faces = Eigenfaces('./datasets/att_faces_compress')
            faces = Eigenfaces('./datasets/att_faces_restore')
            faces.classify_tra_PCA()
        elif args.model == 'RCA-DPCA':
            # aaa = Eigenfaces('./datasets/att_faces_restore')
            if not os.path.exists('results'):                                           # create a folder where to store the results
                os.makedirs('results')
            # aaa.evaluate()
            compressed_dir = os.path.join('datasets', 'att_faces_compress')
            # faces = Eigenfaces('./datasets/att_faces')
            # faces = Eigenfaces('./datasets/att_faces_compress')
            faces = Eigenfaces('./datasets/att_faces_restore')
            faces.ini_pca()
            faces.evaluate()
            # evaluate with psnr
            faces.compute_psnr()
            faces.all_metrics()