# Deep PCA for Eigenfaces

## TODO

1. add a function that computes PSNR to test if the recovered images are good, or at least better than compressed images.
2. Is is possible to adopt deep PCA in RECOGNITION process, not just compressing images?
As for now, I tried using deep PCA when making eigenfaces, but there are some concerns and technical issues. Like it is not feasible to do matrix inverse for a matrix $X X^\top$ with size $(9\times H \times W, 9\times H \times W)$ where H = 112, W =92.

3. Do more experiments for random seeds, different train/test ratios and make illustrative tables and plots.
4. Write report and slides.

## Abstract

This project focused on the methodology of Turk and Pentland's paper, ***Face recognition using eigenfaces*** [1]. We implemented the workflow suing basic algebra function of Numpy and PyTorch, including images preprocessing, eigenfaces construction, eigenspace representation of images, face recognition based on K-nn (K near neighbors) algorithm, performance evaluation. For performance evaluation, we worked AT&T face dataset (formerly 'The ORL Database of Faces').

To save storage, we also apply Deep PCA to each image and test the performance of the recognition. The ideal case is even if we save an image with 4 times information lost, we can still achieve comparable recognition correctness using the same algorithm (eigenfaces + Knn).

## Datasets

The AT&T "The Database of Faces" (formerly "The ORL Database of Faces") dataset [3] contains ten different images of each of 40 distinct subjects.

The images were taken at different times, varying the lighting, facial expressions (open / closed eyes, smiling / not smiling) and facial details (glasses / no glasses). All the images were taken against a dark homogeneous background with the subjects in an upright, frontal position (with tolerance for some side movement)

## Pre-processing

1. For every image, we compress 2x directly. Then we save an up-sampling version to *dataset/att_faces_compress* folder.
Surely directly reducing image by 4 times would lead to a lot of information lost. However, we save a weight based on Deep pca hoping it could recover super resolution. Namely, we minimize $||Y - FU X||_F^2$ where $U$ is a combinations of eigenfectors. Finally the weight used to recover image is $W = FU$.
2. For every compressed image, we apply convolution kernel $W$ and save the restored image to datasets.att_faces_restore

## Methodology

1. For each of the three datasets. Load images and convert every of them into a matrix.

2. Compute the mean face by averaging over all images.

3. Compute the normalized images by subtracting mean face.

4. Compute the Covariance Matrix S, which is different from the covariance matrix, in order to avoid huge matrix for doing eigen decomposition.

5. Compute the eigenvalue and eigenvector. Then we have completed the initialization process of eigenfaces.

## Recognition

The application of eigenfaces implies that we consider each image as the linear combination of the eigenfaces (by projecting onto the eigenspace). The weights of the corresponding eigenfaces therefore represented the image itself.  
We only used the top n eigenfaces to represent an image, where the n was determined by how much variance this sub-eigenspace can represent. We investigated 85% percent of variance for each dataset.  
To recognize an unknown face, we used the Knn algorithm to find the close subject in the database.  

1. For each image in a dataset, we considered it as a query image and the other images in the dataset as training data.  
2. We got the nearest K neighbors and let them vote to determine the label of the query image. Whenever there was a tie, we used the label with the least average distance.  
3. If the prediction label is the same with the ground label, it is a true positive; otherwise, it is a false positive. We calculated the precision as the performance of the recognition of the result.  

## Results

* Using original images, probability of correctness is: 96.25\%
* Using 4 times compressed images, probability of correctness is: 88.75\%
* Using images compressed then recovered by deep PCA, probability of correctness is: 89.375\%

## Discussion

The eigenfaces is one of the most popular approaches to represent an image, with the basic idea that the top k component eigenvectors (eigenfaces) represent as much variance as possible. This criterion need not to be meaningful. It is also susceptible to illumination and background around the face.  
Fisherfaces [2] is considered to be a better representation than eigenfaces since it is more robust to illumination. But both of them do not contain semantic meanings as human to understand a face image. A possible further study is the deep neural network approach that produce the state of the art performance by now.

## Reference

[1] Turk, Matthew A., and Alex P. Pentland. "Face recognition using eigenfaces." Computer Vision and Pattern Recognition, 1991. Proceedings CVPR'91., IEEE Computer Society Conference on. IEEE, 1991.  
[2] Belhumeur, Peter N., Joao Pedo Hespanha, and David J. Kriegman. "Eigenfaces vs. fisherfaces: Recognition using class specific linear projection." IEEE Transactions on pattern analysis and machine intelligence 19.7 (1997): 711-720.  
[3] <http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html>  
[4] <https://github.com/zwChan/Face-recognition-using-eigenfaces>
