# Deep PCA for Eigenfaces

## Abstract

This project focused on the methodology of Turk and Pentland's paper, ***Face recognition using eigenfaces***. We implemented the workflow suing basic algebra function of Numpy, including images preprocessing, eigenfaces construction, eigenspace representation of images, face recognition based on K-nn (K near neighbors) algorithm, performance evaluation. For performance evaluation, we worked AT&T face dataset (formerly 'The ORL Database of Faces').

## Datasets:

### AT&T "The Database of Faces" (formerly "The ORL Database of Faces")

Ten different images of each of 40 distinct subjects. The images were taken at different times, varying the lighting, facial expressions (open / closed eyes, smiling / not smiling) and facial details (glasses / no glasses). All the images were taken against a dark homogeneous background with the subjects in an upright, frontal position (with tolerance for some side movement)

### Methodology, Workflow and Result

1. Load images and convert every of them into a Numpy matrix xi;
2. Compute the mean ?:  
Figure 1 AT&T database mean image  
![Figure 1 AT&T database mean image](/result/att_mean_image.png?raw=true "AT&T database mean image")

3. Compute the normalized images:

4. Compute the Covariance Matrix S, which is different from the covariance matrix, in order to avoid huge matrix for eigen decomposition problem:
5. Compute the Covariance Matrix S, which is different from the covariance matrix, in order to avoid huge matrix for eigen decomposition problem:
6. Compute the eigenvalue and eigenvector

Figure 5 Top 16 eigenfaces (AT&T database)
![result](/result/att_top_16_eigenfaces.png?raw=true)

7. Project an image into eigenspace and reconstruct using K eigenfaces:
W = vTXi  
Xf = V W  

Figure 7 Reconstruct using 80% variance (AT&T)
![result](/result/att_var080_faces43.png?raw=true)

Figure 8 Reconstruct using 90% variance (AT&T)
![result](/result/att_var090_faces110.png?raw=true)

Figure 9 Reconstruct using 95% variance (AT&T)
![result](/result/att_var095_faces189.png?raw=true)

Figure 10 Reconstruct using 99% variance (AT&T)
![result](/result/att_var099_faces324.png?raw=true)

Figure 11 Reconstruct using 80% variance (Yale)
![result](/result/yale_var080_faces4.png?raw=true)

Figure 12 Reconstruct using 90% variance (Yale)
![result](/result/yale_var090_faces22.png?raw=true)

Figure 13 Reconstruct using 95% variance (Yale)
![result](/result/yale_var095_faces62.png?raw=true)

Figure 14 Reconstruct using 99% variance (Yale)
![result](/result/yale_var090_faces189.png?raw=true)

7. Recognition
For an image, we projected it into the eigenspace, and the image was considered as the linear combination of the eigenfaces. The weights of the corresponding eigenfaces therefore represented the image. We only used the top n eigenfaces to represent an image, where the n was determined by how much variance this sub-eigenspace can represent. We investigated 99%, 95%, 90% and 80% percent of variance for both datasets. For AT&T face dataset, the n were 324, 189, 180 and 43, respectively; for Yale face dataset, the n were 297, 62, 22 and 4, respectively.  
To recognize an unknown face, we used the Knn algorithm to find the close subject in the database. For each image in a dataset, we considered it as a query image and the other images in the dataset as training data. We got the nearest K neighbors and let them vote to determine the label of the query image. Whenever there was a tie, we used the label with the least average distance.  
If the predict label is the same with the ground label, it is a true positive; otherwise, it is a false positive. We calculated the precision as the performance of the recognition of the result.  

## Discussion

The eigenfaces is one of the most popular approaches to represent an image, with the basic idea that the top k component eigenvectors (eigenfaces) represent as much variance as possible. This criterion need not to be meaningful. It is also susceptible to illumination and background around the face. Fisherfaces [6] is considered to be a better representation than eigenfaces since it is more robust to illumination. But both of them do not contain semantic meanings as human to understand a face image. A possible further study is the deep neural network approach that produce the state of the art performance by now.

## Reference

* Turk, Matthew A., and Alex P. Pentland. "Face recognition using eigenfaces." Computer Vision and Pattern Recognition, 1991. Proceedings CVPR'91., IEEE Computer Society Conference on. IEEE, 1991.
* Turk, Matthew, and Alex Pentland. "Eigenfaces for recognition." Journal of cognitive neuroscience 3.1 (1991): 71-86.
* Belhumeur, Peter N., Jo?o P. Hespanha, and David J. Kriegman. "Eigenfaces vs. fisherfaces: Recognition using class specific linear projection." IEEE Transactions on pattern analysis and machine intelligence 19.7 (1997): 711-720.
* http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
* http://vision.ucsd.edu/content/extended-yale-face-database-b-b
* Belhumeur, Peter N., Jo?o P. Hespanha, and David J. Kriegman. "Eigenfaces vs. fisherfaces: Recognition using class specific linear projection." IEEE Transactions on pattern analysis and machine intelligence 19.7 (1997): 711-720.
