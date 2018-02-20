#!/usr/bin/python
"""Script to prepare images and labels for VOC Contours"""

import cv2
import imageio
import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy.io import loadmat
from scipy import misc

def load_image(image_path, out_size=(321,481)):
    img = misc.imread(image_path)
    if img.shape[1]<img.shape[0]:
        img = img.transpose(1,0,2)
    if not np.all(img.shape[0:2]==out_size):
        if img.shape[0]<out_size[0] or img.shape[1]<out_size[1]:
            return -1
        h,w = img.shape[0:2]
        h_,w_ = out_size
        h_diff, w_diff = h-h_, w-w_
        h_sub, w_sub = h_diff/2, w_diff/2
        new_img = img[h_sub:-(h_sub+h_diff%2),w_sub:-(w_sub+w_diff%2),:]
        return new_img

def load_label_SBD_resize(mat_path, out_size=(321,481)):
    mat = loadmat(mat_path)
    all_labels = []
    bounds = mat['GTcls'][0,0][0]
    for i in range(20): #Number of classes in VOC is 20
        cls_bound = bounds[i][0].toarray() #Extract binary contour map for class i
        if cls_bound.sum()>0: #Contour exists, array with zeros and ones
            all_labels.append(cls_bound*(i+1))
    all_labels = np.array(all_labels)
    all_labels = all_labels.sum(axis=0)
    #Shift from portrait to landscape
    if all_labels.shape[1] < all_labels.shape[0]:
        all_labels = all_labels.transpose(1,0)
    #Cropping the center of size (321, 481) for SBD
    h, w = all_labels.shape
    h_, w_ = out_size
    h_diff, w_diff = h-h_, w-w_
    h_sub, w_sub = h_diff/2, w_diff/2
    new_label = all_labels[h_sub:-(h_sub+h_diff%2),w_sub:-(w_sub+w_diff%2)]
    all_labels[all_labels>0] = 1
    new_label[new_label>0] = 1
    return all_labels, new_label

def load_label_SBD(mat_path):
    mat = loadmat(mat_path)
    all_labels = []
    bounds = mat['GTcls'][0,0][0]
    for i in range(20): #Number of classes in VOC is 20
        cls_bound = bounds[i][0].toarray() #Extract binary contour map for class i
        if cls_bound.sum()>0: #Contour exists, array with zeros and ones
            all_labels.append(cls_bound*(i+1))
    all_labels = np.array(all_labels)
    all_labels = all_labels.sum(axis=0)
    all_labels[all_labels>0] = 1

def get_label_image(image_path, mat_path, output_size=(321,481)):
    img = load_image(image_path,out_size=output_size)
    if type(img)==int:
        return -1,-1
    mat_,mat_rsz = load_label_SBD_resize(mat_path,out_size=output_size)
    mat = mat_rsz*255.
    assert img.shape[0:2] == mat.shape, 'Different shape for image and label'
    img = img.astype(np.float32)
    mat = mat.astype(np.float32)
    if img.max()>1.:
        img = img/255.
    mat[mat>0] = 1.
    return img, mat

def main():
    import sys
    img_path = sys.argv[1]
    mat_path = sys.argv[1].replace('images','groundTruth').replace('jpg','mat')
    img, mat = get_label_image(img_path,mat_path)
    print img.shape, img.dtype
    print mat.shape, mat.dtype
    plt.subplot(1,2,1); plt.imshow(img); plt.subplot(1,2,2); plt.imshow(mat); plt.show()

if __name__=="__main__":
    main()
