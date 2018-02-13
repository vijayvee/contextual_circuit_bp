#!/usr/bin/python
"""Script to prepare images and labels for VOC Contours"""

import cv2
import imageio
import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy.io import loadmat

def load_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img

def load_label(mat_path):
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
    return all_labels

def get_label_image(image_path, mat_path, output_size=(400,400)):
    img = load_image(image_path)
    mat = load_label(mat_path)
    mat = mat*255.
    img = cv2.resize(img, output_size)
    mat = cv2.resize(mat, output_size)
    mat[mat>0] = 1.
    if img.max()>1.:
        img = img/255.
    return img, mat

def main():
    import sys
    img_path = sys.argv[1]
    mat_path = sys.argv[1].replace('images','groundTruth').split('.')[0] + '.mat'
    img, mat = get_label_image(img_path,mat_path)
    plt.subplot(1,2,1); plt.imshow(img); plt.subplot(1,2,2); plt.imshow(mat); plt.show()

if __name__=="__main__":
    main()
