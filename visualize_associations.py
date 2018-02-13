#!/usr/bin/python
"""Script to visualize association fields, conv kernels etc. """
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib import gridspec

def get_p_r(weight_key):
    """Function to get P_r, the association field weight matrix"""
    p_r = np.load(weight_key)
    rev_p_r = p_r.transpose((2,3,0,1))
    min_pr, max_pr = rev_p_r.min(), rev_p_r.max()
    norm_pr = (rev_p_r - min_pr)/(max_pr - min_pr)
    return norm_pr

def get_conn_mat(weight_key):
    """Function that computes the connectivity matrix for conv1 kernels"""
    norm_pr = get_p_r(weight_key)
    conn_mat = norm_pr.max(2,keepdims=True).max(-1,keepdims=True)
    return conn_mat

def get_conv1(weight_key):
    conv1 = np.load(weight_key)
    conv1_rev = conv1.transpose(3,0,1,2)
    conv1_min = conv1_rev.min(1,keepdims=True).min(2,keepdims=True).min(-1,keepdims=True)
    conv1_max = conv1_rev.max(1,keepdims=True).max(2,keepdims=True).max(-1,keepdims=True)
    conv1_norm = (conv1_rev - conv1_min)/(conv1_max - conv1_min)
    return conv1_norm

def get_conv1_alexnet(weight_key):
    weight_dict = np.load(weight_key)
    weight_conv1 = weight_dict.item().get('conv1')
    conv1 = weight_conv1[0]
    conv1_rev = conv1.transpose(3,0,1,2)
    conv1_min = conv1_rev.min(1,keepdims=True).min(2,keepdims=True).min(-1,keepdims=True)
    conv1_max = conv1_rev.max(1,keepdims=True).max(2,keepdims=True).max(-1,keepdims=True)
    conv1_norm = (conv1_rev - conv1_min)/(conv1_max - conv1_min)
    return conv1_norm

def plot_conv_association(i,j,norm_pr,conv1_norm):
    plt.figure()
    plt.subplot(1,3,1)
    plt.title('Conv filter A')
    plt.imshow(conv1_norm[i])
    plt.subplot(1,3,2)
    plt.title('Conv filter B')
    plt.imshow(conv1_norm[j])
    plt.subplot(1,3,3)
    plt.title('Near surround\n"association field"')
    plt.imshow(norm_pr[i,j,:,:])
    plt.axis('off')
    plt.show()
    plt.close()

def plot_connectivity_matrix(i,j,norm_pr):
    for ii in range(i):
        for jj in range(j):
            plt.subplot(i,j,((ii)*i)+(jj+1))
            plt.imshow(norm_pr[ii,jj,:,:])
            plt.axis('off')
    plt.show()
    plt.close()

def plot_conn_conv(i,j,norm_pr):
    for ii in range(i):
        for jj in range(j):
            plt.subplot(i,j,((ii)*i)+(jj+1))
            plt.imshow(norm_pr[ii,jj,:,:])
            plt.axis('off')
    plt.show()
    plt.close()

def plot_connectivity_matrix(i,j,norm_pr,conv1_norm):
     for ii in range(i+1):
         for jj in range(j+1):
             if ii+jj==0:
                 continue
             if ii==0:
                 #Plotting conv kernels on the first row
                 plt.subplot(i+1,j+1,(ii*i)+(jj+ii)+1)
                 plt.imshow(conv1_norm[jj-1,:,:,:])
                 plt.axis('off')
                 continue
             if jj==0:
                 #Plotting conv kernels on the first column
                 plt.subplot(i+1,j+1,(ii*i)+(jj+ii)+1)
                 plt.imshow(conv1_norm[ii-1,:,:,:])
                 plt.axis('off')
                 continue
             #Plotting association field kernels
             plt.subplot(i+1,j+1,((ii)*(i))+(jj+ii)+1)
             plt.imshow(norm_pr[ii-1,jj-1,:,:])
             plt.axis('off')
     plt.show()
     plt.close()

def plot_conv_kernels(i,j,conv1_norm):
    for ii in range(i):
        for jj in range(j):
            plt.subplot(i,j,(ii*i)+(jj+1))
            plt.imshow(conv1_norm[ii*i+jj,:,:,:])
    plt.show()
    plt.close()

def plot_conv_association_color(i,j,norm_pr,conv1_norm):
    plt.figure()
    plt.subplot(1,3,1)
    plt.title('Conv filter A')
    plt.imshow(conv1_norm[i])
    plt.subplot(1,3,2)
    plt.title('Conv filter B')
    plt.imshow(conv1_norm[j])
    plt.subplot(1,3,3)
    plt.title('Near surround\n"association field"')
    plt.imshow(norm_pr[i,j,:,:])
    plt.axis('off')
    plt.show()
    plt.close(f)
