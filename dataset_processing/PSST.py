import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from glob import glob
from config import Config
from ops import tf_fun
from utils import py_utils
from scipy import io, misc
from random import sample
from psst_master.instances import psst
import sys
sys.path.append(os.path.abspath(os.path.join('..')))

class data_processing(object):
    def __init__(self):
        self.name = 'PSST'
        self.orig_name = 'PSST'
        self.images_dir = 'images'
        self.labels_dir = 'groundTruth'
        self.processed_labels = 'processed_labels'
        self.processed_images = 'processed_images'
        self.config = Config()
        self.im_size = [200, 200]
        self.model_input_image_size = [200, 200]  # [107, 160, 3]
        self.output_size = [1] #classification/regression label
        self.label_size = self.output_size
        self.default_loss_function = 'pearson' #Change to cce later
        self.score_metric = 'pearson'
        self.aux_scores = ['f1']
        self.preprocess = [None]  # ['resize_nn']
        self.folds = {
            'train': 'train',
            'val': 'val'
        }
        self.targets = {
            'image': tf_fun.bytes_feature,
            'label': tf_fun.bytes_feature
        }
        self.tf_dict = {
            'image': tf_fun.fixed_len_feature(dtype='string'),
            'label': tf_fun.fixed_len_feature(dtype='string')
        }
        self.tf_reader = {
            'image': {
                'dtype': tf.float32,
                'reshape': self.im_size
            },
            'label': {
                'dtype': tf.float32,
            }
            }
        self.n_imgs_fold = {
            'train':0,
            'val':0
        }
        self.n_counts = 6
        self.n_img_per_count = 10000


    def get_data(self):
        labels, files = self.get_labels(n_counts=self.n_counts, n_img_per_count=self.n_img_per_count)
        return files, labels

    def get_labels(self, n_counts, n_img_per_count):
        """Process and save label images."""
        labels = {}
        new_files = {}
        counts = range(1,n_counts+1)
        total_n_imgs = n_counts*n_img_per_count
        self.n_imgs_fold['train'] = int(0.7*total_n_imgs)
        self.n_imgs_fold['val'] = int(0.3*total_n_imgs)
        for k in self.folds.keys():
            n_imgs = self.n_imgs_fold[k]
            label_vec = []
            file_vec = []
            fold = k
            # New label dir
            proc_dir = os.path.join(
                self.config.data_root,
                self.name,
                self.labels_dir,
                fold,
                self.processed_labels
                )
            py_utils.make_dir(proc_dir)

            # New image dir
            proc_image_dir = os.path.join(
                self.config.data_root,
                self.name,
                self.images_dir,
                fold,
                self.processed_images)
            py_utils.make_dir(proc_image_dir)
            all_images = []
            for i in tqdm(range(n_imgs),desc='Generating images for %s..'%(fold)):
                # Replace extension and path with labels
                it_label = 'gt_%06d'%(i) + '.npy'
                it_label_path = proc_dir + '/' + it_label
                label_data = sample(counts,1)[0] #Randomly sampling a number of objects
                it_im_name = 'img_%06d_%s.npy'%(i, label_data)
                im_data = psst.generate_image(label_data)
                label_vec += [label_data] #Since labels are just counts, append the label value to label_vec
                proc_im = os.path.join(proc_image_dir, it_im_name)
                np.save(proc_im, im_data)
                file_vec += [proc_im]
            labels[k] = label_vec
            new_files[k] = file_vec
        return labels, new_files
