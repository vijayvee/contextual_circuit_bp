import os
import re
import numpy as np
from glob import glob
from config import Config
from ops import tf_fun
import random


class data_processing(object):
    def __init__(self):
        self.name = 'coco_2014'
        self.aux_dir = 'coco_images'
        self.extension = '.jpg'
        self.config = Config()
        self.folds = {
            'train': 'train2014',
            'val': 'val2014',
            'test': 'test2014'
        }
        self.targets = {
            'image': tf_fun.bytes_feature,
            'label': tf_fun.int64_feature
        }
        self.tf_dict = {
            'image': tf_fun.fixed_len_feature(dtype='string'),
            'label': tf_fun.fixed_len_feature(dtype='int64')
        }
        self.output_size = [10, 1]
        self.im_size = [256, 256, 3]
        self.image_meta_file = 'coco_full_im_processed_labels.npz'
        self.preprocess = ['center_crop']
        self.shuffle = False  # Preshuffle data?

    def get_data(self):
        files = self.get_files()
        labels = self.get_labels(files)
        return files, labels

    def get_files(self):
        files = {}
        for k, fold in self.folds.iteritems():
            it_files = glob(
                os.path.join(
                    self.config.data_root,
                    self.name,
                    self.aux_dir,
                    fold,
                    '*%s' % self.extension))
            if self.shuffle:
                random.shuffle(it_files)
            files[k] = it_files
        return files

    def get_labels(self, files):
        labels = {}
        meta_file = np.load(
            os.path.join(
                self.config.data_root,
                self.name,
                self.aux_dir,
                self.image_meta_file))
        image_map = meta_file['training_image_map']
        mapped_labels = meta_file['training_image_category_id']
        for k, v in files.iteritems():
            it_labels = []
            for f in v:
                it_labels += [mapped_labels[mapped_labels[image_map == '/%s' % f]]]
            labels[k] = it_labels
        return labels

