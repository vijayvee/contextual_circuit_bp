import os
import shutil
import numpy as np
import tensorflow as tf
import pandas as pd
from config import Config
from ops import tf_fun
from utils import py_utils
from glob import glob


class data_processing(object):
    def __init__(self):
        self.name = 'spikefinder'
        self.config = Config()
        self.output_size = [1, 1]
        self.im_size = [1, 1]
        self.model_input_image_size = [1, 1]
        self.default_loss_function = 'l2'
        self.score_metric = 'l2'
        self.preprocess = [None]
        self.folds = {
            'train': 'train',
            'test': 'test'}
        self.targets = {
            'image': tf_fun.float_feature,
            'label': tf_fun.float_feature
        }
        self.tf_dict = {
            'image': tf_fun.fixed_len_feature(dtype='float'),
            'label': tf_fun.fixed_len_feature(dtype='float')
        }
        self.tf_reader = {
            'image': {
                'dtype': tf.float32,
                'reshape': None
            },
            'label': {
                'dtype': tf.float32,
                'reshape': None
            }
        }

    def get_data(self):
        files = self.get_files()
        labels = self.get_labels(files)
        return files, labels

    def get_files(self):
        files = {}
        all_ims = np.load(os.path.join(
            self.config.data_root,
            self.name,
            self.data_file))[self.file_key]
        all_ims = np.asarray([x for x in all_ims if x not in self.ignore_ims])

        # Create folders for training/validation splits
        self.rand_order = np.random.permutation(len(all_ims))
        self.test_split = np.round(len(all_ims) * self.crossval_split).astype(int)
        shuffled_ims = all_ims[self.rand_order]
        test_ims = shuffled_ims[:self.test_split]
        train_ims = shuffled_ims[self.test_split:]
        target_test_ims = [os.path.join(
            self.config.data_root,
            self.name,
            self.folds['test'],
            f) for f in test_ims]
        target_train_ims = [os.path.join(
            self.config.data_root,
            self.name,
            self.folds['train'],
            f) for f in train_ims]
        test_ims = [os.path.join(
            self.config.data_root,
            self.name,
            f) for f in test_ims]
        train_ims = [os.path.join(
            self.config.data_root,
            self.name,
            f) for f in train_ims]

        py_utils.make_dir(os.path.join(
            self.config.data_root,
            self.name,
            self.folds['test']))
        py_utils.make_dir(os.path.join(
            self.config.data_root,
            self.name,
            self.folds['train']))
        [shutil.copyfile(s, t) for s, t in zip(test_ims, target_test_ims)]
        [shutil.copyfile(s, t) for s, t in zip(train_ims, target_train_ims)]
        files = {
            self.folds['test']: test_ims,
            self.folds['train']: train_ims
        }
        return files

    def get_labels(self, files):
        label_dict = np.load(os.path.join(
            self.config.data_root,
            self.name,
            self.data_file))
        all_labels = label_dict[self.label_key][self.rand_order]
        labels = {
            self.folds['test']: all_labels[:self.test_split],
            self.folds['train']: all_labels[self.test_split:]
        }
        return labels
