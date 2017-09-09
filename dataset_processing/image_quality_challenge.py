import os
import shutils
import numpy as np
import tensorflow as tf
from glob import glob
from config import Config
from ops import tf_fun
from utils import py_utils
import random


class data_processing(object):
    def __init__(self):
        self.name = 'ChallengeDB_release'
        self.config = Config()
        self.output_size = [100, 1]
        self.im_size = [500, 500, 3]
        self.model_input_image_size = [256, 256, 3]
        self.data_file = 'processed_scores.npz'
        self.file_key = 'im_files'
        self.label_key = 'im_scores'
        self.default_loss_function = 'mse'
        self.score_metric = 'accuracy'
        self.crossval_split = 0.1
        self.preprocess = ['resize']
        self.shuffle = True  # Preshuffle data?
        self.folds = {
            'train': 'train',
            'test': 'test'}
        self.targets = {
            'image': tf_fun.bytes_feature,
            'label': tf_fun.int64_feature
        }
        self.tf_dict = {
            'image': tf_fun.fixed_len_feature(dtype='string'),
            'label': tf_fun.fixed_len_feature(dtype='int64')
        }
        self.tf_reader = {
            'image': {
                'dtype': tf.float32,
                'reshape': self.im_size
            },
            'label': {
                'dtype': tf.int64,
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
            self.data_file))))[self.file_key]

        # Create folders for training/validation splits
        self.rand_order = np.random.permutations(len(all_ims))]
        self.test_split = np.round(len(all_ims) * self.crossval_split)
        shuffled_ims = all_ims[self.rand_order]
        test_ims = shuffled_ims[:self.test_split]
        train_ims = shuffled_ims[self.test_split:]
        test_ims = [os.path.join(
            self.config.data_root,
            self.name,
            self.folds['test'],
            f) for f in split_test_ims]
        train_ims = [os.path.join(
            self.config.data_root,
            self.name,
            self.folds['train'],
            f) for f in split_train_ims]

        py_utils.make_dir(self.folds['test'])
        [shutils.copyfile(f) for f in test_ims]
        py_utils.make_dir(self.folds['train'])
        [shutils.copyfile(f) for f in train_ims]
        files = {
            self.folds['test']: test_ims,
            self.folds['train']: train_ims
        }
        return files

    def get_labels(self, files):
        label_dict = np.load(os.path.join(
            self.config.data_root,
            self.name,
            self.data_file))))
        all_labels = label_dict[self.label_key][self.rand_order]
        labels = {
            self.folds['test']: all_labels[:self.test_split],
            self.folds['train']: all_labels[self.test_split:]
        }
        return labels
