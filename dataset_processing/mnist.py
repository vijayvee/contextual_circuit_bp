import os
import tensorflow as tf
from glob import glob
from config import Config
from ops import tf_fun
from utils import py_utils
import random


class data_processing(object):
    def __init__(self):
        """MNIST init function."""
        self.name = 'mnist'
        self.extension = '.png'
        self.config = Config()
        self.output_size = [10, 1]
        self.im_size = [28, 28, 1]
        self.model_input_image_size = [28, 28, 1]
        self.default_loss_function = 'cce'
        self.score_metric = 'accuracy'
        self.preprocess = [None]
        self.shuffle = True  # Preshuffle data?
        self.folds = {
            'train': 'training',
            'test': 'testing'
        }
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
        for k, fold in self.folds.iteritems():
            it_files = []
            dirs = glob(
                os.path.join(
                    self.config.data_root,
                    self.name,
                    fold,
                    '*'))
            for d in dirs:
                it_files += [glob(
                    os.path.join(
                        d,
                        '*%s' % self.extension))]
            it_files = py_utils.flatten_list(it_files)
            if self.shuffle:
                random.shuffle(it_files)
            files[k] = it_files
        return files

    def get_labels(self, files):
        labels = {}
        for k, v in files.iteritems():
            it_labels = []
            for f in v:
                it_labels += [int(f.split('/')[-2])]
            labels[k] = it_labels
        return labels
