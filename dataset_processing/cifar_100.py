import os
from glob import glob
from config import Config
from ops import tf_fun
from utils import py_utils
import random


class data_processing(object):
    def __init__(self):
        self.name = 'cifar_100'
        self.extension = '.png'
        self.config = Config()
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
        self.output_size = [100, 1]
        self.im_size = [32, 32, 3]
        self.label_list = 'labels.txt'
        self.default_loss_function = 'cce'
        self.preprocess = [None]
        self.shuffle = True  # Preshuffle data?

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
        with open(
            os.path.join(
                self.config.data_root,
                self.name,
                self.label_list)) as fp:
            label_list = fp.read().split('\n')
        for k, v in files.iteritems():
            it_labels = []
            for f in v:
                it_labels += [label_list.index(f.split('/')[-2])]
            labels[k] = it_labels
        return labels
