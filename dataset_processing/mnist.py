import os
import re
from glob import glob
from config import config
from ops import tf_fun


class data_processing(object):
    def __init__(self):
        self.name = 'mnist'
        self.extension = '.png'
        self.config = config()
        self.folds = {
            'train': 'training',
            'test': 'testing']
        self.targets = {
            'image': tf_fun.bytes_feature,
            'label': tf_fun:int64_feature
        }
        self.im_size = [28, 28, 1]

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
                    fold,
                    '*%s' % self.extension))
            for d in dirs:
                it_files += [glob(
                    os.path.join(
                        self.config.data_root,
                        d,
                        fold,
                        '*%s' % self.extension)
            files[k] = putils.flatten_list(it_files)
        return files

    def get_labels(self, files):
        labels = {}
        for k, v in files.iteritems():
            it_labels = []
            for f in v:
                it_labels += [int(f.split('/')[-2])]
            labels[k] = it_labels
        return labels

