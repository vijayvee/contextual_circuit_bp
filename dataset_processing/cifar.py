import os
import re
from glob import glob
from config import config


class data_processing(object):
    def __init__(self):
        self.name = 'cifar'
        self.extension = '.png'
        self.config = config()
        self.folds = {
            'train': 'train',
            'test': 'test']

    def __enter__(self):
        files = self.get_files()
        labels = self.get_labels(files)
        return files, labels

    def get_files(self):
        files = {}
        for k, fold in self.folds.iteritems():
            files[k] = glob(
                os.path.join(
                    self.config.data_root,
                    fold,
                    '*%s' % self.extension))
        return files

    def get_labels(self, files):
        labels = {}
        for k, v in files.iteritems():
            it_labels = []
            for f in v:
                it_labels += [int(re.split('\.', f.split('_')[-1])[0])]
            labels[k] = it_labels
        return labels

