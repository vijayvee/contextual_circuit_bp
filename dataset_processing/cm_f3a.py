"""Class for creating contextual model stimulus TFrecords."""
import os
import numpy as np
import tensorflow as tf
from config import Config
from ops import tf_fun


class data_processing(object):
    """Tilt-illusion from Contextual modeling paper (fig3a)."""

    def __init__(self):
        """Init global variables for contextual circuit bp."""
        self.name = 'contextual_model_stimuli'
        self.figure_name = 'f3a'
        self.config = Config()
        self.output_size = [1, 1]
        self.im_size = (51, 51, 1)
        self.model_input_image_size = [51, 51, 1]
        self.default_loss_function = 'pearson'
        self.score_metric = 'pearson'
        self.preprocess = [None]
        self.folds = {
            'train': 'train',
            'test': 'test'}
        self.targets = {
            'image': tf_fun.bytes_feature,
            'label': tf_fun.float_feature
        }
        self.tf_dict = {
            'image': tf_fun.fixed_len_feature(dtype='string'),
            'label': tf_fun.fixed_len_feature(dtype='float')
        }
        self.tf_reader = {
            'image': {
                'dtype': tf.float32,
                'reshape': self.im_size
            },
            'label': {
                'dtype': tf.float32,
                'reshape': None
            }
        }

    def get_data(self):
        """Called by encode_dataset.py to create a TFrecords."""
        data_file = os.path.join(
            self.config.data_root,
            self.name,
            '%s_stim.npy' % self.figure_name)
        label_file = os.path.join(
            self.config.data_root,
            self.name,
            '%s_gt.npy' % self.figure_name)
        files = np.expand_dims(np.load(data_file).squeeze(), axis=-1)
        labels = np.load(label_file)
        files = {k: files for k in self.folds.keys()}
        labels = {k: labels for k in self.folds.keys()}
        return files, labels
