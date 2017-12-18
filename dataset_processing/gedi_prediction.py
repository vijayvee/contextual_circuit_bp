import numpy as np
from ops import tf_fun
from config import Config


class data_processing(object):
    """Template file for GEDI."""
    def __init__(self):
        """Init variables for reading from GEDI prediction."""
        self.name = 'GEDI_prediction'
        self.config = Config()
        self.output_size = [2, 1]
        self.im_size = (224, 224, 1)
        self.model_input_image_size = [224, 224, 1]
        self.default_loss_function = 'cce'
        self.score_metric = 'accuracy'
        self.preprocess = [None]

        # Load vars from the meta file
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
