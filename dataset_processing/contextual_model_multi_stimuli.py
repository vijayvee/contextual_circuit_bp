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
        self.name = 'contextual_model_multi_stimuli'
        self.figures = [
            'f3a.npz',
            'f4.npz',
            'f5.npz',
            # 'tbp.npz',
            'tbtcso.npz',
            # 'bw.npz'
            ]
        self.target_data = 'label_dict'
        self.config = Config()
        self.output_size = [42, 1]
        self.im_size = (256, 51, 51, 64)
        self.model_input_image_size = [256, 51, 51, 64]
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
            'label': tf_fun.fixed_len_feature(
                dtype='float',
                length=self.output_size[0])
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

    def flatten_and_pad_dict(self, d, flatten=False, constant=0.):
        """Flatten each dict entry then pad to the largest sized one."""
        if flatten:
            d = {k: v.reshape(-1) for k, v in d.iteritems()}
        sizes = np.concatenate(
                    [np.asarray(v.shape).reshape(1, -1) for v in d.values()])
        m_sizes = np.max(sizes, axis=0)
        for k, v in d.iteritems():
            v_shape = v.shape
            out_array = np.zeros(m_sizes)
            # Hardcoded tensor/vector operations
            if len(m_sizes) == 1:
                out_array[:v_shape[0]] = v
            else:
                out_array[
                    :v_shape[0],
                    :v_shape[1],
                    :v_shape[2],
                    :v_shape[3]] = v
            if flatten:
                out_array = list(out_array)
            d[k] = out_array
        return d

    def get_data(self):
        """Called by encode_dataset.py to create a TFrecords."""
        stim_dict, label_dict, model_dict = {}, {}, {}
        for f in self.figures:
            # Load each and add to a dict
            npz_file = os.path.join(
                self.config.data_root,
                self.name,
                '%s' % f)
            data = np.load(npz_file)
            fig_name = f.split('.')[0]
            stim_dict[fig_name] = data['stim']
            label_dict[fig_name] = data['gt']
            model_dict[fig_name] = data['model_pred']

        # Pad each with 0s to the largest size
        stim_dict = self.flatten_and_pad_dict(stim_dict, flatten=False)
        label_dict = self.flatten_and_pad_dict(label_dict, flatten=True)
        model_dict = self.flatten_and_pad_dict(model_dict, flatten=True)

        # Package for TFrecords
        files = {k: stim_dict.values() for k in self.folds.keys()}
        if self.target_data == 'label_dict':
            labels = {k: label_dict.values() for k in self.folds.keys()}
        elif self.target_data == 'model_dict':
            labels = {k: model_dict.values() for k in self.folds.keys()}
        else:
            raise RuntimeError('Select a target dictionary for TFrecords.')
        return files, labels
