import os
import numpy as np
import tensorflow as tf
import pandas as pd
from config import Config
from ops import tf_fun
from glob import glob


class data_processing(object):
    def __init__(self):
        self.name = 'spikefinder'
        self.config = Config()
        self.file_extension = '.csv'
        self.timepoints = 5  # Data is 100hz
        self.trim_nans = True
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

    def create_events(self, data):
        """Re-encode each dimension of events into a self.timepoints array.

        The goal is to predict the last element in event_array.
        """
        event_array = []
        for idx in range(data.shape[-1]):
            it_neuron = data[:, idx]
            it_event_array = []
            for rs in range(it_neuron.shape[0] - self.timepoints):
                it_event = it_neuron[rs:rs + self.timepoints]
                if self.trim_nans and not np.any(np.isnan(it_event)):
                    # Exclude events with NaNs
                    it_event_array += [[it_event]]
                elif not self.trim_nans:
                    it_event_array += [[it_event]]
                else:
                    raise NotImplementedError
            event_array += [np.asarray(it_event_array)]
        return np.concatenate(event_array, axis=0).squeeze()

    def get_data(self):
        cv_spikes, cv_traces = {}, {}
        for k, v in self.folds.iteritems():
            it_files = np.asarray(
                glob(
                    os.path.join(
                        self.config.data_root,
                        k,
                        '*%s' % self.file_extension)))
            # Trim to the file index
            it_idx = np.asarray(
                [int(
                    x.split(
                        os.path.sep)[-1].split('.')[0]) for x in it_files])

            # Encode into events with self.timepoints
            calcium_data, spike_data = [], []
            for idx in np.unique(it_idx):
                idx_files = it_files[it_idx == idx]
                calcium_file = [x for x in idx_files if 'calcium' in x][0]
                calcium = pd.read_csv(calcium_file).as_matrix()
                calcium_data += [self.create_events(calcium)]
                if k != 'test':
                    spike_file = [x for x in idx_files if 'spikes' in x][0]
                    spikes = pd.read_csv(spike_file).as_matrix()
                    spike_data += [self.create_events(spikes)]

            # Store in the dictionary
            cv_traces[k] = np.concatenate(calcium_data, axis=0).squeeze()
            cv_spikes[k] = np.concatenate(spike_data, axis=0).squeeze()
        return cv_traces, cv_spikes
