import os
import numpy as np
import tensorflow as tf
from config import Config
from ops import tf_fun
from glob import glob
from tqdm import tqdm
import scipy
from scipy import io as spio
from scipy import misc


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs matobject nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


class data_processing(object):
    def __init__(self):
        self.name = 'sheinberg_data'
        self.config = Config()
        self.output_size = [1, 1]
        self.im_size = [1, 1]
        self.model_input_image_size = [1, 1]
        self.default_loss_function = 'l2'
        self.score_metric = 'l2'
        self.preprocess = [None]
        self.im_ext = '.jpg'
        self.im_folder = 'scene_images'
        self.neural_data = 'LFP'  # 'spike'
        self.val_set = -1
        # Recording starts 200msec before onset.
        # Target is 50 - 150ms. = 270 - 370.
        self.spike_range = [250, 350]
        self.resize = [192, 256]
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
        neural_files = glob(
            os.path.join(
                self.config.data_root,
                self.name,
                'scene*.mat'))
        # rf_files = glob(
        #     os.path.join(
        #         self.config.data_root,
        #         self.name,
        #         'rf*.mat'))
        scene_images = glob(
            os.path.join(
                self.config.data_root,
                self.name,
                self.im_folder,
                '*%s' % self.im_ext))
        scene_labels = np.asarray(
            [x.split('/')[-1].split(self.im_ext)[0]
                for x in scene_images])
        files = []
        labels = []
        neural_channels = {}
        bin_range = np.arange(self.spike_range[0], self.spike_range[1])
        for f in tqdm(
                neural_files,
                total=len(neural_files),
                desc='Processing Sheinberg data'):
            data = loadmat(f)['data']
            stim_names = data['trial_info']['stim_names']
            it_images = []
            it_labels = []
            for st in stim_names:
                st = st.split('\x00')[0].replace(' ', '')
                it_image = misc.imread(
                    os.path.join(
                        self.config.data_root,
                        self.name,
                        self.im_folder,
                        '%s%s' % (st, self.im_ext)))
                if self.resize is not None:
                    it_image = misc.imresize(it_image, self.resize)
                it_images += [
                    {st: np.expand_dims(it_image, axis=0)}
                ]
                it_labels += [np.where(np.asarray(st) == scene_labels)]
            it_channel_key = np.asarray(data['spike']['channel'])
            it_neural = np.asarray(data['spike']['data'])
            unique_channels = np.unique(it_channel_key)
            for channel in unique_channels:
                channel_idx = np.where(it_channel_key == channel)[0]
                proc_it_neural = np.expand_dims(
                    it_neural[channel_idx].sum(0)[bin_range].sum(0),
                    axis=-1)
                if channel not in neural_channels.keys():
                    neural_channels[channel] = [proc_it_neural]
                else:
                    neural_channels[channel] += [proc_it_neural]
            # TODO: Visualize gaussian-smoothed spikes here
            files += [it_images]
            labels += [neural_channels]

        # Combine images
        all_image_map = {}
        for fidx, f in enumerate(files):
            for itidx, it_f in enumerate(f):
                if 
                all_image_names += [it_f.keys()[0]]


        all_image_key = {}
        for 

        # Mean spikes across stimuli
        all_channels = np.unique([d.keys() for d in labels])
        for l in labels:


        # Organize images according to this ordering

        # Split labels/files into training/testing (leave one session out).
        out_files = {  # Images
            'train': np.concatenate(files[:self.val_set], axis=0),
            'val': np.concatenate(files[self.val_set:], axis=0)
        }
        out_labels = {  # Neural data
            'train': np.concatenate(labels[:self.val_set], axis=0),
            'val': np.concatenate(labels[self.val_set:], axis=0)
        }
        return files, labels

