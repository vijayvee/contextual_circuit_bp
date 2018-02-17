import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from glob import glob
from config import Config
from ops import tf_fun
from utils import py_utils
from scipy import io, misc
from VOC_Contours import *

class data_processing(object):
    def __init__(self):
        self.name = 'BSDS_SBD'
        self.dataset_roots = ['BSDS500', 'SBD']
        self.orig_name = 'BSDS_SBD'
        self.im_extension = '.jpg'
        self.lab_extension = '.mat'
        self.images_dir = 'images'
        self.labels_dir = 'groundTruth'
        self.processed_labels = 'processed_labels'
        self.processed_images = 'processed_images'
        self.config = Config()
        self.im_size = [320, 480, 3]
        self.lab_size = (320,480) #Opposite to convention, opencv standards
        self.model_input_image_size = [320, 480, 3] #[150, 240, 3]  # [107, 160, 3]
        self.output_size = [320, 480, 1]
        self.label_size = self.output_size
        self.default_loss_function = 'pearson'
        self.score_metric = 'pearson'
        self.aux_scores = ['f1']
        self.preprocess = [None]  # ['resize_nn']
        self.folds = {
            'train': 'train',
            'val': 'val'
        }
        self.fold_options = {
            'train': 'duplicate',
            'val': 'mean'
        }
        self.targets = {
            'image': tf_fun.bytes_feature,
            'label': tf_fun.bytes_feature
        }
        self.tf_dict = {
            'image': tf_fun.fixed_len_feature(dtype='string'),
            'label': tf_fun.fixed_len_feature(dtype='string')
        }
        self.tf_reader = {
            'image': {
                'dtype': tf.float32,
                'reshape': self.im_size
            },
            'label': {
                'dtype': tf.float32,
                'reshape': self.output_size
            }
        }

    def get_data(self):
        files = self.get_files()
        labels, files = self.get_labels(files)

        return files, labels

    def get_files(self):
        """Get the names of files."""
        files = {}
        for k, fold in self.folds.iteritems():
            it_files = []
            for dst_root in self.dataset_roots: #For storing images from multiple datasets
                it_files.extend(glob(
                    os.path.join(
                        self.config.data_root,
                        dst_root,
                        self.images_dir,
                        fold,
                        '*%s' % self.im_extension)))
            files[k] = it_files
        return files

    def create_label_image_dirs(sample_img_path, fold):
        """Create directories for labels and images"""
        proc_dir = os.path.join(
            sample_img_path.split(fold)[0],
            fold,
            self.processed_labels)
        py_utils.make_dir(proc_dir)
        proc_image_dir = os.path.join(
            self.config.data_root,
            self.name,
            self.images_dir,
            fold,
            self.processed_images)
        py_utils.make_dir(proc_image_dir)

    def get_labels_SBD(self, im, it_label_path):
        im_data, ip_lab = get_label_image(im, it_label_path, output_size=self.lab_size)
        transpose_labels = False
        if not np.all(self.im_size == list(im_data.shape)):
            im_data = np.swapaxes(im_data, 0, 1)
            transpose_labels = True
        assert np.all(
            self.im_size == list(im_data.shape)
            ), 'Mismatched dimensions.'
        if transpose_labels:
            ip_lab = np.swapaxes(ip_lab, 0, 1)
        it_im_name = it_label #Copying image name from a previously assigned variable
        it_lab_name = '%s.npy' % it_im_name.split('.')[0]

    def get_labels(self, files):
        """Process and save label images."""
        labels = {}
        new_files = {}
        for k, images in files.iteritems():
            # Replace extension and path with labels
            label_vec = []
            file_vec = []
            fold = k
            # New label dir
            create_label_image_dirs(images[0], fold) #Create new directories for images and labels
            # New image dir
            for im in tqdm(images,total=len(images),desc='Storing %s labels and images for %s'%(self.name,k)):
                it_label = im.split(os.path.sep)[-1] #Get image name
                it_label_path = im.replace(self.im_extension, self.lab_extension)
                it_label_path = it_label_path.replace(self.images_dir, self.labels_dir)
                #Obtain image and contour label in im_size shape
                if 'SBD' in im_data:
                    get_labels_SBD(im, it_label_path)
                else:
                    get_labels_BSDS(im, it_label_path)
                out_lab = os.path.join(proc_dir, it_lab_name)
                np.save(out_lab, ip_lab)
                label_vec += [out_lab]
                # Process images
                proc_im = os.path.join(proc_image_dir, it_im_name)
                #misc.imsave(proc_im, im_data)
                np.save(proc_im + '.npy',im_data.astype(np.float32))
                file_vec += [proc_im + '.npy']
                #Cannot compute z-score for SBD, too many images
            labels[k] = label_vec
            new_files[k] = file_vec
        return labels, new_files
