import os
import re
import sys
import numpy as np
import tensorflow as tf
from db import db
from config import Config
from argparse import ArgumentParser
from datetime import datetime
from utils import logger
from utils import py_utils
from models import experiments
from ops import data_loader


def add_to_config(d, config):
    for k, v in d.iteritems():
        setattr(config, k, v)
    return config


def process_DB_exps(config):
    exp_params, exp_id = db.get_parameters()
    if exp_id is None:
        print 'No empty experiments found. Exiting.'
        sys.exit(1)
    for k, v in exp_params.iteritems():
        if isinstance(v, basestring) and '{' in v and '}' in v:
            v = v.strip('{').strip('}').split(',')
        setattr(config, k, v)
    return config


def get_data_pointers(dataset, base_dir, cv, log):
    data_pointer = os.path.join(base_dir, '%s_%s.tfrecords' % (dataset, cv))
    data_means = os.path.join(base_dir, '%s_%s_means.npy' % (dataset, cv))
    data_means = np.load(data_means)
    log.info(
        'Using %s tfrecords: %s' % (
            cv,
            data_pointer)
        )
    return data_pointer, data_means


def get_dt_stamp():
    return re.split(
        '\.', str(datetime.now()))[0].replace(
        ' ',
        '_').replace(
        ':',
        '_').replace(
        '-',
        '_')


def main(reset_process, initialize_db, experiment_name):
    """Create a tensorflow worker to run experiments in your DB."""
    # Prepare to run the model
    config = Config()
    model_label = '%s_%s' % (experiment_name, get_dt_stamp())
    log = logger.get(os.path.join(config.log_dir, model_label))
    try:
        experiment_dict = experiments[experiment_name]()
        config = add_to_config(d=experiment_dict, config=config)  # Globals
        config = process_DB_exps(config)  # Update config w/ DB params
    except:
        err = 'Cannot understand the experiment name you selected.'
        log.fatal(err)
        raise RuntimeError(err)
    dataset_module = py_utils.import_module(config.dataset)
    dataset_module = dataset_module()
    train_data, train_means = get_data_pointers(
        dataset=config.dataset,
        base_dir=config.tf_records,
        cv=dataset_module['folds'].keys()[1],  # TODO: SEARCH FOR INDEX.
        log=log
    )
    val_data, val_means = get_data_pointers(
        dataset=config.dataset,
        base_dir=config.tf_records,
        cv=dataset_module['folds'].keys()[0],
        log=log
    )
    dir_list = {
        'checkpoints': os.path.join(config.checkpoints, model_label),
        'summaries': os.path.join(config.summaries, model_label),
        'evaluations': os.path.join(config.evaluations, model_label),
        'visualization': os.path.join(config.visualizations, model_label)
    }
    [py_utils.make_dir(v) for v in dir_list.values()]

    # Prepare data loaders
    train_images, train_labels = data_loader.inputs(
        dataset=train_data,
        batch_size=config.batch_size,
        true_image_size=dataset_module.im_size,
        model_input_image_size=dataset_module.im_size,
        tf_dict=dataset_module.tf_dict,
        data_augmentations=config.data_augmentations,
        num_epochs=config.epochs,
        shuffle=config.shuffle
    )

    train_images, train_labels = data_loader.inputs(
        dataset=train_data,
        batch_size=config.batch_size,
        true_image_size=dataset_module.im_size,
        model_input_image_size=dataset_module.im_size,
        tf_dict=dataset_module.tf_dict,
        data_augmentations=config.data_augmentations,
        num_epochs=config.epochs,
        shuffle=config.shuffle
    )



    with tf.device('/cpu:0'):
        train_images, train_labels, train_heatmaps = inputs(
            train_data, config.train_batch, config.image_size,
            config.model_image_size[:2],
            train=config.data_augmentations,
            num_epochs=config.epochs,
            return_heatmaps=True)
        val_images, val_labels = inputs(
            validation_data, config.validation_batch, config.image_size,
            config.model_image_size[:2],
            num_epochs=None,
            return_heatmaps=False)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--dataset',
        dest='dataset',
        help='Name of the dataset.')
    args = parser.parse_args()
    main(**vars(args))
