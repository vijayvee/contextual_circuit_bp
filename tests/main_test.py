import os
import re
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
from ops import model_utils
from ops import loss_utils
from ops import eval_metrics
from ops import test_training as training
from ops import plotting
from scipy import misc
from tqdm import tqdm
import time


def image_batcher(
        start,
        num_batches,
        images,
        labels,
        batch_size):
    """Placeholder image/label batch loader."""
    for b in range(num_batches):
        next_image_batch = images[start:start + batch_size]
        image_stack = []
        label_stack = labels[start:start + batch_size]
        for f in next_image_batch:
            # 1. Load image patch
            patch = misc.imread(f)
            if len(patch.shape) == 2:
                image_stack += [patch[None, :, :, None]]
            else:
                image_stack += [patch[None, :, :, :]]
        # Add dimensions and concatenate
        start += batch_size
        yield np.concatenate(image_stack, axis=0), label_stack, next_image_batch


def print_model_architecture(model_summary):
    print '_' * 20
    print 'Model architecture:'
    print '_' * 20
    for s in model_summary:
        print s
    print '_' * 20


def add_to_config(d, config):
    for k, v in d.iteritems():
        setattr(config, k, v)
    return config


def process_DB_exps(experiment_name, log, config):
    exp_params, exp_id = db.get_parameters(
        experiment_name=experiment_name,
        log=log)
    if exp_id is None:
        err = 'No empty experiments found.' + \
            'Did you select the correct experiment name?'
        log.fatal(err)
        raise RuntimeError(err)
    for k, v in exp_params.iteritems():
        if isinstance(v, basestring) and '{' in v and '}' in v:
            v = v.strip('{').strip('}').split(',')
        setattr(config, k, v)
    return config


def get_data_pointers(dataset, base_dir, cv, log):
    data_pointer = os.path.join(base_dir, '%s_%s.tfrecords' % (dataset, cv))
    data_means = os.path.join(base_dir, '%s_%s_means.npy' % (dataset, cv))
    log.info(
        'Using %s tfrecords: %s' % (
            cv,
            data_pointer)
        )
    py_utils.check_path(data_pointer, log, '%s not found.' % data_pointer)
    py_utils.check_path(data_means, log, '%s not found.' % data_means)
    data_means = np.load(data_means)
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


def main(experiment_name, list_experiments=False):
    """Create a tensorflow worker to run experiments in your DB."""
    if list_experiments:
        exps = db.list_experiments()
        print '_' * 30
        print 'Initialized experiments:'
        print '_' * 30
        for l in exps:
            print l.values()[0]
        print '_' * 30
        return
    # Prepare to run the model
    config = Config()
    condition_label = '%s_%s' % (experiment_name, get_dt_stamp())
    experiment_label = '%s' % (experiment_name)
    log = logger.get(os.path.join(config.log_dir, condition_label))
    experiment_dict = experiments.experiments()[experiment_name]()
    config = add_to_config(d=experiment_dict, config=config)  # Globals
    config = process_DB_exps(
        experiment_name=experiment_name,
        log=log,
        config=config)  # Update config w/ DB params
    dataset_module = py_utils.import_module(
        model_dir=config.dataset_info,
        dataset=config.dataset)
    dataset_module = dataset_module.data_processing()  # hardcoded class name

    # Prepare data loaders on the cpu
    with tf.device('/cpu:0'):
        # Test issues with data loading? Try placeholders instead.
        train_images = tf.placeholder(
            tf.float32,
            name='train_images',
            shape=[config.batch_size] + dataset_module.im_size)
        train_labels = tf.placeholder(
            tf.int64,
            name='train_labels',
            shape=[config.batch_size])
        val_images = tf.placeholder(
            tf.float32,
            name='val_images',
            shape=[config.batch_size] + dataset_module.im_size)
        val_labels = tf.placeholder(
            tf.int64,
            name='val_labels',
            shape=[config.batch_size])
    log.info('Created tfrecord dataloader tensors.')

    # Prepare model on GPU
    with tf.device('/gpu:0'):
        with tf.variable_scope('cnn') as scope:

            # Training model
            if len(dataset_module.output_size) > 1:
                log.warning(
                    'Found > 1 dimension for your output size.'
                    'Converting to a scalar.')
                dataset_module.output_size = np.prod(
                    dataset_module.output_size)
            # Click weighting
            flat_ims = tf.reshape(train_images, [config.batch_size, np.prod(dataset_module.im_size)])
            W = tf.get_variable(
                name='W',
                initializer=tf.truncated_normal_initializer(stddev=0.1),
                shape=[np.prod(dataset_module.im_size), dataset_module.output_size])
            b = tf.get_variable(
                name='b',
                initializer=tf.truncated_normal_initializer(stddev=0.1),
                shape=[dataset_module.output_size])
            output_scores = tf.matmul(flat_ims, W) + b

            # Prepare the loss function
            train_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=train_labels, logits=output_scores))
            train_op = tf.train.GradientDescentOptimizer(config.lr).minimize(train_loss)
            log.info('Built training loss function.')

    # Set up summaries and saver
    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()

    # Initialize the graph
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    # Need to initialize both of these if supplying num_epochs to inputs
    sess.run(
        tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer())
        )

    # Start training loop
    step, time_elapsed = 0, 0
    train_losses, train_accs, val_losses, val_accs, timesteps = {}, {}, {}, {}, {}
    files, labels = dataset_module.get_data()
    combined_files = files['train']
    combined_labels = labels['train']
    batch_size = config.batch_size
    num_batches = len(combined_files) // batch_size
    log.info('Finished training.')
    for image_batch, label_batch, _ in tqdm(
            image_batcher(
                start=0,
                num_batches=num_batches,
                images=combined_files,
                labels=combined_labels,
                batch_size=batch_size),
            total=num_batches):
        feed_dict = {
            train_images: image_batch.astype(np.float32),
            train_labels: np.asarray(label_batch).astype(int)
        }
        import ipdb;ipdb.set_trace()
        start_time = time.time()
        _, loss_value = sess.run(
            [
                train_op,
                train_loss,
            ],
            feed_dict=feed_dict)

    files_to_save = {
        'training_loss': tr_loss,
        'validation_loss': val_loss,
        'training_acc': tr_accs,
        'validation_acc': val_accs,
        'timesteps': timesteps
    }

    model_name = config.model_struct.replace('/', '_')
    py_utils.save_npys(
        data=files_to_save,
        model_name=model_name,
        output_string=dir_list['experiment_evaluations']
        )

    # Compare this condition w/ all others.
    plotting.plot_data(
        train_loss=tr_loss,
        val_loss=val_loss,
        model_name=model_name,
        timesteps=timesteps,
        config=config,
        output=os.path.join(
            dir_list['condition_evaluations'], 'loss'),
        output_ext='.pdf',
        data_type='loss')
    plotting.plot_data(
        tr_accs=tr_accs,
        val_accs=val_accs,
        model_name=model_name,
        timesteps=timesteps,
        config=config,
        output=os.path.join(
            dir_list['condition_evaluations'], 'acc'),
        output_ext='.pdf',
        data_type='acc')
    log.info('Completed plots.')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--experiment',
        dest='experiment_name',
        type=str,
        default=None,
        help='Name of the experiment.')
    parser.add_argument(
        '--list_experiments',
        dest='list_experiments',
        action='store_true',
        help='Name of the experiment.')
    args = parser.parse_args()
    main(**vars(args))
