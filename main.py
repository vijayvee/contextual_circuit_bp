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
from ops import model_utils
from ops import loss_utils
from ops import eval_metrics
from ops import training
from ops import plotting


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

    # Prepare data loaders on the cpu
    with tf.device('/cpu:0'):
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
        val_images, val_labels = data_loader.inputs(
            dataset=val_data,
            batch_size=config.batch_size,
            true_image_size=dataset_module.im_size,
            model_input_image_size=dataset_module.im_size,
            tf_dict=dataset_module.tf_dict,
            data_augmentations=config.data_augmentations,
            num_epochs=config.epochs,
            shuffle=config.shuffle
        )
    log.info('Created tfrecord dataloader tensors.')

    # Prepare model on GPU
    model_dict = py_utils.import_module(
        dataset=config.model_struct,
        model_dir='models/structs/')
    with tf.device('/gpu:0'):
        with tf.variable_scope('cnn') as scope:
            # Training model
            model = model_utils.model_class(
                mean=train_means,
                training=tf.constant(True),
                output_size=dataset_module.output_size)
            output_scores = model.build(
                data=train_images,
                layer_structure=model_dict,
                tower_name='cnn')
            log.info('Built training model.')

            # Prepare the loss function
            train_loss, train_scores = loss_utils.loss_interpreter(
                logits=output_scores,
                labels=train_labels,
                loss_type=config.loss_function)

            # Add weight decay if requested
            if len(model.regularizations) > 0:
                train_loss = loss_utils.wd_loss(
                    model=model,
                    loss=train_loss,
                    wd_penalty=config.wd_penalty)
            train_op = loss_utils.optimizer_interpreter(
                loss=loss,
                lr=config.lr,
                optimizer=config.optimizer)
            log.info('Built training loss function.')

            train_accuracy = eval_metrics.class_accuracy(
                pred=train_scores,
                labels=train_labels)  # training accuracy
            tf.summary.image('train images', train_images)
            tf.summary.scalar('training loss', train_loss)
            tf.summary.scalar('training accuracy', train_accuracy)
            log.info('Added training summaries.')

            # Validation model
            scope.reuse_variables()
            val_model = model_utils.model_class(
                mean=val_means,
                training=tf.constant(True),
                output_size=dataset_module.output_size)
            val_output_scores = val_model.build(
                data=val_images,
                layer_structure=model_dict,
                tower_name='cnn')
            log.info('Built validation model.')

            val_loss, val_scores = loss_utils.loss_interpreter(
                logits=val_output_scores,
                labels=val_labels,
                loss_type=config.loss_function)
            val_accuracy = eval_metrics.class_accuracy(
                pred=val_scores,
                labels=train_labels)  # training accuracy
            tf.summary.image('val images', val_images)
            tf.summary.scalar('validation loss', val_loss)
            tf.summary.scalar('validation accuracy', val_accuracy)
            log.info('Added validation summaries.')

    # Set up summaries and saver
    saver = tf.train.Saver(
        tf.global_variables(), max_to_keep=config.keep_checkpoints)
    summary_op = tf.summary.merge_all()

    # Initialize the graph
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # Need to initialize both of these if supplying num_epochs to inputs
    sess.run(
        tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer())
        )
    summary_writer = tf.summary.FileWriter(dir_list['summaries'], sess.graph)

    # Set up exemplar threading
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Start training loop
    np.save(
        os.path.join(dir_list['evaluations'], 'training_config_file'),
        config)
    log.info('Starting training')
    tr_loss, val_loss, tr_accs, val_accs, timesteps = training.training_loop(
        config=config,
        db=db,
        coord=coord,
        sess=sess,
        train_op=train_op,
        summary_op=summary_op,
        summary_writer=summary_writer,
        train_loss=train_loss,
        val_loss=val_loss,
        saver=saver,
        threads=threads,
        summary_dir=dir_list['summaries'],
        val_accuracy=val_accuracy,
        train_accuracy=train_accuracy)
    log.info('Finished training.')

    plotting.plot_losses(
        train_loss=tr_loss,
        val_loss=val_loss,
        timesteps=timesteps,
        config=config,
        output=os.path.join(
            dir_list['evaluations'], 'losses'),
        output_ext='.pdf')
    plotting.plot_accuracies(
        tr_accs=tr_accs,
        val_accs=val_accs,
        timesteps=timesteps,
        config=config,
        output=os.path.join(
            dir_list['evaluations'], 'accuracies'),
        output_ext='.pdf')
    log.info('Completed plots.')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--dataset',
        dest='dataset',
        help='Name of the dataset.')
    args = parser.parse_args()
    main(**vars(args))
