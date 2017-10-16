import os
import json
import numpy as np
import tensorflow as tf
import experiments
from db import db
from config import Config
from argparse import ArgumentParser
from utils import logger
from utils import py_utils
from ops import data_loader
from ops import model_utils
from ops import loss_utils
from ops import eval_metrics
from ops import training
from ops import hp_opt_utils


def print_model_architecture(model_summary):
    """Print a list fancy."""
    print '_' * 20
    print 'Model architecture:'
    print '_' * 20
    for s in model_summary:
        print s
    print '_' * 20


def add_to_config(d, config):
    """Add attributes to config class."""
    for k, v in d.iteritems():
        setattr(config, k, v)
    return config


def process_DB_exps(experiment_name, log, config):
    """Interpret and prepare hyperparams at runtime."""
    exp_params, exp_id = db.get_parameters(
        experiment_name=experiment_name,
        log=log)
    if 'hp_optim' in exp_params.keys() and exp_params['hp_optim'] is not None:
        performance_history, hp_hist = db.query_hp_hist(exp_params)
        proc_exp_params = hp_opt_utils.hp_optim_interpreter(
            hp_hist=hp_hist,
            exp_params=exp_params,
            performance_history=performance_history
        )
    else:
        proc_exp_params = exp_params

    if exp_id is None:
        err = 'No empty experiments found.' + \
            'Did you select the correct experiment name?'
        log.fatal(err)
        raise RuntimeError(err)
    for k, v in exp_params.iteritems():
        if isinstance(v, basestring) and '{' in v and '}' in v:
            v = v.strip('{').strip('}').split(',')
        setattr(config, k, v)
    if not hasattr(config, '_id'):
        config._id = exp_id
    return config, proc_exp_params


def get_data_pointers(dataset, base_dir, cv, log):
    """Get data file pointers."""
    data_pointer = os.path.join(base_dir, '%s_%s.tfrecords' % (dataset, cv))
    data_means = os.path.join(base_dir, '%s_%s_means.npy' % (dataset, cv))
    log.info(
        'Using %s tfrecords: %s' % (
            cv,
            data_pointer)
        )
    py_utils.check_path(
        data_pointer, log, '%s not found.' % data_pointer)
    mean_loc = py_utils.check_path(
        data_means, log, '%s not found for cv: %s.' % (data_means, cv))
    if not mean_loc:
        alt_data_pointer = data_means.replace('.npy', '.npz')
        alt_data_pointer = py_utils.check_path(
            alt_data_pointer, log, '%s not found.' % alt_data_pointer)
        # TODO: Fix this API and make it more flexible. Kill npzs in Allen?
        if not alt_data_pointer:
            # No mean for this dataset
            data_means = None
        else:
            log.info('Loading means from npz for cv: %s.' % cv)
            data_means = np.load(alt_data_pointer)
            data_means = data_means[data_means.keys()[0]].item()['image']
    else:
        data_means = np.load(data_means)
    return data_pointer, data_means


def main(
        experiment_name,
        list_experiments=False,
        gpu_device='/gpu:0'):
    """Create a tensorflow worker to run experiments in your DB."""
    if list_experiments:
        exps = db.list_experiments()
        print '_' * 30
        print 'Initialized experiments:'
        print '_' * 30
        for l in exps:
            print l.values()[0]
        print '_' * 30
        print 'You can add to the DB with: '\
            'python prepare_experiments.py --experiment=%s' % \
            exps[0].values()[0]
        return
    if experiment_name is None:
        print 'No experiment specified. Pulling one out of the DB.'
        experiment_name = db.get_experiment_name()

    # Prepare to run the model
    config = Config()
    condition_label = '%s_%s' % (experiment_name, py_utils.get_dt_stamp())
    experiment_label = '%s' % (experiment_name)
    log = logger.get(os.path.join(config.log_dir, condition_label))
    experiment_dict = experiments.experiments()[experiment_name]()
    config = add_to_config(d=experiment_dict, config=config)  # Globals
    config, exp_params = process_DB_exps(
        experiment_name=experiment_name,
        log=log,
        config=config)  # Update config w/ DB params
    dataset_module = py_utils.import_module(
        model_dir=config.dataset_info,
        dataset=config.dataset)
    dataset_module = dataset_module.data_processing()  # hardcoded class name
    train_data, train_means = get_data_pointers(
        dataset=config.dataset,
        base_dir=config.tf_records,
        cv=dataset_module.folds.keys()[1],  # TODO: SEARCH FOR INDEX.
        log=log
    )
    val_data, val_means = get_data_pointers(
        dataset=config.dataset,
        base_dir=config.tf_records,
        cv=dataset_module.folds.keys()[0],
        log=log
    )

    # Initialize output folders
    dir_list = {
        'checkpoints': os.path.join(
            config.checkpoints, condition_label),
        'summaries': os.path.join(
            config.summaries, condition_label),
        'condition_evaluations': os.path.join(
            config.condition_evaluations, condition_label),
        'experiment_evaluations': os.path.join(  # DEPRECIATED
            config.experiment_evaluations, experiment_label),
        'visualization': os.path.join(
            config.visualizations, condition_label),
        'weights': os.path.join(
            config.condition_evaluations, condition_label, 'weights')
    }
    [py_utils.make_dir(v) for v in dir_list.values()]

    # Prepare data loaders on the cpu
    config.data_augmentations = py_utils.flatten_list(
        config.data_augmentations,
        log)
    with tf.device('/cpu:0'):
        train_images, train_labels = data_loader.inputs(
            dataset=train_data,
            batch_size=config.batch_size,
            model_input_image_size=dataset_module.model_input_image_size,
            tf_dict=dataset_module.tf_dict,
            data_augmentations=config.data_augmentations,
            num_epochs=config.epochs,
            tf_reader_settings=dataset_module.tf_reader,
            shuffle=config.shuffle
        )
        val_images, val_labels = data_loader.inputs(
            dataset=val_data,
            batch_size=config.batch_size,
            model_input_image_size=dataset_module.model_input_image_size,
            tf_dict=dataset_module.tf_dict,
            data_augmentations=config.data_augmentations,
            num_epochs=config.epochs,
            tf_reader_settings=dataset_module.tf_reader,
            shuffle=config.shuffle
        )
    log.info('Created tfrecord dataloader tensors.')

    # Load model specification
    struct_name = config.model_struct.split(os.path.sep)[-1]
    try:
        model_dict = py_utils.import_module(
            dataset=struct_name,
            model_dir=os.path.join(
                'models',
                'structs',
                experiment_name).replace(os.path.sep, '.')
            )
    except IOError:
        print 'Could not find the model structure: %s' % experiment_name

    # Inject model_dict with hyperparameters if requested
    model_dict.layer_structure = hp_opt_utils.inject_model_with_hps(
        layer_structure=model_dict.layer_structure,
        exp_params=exp_params)

    # Prepare model on GPU
    with tf.device(gpu_device):
        with tf.variable_scope('cnn') as scope:

            # Training model
            if len(dataset_module.output_size) > 1:
                log.warning(
                    'Found > 1 dimension for your output size.'
                    'Converting to a scalar.')
                dataset_module.output_size = np.prod(
                    dataset_module.output_size)

            if hasattr(model_dict, 'output_structure'):
                # Use specified output layer
                output_structure = model_dict.output_structure
            else:
                output_structure = None
            model = model_utils.model_class(
                mean=train_means,
                training=True,
                output_size=dataset_module.output_size)
            train_scores, model_summary = model.build(
                data=train_images,
                layer_structure=model_dict.layer_structure,
                output_structure=output_structure,
                log=log,
                tower_name='cnn')
            log.info('Built training model.')
            log.debug(
                json.dumps(model_summary, indent=4),
                verbose=0)
            print_model_architecture(model_summary)

            # Prepare the loss function
            train_loss, _ = loss_utils.loss_interpreter(
                logits=train_scores,
                labels=train_labels,
                loss_type=config.loss_function,
                dataset_module=dataset_module)

            # Add weight decay if requested
            if len(model.regularizations) > 0:
                train_loss = loss_utils.wd_loss(
                    regularizations=model.regularizations,
                    loss=train_loss,
                    wd_penalty=config.regularization_strength)
            train_op = loss_utils.optimizer_interpreter(
                loss=train_loss,
                lr=config.lr,
                optimizer=config.optimizer)
            log.info('Built training loss function.')

            train_accuracy = eval_metrics.metric_interpreter(
                metric=dataset_module.score_metric,
                pred=train_scores,
                labels=train_labels)  # training accuracy
            if int(train_images.get_shape()[-1]) <= 3:
                tf.summary.image('train images', train_images)
            tf.summary.scalar('training loss', train_loss)
            tf.summary.scalar('training accuracy', train_accuracy)
            log.info('Added training summaries.')

            # Validation model
            scope.reuse_variables()
            val_model = model_utils.model_class(
                mean=val_means,
                training=True,
                output_size=dataset_module.output_size)
            val_scores, _ = val_model.build(  # Ignore summary
                data=val_images,
                layer_structure=model_dict.layer_structure,
                output_structure=output_structure,
                log=log,
                tower_name='cnn')
            log.info('Built validation model.')

            val_loss, _ = loss_utils.loss_interpreter(
                logits=val_scores,
                labels=val_labels,
                loss_type=config.loss_function,
                dataset_module=dataset_module)
            val_accuracy = eval_metrics.metric_interpreter(
                metric=dataset_module.score_metric,
                pred=val_scores,
                labels=val_labels)  # training accuracy
            if int(train_images.get_shape()[-1]) <= 3:
                tf.summary.image('val images', val_images)
            tf.summary.scalar('validation loss', val_loss)
            tf.summary.scalar('validation accuracy', val_accuracy)
            log.info('Added validation summaries.')

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
    summary_writer = tf.summary.FileWriter(dir_list['summaries'], sess.graph)

    # Set up exemplar threading
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Create dictionaries of important training and validation information
    train_dict = {
        'train_loss': train_loss,
        'train_accuracy': train_accuracy,
        'train_images': train_images,
        'train_labels': train_labels,
        'train_op': train_op,
        'train_scores': train_scores
    }
    val_dict = {
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'val_images': val_images,
        'val_labels': val_labels,
        'val_scores': val_scores,
    }

    # Start training loop
    np.save(
        os.path.join(
            dir_list['condition_evaluations'], 'training_config_file'),
        config)
    log.info('Starting training')
    output_dict = training.training_loop(
        config=config,
        db=db,
        coord=coord,
        sess=sess,
        summary_op=summary_op,
        summary_writer=summary_writer,
        saver=saver,
        threads=threads,
        summary_dir=dir_list['summaries'],
        checkpoint_dir=dir_list['checkpoints'],
        weight_dir=dir_list['weights'],
        train_dict=train_dict,
        val_dict=val_dict,
        train_model=model,
        val_model=val_model,
        exp_params=exp_params)
    log.info('Finished training.')

    model_name = config.model_struct.replace('/', '_')
    py_utils.save_npys(
        data=output_dict,
        model_name=model_name,
        output_string=dir_list['experiment_evaluations'])


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
    # TODO: Add the ability to specify multiple GPUs for parallelization.
    args = parser.parse_args()
    main(**vars(args))
