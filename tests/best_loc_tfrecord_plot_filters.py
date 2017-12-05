import os
import json
import sys
import cv2
import argparse
import numpy as np
import tensorflow as tf
from glob import glob
from matplotlib import pyplot as plt
from matplotlib import gridspec
# import pandas as pd
# from ggplot import *
import seaborn as sns
from allensdk.brain_observatory.receptive_field_analysis import receptive_field
from config import Config
from ops import data_loader, model_utils
from utils import logger, py_utils
from db import credentials
import numpy.matlib as ml
import sys
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
from numpy import NaN, Inf, arange, isscalar, asarray, array


@ops.RegisterGradient('GradLRP')
def _GradLRP(op, grad):
    eps = 1e-12
    op_out = op.outputs[0]
    op_in = op.inputs[0]
    return grad * op_out / (op_in + eps)


@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    return tf.select(
        grad > 0.,
        gen_nn_ops._relu_grad(grad, op.outputs[0]),
        tf.zeros(grad.get_shape()))


def peakdet(v, delta, x = None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    
    Returns two arrays
    
    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %      
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.
    
    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.
    
    """
    maxtab = []
    mintab = []
       
    if x is None:
        x = arange(len(v))
    
    v = asarray(v)
    
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    
    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    
    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN
    
    lookformax = True
    
    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return array(maxtab), array(mintab)

def save_mosaic(
        maps,
        output,
        title='Mosaic',
        rc=None,
        cc=None):
    if rc is None:
        rc = np.ceil(np.sqrt(len(maps))).astype(int)
        cc = np.ceil(np.sqrt(len(maps))).astype(int)
    plt.figure(figsize=(10, 10))
    plt.suptitle(title, fontsize=20)
    gs1 = gridspec.GridSpec(rc, cc)
    gs1.update(wspace=0.01, hspace=0.01)  # set the spacing between axes.
    for idx, im in enumerate(maps):
        ax1 = plt.subplot(gs1[idx])
        plt.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')
        ax1.imshow(im.squeeze(), interpolation='nearest')
    plt.savefig(output)
    plt.savefig(output.split('.')[0] + '.pdf')


def plot_fits(
        experiment='760_cells_2017_11_04_16_29_09',
        query_db=False,
        num_models=3,
        template_exp='ALLEN_selected_cells_1',
        process_pnodes=False,
        allen_dir='/home/drew/Documents/Allen_Brain_Observatory',
        output_dir='tests/ALLEN_files',
        stimulus_type='tfrecord',
        top_n=1,
        grad='lrp',
        target_layer='conv1_1',  # conv1_1, sep_conv1_1, dog1_1
        target_model='conv2d'):  # conv2d, sep_conv2d, dog
    """Plot fits across the RF.
    experiment: Name of Allen experiment you're plotting.
    query_db: Use data from DB versus data in Numpys.
    num_models: The number of architectures you're testing.
    template_exp: The name of the contextual_circuit model template used."""
    sys.path.append(allen_dir)
    from allen_config import Allen_Brain_Observatory_Config
    if process_pnodes:
        from pnodes_declare_datasets_loop import query_hp_hist, sel_exp_query
    else:
        from declare_datasets_loop import query_hp_hist, sel_exp_query
    config = Config()
    main_config = Allen_Brain_Observatory_Config()
    db_config = credentials.postgresql_connection()
    files = glob(
        os.path.join(
            allen_dir,
            main_config.multi_exps,
            experiment, '*.npz'))
    assert len(files), 'Couldn\'t find files.'
    out_data, xs, ys = [], [], []
    perfs, model_types, exps, arg_perf = [], [], [], []
    count = 0
    for f in files:
        data = np.load(f)
        d = {
            'x': data['rf_data'].item()['on_center_x'],
            'y': data['rf_data'].item()['on_center_y'],
            # x: files['dataset_method'].item()['x_min'],
            # y: files['dataset_method'].item()['y_min'],
        }
        exp_name = {
            'experiment_name': data['dataset_method'].item()[
                'experiment_name']}
        if query_db:
            perf = query_hp_hist(
                exp_name['experiment_name'],
                db_config=db_config)
            if perf is None:
                print 'No fits for: %s' % exp_name['experiment_name']
            else:
                raise NotImplementedError
                d['perf'] = perf
                d['max_val'] = np.max(perf)
                out_data += [d]
                xs += [np.round(d['x'])]
                ys += [np.round(d['y'])]
                perfs += [np.max(d['perf'])]
                count += 1
        else:
            data_files = glob(
                os.path.join(
                    main_config.ccbp_exp_evals,
                    exp_name['experiment_name'],
                    '*val_losses.npy'))  # Scores has preds, labels has GT
            for gd in data_files:
                mt = gd.split(
                    os.path.sep)[-1].split(
                        template_exp + '_')[-1].split('_' + 'val')[0]
                it_data = np.load(gd).item()
                sinds = np.asarray(it_data.keys())[np.argsort(it_data.keys())]
                sit_data = [it_data[idx] for idx in sinds]
                d['perf'] = sit_data
                d['max_val'] = np.max(sit_data)
                d['max_idx'] = np.argmax(sit_data)
                d['mt'] = mt
                out_data += [d]
                xs += [np.round(d['x'])]
                ys += [np.round(d['y'])]
                perfs += [np.max(sit_data)]
                arg_perf += [np.argmax(sit_data)]
                exps += [gd.split(os.path.sep)[-2]]
                model_types += [mt]
                count += 1

    # Package as a df
    xs = np.round(np.asarray(xs)).astype(int)
    ys = np.round(np.asarray(ys)).astype(int)
    perfs = np.asarray(perfs)
    arg_perf = np.asarray(arg_perf)
    exps = np.asarray(exps)
    model_types = np.asarray(model_types)

    # Filter to only keep top-scoring values at each x/y (dirty trick)
    fxs, fys, fperfs, fmodel_types, fexps, fargs = [], [], [], [], [], []
    xys = np.vstack((xs, ys)).transpose()
    cxy = np.ascontiguousarray(  # Unique rows
        xys).view(
        np.dtype((np.void, xys.dtype.itemsize * xys.shape[1])))
    _, idx = np.unique(cxy, return_index=True)
    uxys = xys[idx]
    scores = []
    for xy in uxys:
        sel_idx = (xys == xy).sum(axis=-1) == 2
        sperfs = perfs[sel_idx]
        sexps = exps[sel_idx]
        sargs = arg_perf[sel_idx]
        sel_mts = model_types[sel_idx]
        # Only get top conv/sep spots
        sperfs = sperfs[sel_mts != 'dog']
        sperfs = sperfs[sel_mts != 'DoG']
        scores += [sperfs.mean() / sperfs.std()]
    best_fits = np.argmax(np.asarray(scores))
    xs = np.asarray([uxys[best_fits][0]])
    ys = np.asarray([uxys[best_fits][1]])
    sel_idx = (xys == uxys[best_fits]).sum(axis=-1) == 2
    perfs = np.asarray(perfs[sel_idx])
    exps = np.asarray(exps[sel_idx])
    model_types = np.asarray(model_types[sel_idx])
    umt, model_types_inds = np.unique(model_types, return_inverse=True)

    # Get weights for the top-n fitting models of each type
    it_perfs = perfs[model_types == target_model]
    it_exps = exps[model_types == target_model]
    # it_args = arg_perf[model_types == target_model]
    sorted_perfs = np.argsort(it_perfs)[::-1][:top_n]
    for idx in sorted_perfs:
        perf = sel_exp_query(
            experiment_name=it_exps[idx],
            model=target_model,
            db_config=db_config)
        # perf_steps = np.argsort([v['training_step'] for v in perf])[::-1]
        perf_steps = [v['validation_loss'] for v in perf]
        max_score = np.max(perf_steps)
        arg_perf_steps = np.argmax(perf_steps)
        sel_model = perf[arg_perf_steps]  # perf_steps[it_args[idx]]]
        print 'Using %s' % sel_model
        model_file = sel_model['ckpt_file'].split('.')[0]
        model_ckpt = '%s.ckpt-%s' % (
            model_file,
            model_file.split(os.path.sep)[-1].split('_')[-1])
        model_meta = '%s.meta' % model_ckpt

        # Pull stimuli
        stim_dir = os.path.join(
            main_config.tf_record_output,
            sel_model['experiment_name'])
        stim_files = glob(stim_dir + '*')
        stim_meta_file = [x for x in stim_files if 'meta' in x][0]
        # stim_val_data = [x for x in stim_files if 'val.tfrecords' in x][0]
        stim_val_data = [x for x in stim_files if 'train.tfrecords' in x][0]
        stim_val_mean = [x for x in stim_files if 'train_means' in x][0]
        assert stim_meta_file is not None
        assert stim_val_data is not None
        assert stim_val_mean is not None
        stim_meta_data = np.load(stim_meta_file).item()
        rf_stim_meta_data = stim_meta_data['rf_data']
        stim_mean_data = np.load(
            stim_val_mean).items()[0][1].item()['image']['mean']

        # Store sparse noise for reference
        sparse_rf_on = {
            'center_x': rf_stim_meta_data.get('on_center_x', None),
            'center_y': rf_stim_meta_data.get('on_center_y', None),
            'width_x': rf_stim_meta_data.get('on_width_x', None),
            'width_y': rf_stim_meta_data.get('on_width_y', None),
            'distance': rf_stim_meta_data.get('on_distance', None),
            'area': rf_stim_meta_data.get('on_area', None),
            'rotation': rf_stim_meta_data.get('on_rotation', None),
        }
        sparse_rf_off = {
            'center_x': rf_stim_meta_data.get('off_center_x', None),
            'center_y': rf_stim_meta_data.get('off_center_y', None),
            'width_x': rf_stim_meta_data.get('off_width_x', None),
            'width_y': rf_stim_meta_data.get('off_width_y', None),
            'distance': rf_stim_meta_data.get('off_distance', None),
            'area': rf_stim_meta_data.get('off_area', None),
            'rotation': rf_stim_meta_data.get('off_rotation', None),
        }
        sparse_rf = {'on': sparse_rf_on, 'off': sparse_rf_off}

        # Pull responses
        dataset_module = py_utils.import_module(
            model_dir=config.dataset_info,
            dataset=sel_model['experiment_name'])
        dataset_module = dataset_module.data_processing()
        with tf.device('/cpu:0'):
            if stimulus_type == 'sparse_noise':
                pass
            elif stimulus_type == 'drifting_grating':
                pass
            elif stimulus_type == 'tfrecord':
                val_images, val_labels = data_loader.inputs(
                    dataset=stim_val_data,
                    batch_size=1,
                    model_input_image_size=dataset_module.model_input_image_size,
                    tf_dict=dataset_module.tf_dict,
                    data_augmentations=[None],  # dataset_module.preprocess,
                    num_epochs=1,
                    tf_reader_settings=dataset_module.tf_reader,
                    shuffle=False
                )

        # Mean normalize
        log = logger.get(os.path.join(output_dir, 'sta_logs', target_model))
        data_dir = os.path.join(output_dir, 'data', target_model)
        py_utils.make_dir(data_dir)
        sys.path.append(os.path.join('models', 'structs', sel_model['experiment_name']))
        model_dict = __import__(target_model) 
        if hasattr(model_dict, 'output_structure'):
            # Use specified output layer
            output_structure = model_dict.output_structure
        else:
            output_structure = None
        model = model_utils.model_class(
            mean=stim_mean_data,
            training=True,  # FIXME
            output_size=dataset_module.output_size)
        with tf.device('/gpu:0'):
            with tf.variable_scope('cnn') as scope:
                val_scores, model_summary = model.build(
                    data=val_images,
                    layer_structure=model_dict.layer_structure,
                    output_structure=output_structure,
                    log=log,
                    tower_name='cnn')
                if grad == 'vanilla':
                    grad_image = tf.gradients(model.output, val_images)[0]
                elif grad == 'lrp':
                    eval_graph = tf.Graph()
                    with eval_graph.as_default():
                        with eval_graph.gradient_override_map(
                            {'Relu': 'GradLRP'}):
                            grad_image = tf.gradients(model.output, val_images)[0]
                elif grad == 'cam':
                    eval_graph = tf.Graph()
                    with eval_graph.as_default():
                        with eval_graph.gradient_override_map(
                            {'Relu': 'GuidedRelu'}):
                            grad_image = tf.gradients(model.output, val_images)[0]
                else:
                    raise NotImplementedError
        print(json.dumps(model_summary, indent=4))

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
        saver.restore(sess, model_ckpt)

        # Set up exemplar threading
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        if target_model == 'conv2d':
            fname = [
                x for x in tf.global_variables()
                if 'conv1_1_filters:0' in x.name]
        elif target_model == 'sep_conv2d':
            fname = [
                x for x in tf.global_variables()
                if 'sep_conv1_1_filters:0' in x.name]
        elif target_model == 'dog' or target_model == 'DoG':
            fname = [
                x for x in tf.global_variables()
                if 'dog1_1_filters:0' in x.name]
        else:
            raise NotImplementedError
        val_tensors = {
            'images': val_images,
            'labels': val_labels,
            'filts': fname,
            'responses': model.output,  # model[target_layer],
            'grads': grad_image
        }
        all_images, all_preds, all_grads, all_responses = [], [], [], []
        step = 0
        try:
            while not coord.should_stop():
                val_vals = sess.run(val_tensors.values())
                val_dict = {k: v for k, v in zip(val_tensors.keys(), val_vals)}
                all_images += [val_dict['images']]
                all_responses += [val_dict['responses']]
                all_preds += [val_dict['labels'].squeeze()]
                all_grads += [val_dict['grads'].squeeze()]
                print 'Finished step %s' % step
                step += 1
        except:
            print 'Finished tfrecords'
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()

        # Process and save data
        # if target_model != 'dog':
        #     filters = val_dict['filts'][0].squeeze().transpose(2, 0, 1)
        all_images = np.concatenate(all_images).squeeze()
        all_grads = np.asarray(all_grads)
        all_preds = np.asarray(all_preds).reshape(-1, 1)
        all_responses = np.asarray(all_responses).squeeze()

        np.savez(
            os.path.join(data_dir, 'data'),
            images=all_images,
            pred=all_preds,
            # filters=filters,
            grads=all_grads)
        # if target_model != 'dog':
        #     save_mosaic(
        #         maps=filters,  # [0].squeeze().transpose(2, 0, 1),
        #         output=os.path.join(data_dir, '%s_filters' % target_layer),
        #         rc=8,
        #         cc=4,
        #         title='%s filters' % (
        #             target_layer))
        print 'Complete.'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment',
        type=str,
        default='760_cells_2017_11_05_16_36_55',
        dest='experiment',
        help='Name of experiment.')
    parser.add_argument(
        '--pnode',
        action='store_true',
        dest='process_pnodes',
        help='Access the pnode DB.')
    parser.add_argument(
        '--target_model',
        type=str,
        default='conv2d',
        dest='target_model',
        help='Name of experiment.')
    parser.add_argument(
        '--target_layer',
        type=str,
        default='conv2d',
        dest='target_layer',
        help='Name of experiment.')
    args = parser.parse_args()
    plot_fits(**vars(args))

