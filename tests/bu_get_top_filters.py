import os
import json
import sys
import cv2
import argparse
import numpy as np
import tensorflow as tf
import cPickle as pickle
import scipy as sp
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
from numpy import NaN, Inf, arange, isscalar, asarray, array
from scipy import misc
import cv2
import scipy


def gaussian(height, center_x, center_y, width_x, width_y, rotation):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)

    rotation = np.deg2rad(rotation)
    center_x = center_x * np.cos(rotation) - center_y * np.sin(rotation)
    center_y = center_x * np.sin(rotation) + center_y * np.cos(rotation)

    def rotgauss(x,y):
        xp = x * np.cos(rotation) - y * np.sin(rotation)
        yp = x * np.sin(rotation) + y * np.cos(rotation)
        g = height*np.exp(
            -(((center_x-xp)/width_x)**2+
              ((center_y-yp)/width_y)**2)/2.)
        return g
    return rotgauss

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y, 0.0


def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    import ipdb;ipdb.set_trace()
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) - data)
    p, success = sp.optimize.leastsq(errorfunction, params)
    return p
        

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
        template_exp='ALLEN_selected_cells_1',
        process_pnodes=False,
        allen_dir='/home/drew/Documents/Allen_Brain_Observatory',
        output_dir='tests/ALLEN_files',
        stimulus_dir='/media/data_cifs/AllenData/DataForTrain/all_stimulus_template',
        stimulus_type='tfrecord',
        top_n=2,
        preload_stim=False,
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
            score_files = glob(
                os.path.join(
                    main_config.ccbp_exp_evals,
                    exp_name['experiment_name'],
                    '*val_scores.npy'))  # Scores has preds, labels has GT
            lab_files = glob(
                os.path.join(
                    main_config.ccbp_exp_evals,
                    exp_name['experiment_name'],
                    '*val_labels.npy'))  # Scores has preds, labels has GT
            for gd, sd, ld in zip(data_files, score_files, lab_files):
                mt = gd.split(
                    os.path.sep)[-1].split(
                        template_exp + '_')[-1].split('_' + 'val')[0]
                # it_data = np.load(gd).item()
                lds = np.load(ld).item()
                sds = np.load(sd).item()
                it_data = {k: np.corrcoef(lds[k], sds[k])[0, 1] for k in sds.keys()}
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
    for xy in uxys:
        sel_idx = (xys == xy).sum(axis=-1) == 2
        sperfs = perfs[sel_idx]
        sexps = exps[sel_idx]
        sargs = arg_perf[sel_idx]
        sel_mts = model_types[sel_idx]
        bp = np.argmax(sperfs)
        fxs += [xy[0]]
        fys += [xy[1]]
        fperfs += [sperfs[bp]]
        fargs += [sargs[bp]]
        fmodel_types += [sel_mts[bp]]
        fexps += [sexps[bp]]
    xs = np.asarray(fxs)
    ys = np.asarray(fys)
    perfs = np.asarray(fperfs)
    arg_perf = np.asarray(fargs)
    exps = np.asarray(fexps)
    model_types = np.asarray(fmodel_types)
    umt, model_types_inds = np.unique(model_types, return_inverse=True)

    # Get weights for the top-n fitting models of each type
    it_perfs = perfs[model_types == target_model]
    it_exps = exps[model_types == target_model]
    # it_args = arg_perf[model_types == target_model]
    sorted_perfs = np.argsort(it_perfs)[::-1][:top_n]
    perf = sel_exp_query(
        experiment_name=it_exps[sorted_perfs[0]],
        model=target_model,
        db_config=db_config)
    dummy_sel_model = perf[-1]
    print 'Using %s' % dummy_sel_model    
    model_file = dummy_sel_model['ckpt_file'].split('.')[0]
    model_ckpt = '%s.ckpt-%s' % (
        model_file,
        model_file.split(os.path.sep)[-1].split('_')[-1])
    model_meta = '%s.meta' % model_ckpt

    # Pull stimuli
    stim_dir = os.path.join(
        main_config.tf_record_output,
        dummy_sel_model['experiment_name'])
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


    # Pull responses
    dataset_module = py_utils.import_module(
        model_dir=config.dataset_info,
        dataset=dummy_sel_model['experiment_name'])
    dataset_module = dataset_module.data_processing()
    with tf.device('/cpu:0'):
        val_images = tf.placeholder(
            tf.float32,
            shape=[1] + [x for x in dataset_module.im_size])

    # Mean normalize
    log = logger.get(os.path.join(output_dir, 'sta_logs', target_model))
    data_dir = os.path.join(output_dir, 'data', target_model)
    py_utils.make_dir(data_dir)
    sys.path.append(os.path.join('models', 'structs', dummy_sel_model['experiment_name']))
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
            grad_image = tf.gradients(model.output, val_images)[0]
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
    all_filters = []
    all_rfs = []
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

        # Set up exemplar threading
        if target_model == 'conv2d':
            fname = [
                x for x in tf.global_variables()
                if 'conv1_1_filters:0' in x.name]
        elif target_model == 'sep_conv2d':
            fname = [
                x for x in tf.global_variables()
                if 'sep_conv1_1_filters:0' in x.name]
        elif target_model == 'dog':
            fname = [
                x for x in tf.global_variables()
                if 'dog1_1_filters:0' in x.name]
        else:
            raise NotImplementedError

        print 'Using %s' % sel_model
        model_file = sel_model['ckpt_file'].split('.')[0]
        model_ckpt = '%s.ckpt-%s' % (
            model_file,
            model_file.split(os.path.sep)[-1].split('_')[-1])
        saver.restore(sess, model_ckpt)
        all_filters += [sess.run(fname)]
        all_rfs += [sparse_rf]
    np.savez(
        'tests/ALLEN_files/filters/%s_%s' % (experiment, target_model),
        rfs=all_rfs,
        filters=all_filters) 
    if target_model != 'dog':
        save_mosaic(
            maps=all_filters[0][0].squeeze().transpose(2, 0, 1),
            output=os.path.join(data_dir, '%s_filters' % target_layer),
            rc=8,
            cc=4,
            title='%s filters' % (
                target_layer))
    print 'SAVED'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment',
        type=str,
        default='760_cells_2017_11_05_16_36_55',
        dest='experiment',
        help='Name of experiment.')
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
    parser.add_argument(
        '--pnode',
        action='store_true',
        dest='process_pnodes',
        help='Access the pnode DB.')
    args = parser.parse_args()
    plot_fits(**vars(args))


