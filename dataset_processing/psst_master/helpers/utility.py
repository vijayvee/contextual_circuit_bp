import tensorflow as tf
import numpy as np
import sys
import os
import subprocess
import webbrowser
import shutil
from tensorflow.python.client import device_lib
import ipdb

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def concatenate_path_list(lst):

    cct = lst[0]
    for l in lst[1:]:
        cct+='/' + l
    return cct

def find_files(files, dirs=[], contains=[]):
    for d in dirs:
        onlyfiles = [os.path.join(d, f) for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
        for i, part in enumerate(contains):
            files += [os.path.join(d, f) for f in onlyfiles if part in f]
        onlydirs = [os.path.join(d, dd) for dd in os.listdir(d) if os.path.isdir(os.path.join(d, dd))]
        if onlydirs:
            files += find_files([], onlydirs, contains)

    return files

def pairwise_add(u, v=None, is_batch=False):
    """
    performs a pairwise summation between vectors (possibly the same)

    Parameters:
    ----------
    u: Tensor (n, ) | (n, 1)
    v: Tensor (n, ) | (n, 1) [optional]
    is_batch: bool
        a flag for whether the vectors come in a batch
        ie.: whether the vectors has a shape of (b,n) or (b,n,1)

    Returns: Tensor (n, n)
    Raises: ValueError
    """
    u_shape = u.get_shape().as_list()

    if len(u_shape) > 2 and not is_batch:
        raise ValueError("Expected at most 2D tensors, but got %dD" % len(u_shape))
    if len(u_shape) > 3 and is_batch:
        raise ValueError("Expected at most 2D tensor batches, but got %dD" % len(u_shape))

    if v is None:
        v = u
    else:
        v_shape = v.get_shape().as_list()
        if u_shape != v_shape:
            raise ValueError("Shapes %s and %s do not match" % (u_shape, v_shape))

    n = u_shape[0] if not is_batch else u_shape[1]

    column_u = tf.reshape(u, (-1, 1) if not is_batch else (-1, n, 1))
   # U = tf.concat(1 if not is_batch else 2, [column_u] * n)
    U = tf.concat( [column_u] * n, 1 if not is_batch else 2)
    if v is u:
        return U + tf.transpose(U, None if not is_batch else [0, 2, 1])
    else:
        row_v = tf.reshape(v, (1, -1) if not is_batch else (-1, 1, n))
        V = tf.concat(0 if not is_batch else 1, [row_v] * n)

        return U + V

def decaying_softmax(shape, axis):
    rank = len(shape)
    max_val = shape[axis]

    weights_vector = np.arange(1, max_val + 1, dtype=np.float32)
    weights_vector = weights_vector[::-1]
    weights_vector = np.exp(weights_vector) / np.sum(np.exp(weights_vector))

    container = np.zeros(shape, dtype=np.float32)
    broadcastable_shape = [1] * rank
    broadcastable_shape[axis] = max_val

    return container + np.reshape(weights_vector, broadcastable_shape)

def unpack_into_tensorarray(value, axis, size=None):
    """
    unpacks a given tensor along a given axis into a TensorArray

    Parameters:
    ----------
    value: Tensor
        the tensor to be unpacked
    axis: int
        the axis to unpack the tensor along
    size: int
        the size of the array to be used if shape inference resulted in None

    Returns: TensorArray
        the unpacked TensorArray
    """

    shape = value.get_shape().as_list()
    rank = len(shape)
    dtype = value.dtype
    array_size = shape[axis] if not shape[axis] is None else size

    if array_size is None:
        raise ValueError("Can't create TensorArray with size None")
    array = tf.TensorArray(dtype=dtype, size=array_size)
    dim_permutation = [axis] + range(0, axis) + range(axis + 1, rank)
    unpack_axis_major_value = tf.transpose(value, dim_permutation)
    full_array = array.unstack(unpack_axis_major_value)

    return full_array

def pack_into_tensor(array, axis):
    """
    packs a given TensorArray into a tensor along a given axis

    Parameters:
    ----------
    array: TensorArray
        the tensor array to pack
    axis: int
        the axis to pack the array along

    Returns: Tensor
        the packed tensor
    """

    packed_tensor = array.stack()
    
    shape = packed_tensor.get_shape()

    rank = len(shape)

    dim_permutation = [axis] + range(0, axis) + range(axis + 1, rank)
    
    correct_shape_tensor = tf.transpose(packed_tensor, dim_permutation)

    return correct_shape_tensor

def pack_into_tensor_sized(array, axis, size):

    tensor_list = []
   
    for ss in range(size):
        tensor = array.read(ss)
        tensor_list.append(tensor)
    
    stacked_tensor = tf.stack(tensor_list)
    shape = stacked_tensor.get_shape()
    rank = len(shape)

    dim_permutation = [axis] + range(0, axis) + range(axis + 1, rank)
    correct_shape_tensor = tf.transpose(stacked_tensor, dim_permutation)

    return correct_shape_tensor

def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()

def prepare_summary(value, sum_name, sum_type):
    shape = value.get_shape().as_list()
    rank = len(shape)
    if sum_type is "Image":
        if rank is 5:
            if shape[3] == 1:
                value = tf.squeeze(value, axis=3)
            else:
                value = value[:,0,:,:,:]
        elif rank is 3:
            value = tf.expand_dims(value, axis=-1)
        perm = [0, 2, 1, 3]
        prepared_summary = tf.summary.image(sum_name, tf.transpose(value, perm), max_outputs=1)
    elif sum_type is "Category":
        if rank is not 4:
            ValueError("Time-series data not yet implemented for this type of summary")
        perm = [0, 3, 1, 2]
        prepared_summary = tf.summary.image(sum_name, tf.transpose(value, perm), max_outputs=1)
    elif sum_type is "Scalar":
        prepared_summary = tf.summary.scalar(sum_name, value)
    elif sum_type is "Histogram":
        prepared_summary = tf.summary.histogram(sum_name, value)

    return prepared_summary

def tb_setup_BasicFFClassifier(tb_logs_dir, input_data, model_output, target_output, gradients, loss, accuracy, average, variance, session):

    summaries = []

    # Loss summary
    summaries.append(prepare_summary(loss, "Loss", "Scalar"))
    summaries.append(prepare_summary(accuracy, "Accuracy", "Scalar"))
    summaries.append(prepare_summary(average, "Average", "Scalar"))
    summaries.append(prepare_summary(variance, "Variance", "Scalar"))

    # Gradient summary
    for i, (grad, var) in enumerate(gradients):
        if grad is not None:
            summaries.append(prepare_summary(grad, var.name + '/grad', "Histogram"))

    # IO summary
    summaries.append(prepare_summary(input_data, "IO/Input", "Image"))
    summaries.append(prepare_summary(model_output, "IO/Output", "Category"))
    summaries.append(prepare_summary(target_output, "IO/Target", "Category"))

    # Create summarize op
    summarize_op = tf.summary.merge(summaries)
    summarizer = tf.summary.FileWriter(tb_logs_dir, session.graph)

    # Open browser
    subprocess.call('fuser 6006/tcp -k &', shell=True)
    subprocess.call('tensorboard --logdir=' + tb_logs_dir + ' --reload_interval=10' + ' &', shell=True)
    ff = webbrowser.get('firefox')
    ff.open_new("http://127.0.1.1:6006/")

    return summarize_op, summarizer, summaries

def tb_flush(tb_logs_dir):
    shutil.rmtree(tb_logs_dir)

def display_to_console(i, session_results, summarizer, summarize, summarize_op, losses, accs):

    summarizer.add_summary(summarize_op, i)

    if summarize:
        llprint("\n\tAvg. Logistic Loss: %.4f\n" % (np.mean(losses)))
        llprint("\n\tAvg. Accuracy: %.4f\n" % (np.mean(accs)))
        losses = []
        accs = []

    return losses, accs

def take_checkpoint(ncomputer, session, ckpts_dir, task, iterations):

    llprint("\nSaving Checkpoint ... "),\
    ncomputer.save(session, os.path.join(ckpts_dir, task), 'step-%d' % (iterations))
    llprint("Done!\n")

def softmax_safe(X, axis=[0], eps=1e-6):

    num = tf.exp(X)
    den = tf.reduce_sum(num, axis=axis)

    for a in axis:
        den = tf.expand_dims(den, axis=a)
    sm = tf.div(num, den + eps)
    return sm

def bad_catcher(rv, g, var_names):

    rv_inf_catcher = {str(k): [] for k in range(len(rv))}
    rv_nan_catcher = {str(k): [] for k in range(len(rv))}
    g_nan_catcher = {var_names[k]: [] for k in range(len(g))}
    g_inf_catcher = {var_names[k]: [] for k in range(len(g))}

    rv_nan_names = []
    rv_inf_names = []
    g_nan_names = []
    g_inf_names = []

    # Find RV NaNs and Infs
    for ff in range(len(rv)):
        the_nans = np.argwhere(np.isnan(rv[ff]))
        the_infs = np.argwhere(np.isinf(rv[ff]))
        if len(the_nans) > 0:
            rv_nan_catcher[str(ff)].append(the_nans)
            rv_nan_names.append(str(ff))
        if len(the_infs) > 0:
            rv_inf_catcher[str(ff)].append(the_infs)
            rv_inf_names.append(str(ff))

    # Find G NaNs and Infs
    for ff in range(len(g)):
        the_nans = np.argwhere(np.isnan(g[ff]))
        the_infs = np.argwhere(np.isinf(g[ff]))
        if len(the_nans) > 0:
            g_nan_catcher[var_names[ff]].append(the_nans)
            g_nan_names.append(var_names[ff])
        if len(the_infs) > 0:
            g_inf_catcher[var_names[ff]].append(the_infs)
            g_inf_names.append(var_names[ff])

    return [rv_nan_catcher, rv_inf_catcher, g_nan_catcher, g_inf_catcher, rv_nan_names, rv_inf_names, g_nan_names, g_inf_names]


