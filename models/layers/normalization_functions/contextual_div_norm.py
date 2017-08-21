"""
Divisive normalization implementation.

See paper Normalizing the Normalizers: Comparing and Extending Network
Normalization Schemes. Mengye Ren*, Renjie Liao*, Raquel Urtasun, Fabian H.
Sinz, Richard S. Zemel. 2016. https://arxiv.org/abs/1611.04520
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf
from utils import py_utils


def contextual_div_norm_2d(
        x,
        CRF_sum_window,
        CRF_sup_window,
        eCRF_sum_window,
        eCRF_sup_window,
        strides,
        padding,
        gamma=None,
        beta=None,
        eps=1.0,
        scope="dn",
        name="dn_out",
        return_mean=False):
    """Applies divisive normalization on CNN feature maps.
    Collect mean and variances on x on a local window across channels.
    And apply normalization as below:
      x_ = gamma * (x - mean) / sqrt(var + eps) + beta
    https://github.com/renmengye/div-norm/blob/master/div_norm.py

    Args:
      x: Input tensor, [B, H, W, C].
      sum_window: Summation window size, [H_sum, W_sum].
      sup_window: Suppression window size, [H_sup, W_sup].
      gamma: Scaling parameter.
      beta: Bias parameter.
      eps: Denominator bias.
      return_mean: Whether to also return the computed mean.

    Returns:
      normed: Divisive-normalized variable.
      mean: Mean used for normalization (optional).
    """
    if not isinstance(CRF_sum_window, list):
        CRF_sum_window = list(np.repeat(CRF_sum_window, 2))
    if not isinstance(CRF_sup_window, list):
        CRF_sup_window = list(np.repeat(CRF_sup_window, 2))
    if not isinstance(eCRF_sum_window, list):
        eCRF_sum_window = list(np.repeat(eCRF_sum_window, 2))
    if not isinstance(eCRF_sup_window, list):
        eCRF_sup_window = list(np.repeat(eCRF_sup_window, 2))
    k = int(x.get_shape()[-1])
    with tf.variable_scope(scope):

        # Q
        q_array = np.ones((CRF_sum_window + [k, k]))
        q_array /= q_array.sum()
        w_sum = tf.cast(tf.constant(q_array), tf.float32)
        # U
        u_array = np.ones((CRF_sum_window + [k, 1]))
        u_array /= u_array.sum()
        w_sup = tf.cast(tf.constant(u_array), tf.float32)
        CRF_sum_window = CRF_sum_window[0]
        CRF_sup_window = CRF_sup_window[0]
        # P
        p_shape = eCRF_sum_window + [k, k]
        eCRF_sum_window = eCRF_sum_window[0]
        p_array = np.zeros(p_shape)
        for pdx in range(k):
            p_array[:eCRF_sum_window, :eCRF_sum_window, pdx, pdx] = 1.0
        p_array[
            eCRF_sum_window // 2 - py_utils.ifloor(
                CRF_sum_window / 2.0):eCRF_sum_window // 2 + py_utils.iceil(
                CRF_sum_window / 2.0),
            CRF_sum_window // 2 - py_utils.ifloor(
                CRF_sum_window / 2.0):eCRF_sum_window // 2 + py_utils.iceil(
                CRF_sum_window / 2.0),
            :,  # exclude CRF!
            :] = 0.0
        w_esum = tf.cast(
            tf.constant(p_array) / p_array.sum(), tf.float32)

        # T
        t_shape = eCRF_sup_window + [k, k]
        eCRF_sup_window = eCRF_sup_window[0]
        t_array = np.zeros(t_shape)
        for tdx in range(k):
            t_array[:eCRF_sup_window, :eCRF_sup_window, tdx, tdx] = 1.0
        t_array[
            eCRF_sup_window // 2 - py_utils.ifloor(
                CRF_sup_window / 2.0):eCRF_sup_window // 2 + py_utils.iceil(
                CRF_sup_window / 2.0),
            eCRF_sup_window // 2 - py_utils.ifloor(
                CRF_sup_window / 2.0):eCRF_sup_window // 2 + py_utils.iceil(
                CRF_sup_window / 2.0),
            :,  # exclude near surround!
            :] = 0.0

        w_esup = tf.cast(
            tf.constant(t_array) / t_array.sum(), tf.float32)

        # SUM
        x_mean_CRF = tf.nn.conv2d(
            x,
            w_sum,
            strides=strides,
            padding=padding)
        x_mean_eCRF = tf.nn.conv2d(
            x,
            w_esum,
            strides=strides,
            padding=padding)
        normed = x - x_mean_CRF - x_mean_eCRF
        x2 = tf.square(normed)

        # SUP
        x2_mean_CRF = tf.nn.conv2d(
            x2,
            w_sup,
            strides=strides,
            padding=padding)
        x2_mean_eCRF = tf.nn.conv2d(
            x2,
            w_esup,
            strides=strides,
            padding=padding)
        denom = tf.sqrt(x2_mean_CRF + x2_mean_eCRF + eps)
        normed = normed / denom
        if gamma is not None:
            normed *= gamma
        if beta is not None:
            normed += beta
    normed = tf.identity(normed, name=name)
    if return_mean:
        return normed, x2
    else:
        return normed
