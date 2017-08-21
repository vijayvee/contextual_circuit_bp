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
    with tf.variable_scope(scope):
        w_sum = tf.ones(
            CRF_sum_window + [1, 1]) / np.prod(np.array(CRF_sum_window))
        w_sup = tf.ones(
            CRF_sup_window + [1, 1]) / np.prod(np.array(CRF_sup_window))
        w_esum = tf.ones(
            eCRF_sum_window + [1, 1]) / np.prod(np.array(eCRF_sum_window))
        w_esup = tf.ones(
            eCRF_sup_window + [1, 1]) / np.prod(np.array(eCRF_sup_window))
        x_mean = tf.reduce_mean(x, [3], keep_dims=True)

        # SUM
        x_mean_CRF = tf.nn.conv2d(
            x_mean,
            w_sum,
            strides=strides,
            padding=padding)
        x_mean_eCRF = tf.nn.conv2d(
            x_mean,
            w_esum,
            strides=strides,
            padding=padding)
        normed = x - x_mean_CRF - x_mean_eCRF
        x2 = tf.square(normed)
        x2_mean = tf.reduce_mean(x2, [3], keep_dims=True)

        # SUP
        x2_mean_CRF = tf.nn.conv2d(
            x2_mean,
            w_sup,
            strides=strides,
            padding=padding)
        x2_mean_eCRF = tf.nn.conv2d(
            x2_mean,
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
        return normed, x_mean
    else:
        return normed
