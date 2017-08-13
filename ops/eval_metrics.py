import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops


def class_accuracy(pred, labels):
    """Accuracy of 1/n*sum(pred_i == label_i)."""
    return tf.reduce_mean(
        tf.to_float(tf.equal(tf.argmax(pred, 1), tf.cast(
            labels, dtype=tf.int64))))


def tf_confusion_matrix(pred, targets):
    """Wrapper for calculating confusion matrix."""
    return tf.contrib.metrics.confusion_matrix(pred, targets)


def pearson_correlation(x, y):
    """Pearson correlation: rho(x, y)."""
    x_shape = [int(xi) for xi in x.get_shape()]
    y_shape = [int(yi) for yi in y.get_shape()]
    x = tf.reshape(x, [x_shape[0], np.prod(x_shape[1:])])
    y = tf.reshape(y, [y_shape[0], np.prod(y_shape[1:])])
    cov = math_ops.matmul(x, y, transpose_b=True) / (int(x.get_shape()[0]) - 1)
    _, sd_x = tf.nn.moments(x, axes=[1])
    _, sd_y = tf.nn.moments(y, axes=[1])
    return cov / (sd_x * sd_y)
