import numpy as np
import tensorflow as tf


def avg_pool(
        bottom,
        name,
        k=[1, 2, 2, 1],
        s=[1, 2, 2, 1],
        p='SAME'):
    return tf.nn.avg_pool(
        bottom,
        ksize=k,
        strides=s,
        padding=p,
        name=name)


def max_pool(
        bottom,
        name,
        k=[1, 2, 2, 1],
        s=[1, 2, 2, 1],
        p='SAME'):
    return tf.nn.max_pool(
        bottom,
        ksize=k,
        strides=s,
        padding=p,
        name=name)

