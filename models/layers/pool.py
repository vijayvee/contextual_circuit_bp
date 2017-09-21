"""Pooling wrapper. TODO: add interpreter."""
import tensorflow as tf


def avg_pool(
        self,
        bottom,
        name,
        k=[1, 2, 2, 1],
        s=[1, 2, 2, 1],
        p='SAME'):
    """Local average pooling."""
    return self, tf.nn.avg_pool(
        bottom,
        ksize=k,
        strides=s,
        padding=p,
        name=name)


def max_pool(
        self,
        bottom,
        name,
        k=[1, 2, 2, 1],
        s=[1, 2, 2, 1],
        p='SAME'):
    """Local max pooling."""
    return self, tf.nn.max_pool(
        bottom,
        ksize=k,
        strides=s,
        padding=p,
        name=name)
