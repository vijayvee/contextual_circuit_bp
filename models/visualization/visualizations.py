import tensorflow as tf
from models.visualization.visualizations import gradient_methods as gm


def identity(x):
    return x


def sum_abs(x):
    return tf.reduce_sum(tf.abs(x), axis=-1)


def sum_p(x, p=2):
    return tf.reduce_sum(tf.pow(x, p), axis=-1) 


def max_p(x, p=2):
    return tf.reduce_max(tf.pow(x, p), axis=-1)  


class visualizations(object):
    """Visualization class."""
    def __init__(self, kwargs):
        """Class initialization."""
        self.summary = sum_abs 

    def gradient_image(self, x, layer):
        """Wrapper for simonyan 2014 gradient image."""
        return gm.gradient_image(x, layer)

    def stochastic_gradient_image(self, x, layer, num_iterations=1000):
        return gm.sgi(x, layer, num_iterations)

