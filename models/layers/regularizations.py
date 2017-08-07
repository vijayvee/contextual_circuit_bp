import numpy as np
import tensorflow as tf
from models.layers import regularization_functions


class regularizations(object):
    """Wrapper class for activation functions."""
    def __init__(self, kwargs**):
        """Globals for activation functions."""
        self.training = True
        self.update_params(kwargs**)
        
    def update_params(self, d):
        for k, v in d.iteritems():
            update = self.get(k)
            if update is not None:
                self[k] = v

    def dropout(self, x, keep_prob, kwargs**):
        """Dropout."""
        return tf.nn.dropout(x, keep_prob=keep_prob)

    def l1(self, x, x_mean=0):
        return tf.reduce_mean(tf.abs(x - x_mean))

    def l2(self, x, x_mean=0):
        return tf.nn.l2_loss(x - x_mean)

    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)

