import numpy as np
import tensorflow as tf
from models.layers import activation_functions


class activations(object):
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

    def relu(self, x, kwargs**):   
        """Rectified linearity."""
        return tf.nn.relu(x)

    def selu(self, x, kwargs**):   
        """Scaled exponential linear unit."""
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(tf.greater_equal(x, 0.0), x, alpha * tf.nn.elu(x))

    def soft_plus(self, kwargs**):
        """Soft logistic."""
        return tf.nn.softplus(x)

    def crelu(self, x, kwargs**):
        """Concatenated +/- relu."""
        return tf.nn.crelu(x)

    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)

