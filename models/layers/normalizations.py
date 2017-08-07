import numpy as np
import tensorflow as tf
from models.layers import normalization_functions as nf


class normalizations(object):
    """Wrapper class for activation functions."""
    def __init__(self, kwargs**):
        """Globals for normalization functions."""
        self.timesteps = 1
        self.CRF_excitation = 1
        self.CRF_inhibition = 1
        self.scale_CRF = True
        self.bias_CRF = True
        self.training = True
        self.update_params(kwargs**)

    def update_params(self, d):
        for k, v in d.iteritems():
            update = self.get(k)
            if update is not None:
                self[k] = v

    def contextual(self, x, kwargs**):   
        """Contextual model 2D."""
        return nf.contextual(x, timesteps=self.timesteps)

    def divisive(self, x, kwargs**):   
        """Divisive normalization 2D."""
        return div_norm_2d(
            x,
            sum_window=self.CRF_excitation,
            sup_window=self.CRF_inhibition,
            gamma=self.scale_CRF,
            beta=self.bias_CRF) 

    def batch(self, x, kwargs**):
        """Batch normalization."""
        return tf.layers.batch_normalization(
            x,
            scale=self.scale_CRF,
            center=self.bias_CRF,
            training=self.training)

    def layer(self, x, kwargs**):
        """Layer normalization."""
        return layer_norm(
            x,
            gamma=self.scale_CRF,
            beta=self.bias_CRF) 

    def lrn(self, x, kwargs**):
        """Local response normalization."""

    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)

