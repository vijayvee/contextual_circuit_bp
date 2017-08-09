import numpy as np
import tensorflow as tf
from models.layers import normalization_functions as nf


class normalizations(object):
    """Wrapper class for activation functions."""
    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)

    def __init__(self, kwargs=None):
        """Globals for normalization functions."""
        self.timesteps = 1
        self.CRF_excitation = 1
        self.CRF_inhibition = 1
        self.eCRF_excitation = 3
        self.eCRF_inhibition = 9
        self.scale_CRF = True
        self.bias_CRF = True
        self.training = None
        if kwargs is not None:
            self.update_params(**kwargs)

    def update_params(self, **kwargs):
        for k, v in kwargs.iteritems():
            setattr(self, k, v)

    def contextual(self, x, **kwargs):
        """Contextual model 2D."""
        contextual_layer = nf.ContextualCircuit()
        if self.CRF_excitation != self.CRF_inhibition:
            CRF_size = np.max([self.CRF_excitation, self.CRF_inhibition])
            print 'CRF inhibition/excitation RFs are uneven.' + \
                'Using the max extent for the contextual model CRF.'
        else:
            CRF_size = self.CRF_inhibition
        return contextual_layer(
            x,
            timesteps=self.timesteps,
            lesions=None,
            SRF=CRF_size,
            SSN=self.eCRF_excitation,
            SSF=self.eCRF_inhibition)

    def divisive(self, x, **kwargs):
        """Divisive normalization 2D."""
        return nf.div_norm.div_norm_2d(
            x,
            sum_window=self.CRF_excitation,
            sup_window=self.CRF_inhibition,
            gamma=self.scale_CRF,
            beta=self.bias_CRF)

    def divisive_1d(self, x, **kwargs):
        """Divisive normalization 2D."""
        return nf.div_norm.div_norm_1d(
            x,
            sum_window=self.CRF_excitation,
            sup_window=self.CRF_inhibition,
            gamma=self.scale_CRF,
            beta=self.bias_CRF)

    def batch(self, x, **kwargs):
        """Batch normalization."""
        return tf.layers.batch_normalization(
            x,
            scale=self.scale_CRF,
            center=self.bias_CRF,
            training=self.training)

    def batch_renorm(self, x, **kwargs):
        """Batch re-normalization."""
        return tf.layers.batch_normalization(
            x,
            scale=self.scale_CRF,
            center=self.bias_CRF,
            training=self.training,
            renorm=True)

    def layer(self, x, **kwargs):
        """Layer normalization."""
        return nf.layer_norm.layer_norm(
            x,
            gamma=self.scale_CRF,
            beta=self.bias_CRF)

    def lrn(self, x, **kwargs):
        """Local response normalization."""
        return tf.nn.local_response_normalization(
            x,
            depth_radius=self.CRF_inhibition,
            alpha=self.scale_CRF,
            beta=self.bias_CRF)
