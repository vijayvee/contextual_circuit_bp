import numpy as np
import tensorflow as tf
from models.layers import normalization_functions as nf
from utils import pyutils


class normalizations(object):
    """Wrapper class for activation functions."""
    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)

    def __init__(self, kwargs=None):
        """Globals for normalization functions."""
        self.timesteps = 1
        self.scale_CRF = True
        self.bias_CRF = True
        self.lesions = None
        self.training = None
        if kwargs is not None:
            self.update_params(**kwargs)

    def update_params(self, **kwargs):
        for k, v in kwargs.iteritems():
            setattr(self, k, v)

    def set_RFs(self, eSRF, iSRF, SSN, SSF):
        # Angelucci & Shushruth 2013 V1 RFs:
        # CRF = 0.26 degrees (1x)
        # eCRF near = 0.54 degrees (2x)
        # eCRF far = 1.41 degrees (5.5x)
        assert SRF is not None, 'Provide an SRF (the layer\'s effective RF).'
        self.SRF = SRF
        if SSN is None:
            self.SSN = self.SRF * pyutils.iround(0.54 / 0.26)
        if SSF is None:
            self.SSF = pyutils.iround(self.SRF * 5.42)
        self.SSN = SSN
        self.SSF = SSF

    def contextual(self, x, **kwargs):
        """Contextual model from paper with learnable weights."""
        self.set_CRF_eCRF_sizes(
            CRF_excitation=self.CRF_excitation,
            CRF_inhibition=self.CRF_inhibition,
            eCRF_excitation=self.eCRF_excitation,
            eCRF_inhibition=self.eCRF_inhibition,
            )
        if self.CRF_excitation != self.CRF_inhibition:
            CRF_size = np.max([self.CRF_excitation, self.CRF_inhibition])
            print 'CRF inhibition/excitation RFs are uneven.' + \
                'Using the max extent for the contextual model CRF.'
        else:
            CRF_size = self.CRF_inhibition
        contextual_layer = nf.ContextualCircuit(
            timesteps=self.timesteps,
            lesions=None,
            SRF=CRF_size,
            SSN=self.eCRF_excitation,
            SSF=self.eCRF_inhibition)
        return contextual_layer(x)

    def contextual_rnn(self, x, **kwargs):
        """Contextual model translated into a RNN architecture."""
        self.set_CRF_eCRF_sizes(
            CRF_excitation=self.CRF_excitation,
            CRF_inhibition=self.CRF_inhibition,
            eCRF_excitation=self.eCRF_excitation,
            eCRF_inhibition=self.eCRF_inhibition,
            )
        contextual_layer = nf.contextual_rnn.ContextualCircuit(
            timesteps=self.timesteps,
            lesions=None,
            SRF=CRF_size,
            SSN=self.eCRF_excitation,
            SSF=self.eCRF_inhibition)
        return contextual_layer(x)

    def divisive(self, x, **kwargs):
        """Divisive normalization 2D."""
        self.set_CRF_eCRF_sizes(
            CRF_excitation=self.CRF_excitation,
            CRF_inhibition=self.CRF_inhibition
            )
        return nf.div_norm.div_norm_2d(
            x,
            sum_window=self.CRF_excitation,
            sup_window=self.CRF_inhibition,
            gamma=self.scale_CRF,
            beta=self.bias_CRF)

    def divisive_1d(self, x, **kwargs):
        """Divisive normalization 2D."""
        self.set_CRF_eCRF_sizes(
            CRF_excitation=self.CRF_excitation,
            CRF_inhibition=self.CRF_inhibition
            )
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
        self.set_CRF_eCRF_sizes(
            CRF_inhibition=self.CRF_inhibition
            )
        return tf.nn.local_response_normalization(
            x,
            depth_radius=self.CRF_inhibition,
            alpha=self.scale_CRF,
            beta=self.bias_CRF)
