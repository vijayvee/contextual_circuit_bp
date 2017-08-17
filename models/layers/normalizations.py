import numpy as np
import tensorflow as tf
from utils import py_utils
from models.layers.normalization_functions import div_norm
from models.layers.normalization_functions import layer_norm
from models.layers.normalization_functions import contextual
from models.layers.normalization_functions import contextual_rnn


class normalizations(object):
    """Wrapper class for activation functions."""
    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)

    def __init__(self, kwargs=None):
        """Globals for normalization functions."""
        self.timesteps = 2
        self.scale_CRF = True
        self.bias_CRF = True
        self.lesions = [None]
        self.training = None
        self.strides = [1, 1, 1, 1]
        self.padding = 'SAME'
        if kwargs is not None:
            self.update_params(**kwargs)

    def update_params(self, **kwargs):
        for k, v in kwargs.iteritems():
            setattr(self, k, v)

    def set_RFs(
            self,
            eSRF,
            iSRF,
            SSN=None,
            SSF=None,
            V1_CRF=0.26,
            V1_neCRF=0.54,
            V1_feCRF=1.00):  # 1.41
        """ Set RF sizes for the normalizations.
        Based on calculation of an effective RF (i.e. not
        simply kernel size of the current layer, but instead
        w.r.t. input).

        Angelucci & Shushruth 2013 V1 RFs:
        https://www.shadlenlab.columbia.edu/people/Shushruth/Lab_Page/Home_files/GalleyProof.pdf
        CRF = 0.26 degrees (1x)
        eCRF near = 0.54 degrees (2x)
        eCRF far = 1.41 degrees (5.5x)
        """
        if eSRF != iSRF:
            if eSRF is not None and iSRF is not None:
                CRF_size = np.max([eSRF, iSRF])
            elif eSRF is None:
                CRF_size = iSRF
            elif iSRF is None:
                CRF_size = eSRF
            print 'CRF inhibition/excitation RFs are uneven.' + \
                'Using the max extent for the contextual model CRF.'
        else:
            CRF_size = iSRF
        self.CRF_excitation = CRF_size
        self.CRF_inhibition = CRF_size
        self.SRF = CRF_size
        if SSN is None:
            self.SSN = py_utils.iceil(self.SRF * (V1_neCRF / V1_CRF))
        else:
            self.SSN = SSN
        if SSF is None:
            self.SSF = py_utils.iceil(self.SRF * (V1_feCRF / V1_CRF))
        else:
            self.SSF = SSF

    def contextual(self, x, eRF):
        """Contextual model from paper with learnable weights."""
        self.set_RFs(
            eSRF=eRF,
            iSRF=eRF)
        contextual_layer = contextual.ContextualCircuit(
            X=x,
            timesteps=self.timesteps,
            lesions=self.lesions,
            SRF=self.SRF,
            SSN=self.SSN,
            SSF=self.SSF,
            strides=self.strides,
            padding=self.padding)
        return contextual_layer.build()

    def contextual_rnn(self, x, eRF):
        """Contextual model translated into a RNN architecture."""
        self.set_RFs(
            eSRF=eRF,
            iSRF=eRF)
        contextual_layer = contextual_rnn.ContextualCircuit(
            X=x,
            timesteps=self.timesteps,
            lesions=self.lesions,
            SRF=self.SRF,
            SSN=self.SSN,
            SSF=self.SSF,
            strides=self.strides,
            padding=self.padding)
        return contextual_layer.build()

    def divisive_2d(self, x, eRF):
        """Divisive normalization 2D."""
        self.set_RFs(
            eSRF=eRF,
            iSRF=eRF)
        return div_norm.div_norm_2d(
            x,
            sum_window=self.CRF_excitation,
            sup_window=self.CRF_inhibition,
            gamma=self.scale_CRF,
            beta=self.bias_CRF,
            strides=self.strides,
            padding=self.padding)

    def divisive_1d(self, x, eRF):
        """Divisive normalization 2D."""
        self.set_RFs(
            eSRF=eRF,
            iSRF=eRF)
        return div_norm.div_norm_1d(
            x,
            sum_window=self.CRF_excitation,
            sup_window=self.CRF_inhibition,
            gamma=self.scale_CRF,
            beta=self.bias_CRF)

    def batch(self, x, eRF):
        """Batch normalization."""
        return tf.layers.batch_normalization(
            x,
            scale=self.scale_CRF,
            center=self.bias_CRF,
            training=self.training)

    def batch_renorm(self, x, eRF):
        """Batch re-normalization."""
        return tf.layers.batch_normalization(
            x,
            scale=self.scale_CRF,
            center=self.bias_CRF,
            training=self.training,
            renorm=True)

    def layer(self, x, eRF):
        """Layer normalization."""
        return layer_norm.layer_norm(
            x,
            gamma=self.scale_CRF,
            beta=self.bias_CRF)

    def lrn(self, x, eRF):
        """Local response normalization."""
        self.set_RFs(
            eSRF=eRF,
            iSRF=eRF)
        return tf.nn.local_response_normalization(
            x,
            depth_radius=self.CRF_inhibition,
            alpha=self.scale_CRF,
            beta=self.bias_CRF)
