import tensorflow as tf
from utils import py_utils
from ops.eRF_calculator import eRF_calculator
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
        self.timesteps = 3
        self.scale_CRF = True
        self.bias_CRF = True
        self.lesions = [None]
        self.training = None
        self.strides = [1, 1, 1, 1]
        self.padding = 'SAME'
        self.update_params(kwargs)

    def update_params(self, kwargs):
        if kwargs is not None:
            for k, v in kwargs.iteritems():
                setattr(self, k, v)

    def set_RFs(
            self,
            eRF,
            layer,
            eSRF=None,
            iSRF=None,
            SSN=None,
            SSF=None,
            V1_CRF=0.26,
            V1_neCRF=0.54,
            V1_feCRF=1.41,
            default_stride=1,
            padding=1):
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
        eRFc = eRF_calculator()
        conv = {
            'padding': 1,
            'kernel': layer['filter_size'][0],
            'stride': 1
        }
        if len(layer['filter_size']) > 1:
            raise RuntimeError(
                'API not implemented for layers with > 1 module.')
        self.SRF = layer['filter_size'][0]
        self.CRF_excitation = layer['filter_size'][0]
        self.CRF_inhibition = layer['filter_size'][0]
        if eSRF is not None:
            self.CRF_excitation
        if iSRF is not None:
            self.CRF_inhibition
        if SSN is None:
            SSN_eRF = py_utils.iceil(eRF['r_i'] * (V1_neCRF / V1_CRF))
            self.SSN = eRFc.outFromIn(conv=conv, layer=eRF, fix_r_out=SSN_eRF)
        else:
            self.SSN = SSN
        if SSF is None:
            SSF_eRF = py_utils.iceil(eRF['r_i'] * (V1_feCRF / V1_CRF))
            self.SSF = eRFc.outFromIn(conv=conv, layer=eRF, fix_r_out=SSF_eRF)
        else:
            self.SSF = SSF

    def contextual(self, x, layer, eRF, aux):
        """Contextual model from paper with learnable weights."""
        self.update_params(aux)
        self.set_RFs(layer=layer, eRF=eRF)
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

    def contextual_rnn(self, x, layer, eRF, aux):
        """Contextual model translated into a RNN architecture."""
        self.update_params(aux)
        self.set_RFs(layer=layer, eRF=eRF)
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

    def divisive_2d(self, x, layer, eRF, aux):
        """Divisive normalization 2D."""
        self.update_params(aux)
        self.set_RFs(layer=layer, eRF=eRF)
        return div_norm.div_norm_2d(
            x,
            sum_window=self.CRF_excitation,
            sup_window=self.CRF_inhibition,
            gamma=self.scale_CRF,
            beta=self.bias_CRF,
            strides=self.strides,
            padding=self.padding), None

    def divisive_1d(self, x, layer, eRF, aux):
        """Divisive normalization 2D."""
        self.update_params(aux)
        self.set_RFs(layer=layer, eRF=eRF)
        return div_norm.div_norm_1d(
            x,
            sum_window=self.CRF_excitation,
            sup_window=self.CRF_inhibition,
            gamma=self.scale_CRF,
            beta=self.bias_CRF), None

    def batch(self, x, layer, eRF, aux):
        """Batch normalization."""
        self.update_params(aux)
        return tf.layers.batch_normalization(
            x,
            scale=self.scale_CRF,
            center=self.bias_CRF,
            training=self.training), None

    def batch_renorm(self, x, layer, eRF, aux):
        """Batch re-normalization."""
        self.update_params(aux)
        return tf.layers.batch_normalization(
            x,
            scale=self.scale_CRF,
            center=self.bias_CRF,
            training=self.training,
            renorm=True), None

    def layer(self, x, layer, eRF, aux):
        """Layer normalization."""
        self.update_params(aux)
        return layer_norm.layer_norm(
            x,
            gamma=self.scale_CRF,
            beta=self.bias_CRF), None

    def lrn(self, x, layer, eRF, aux):
        """Local response normalization."""
        self.update_params(aux)
        self.set_RFs(layer=layer, eRF=eRF)
        return tf.nn.local_response_normalization(
            x,
            depth_radius=self.CRF_inhibition,
            alpha=self.scale_CRF,
            beta=self.bias_CRF), None

