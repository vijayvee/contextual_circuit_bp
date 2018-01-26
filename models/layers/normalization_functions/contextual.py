"""Contextual model with partial filters."""
import numpy as np
import tensorflow as tf
from utils import py_utils
from ops import initialization


# TODO:
# 1. Separable HW/Feature weights
# 2. Combine P/Q into single weight matrix (prob a dead end)
# 3. Association field by turning P into a Full conv matrix
# 4. Frobenius norm regularization for the association field
# 5. Accept timeseries data
# 6. Spatial Anisotropies between P and T tensors
def auxilliary_variables():
    """A dictionary containing defaults for auxilliary variables.

    These are adjusted by a passed aux dict variable."""
    return {
        'lesions': [None],  # ['Q', 'T', 'P', 'U'],
        'dtype': tf.float32,
        'return_weights': True,
        'hidden_init': 'random',
        'tuning_init': 'cov',  # TODO: Initialize tuning as input covariance
        'association_field': False,
        'nonnegative_association': False,
        'tuning_nl': 'relu',
        'train': True,
        'dropout': None,
        'separable': False,  # Need C++ implementation.
        'recurrent_nl': tf.nn.selu,  # tf.nn.leakyrelu, tf.nn.relu, tf.nn.selu
        'gate_nl': tf.nn.sigmoid,
        'normal_initializer': False,
        'gate_filter': 1
    }
# TODO: Regularization on the activations


def interpret_nl(nl):
    """Returns appropriate nonlinearity."""
    if nl is not None or nl is not 'pass':
        # Rectification on the "tuned" activities
        if nl == 'relu':
            return tf.nn.relu
        elif nl == 'selu':
            return tf.nn.selu
        else:
            raise NotImplementedError


class ContextualCircuit(object):
    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)

    def __init__(
            self,
            X,
            timesteps=1,
            SRF=1,
            SSN=9,
            SSF=29,
            strides=[1, 1, 1, 1],
            padding='SAME',
            aux=None):
        """Global initializations and settings."""
        self.X = X
        self.n, self.h, self.w, self.k = [int(x) for x in X.get_shape()]
        self.timesteps = timesteps
        self.strides = strides
        self.padding = padding

        # Sort through and assign the auxilliary variables
        aux_vars = auxilliary_variables()
        if aux is not None and isinstance(aux, dict):
            for k, v in aux.iteritems():
                aux_vars[k] = v
        self.update_params(aux_vars)

        # Kernel shapes
        self.SRF, self.SSN, self.SSF = SRF, SSN, SSF
        self.SSN_ext = 2 * py_utils.ifloor(SSN / 2.0) + 1
        self.SSF_ext = 2 * py_utils.ifloor(SSF / 2.0) + 1
        if self.SSN is None:
            self.SSN = self.SRF * 3
        if self.SSF is None:
            self.SSF = self.SRF * 5
        if self.separable:
            self.q_shape = [self.SRF, self.SRF, 1, 1]
            self.u_shape = [self.SRF, self.SRF, 1, 1]
            self.p_shape = [self.SSN_ext, self.SSN_ext, 1, 1]
            self.t_shape = [self.SSF_ext, self.SSF_ext, 1, 1]
        else:
            self.q_shape = [self.SRF, self.SRF, self.k, self.k]
            self.u_shape = [self.SRF, self.SRF, self.k, 1]
            self.p_shape = [self.SSN_ext, self.SSN_ext, self.k, self.k]
            self.t_shape = [self.SSF_ext, self.SSF_ext, self.k, self.k]
        self.i_shape = [self.gate_filter, self.gate_filter, self.k, self.k]
        self.o_shape = [self.gate_filter, self.gate_filter, self.k, self.k]
        self.bias_shape = [1, 1, 1, self.k]

        if self.association_field:
            self.tuning_params = ['Q', 'T']
            self.p_shape = [self.SSN_ext, self.SSN_ext, self.k, self.k]
        else:
            self.tuning_params = ['Q', 'P', 'T']  # Learned connectivity
        self.tuning_shape = [1, 1, self.k, self.k]

        # Nonlinearities and initializations
        self.u_nl = tf.identity
        self.t_nl = tf.identity
        self.q_nl = tf.identity
        self.p_nl = tf.identity
        self.tuning_nl = interpret_nl(self.tuning_nl)

    def update_params(self, kwargs):
        """Update the class attributes with kwargs."""
        if kwargs is not None:
            for k, v in kwargs.iteritems():
                setattr(self, k, v)

    def prepare_tensors(self):
        """ Prepare recurrent/forward weight matrices."""
        self.weight_dict = {  # Weights lower/activity upper
            'U': {
                'r': {
                    'weight': 'u_r',
                    'activity': 'U_r'
                    }
                },
            'T': {
                'r': {
                    'weight': 't_r',
                    'activity': 'T_r',
                    'tuning': 't_t'
                    }
                },
            'P': {
                'r': {
                    'weight': 'p_r',
                    'activity': 'P_r',
                    'tuning': 'p_t'
                    }
                },
            'Q': {
                'r': {
                    'weight': 'q_r',
                    'activity': 'Q_r',
                    'tuning': 'q_t'
                    }
                },
            'I': {
                'r': {  # Recurrent state
                    'weight': 'i_r',
                    'bias': 'i_b',
                    'activity': 'I_r'
                },
                'f': {  # Recurrent state
                    'weight': 'i_f',
                    'activity': 'I_f'
                },
            },
            'O': {
                'r': {  # Recurrent state
                    'weight': 'o_r',
                    'bias': 'o_b',
                    'activity': 'O_r'
                },
                'f': {  # Recurrent state
                    'weight': 'o_f',
                    'activity': 'O_f'
                },
            },
            'xi': {
                'r': {  # Recurrent state
                    'weight': 'xi',
                }
            },
            'alpha': {
                'r': {  # Recurrent state
                    'weight': 'alpha',
                }
            },
            'beta': {
                'r': {  # Recurrent state
                    'weight': 'beta',
                }
            },
            'mu': {
                'r': {  # Recurrent state
                    'weight': 'mu',
                }
            },
            'nu': {
                'r': {  # Recurrent state
                    'weight': 'nu',
                }
            },
            'zeta': {
                'r': {  # Recurrent state
                    'weight': 'zeta',
                }
            },
            'gamma': {
                'r': {  # Recurrent state
                    'weight': 'gamma',
                }
            },
            'delta': {
                'r': {  # Recurrent state
                    'weight': 'delta',
                }
            }
        }

        # tuned summation: pooling in h, w dimensions
        #############################################
        q_array = np.ones(self.q_shape) / np.prod(self.q_shape)
        if 'Q' in self.lesions:
            q_array = np.zeros_like(q_array).astype(np.float32)
            print 'Lesioning CRF excitation.'
        setattr(
            self,
            self.weight_dict['Q']['r']['weight'],
            tf.get_variable(
                name=self.weight_dict['Q']['r']['weight'],
                dtype=self.dtype,
                initializer=q_array.astype(np.float32),
                trainable=False)
            )

        # untuned suppression: reduction across feature axis
        ####################################################
        u_array = np.ones(self.u_shape) / np.prod(self.u_shape)
        if 'U' in self.lesions:
            u_array = np.zeros_like(u_array).astype(np.float32)
            print 'Lesioning CRF inhibition.'
        setattr(
            self,
            self.weight_dict['U']['r']['weight'],
            tf.get_variable(
                name=self.weight_dict['U']['r']['weight'],
                dtype=self.dtype,
                initializer=u_array.astype(np.float32),
                trainable=False)
            )

        # weakly tuned summation: pooling in h, w dimensions
        #############################################
        p_array = np.ones(self.p_shape)
        p_array[
            self.SSN // 2 - py_utils.ifloor(
                self.SRF / 2.0):self.SSN // 2 + py_utils.iceil(
                self.SRF / 2.0),
            self.SSN // 2 - py_utils.ifloor(
                self.SRF / 2.0):self.SSN // 2 + py_utils.iceil(
                self.SRF / 2.0),
            :,  # exclude CRF!
            :] = 0.0
        p_array = p_array / p_array.sum()
        if 'P' in self.lesions:
            print 'Lesioning near eCRF.'
            p_array = np.zeros_like(p_array).astype(np.float32)

        # Association field is fully learnable
        if self.association_field and 'P' not in self.lesions:
            setattr(
                self,
                self.weight_dict['P']['r']['weight'],
                tf.get_variable(
                    name=self.weight_dict['P']['r']['weight'],
                    dtype=self.dtype,
                    initializer=initialization.xavier_initializer(
                        shape=self.p_shape,
                        uniform=self.normal_initializer),
                    trainable=True))
        else:
            setattr(
                self,
                self.weight_dict['P']['r']['weight'],
                tf.get_variable(
                    name=self.weight_dict['P']['r']['weight'],
                    dtype=self.dtype,
                    initializer=p_array.astype(np.float32),
                    trainable=False))

        # weakly tuned suppression: pooling in h, w dimensions
        ###############################################
        t_array = np.ones(self.t_shape)
        t_array[
            self.SSF // 2 - py_utils.ifloor(
                self.SSN / 2.0):self.SSF // 2 + py_utils.iceil(
                self.SSN / 2.0),
            self.SSF // 2 - py_utils.ifloor(
                self.SSN / 2.0):self.SSF // 2 + py_utils.iceil(
                self.SSN / 2.0),
            :,  # exclude near surround!
            :] = 0.0
        t_array = t_array / t_array.sum()
        if 'T' in self.lesions:
            print 'Lesioning Far eCRF.'
            t_array = np.zeros_like(t_array).astype(np.float32)
        setattr(
            self,
            self.weight_dict['T']['r']['weight'],
            tf.get_variable(
                name=self.weight_dict['T']['r']['weight'],
                dtype=self.dtype,
                initializer=t_array.astype(np.float32),
                trainable=False))

        # Connectivity tensors -- Q/P/T
        if 'Q' in self.lesions:
            print 'Lesioning CRF excitation connectivity.'
            setattr(
                self,
                self.weight_dict['Q']['r']['tuning'],
                tf.get_variable(
                    name=self.weight_dict['Q']['r']['tuning'],
                    dtype=self.dtype,
                    trainable=False,
                    initializer=np.zeros(
                        self.tuning_shape).astype(np.float32)))
        else:
            setattr(
                self,
                self.weight_dict['Q']['r']['tuning'],
                tf.get_variable(
                    name=self.weight_dict['Q']['r']['tuning'],
                    dtype=self.dtype,
                    trainable=True,
                    initializer=initialization.xavier_initializer(
                        shape=self.tuning_shape,
                        uniform=self.normal_initializer,
                        mask=None)))
        if not self.association_field:
            # Need a tuning tensor for near surround
            if 'P' in self.lesions:
                print 'Lesioning near eCRF connectivity.'
                setattr(
                    self,
                    self.weight_dict['P']['r']['tuning'],
                    tf.get_variable(
                        name=self.weight_dict['P']['r']['tuning'],
                        dtype=self.dtype,
                        trainable=False,
                        initializer=np.zeros(
                            self.tuning_shape).astype(np.float32)))
            else:
                setattr(
                    self,
                    self.weight_dict['P']['r']['tuning'],
                    tf.get_variable(
                        name=self.weight_dict['P']['r']['tuning'],
                        dtype=self.dtype,
                        trainable=True,
                        initializer=initialization.xavier_initializer(
                            shape=self.tuning_shape,
                            uniform=self.normal_initializer,
                            mask=None)))
        if 'T' in self.lesions:
            print 'Lesioning far eCRF connectivity.'
            setattr(
                self,
                self.weight_dict['T']['r']['tuning'],
                tf.get_variable(
                    name=self.weight_dict['T']['r']['tuning'],
                    dtype=self.dtype,
                    trainable=False,
                    initializer=np.zeros(
                        self.tuning_shape).astype(np.float32)))
        else:
            setattr(
                self,
                self.weight_dict['T']['r']['tuning'],
                tf.get_variable(
                    name=self.weight_dict['T']['r']['tuning'],
                    dtype=self.dtype,
                    trainable=True,
                    initializer=initialization.xavier_initializer(
                        shape=self.tuning_shape,
                        uniform=self.normal_initializer,
                        mask=None)))

        # Input
        setattr(
            self,
            self.weight_dict['I']['r']['weight'],
            tf.get_variable(
                name=self.weight_dict['I']['r']['weight'],
                dtype=self.dtype,
                trainable=True,
                initializer=initialization.xavier_initializer(
                    shape=self.i_shape,
                    uniform=self.normal_initializer,
                    mask=None)))
        setattr(
            self,
            self.weight_dict['I']['f']['weight'],
            tf.get_variable(
                name=self.weight_dict['I']['f']['weight'],
                dtype=self.dtype,
                trainable=True,
                initializer=initialization.xavier_initializer(
                    shape=self.i_shape,
                    uniform=self.normal_initializer,
                    mask=None)))
        setattr(
            self,
            self.weight_dict['I']['r']['bias'],
            tf.get_variable(
                name=self.weight_dict['I']['r']['bias'],
                dtype=self.dtype,
                trainable=True,
                initializer=tf.ones(self.bias_shape)))

        # Output
        setattr(
            self,
            self.weight_dict['O']['r']['weight'],
            tf.get_variable(
                name=self.weight_dict['O']['r']['weight'],
                dtype=self.dtype,
                trainable=True,
                initializer=initialization.xavier_initializer(
                    shape=self.o_shape,
                    uniform=self.normal_initializer,
                    mask=None)))
        setattr(
            self,
            self.weight_dict['O']['f']['weight'],
            tf.get_variable(
                name=self.weight_dict['O']['f']['weight'],
                dtype=self.dtype,
                trainable=True,
                initializer=initialization.xavier_initializer(
                    shape=self.o_shape,
                    uniform=self.normal_initializer,
                    mask=None)))
        setattr(
            self,
            self.weight_dict['O']['r']['bias'],
            tf.get_variable(
                name=self.weight_dict['O']['r']['bias'],
                dtype=self.dtype,
                trainable=True,
                initializer=tf.ones(self.bias_shape)))

        # Vector weights
        w_array = np.ones([1, 1, 1, self.k]).astype(np.float32)
        b_array = np.zeros([1, 1, 1, self.k]).astype(np.float32)
        self.xi = tf.get_variable(name='xi', initializer=w_array)
        self.alpha = tf.get_variable(name='alpha', initializer=w_array)
        self.beta = tf.get_variable(name='beta', initializer=w_array)
        self.mu = tf.get_variable(name='mu', initializer=b_array)
        self.nu = tf.get_variable(name='nu', initializer=b_array)
        self.zeta = tf.get_variable(name='zeta', initializer=w_array)
        self.gamma = tf.get_variable(name='gamma', initializer=w_array)
        self.delta = tf.get_variable(name='delta', initializer=w_array)

    def conv_2d_op(self, data, weight_key, out_key=None, weights=None):
        """2D convolutions, lesion, return or assign activity as attribute."""
        if weights is None:
            weights = self[weight_key]
        w_shape = [int(w) for w in weights.get_shape()]
        if len(w_shape) > 1 and int(w_shape[-2]) > 1:
            # Full convolutions
            activities = tf.nn.conv2d(
                data,
                weights,
                self.strides,
                padding=self.padding)
        elif len(w_shape) > 1 and int(w_shape[-2]) == 1:
            # Separable spacial
            d = int(data.get_shape()[-1])
            split_data = tf.split(data, d, axis=3)
            sep_convs = []
            for idx in range(len(split_data)):
                sep_convs += [tf.nn.conv2d(
                    split_data[idx],
                    weights,
                    self.strides,
                    padding=self.padding)]

            # TODO: Write the c++ for this.
            activities = tf.concat(sep_convs, axis=-1)
        else:
            raise RuntimeError

        # Do a split convolution
        if out_key is None:
            return activities
        else:
            setattr(
                self,
                out_key,
                activities)

    def apply_tuning(self, data, wm, nl=True):
        for k in self.tuning_params:
            if wm == k:
                data = self.conv_2d_op(
                    data=data,
                    weight_key=self.weight_dict[wm]['r']['tuning']
                    )
                if nl:
                    return self.tuning_nl(data)
                else:
                    return data
        return data

    def zoneout(self, dropout):
        """Calculate a dropout mask for update gates."""
        return tf.cast(
            tf.greater(tf.random_uniform(
                [1, 1, 1, self.k],
                minval=0,
                maxval=1.),
                dropout),  # zone-out dropout mask
            tf.float32)

    def full(self, i0, O, I):
        """Published CM with learnable weights.

        Swap out scalar weights for GRU-style update gates:
        # Eps_eta is I forget gate
        # Eta is I input gate
        # sig_tau is O forget gate
        # tau is O input gate
        """

        # Connectivity convolutions
        U = self.conv_2d_op(
            data=self.apply_tuning(O, 'U'),
            weight_key=self.weight_dict['U']['r']['weight']
        )
        T = self.conv_2d_op(
            data=self.apply_tuning(O, 'T'),
            weight_key=self.weight_dict['T']['r']['weight']
        )

        # Gates
        I_update_input = self.conv_2d_op(
            data=self.X,
            weight_key=self.weight_dict['I']['f']['weight']
        )
        I_update_recurrent = self.conv_2d_op(
            data=O,
            weight_key=self.weight_dict['I']['r']['weight']
        )
        I_update = self.gate_nl(
            I_update_input + I_update_recurrent + self[
                self.weight_dict['I']['r']['bias']])

        # Calculate and apply dropout if requested
        if self.train and self.dropout is not None:
            I_update = self.zoneout(self.dropout) * self.gate_nl(
                I_update_input + I_update_recurrent)
        elif not self.train and self.dropout is not None:
            I_update = (1 / self.dropout) * self.gate_nl(
                I_update_input + I_update_recurrent)

        # Circuit input
        I_summand = self.recurrent_nl(
            (self.xi * self.X)
            - ((self.alpha * I + self.mu) * U)
            - ((self.beta * I + self.nu) * T))
        I = (I_update * I) + ((1 - I_update) * I_summand)

        # Circuit output
        if self.association_field:
            # Ensure that CRF for association field is masked
            p_weights = self[
                self.weight_dict['P']['r']['weight']]
            if self.nonnegative_association:
                p_weights = tf.nn.relu(p_weights)  # Force excitatory conns
            P = self.conv_2d_op(
                data=I,
                weight_key=self.weight_dict['P']['r']['weight'],
                weights=p_weights
            )
        else:
            P = self.conv_2d_op(
                data=self.apply_tuning(I, 'P'),
                weight_key=self.weight_dict['P']['r']['weight']
            )
        Q = self.conv_2d_op(
            data=self.apply_tuning(I, 'Q'),
            weight_key=self.weight_dict['Q']['r']['weight']
        )
        O_update_input = self.conv_2d_op(
            data=self.X,
            weight_key=self.weight_dict['O']['f']['weight']
        )
        O_update_recurrent = self.conv_2d_op(
            data=I,
            weight_key=self.weight_dict['O']['r']['weight']
        )
        O_update = self.gate_nl(
            O_update_input + O_update_recurrent + self[
                self.weight_dict['O']['r']['bias']])

        # Calculate and apply dropout if requested
        if self.train and self.dropout is not None:
            O_update = self.zoneout(self.dropout) * self.gate_nl(
                O_update_input + O_update_recurrent)
        elif not self.train and self.dropout is not None:
            O_update = (1 / self.dropout) * self.gate_nl(
                O_update_input + O_update_recurrent)
        O_summand = self.recurrent_nl(
            self.zeta * I
            + self.gamma * P
            + self.delta * Q)
        O = (O_update * O) + ((1 - O_update) * O_summand)
        i0 += 1  # Iterate loop
        return i0, O, I

    def condition(self, i0, O, I):
        """While loop halting condition."""
        return i0 < self.timesteps

    def gather_tensors(self, wak='weight'):
        weights = {}
        for k, v in self.weight_dict.iteritems():
            for wk, wv in v.iteritems():
                if wak in wv.keys() and hasattr(self, wv[wak]):
                    weights['%s_%s' % (k, wk)] = self[wv[wak]]

        return weights

    def build(self, reduce_memory=False):
        """Run the backprop version of the CCircuit."""
        self.prepare_tensors()
        i0 = tf.constant(0)
        if self.hidden_init == 'identity':
            I = tf.identity(self.X)
            O = tf.identity(self.X)
        elif self.hidden_init == 'random':
            I = initialization.xavier_initializer(
                shape=[self.n, self.h, self.w, self.k],
                uniform=self.normal_initializer,
                mask=None)
            O = initialization.xavier_initializer(
                shape=[self.n, self.h, self.w, self.k],
                uniform=self.normal_initializer,
                mask=None)
        elif self.hidden_init == 'zeros':
            I = tf.zeros_like(self.X)
            O = tf.zeros_like(self.X)
        else:
            raise RuntimeError

        if reduce_memory:
            print 'Warning: Using FF version of the model.'
            for t in range(self.timesteps):
                i0, O, I = self.full(i0, O, I)
        else:
            # While loop
            elems = [
                i0,
                O,
                I
            ]

            returned = tf.while_loop(
                self.condition,
                self.full,
                loop_vars=elems,
                back_prop=True,
                swap_memory=False)

            # Prepare output
            i0, O, I = returned  # i0, O, I

        if self.return_weights:
            weights = self.gather_tensors(wak='weight')
            tuning = self.gather_tensors(wak='tuning')
            new_tuning = {}
            for k, v in tuning.iteritems():
                key_name = v.name.split('/')[-1].split(':')[0]
                new_tuning[key_name] = v
            weights = dict(weights, **new_tuning)
            activities = self.gather_tensors(wak='activity')
            # Attach weights if using association field
            if self.association_field:
                weights['p_t'] = self.p_r  # Make available for regularization
            return O, weights, activities
        else:
            return O
