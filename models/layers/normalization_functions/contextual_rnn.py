import numpy as np
import tensorflow as tf
from utils import py_utils
from ops import initialization


class ContextualCircuit():
    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)

    def __init__(
            self,
            X,
            model_version='full',
            timesteps=1,
            lesions=None,
            SRF=1,
            SSN=9,
            SSF=29,
            strides=[1, 1, 1, 1],
            padding='SAME',
            dtype=tf.float32):

        self.X = X
        self.n, self.h, self.w, self.k = [int(x) for x in X.get_shape()]
        self.model_version = model_version
        self.timesteps = timesteps
        self.lesions = lesions
        self.strides = strides
        self.padding = padding
        self.dtype = dtype
        self.SRF, self.SSN, self.SSF = SRF, SSN, SSF

        self.SSN_ext = 2 * py_utils.ifloor(SSN / 2.0) + 1
        self.SSF_ext = 2 * py_utils.ifloor(SSF / 2.0) + 1
        self.q_shape = [self.SRF, self.SRF, self.k, self.k]
        self.u_shape = [self.SRF, self.SRF, self.k, 1]
        self.p_shape = [self.SSN_ext, self.SSN_ext, self.k, self.k]
        self.t_shape = [self.SSF_ext, self.SSF_ext, self.k, self.k]
        self.i_shape = [self.SRF, self.SRF, self.k]
        self.o_shape = [self.SRF, self.SRF, self.k]
        self.u_nl = tf.identity
        self.t_nl = tf.identity
        self.q_nl = tf.identity
        self.p_nl = tf.identity
        self.i_nl = tf.nn.relu  # input non linearity
        self.o_nl = tf.nn.relu  # output non linearity

        self.normal_initializer = False
        if self.SSN is None:
            self.SSN = self.SRF * 3
        if self.SSF is None:
            self.SSF = self.SRF * 5

    def prepare_tensors(self):
        """ Prepare recurrent/forward weight matrices."""
        self.weight_dict = {  # Weights lower/activity upper
            'U': {
                'r': {
                    'weight': 'u_r',
                    'activity': 'U_r'
                    },
                'f': {
                    'weight': 'u_f',
                    'bias': 'ub_f',
                    'activity': 'U_f'
                    }
                },
            'T': {
                'r': {
                    'weight': 't_r',
                    'activity': 'T_r'
                    },
                'f': {
                    'weight': 't_f',
                    'bias': 'tb_f',
                    'activity': 'T_f'
                    }
                },
            'P': {
                'r': {
                    'weight': 'p_r',
                    'activity': 'P_r'
                    },
                'f': {
                    'weight': 'p_f',
                    'bias': 'pb_f',
                    'activity': 'P_f'
                    }
                },
            'Q': {
                'r': {
                    'weight': 'q_r',
                    'activity': 'Q_r'
                    },
                'f': {
                    'weight': 'q_f',
                    'bias': 'qb_f',
                    'activity': 'Q_f'
                    }
                },
            'I': {
                'r': {  # Recurrent state
                    'weight': 'i_r',
                    'activity': 'I_r'
                }
            },
            'O': {
                'r': {  # Recurrent state
                    'weight': 'o_r',
                    'activity': 'O_r'
                }
            }
        }

        # tuned summation: pooling in h, w dimensions
        #############################################
        setattr(
            self,
            self.weight_dict['Q']['r']['weight'],
            tf.get_variable(
                name=self.weight_dict['Q']['r']['weight'],
                dtype=self.dtype,
                initializer=initialization.xavier_initializer(
                    shape=self.q_shape,
                    uniform=self.normal_initializer,
                    mask=None)))

        # untuned suppression: reduction across feature axis
        ####################################################
        setattr(
            self,
            self.weight_dict['U']['r']['weight'],
            tf.get_variable(
                name=self.weight_dict['U']['r']['weight'],
                dtype=self.dtype,
                initializer=initialization.xavier_initializer(
                    shape=self.u_shape,
                    uniform=self.normal_initializer,
                    mask=None)))

        # tuned summation: pooling in h, w dimensions
        #############################################
        p_array = np.zeros(self.p_shape)
        for pdx in range(self.k):
            p_array[:self.SSN, :self.SSN, pdx, pdx] = 1.0
        p_array[
            self.SSN // 2 - py_utils.ifloor(
                self.SRF / 2.0):self.SSN // 2 + py_utils.iceil(
                self.SRF / 2.0),
            self.SSN // 2 - py_utils.ifloor(
                self.SRF / 2.0):self.SSN // 2 + py_utils.iceil(
                self.SRF / 2.0),
            :,  # exclude CRF!
            :] = 0.0

        setattr(
            self,
            self.weight_dict['P']['r']['weight'],
            tf.get_variable(
                name=self.weight_dict['P']['r']['weight'],
                dtype=self.dtype,
                initializer=initialization.xavier_initializer(
                    shape=self.p_shape,
                    uniform=self.normal_initializer,
                    mask=p_array)))

        # tuned suppression: pooling in h, w dimensions
        ###############################################
        t_array = np.zeros(self.t_shape)
        for tdx in range(self.k):
            t_array[:self.SSF, :self.SSF, tdx, tdx] = 1.0
        t_array[
            self.SSF // 2 - py_utils.ifloor(
                self.SSN / 2.0):self.SSF // 2 + py_utils.iceil(
                self.SSN / 2.0),
            self.SSF // 2 - py_utils.ifloor(
                self.SSN / 2.0):self.SSF // 2 + py_utils.iceil(
                self.SSN / 2.0),
            :,  # exclude near surround!
            :] = 0.0
        setattr(
            self,
            self.weight_dict['T']['r']['weight'],
            tf.get_variable(
                name=self.weight_dict['T']['r']['weight'],
                dtype=self.dtype,
                initializer=initialization.xavier_initializer(
                    shape=self.t_shape,
                    uniform=self.normal_initializer,
                    mask=t_array)))

        if self.model_version != 'no_input_facing':
            # Also create input-facing weight matrices
            # Q
            setattr(
                self,
                self.weight_dict['Q']['f']['weight'],
                tf.get_variable(
                    name=self.weight_dict['Q']['f']['weight'],
                    dtype=self.dtype,
                    initializer=initialization.xavier_initializer(
                        shape=self.q_shape,
                        uniform=self.normal_initializer)))
            setattr(
                self,
                self.weight_dict['Q']['f']['bias'],
                tf.get_variable(
                    name=self.weight_dict['Q']['f']['bias'],
                    dtype=self.dtype,
                    initializer=initialization.xavier_initializer(
                        shape=self.q_shape[-1],
                        uniform=self.normal_initializer)))

            # U
            setattr(
                self,
                self.weight_dict['U']['f']['weight'],
                tf.get_variable(
                    name=self.weight_dict['U']['f']['weight'],
                    dtype=self.dtype,
                    initializer=initialization.xavier_initializer(
                        shape=self.u_shape,
                        uniform=self.normal_initializer)))
            setattr(
                self,
                self.weight_dict['U']['f']['bias'],
                tf.get_variable(
                    name=self.weight_dict['U']['f']['bias'],
                    dtype=self.dtype,
                    initializer=initialization.xavier_initializer(
                        self.u_shape[-1],
                        uniform=self.normal_initializer)))

            # P
            setattr(
                self,
                self.weight_dict['P']['f']['weight'],
                tf.get_variable(
                    name=self.weight_dict['P']['f']['weight'],
                    dtype=self.dtype,
                    initializer=initialization.xavier_initializer(
                        self.p_shape,
                        uniform=self.normal_initializer,
                        mask=p_array)))
            setattr(
                self,
                self.weight_dict['P']['f']['bias'],
                tf.get_variable(
                    name=self.weight_dict['P']['f']['bias'],
                    dtype=self.dtype,
                    initializer=initialization.xavier_initializer(
                        self.p_shape[-1],
                        uniform=self.normal_initializer,
                        mask=None)))

            # T
            setattr(
                self,
                self.weight_dict['T']['f']['weight'],
                tf.get_variable(
                    name=self.weight_dict['T']['f']['weight'],
                    dtype=self.dtype,
                    initializer=initialization.xavier_initializer(
                        shape=self.t_shape,
                        uniform=self.normal_initializer,
                        mask=t_array)))
            setattr(
                self,
                self.weight_dict['T']['f']['bias'],
                tf.get_variable(
                    name=self.weight_dict['T']['f']['bias'],
                    dtype=self.dtype,
                    initializer=initialization.xavier_initializer(
                        shape=self.t_shape[-1],
                        uniform=self.normal_initializer,
                        mask=None)))
        if self.model_version == 'full_with_cell_states':

            # Input
            setattr(
                self,
                self.weight_dict['I']['r']['weight'],
                tf.get_variable(
                    name=self.weight_dict['I']['r']['weight'],
                    dtype=self.dtype,
                    initializer=initialization.xavier_initializer(
                        shape=self.i_shape,
                        uniform=self.normal_initializer,
                        mask=t_array)))
            setattr(
                self,
                self.weight_dict['I']['r']['bias'],
                tf.get_variable(
                    name=self.weight_dict['I']['r']['bias'],
                    dtype=self.dtype,
                    initializer=initialization.xavier_initializer(
                        shape=self.k,
                        uniform=self.normal_initializer,
                        mask=None)))

            # Output
            setattr(
                self,
                self.weight_dict['O']['r']['weight'],
                tf.get_variable(
                    name=self.weight_dict['O']['r']['weight'],
                    dtype=self.dtype,
                    initializer=initialization.xavier_initializer(
                        shape=self.o_shape,
                        uniform=self.normal_initializer,
                        mask=t_array)))
            setattr(
                self,
                self.weight_dict['O']['r']['bias'],
                tf.get_variable(
                    name=self.weight_dict['O']['r']['bias'],
                    dtype=self.dtype,
                    initializer=initialization.xavier_initializer(
                        shape=self.k,
                        uniform=self.normal_initializer,
                        mask=None)))

        # Scalar weights
        self.alpha = tf.get_variable(name='alpha', initializer=1.)
        self.tau = tf.get_variable(name='tau', initializer=1.)
        self.eta = tf.get_variable(name='eta', initializer=1.)
        self.omega = tf.get_variable(name='omega', initializer=1.)
        self.eps = tf.get_variable(name='eps', initializer=1.)
        self.gamma = tf.get_variable(name='gamma', initializer=1.)

    def conv_2d_op(self, data, weight_key, out_key=None):
        """2D convolutions, lesion, return or assign activity as attribute."""
        if weight_key in self.lesions:
            weights = tf.constant(0.)
        else:
            weights = self[weight_key]
        activities = tf.nn.conv2d(
                data,
                weights,
                self.strides,
                padding=self.padding)
        if out_key is None:
            return activities
        else:
            setattr(
                self,
                out_key,
                activities)

    def full(self, i0, O, I):
        """Fully parameterized contextual RNN model."""
        U = self.conv_2d_op(
            data=O,
            weight_key=self.weight_dict['U']['r']['weight']
        )
        T = self.conv_2d_op(
            data=O,
            weight_key=self.weight_dict['T']['r']['weight']
        )
        P = self.conv_2d_op(
            data=I,
            weight_key=self.weight_dict['P']['r']['weight']
        )
        Q = self.conv_2d_op(
            data=I,
            weight_key=self.weight_dict['Q']['r']['weight']
        )

        # Input
        U = self.u_nl(
            tf.nn.bias_add(
                U +
                self[self.weight_dict['U']['f']['activity']],
                self[self.weight_dict['U']['f']['bias']])
            )
        T = self.t_nl(
            tf.nn.bias_add(
                T +
                self[self.weight_dict['T']['f']['activity']],
                self[self.weight_dict['T']['f']['bias']])
            )
        I_summand = self.eta * self.i_nl(self.alpha * self.X - U - T)
        I = (self.eps * I) + I_summand

        # Output
        Q = self.q_nl(
            Q +
            self[self.weight_dict['Q']['f']['activity']])
        P = self.p_nl(
            P +
            self[self.weight_dict['P']['f']['activity']])
        O_summand = self.tau * self.o_nl(Q + P + (self.gamma * I))
        O = (self.omega * O) + O_summand
        return i0, O, I

    def no_input_facing(self, i0, O, I):
        """Remove the direct FF drive to the CRF and eCRFs."""
        U = self.conv_2d_op(
            data=O,
            weight_key=self.weight_dict['U']['r']['weight']
        )
        T = self.conv_2d_op(
            data=O,
            weight_key=self.weight_dict['T']['r']['weight']
        )
        P = self.conv_2d_op(
            data=I,
            weight_key=self.weight_dict['P']['r']['weight']
        )
        Q = self.conv_2d_op(
            data=I,
            weight_key=self.weight_dict['Q']['r']['weight']
        )

        # Input
        U = self.u_nl(tf.nn.bias_add(
            U,
            self[self.weight_dict['U']['f']['bias']])
        )
        T = self.t_nl(tf.nn.bias_add(
            T,
            self[self.weight_dict['T']['f']['bias']])
        )
        I_summand = self.eta(self.i_nl(self.alpha * self.X - U - T))
        I = (self.eps * I) + I_summand

        # Output
        Q = self.q_nl(Q)
        P = self.p_nl(P)
        O_summand = self.tau(self.o_nl(Q + P + (self.gamma * I)))
        O = (self.omega * O) + O_summand
        return i0, O, I

    def no_input_scaling(self, i0, O, I):
        """Remove direct FF input to the I."""
        U = self.conv_2d_op(
            data=O,
            weight_key=self.weight_dict['U']['r']['weight']
        )
        T = self.conv_2d_op(
            data=O,
            weight_key=self.weight_dict['T']['r']['weight']
        )
        P = self.conv_2d_op(
            data=I,
            weight_key=self.weight_dict['P']['r']['weight']
        )
        Q = self.conv_2d_op(
            data=I,
            weight_key=self.weight_dict['Q']['r']['weight']
        )

        # Input
        U = self.u_nl(
            tf.nn.bias_add(
                U +
                self[self.weight_dict['U']['f']['activity']],
                self[self.weight_dict['U']['f']['bias']])
            )
        T = self.t_nl(
            tf.nn.bias_add(
                T +
                self[self.weight_dict['T']['f']['activity']],
                self[self.weight_dict['T']['f']['bias']])
            )
        I_summand = self.eta(self.i_nl(U - T))
        I = (self.eps * I) + I_summand

        # Output
        Q = self.q_nl(
            Q +
            self[self.weight_dict['Q']['f']['activity']])
        P = self.p_nl(
            P +
            self[self.weight_dict['P']['f']['activity']])
        O_summand = self.tau(self.o_nl(Q + P + (self.gamma * I)))
        O = (self.omega * O) + O_summand
        return i0, O, I

    def full_with_cell_states(self, i0, O, I):
        """Replace I/O scaling with weight matrices (cell states)."""
        U = self.conv_2d_op(
            data=O,
            weight_key=self.weight_dict['U']['r']['weight']
        )
        T = self.conv_2d_op(
            data=O,
            weight_key=self.weight_dict['T']['r']['weight']
        )
        P = self.conv_2d_op(
            data=I,
            weight_key=self.weight_dict['P']['r']['weight']
        )
        Q = self.conv_2d_op(
            data=I,
            weight_key=self.weight_dict['Q']['r']['weight']
        )
        I = tf.nn.bias_add(
            self.conv_2d_op(
                data=I,
                weight_key=self.weight_dict['I']['r']['weight']
            ), self[self.weight_dict['I']['r']['bias']])
        O = tf.nn.bias_add(
            self.conv_2d_op(
                data=O,
                weight_key=self.weight_dict['O']['r']['weight']
            ), self[self.weight_dict['O']['r']['bias']])

        # Input
        U = self.u_nl(
            tf.nn.bias_add(
                U +
                self[self.weight_dict['U']['f']['activity']],
                self[self.weight_dict['U']['f']['bias']])
            )
        T = self.t_nl(
            tf.nn.bias_add(
                T +
                self[self.weight_dict['T']['f']['activity']],
                self[self.weight_dict['T']['f']['bias']])
            )
        I_summand = self.eta(self.i_nl(self.alpha * self.X - U - T))
        I += I_summand

        # Output
        Q = self.q_nl(
            Q +
            self[self.weight_dict['Q']['f']['activity']])
        P = self.p_nl(
            P +
            self[self.weight_dict['P']['f']['activity']])
        O_summand = self.tau(self.o_nl(Q + P + (self.gamma * I)))
        O += O_summand
        return i0, O, I

    def condition(self, i0, O, I):
        """While loop halting condition."""
        return i0 < self.timesteps

    def build(self, reduce_memory=True):
        """Run the backprop version of the CCircuit."""
        self.prepare_tensors()
        i0 = tf.constant(0)
        O = tf.identity(self.X)
        I = tf.identity(self.X)

        # While loop
        elems = [
            i0,
            O,
            I
        ]

        if self.model_version == 'full':
            self.conv_2d_op(
                data=self.X,
                weight_key=self.weight_dict['U']['f']['weight'],
                out_key=self.weight_dict['U']['f']['activity'])
            self.conv_2d_op(
                data=self.X,
                weight_key=self.weight_dict['T']['f']['weight'],
                out_key=self.weight_dict['T']['f']['activity'])
            self.conv_2d_op(
                data=self.X,
                weight_key=self.weight_dict['P']['f']['weight'],
                out_key=self.weight_dict['P']['f']['activity'])
            self.conv_2d_op(
                data=self.X,
                weight_key=self.weight_dict['Q']['f']['weight'],
                out_key=self.weight_dict['Q']['f']['activity'])

        if reduce_memory:
            print 'Warning: Using FF version of the model.'
            for t in range(self.timesteps):
                i0, O, I = self[self.model_version](i0, O, I)
                i0 = tf.constant(0)
        else:
            returned = tf.while_loop(
                self.condition,
                self[self.model_version],
                loop_vars=elems,
                back_prop=True,
                swap_memory=False)

            # Prepare output
            _, _, I = returned  # i0, O, I
        return I
