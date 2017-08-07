import numpy as np
import tensorflow as tf


class ContextualCircuit(object):

    def __init__(
            self,
            X,
            model_version='full',
            timesteps=1,
            lesions=None,
            SRF=1,
            SSN=9,
            SSF=29):
        self.X = X
        self.input_shape = [int(x) for x in self.X.shape()]
        self.model_version = model_version
        self.timesteps = timesteps
        self.lesions = lesions
        self.SRF = SRF
        self.SSN = SSN
        self.SSF = SSF
        if self.SSN is None:
            self.SSN = self.SRF * 3
        if self.SSF is None:
            self.SSF = self.SRF * 5

    def prepare_tensors(self):
        """ Allocate buffer space on the GPU, etc. """
        n, h, w, k = self.input_shape
        self.q = tf.get_variable(
            name='q',
            shape=[1, 1, k, k],
            dtype=self.tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())

        # untuned suppression: reduction across feature axis
        ####################################################
        self.u = tf.get_variable(
            name='u',
            shape=[1, 1, k, 1],
            dtype=self.tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())

        # Gaussian weights
        self.p = tf.get_variable(
            name='p',
            shape=[k, k, SSN_, SSN_],
            dtype=self.tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())
        # Uniform weights
        for pdx in range(k):
            p_array[pdx, pdx, :SSN, :SSN] = 1.0
        p_array[
            :,
            :,
            SSN // 2 - ifloor(SRF / 2.0):SSN // 2 + iceil(SRF / 2.0),
            SSN // 2 - ifloor(SRF / 2.0):SSN // 2 + iceil(SRF / 2.0)] = 0.0
        self.p *= p_array  # Zero out classical receptive field

        # tuned suppression: pooling in h, w dimensions
        ###############################################
        SSF_ = 2 * ifloor(self.SSF/2.0) + 1
        self.t = tf.get_variable(
            name='t',
            shape=[k, k, SSF_, SSF_],
            dtype=self.tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())

        # Uniform weights
        for tdx in range(k):
            t_array[tdx, tdx, :SSF, :SSF] = 1.0
        t_array[
            :,
            :,  # exclude near surround!
            SSF // 2 - ifloor(SSN / 2.0):SSF // 2 + iceil(SSN / 2.0),
            SSF // 2 - ifloor(SSN / 2.0):SSF // 2 + iceil(SSN / 2.0)] = 0.0
        self.t *= t_array  # Zero out classical receptive field

        # Scalar weights
        self.xi = tf.get_variable(shape=[], initializer=1.)
        self.zeta = tf.get_variable(shape=[], initializer=1.)
        self.eps = tf.get_variable(shape=[], initializer=1.)
        self.sig = tf.get_variable(shape=[], initializer=1.)
        self.tau = tf.get_variable(shape=[], initializer=1.)
        self.alpha = tf.get_variable(shape=[], initializer=1.)
        self.beta = tf.get_variable(shape=[], initializer=1.)
        self.mu = tf.get_variable(shape=[], initializer=1.)
        self.nu = tf.get_variable(shape=[], initializer=1.)

    def full(self, O, I, U, T, P, Q):
        """pre_inhib = (gamma(FF_DRIVE) - conv(I*U) - conv(I*P)) * eps +
            ((gamma(FF_DRIVE) - conv(I*U) - conv(I*P)) * eta)
        post_inhib = relu(pre_inhib + conv(O*P) + conv(O*Q))
        output = sig * O + tau * post_inhib"""
        pre = (self.xi * self.X) - ((
            self.alpha * I + self.mu) * P) - (
            (self.beta * I + self.mu) * T)
        pre = self.eps * I + self.eta * pre
        post = tf.nn.relu((self.zeta * I) + self.zeta(P + Q))
        return self.sig * O + self.tau * post

    def reduced_1(self, O, I, U, T, P, Q):
        """pre_inhib = (gamma(FF_DRIVE) - conv(I*U) - conv(I*P)) +
            ((gamma(FF_DRIVE) - conv(I*U) - conv(I*P)))
        output = relu(pre_inhib + conv(O*P) + conv(O*Q)) + (tau * O)"""
        mod_X = (self.gamma * self.X)
        pre = mod_X - U - T
        post = tf.nn.relu(pre + P + Q)
        return self.sig * O + self.tau * post

    def body(
            self,
            i0,
            O,
            I):
        """Executes CC."""

        if 'U' in self.lesions:
            U = tf.constant(0.)
        else:
            U = tf.nn.conv2d(
                O, self._gpu_u, self.parameters.strides, padding='SAME')

        if 'T' in self.lesions:
            T = tf.constant(0.)
        else:
            T = tf.nn.conv2d(
                O, self._gpu_t, self.parameters.strides, padding='SAME')

        if 'P' in self.lesions:
            P = tf.constant(0.)
        else:
            P = tf.nn.conv2d(
                I, self._gpu_p, self.parameters.strides, padding='SAME')

        if 'Q' in self.lesions:
            Q = tf.constant(0.)
        else:
            Q = tf.nn.conv2d(
                I, self._gpu_q, self.parameters.strides, padding='SAME')

        O, I = self[self.model_version](
            O=O,
            I=I,
            U=U,
            T=T,
            P=P,
            Q=Q)

        i0 += 1
        return i0, O, I

    def condition(
            self, i0, O, I, alpha, beta, mu, nu, gamma, delta):
        """While loop halting condition."""
        return i0 < self.timesteps

    def run(self, in_array):
        """Run the backprop version of the CCircuit."""
        # Using run_reference implementation
        i0 = tf.constant(0)
        O = tf.identity(self.X)
        I = tf.identity(self.X)

        # While loop
        elems = [
            i0,
            O,
            I
        ]

        if self.overlap_CRF_eCRF:
            body_fun = self.body_overlap_CRF_eCRF
        else:
            body_fun = self.body

        returned = tf.while_loop(
            self.condition,
            body_fun,
            loop_vars=elems,
            back_prop=True,
            swap_memory=False)

        # Prepare output
        return returned  # iteration, I, O
