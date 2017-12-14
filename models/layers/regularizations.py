import tensorflow as tf
# from models.layers import regularization_functions


class regularizations(object):
    """Wrapper class for regularization functions."""
    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)

    def __init__(self, kwargs=None):
        """Globals for activation functions."""
        self.training = None
        if kwargs is not None:
            self.update_params(**kwargs)

    def update_params(self, **kwargs):
        for k, v in kwargs.iteritems():
            setattr(self, k, v)

    def dropout(self, x, keep_prob, **kwargs):
        """Dropout."""
        return tf.nn.dropout(x, keep_prob=keep_prob)

    def l1(self, x, x_mean=0, **kwargs):
        """L1 weight decay."""
        return tf.reduce_mean(tf.abs(x - x_mean))

    def l2(self, x, x_mean=0, **kwargs):
        """L2 weight decay."""
        return tf.nn.l2_loss(x - x_mean)

    def frobenius(self, x, x_mean=None, **kwargs):
        """Norm on 2 - the trace of the correlation of x."""
        raise NotImplementedError
        x_shape = [int(dx) for dx in x.get_shape()]
        res_x = x.reshape([x_shape[0], -1])
        mean_t = tf.reduce_mean(x, axis=1, keep_dims=True)
        cov_t = tf.matmul((x-mean_t), tf.transpose(x - mean_t))/(
            int(res_x.get_shape()[-1]) - 1)
        cov2_t = tf.diag(1. / tf.sqrt(tf.diag_part(cov_t)))
        cor = tf.matmul(tf.matmul(cov2_t, cov_t), cov2_t)
        return tf.reduce_mean(tf.trace((2 - cor) ** 2))
