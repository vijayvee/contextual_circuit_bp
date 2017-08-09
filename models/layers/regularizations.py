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
        return tf.reduce_mean(tf.abs(x - x_mean))

    def l2(self, x, x_mean=0, **kwargs):
        return tf.nn.l2_loss(x - x_mean)
