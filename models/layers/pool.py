"""Pooling wrapper. TODO: add interpreter."""
import tensorflow as tf


class pool(object):
    """Wrapper class for pooling operations."""

    def __getitem__(self, name):
        """Get attribute from class."""
        return getattr(self, name)

    def __contains__(self, name):
        """Check if class contains attribute."""
        return hasattr(self, name)

    def __init__(self, kwargs=None):
        """Global variables for pooling functions."""
        self.k = [1, 2, 2, 1]  # kernel size
        self.s = [1, 2, 2, 1]  # stride
        self.p = 'SAME'  # padding
        self.update_params(kwargs)

    def update_params(self, kwargs):
        """Update the class attributes with kwargs."""
        if kwargs is not None:
            for k, v in kwargs.iteritems():
                setattr(self, k, v)

    def max_pool(self, context, act, name, it_dict, kwargs=None):
        """Max pooling operation. TODO: Move to its own pool layer."""
        if kwargs is not None:
            self.update_params(kwargs)
        context, act = max_pool(
            self=context,
            bottom=act,
            name=name,
            k=self.k,
            s=self.s,
            p=self.p)
        return context, act

    def avg_pool(self, context, act, name, it_dict):
        """Max pooling operation. TODO: Move to its own pool layer."""
        context, act = avg_pool(
            self=context,
            bottom=act,
            name=name,
            k=self.k,
            s=self.s,
            p=self.p)
        return context, act

    def avg_pool_3d(
            self,
            context,
            bottom,
            name,
            k=[1, 2, 2, 1],
            s=[1, 2, 2, 1],
            p='SAME'):
        """Spatiotemporal average pooling."""
        return context, avg_pool_3d(
            self=context,
            bottom=bottom,
            ksize=k,
            strides=s,
            padding=p,
            name=name)

    def max_pool_3d(
            self,
            context,
            bottom,
            name,
            k=[1, 2, 2, 2, 1],  # D/H/W
            s=[1, 2, 2, 2, 1],  # D/H/W
            p='SAME'):
        """Spatiotemporal max pooling."""
        return context, max_pool_3d(
            self=context,
            bottom=bottom,
            ksize=k,
            strides=s,
            padding=p,
            name=name)


def avg_pool(
        self,
        bottom,
        name,
        k=[1, 2, 2, 1],
        s=[1, 2, 2, 1],
        p='SAME'):
    """Local average pooling."""
    return self, tf.nn.avg_pool(
        bottom,
        ksize=k,
        strides=s,
        padding=p,
        name=name)


def max_pool(
        self,
        bottom,
        name,
        k=[1, 2, 2, 1],
        s=[1, 2, 2, 1],
        p='SAME'):
    """Local max pooling."""
    return self, tf.nn.max_pool(
        bottom,
        ksize=k,
        strides=s,
        padding=p,
        name=name)


def avg_pool_3d(
        self,
        bottom,
        name,
        k=[1, 2, 2, 1],
        s=[1, 2, 2, 1],
        p='SAME'):
    """Spatiotemporal average pooling."""
    return self, tf.nn.avg_pool3d(
        bottom,
        ksize=k,
        strides=s,
        padding=p,
        name=name)


def max_pool_3d(
        self,
        bottom,
        name,
        k=[1, 2, 2, 2, 1],  # D/H/W
        s=[1, 2, 2, 2, 1],  # D/H/W
        p='SAME'):
    """Spatiotemporal max pooling."""
    return self, tf.nn.max_pool3d(
        bottom,
        ksize=k,
        strides=s,
        padding=p,
        name=name)
