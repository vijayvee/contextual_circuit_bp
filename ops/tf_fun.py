import tensorflow as tf


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def float_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def fixed_len_feature(length=[], dtype='int64'):
    if dtype == 'int64':
        return tf.FixedLenFeature(length, tf.int64)
    elif dtype == 'string':
        return tf.FixedLenFeature(length, tf.string)
    elif dtype == 'float':
        return tf.FixedLenFeature(length, tf.float32)
    else:
        raise RuntimeError('Cannot understand the fixed_len_feature dtype.')
