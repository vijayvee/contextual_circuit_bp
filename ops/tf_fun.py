import tensorflow as tf


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def fixed_len_feature(len=[], dtype='int64'):
    if dtype == 'int64':
        return tf.FixedLenFeature([], tf.int64)
    elif dtype == 'string':
        return tf.FixedLenFeature([], tf.string)
