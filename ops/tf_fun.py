import tensorflow as tf


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def fixed_len_feature(length=[], dtype='int64'):
    if dtype == 'int64':
        return tf.FixedLenFeature(length, tf.int64)
    elif dtype == 'string':
        return tf.FixedLenFeature(length, tf.string)


def float32():
    return tf.float32


def int64():
    return tf.int64


def int32():
    return tf.int32
