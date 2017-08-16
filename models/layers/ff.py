import tensorflow as tf
from models.layers.activations import activations
from models.layers.normalizations import normalizations


def resnet_layer(
        self,
        bottom,
        layer_weights,
        name,
        activation=None,
        normalization=None,
        combination=tf.add):  # tf.multiply
    ln = '%s_branch' % name
    rlayer = tf.identity(bottom)
    if normalization is not None:
        nm = normalizations()[normalization]
    if activation is not None:
        ac = activations()[activation]
    for idx, lw in enumerate(layer_weights):
        ln = '%s_%s' % (name, idx)
        rlayer = conv_layer(
            self=self,
            bottom=rlayer,
            in_channels=int(rlayer.get_shape()[-1]),
            out_channels=lw,
            name=ln)
        rlayer = nm(ac(rlayer))
    return combination(rlayer, bottom)


def conv_layer(
        self,
        bottom,
        out_channels,
        name,
        in_channels=None,
        filter_size=3,
        stride=[1, 1, 1, 1],
        padding='SAME'):
    with tf.variable_scope(name):
        if in_channels is None:
            in_channels = int(bottom.get_shape()[-1])
        filt, conv_biases = get_conv_var(
            self=self,
            filter_size=filter_size,
            in_channels=in_channels,
            out_channels=out_channels,
            name=name)
        conv = tf.nn.conv2d(bottom, filt, stride, padding=padding)
        bias = tf.nn.bias_add(conv, conv_biases)
        return bias


def fc_layer(self, bottom, out_channels, name, in_channels=None):
    with tf.variable_scope(name):
        if in_channels is None:
            in_channels = int(bottom.get_shape()[-1])
        weights, biases = get_fc_var(
            self=self,
            in_size=in_channels,
            out_size=out_channels,
            name=name)

        x = tf.reshape(bottom, [-1, in_channels])
        fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

        return fc


def get_conv_var(
        self,
        filter_size,
        in_channels,
        out_channels,
        name,
        init_type='xavier'):
    if init_type == 'xavier':
        weight_init = [
            [filter_size, filter_size, in_channels, out_channels],
            tf.contrib.layers.xavier_initializer_conv2d(uniform=False)]
    else:
        weight_init = tf.truncated_normal(
            [filter_size, filter_size, in_channels, out_channels],
            0.0, 0.001)
    bias_init = tf.truncated_normal([out_channels], .0, .001)
    filters = get_var(
        self=self,
        initial_value=weight_init,
        name=name,
        idx=0,
        var_name=name + "_filters")
    biases = get_var(
        self=self,
        initial_value=bias_init,
        name=name,
        idx=1,
        var_name=name + "_biases")
    return filters, biases


def get_fc_var(
        self,
        in_size,
        out_size,
        name,
        init_type='xavier'):
    if init_type == 'xavier':
        weight_init = [
            [in_size, out_size],
            tf.contrib.layers.xavier_initializer(uniform=False)]
    else:
        weight_init = tf.truncated_normal(
            [in_size, out_size], 0.0, 0.001)
    bias_init = tf.truncated_normal([out_size], .0, .001)
    weights = get_var(
        self=self,
        initial_value=weight_init,
        name=name,
        idx=0,
        var_name=name + "_weights")
    biases = get_var(
        self=self,
        initial_value=bias_init,
        name=name,
        idx=1,
        var_name=name + "_biases")
    return weights, biases


def get_var(
        self,
        initial_value,
        name,
        idx,
        var_name,
        in_size=None,
        out_size=None):
    if self.data_dict is not None and name in self.data_dict:
        value = self.data_dict[name][idx]
    else:
        value = initial_value

    if self.training:
        # get_variable, change the boolian to numpy
        if type(value) is list:
            var = tf.get_variable(
                name=var_name,
                shape=value[0],
                initializer=value[1],
                trainable=True)
        else:
            var = tf.get_variable(
                name=var_name,
                initializer=value,
                trainable=True)
    else:
        var = tf.constant(
            value,
            dtype=tf.float32,
            name=var_name)
    self.var_dict[(name, idx)] = var
    return var

