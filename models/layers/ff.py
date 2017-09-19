import tensorflow as tf
from models.layers.activations import activations
from models.layers.normalizations import normalizations


def dog_layer(
        self,
        bottom,
        layer_weights,
        name,
        model_dtype=tf.float32,
        init_bounds=):
    """Antilok et al 2016 difference of gaussians."""
   
    def DoG(self, x, y, sc, ss, rc, rs):
        """DoG operation."""
        pi = tf.constant(np.pi, dtype=self.model_dtype)
        pos = ((self.grid_xx - x)**2 + (self.grid_yy - y)**2)
        center = tf.exp(-pos/2/sc) / (2*(sc)*pi)
        surround = tf.exp(-pos/2/(sc + ss)) / (2*(sc + ss)*pi)
        weight_vec = tf.reshape((rc*(center)) - (rs*(surround)), [-1, 1])
        return tf.matmul(self.images, weight_vec)

    self['%s_num_lgn'] = layer_weights
    act_size = [int(x) for x in bottom.get_shape()]
    initializers = [
        'x_pos': np.linspace(0., act_size[1], layer_weights),
        'y_pos': np.linspace(0., act_size[2], layer_weights),
        'size_center': np.linspace(0.1, act_size[1], layer_weights),
        'size_surround': np.linspace(0., act_size[1] // 3, layer_weights),
        'center_weight': np.linspace(0., act_size[1] // 3, layer_weights),
        'surround_weight': np.linspace(0., act_size[1] // 3, layer_weights)
    ]
    lgn_x = tf.get_variable(
        name='%s_x_pos' % name,
        dtype=model_dtype,
        initializer=initializers['x_pos'],
        trainable=True)
    lgn_y = tf.get_variable(
        name='%s_y_pos' % name,
        dtype=model_dtype,
        initializer=initializers['y_pos'],
        trainable=True)
    lgn_sc = tf.get_variable(
        name='%s_size_center' % name,
        dtype=model_dtype,
        initializer=initializers['size_center'],
        trainable=True)
    lgn_ss = tf.get_variable(
        name='%s_size_surround' % name,
        dtype=model_dtype,
        initializer=initializers['size_surround'],
        trainable=True) 
    lgn_rc = tf.get_variable(
        name='%s_center_weight',
        dtype=model_dtype,
        initializer=initializers['center_weight'],
        trainable=True)
    lgn_rs = tf.get_variable(
        name='%s_surround_weight',
        dtype=model_dtype,
        initializer=initializers['surround_weight'],
        trainable=True)

    output = []    
    for i in range(layer_weights):
        output += [
            DoG(
                x=x_pos[i],
                y=y_pos[i],
                sc=lgn_sc[i],
                ss=lgn_ss[i],
                rc=lgn_rc[i],
                rs=lgn_rs[i])
            ]
    return tf.concat(axis=1, values=output)


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
