"""Functions for handling feedforward and pooling operations."""
import numpy as np
import tensorflow as tf
from ops import initialization
from models.layers.activations import activations
from models.layers.normalizations import normalizations
from models.layers import pool


def pool_ff_interpreter(
        self,
        it_neuron_op,
        act,
        it_name,
        it_dict):
    """Wrapper for FF and pooling functions. TODO: turn into a class."""
    if it_neuron_op == 'pool':  # TODO create wrapper for FF ops.
        self, act = pool.max_pool(
            self=self,
            bottom=act,
            name=it_name)
    elif it_neuron_op == 'dog' or it_neuron_op == 'DoG':
        self, act = dog_layer(
            self=self,
            bottom=act,
            layer_weights=it_dict['weights'],
            name=it_name,
        )
    elif it_neuron_op == 'conv':
        self, act = conv_layer(
            self=self,
            bottom=act,
            in_channels=int(act.get_shape()[-1]),
            out_channels=it_dict['weights'][0],
            name=it_name,
            filter_size=it_dict['filter_size'][0]
        )
    elif it_neuron_op == 'conv3d':
        self, act = conv_3d_layer(
            self=self,
            bottom=act,
            in_channels=int(act.get_shape()[-1]),
            out_channels=it_dict['weights'][0],
            name=it_name,
            filter_size=it_dict['filter_size'][0]
        )
    elif it_neuron_op == 'fc':
        self, act = fc_layer(
            self=self,
            bottom=act,
            in_channels=int(act.get_shape()[-1]),
            out_channels=it_dict['weights'][0],
            name=it_name)
    elif it_neuron_op == 'sparse_pool':
        self, act = sparse_pool_layer(
            self=self,
            bottom=act,
            in_channels=int(act.get_shape()[-1]),
            out_channels=it_dict['weights'][0],
            aux=it_dict,
            name=it_name)
    elif it_neuron_op == 'res':
        self, act = resnet_layer(
            self=self,
            bottom=act,
            layer_weights=it_dict['weights'],
            name=it_name)
    elif it_neuron_op == 'gather':
        self, act = gather_value_layer(
            self=self,
            bottom=act,
            aux=it_dict,
            name=it_name)
    elif it_neuron_op == 'pass':
        pass
    else:
        raise RuntimeError(
            'Your specified operation %s is not implemented' % it_neuron_op)
    return self, act


def gather_value_layer(
        self,
        bottom,
        aux,
        name):
    """Gather a value from a location in an activity tensor."""
    assert aux is not None, 'Gather op needs an aux dict with x/y coordinates.'
    assert 'x' in aux.keys() and 'y' in aux.keys(), 'Gather op dict needs x/y key value pairs'
    import ipdb;ipdb.set_trace()
    x = aux['x']
    y = aux['y']
    out = tf.gather_nd(bottom, [x, y])
    return self, out


# TODO: move each of these ops into a script in the functions folder.
def dog_layer(
        self,
        bottom,
        layer_weights,
        name,
        init_weight=10.,
        model_dtype=tf.float32):
    """Antilok et al 2016 difference of gaussians."""
    tshape = [int(x) for x in bottom.get_shape()]  # Flatten input
    rows = tshape[0]
    cols = np.prod(tshape[1:])
    flat_bottom = tf.reshape(bottom, [rows, cols])
    hw = tshape[1:3][::-1]
    min_dim = np.min(hw)
    act_size = [int(x) for x in flat_bottom.get_shape()]  # Use flattened input
    len_act = len(act_size)
    assert len_act == 2, 'DoG layer needs 2D matrix not %sD tensor.' % len_act
    grid_xx, grid_yy = tf.meshgrid(
        tf.range(hw[0]),
        tf.range(hw[1]))
    grid_xx = tf.cast(grid_xx, tf.float32)
    grid_yy = tf.cast(grid_yy, tf.float32)
    pi = tf.constant(np.pi, dtype=model_dtype)

    def randomize_init(
            bounds,
            layer_weights,
            d1=4.0,
            d2=2.0,
            dtype=np.float32):
        """Initialize starting positions of DoG parameters as uniform rand."""
        init_dict = {}
        for k, v in bounds.iteritems():
            it_inits = []
            r = v[1] - v[0]
            for idx in range(layer_weights):
                it_inits += [v[0] + (r / d1) + np.random.rand() * (r / d2)]
            init_dict[k] = np.asarray(it_inits, dtype=dtype)
        return init_dict

    def DoG(bottom, x, y, sc, ss, rc, rs):
        """DoG operation."""
        pos = ((grid_xx - x)**2 + (grid_yy - y)**2)
        center = tf.exp(-pos / 2 / sc) / (2 * (sc) * pi)
        surround = tf.exp(-pos / 2 / (sc + ss)) / (2 * (sc + ss) * pi)
        weight_vec = tf.reshape((rc * (center)) - (rs * (surround)), [-1, 1])
        return tf.matmul(bottom, weight_vec), weight_vec

    if isinstance(layer_weights, list):
        layer_weights = layer_weights[0]

    # Construct model bounds
    bounds = {
        'x_pos': [
            0.,
            hw[0],
        ],
        'y_pos': [
            0.,
            hw[1],
        ],
        'size_center': [
            0.1,
            min_dim,
        ],
        'size_surround': [
            0.,
            min_dim,
        ],
        'center_weight': [
            0.,
            init_weight,
        ],
        'surround_weight': [
            0.,
            init_weight,
        ],
    }

    # Create tensorflow weights
    initializers = randomize_init(
        bounds=bounds,
        layer_weights=layer_weights)
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
        name='%s_center_weight' % name,
        dtype=model_dtype,
        initializer=initializers['center_weight'],
        trainable=True)
    lgn_rs = tf.get_variable(
        name='%s_surround_weight' % name,
        dtype=model_dtype,
        initializer=initializers['surround_weight'],
        trainable=True)

    output, dog_weights = [], []
    for i in range(layer_weights):
        activities, weight_vec = DoG(
            bottom=flat_bottom,
            x=lgn_x[i],
            y=lgn_y[i],
            sc=lgn_sc[i],
            ss=lgn_ss[i],
            rc=lgn_rc[i],
            rs=lgn_rs[i])
        output += [activities]
        dog_weights += [weight_vec]
    self.var_dict[('%s_weights' % name, 0)] = dog_weights
    return self, tf.concat(axis=1, values=output)


def resnet_layer(
        self,
        bottom,
        layer_weights,
        name,
        activation=None,
        normalization=None,
        combination=tf.add):  # tf.multiply
    """Residual layer."""
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
    return self, combination(rlayer, bottom)


def conv_layer(
        self,
        bottom,
        out_channels,
        name,
        in_channels=None,
        filter_size=3,
        stride=[1, 1, 1, 1],
        padding='SAME'):
    """2D convolutional layer."""
    with tf.variable_scope(name):
        if in_channels is None:
            in_channels = int(bottom.get_shape()[-1])
        self, filt, conv_biases = get_conv_var(
            self=self,
            filter_size=filter_size,
            in_channels=in_channels,
            out_channels=out_channels,
            name=name)
        conv = tf.nn.conv2d(bottom, filt, stride, padding=padding)
        bias = tf.nn.bias_add(conv, conv_biases)
        return self, bias


def conv_3d_layer(
        self,
        bottom,
        out_channels,
        name,
        in_channels=None,
        filter_size=3,
        stride=[1, 1, 1, 1],
        padding='SAME'):
    """NOT YET IMPLEMENTED: 3D convolutional layer."""
    with tf.variable_scope(name):
        if in_channels is None:
            in_channels = int(bottom.get_shape()[-1])
        self, filt, conv_biases = get_conv_var(
            self=self,
            filter_size=filter_size,
            in_channels=in_channels,
            out_channels=out_channels,
            name=name)
        conv = tf.nn.conv3d(bottom, filt, stride, padding=padding)
        bias = tf.nn.bias_add(conv, conv_biases)
        return self, bias


def fc_layer(self, bottom, out_channels, name, in_channels=None):
    """Fully connected layer."""
    with tf.variable_scope(name):
        if in_channels is None:
            in_channels = int(bottom.get_shape()[-1])
        self, weights, biases = get_fc_var(
            self=self,
            in_size=in_channels,
            out_size=out_channels,
            name=name)

        x = tf.reshape(bottom, [-1, in_channels])
        fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

        return self, fc


def sparse_pool_layer(
        self,
        bottom,
        out_channels,
        name,
        in_channels=None,
        aux=None):
    """Sparse pooling layer."""
    def create_gaussian_rf(xy, h, w):
        """Create a gaussian bump for initializing the spatial weights."""
        # TODO: implement this.
        pass

    with tf.variable_scope(name):
        bottom_shape = [int(x) for x in bottom.get_shape()]
        if in_channels is None:
            in_channels = bottom_shape[-1]

        # K channel weights
        channel_weights = tf.get_variable(
            name='%s_channel' % name,
            dtype=tf.float32,
            initializer=initialization.xavier_initializer(
                shape=[in_channels, out_channels],
                uniform=True,
                mask=None))

        # HxW spatial weights
        spatial_weights = tf.get_variable(
            name='%s_spatial' % name,
            dtype=tf.float32,
            initializer=initialization.xavier_initializer(
                shape=[1, bottom_shape[1], bottom_shape[2], 1],
                mask=None))

        # If supplied, initialize the spatial weights with RF info
        if aux is not None and 'xy' in aux.keys():
            gaussian_xy = aux['xy']
            if 'h' in aux.keys():
                gaussian_h = aux['h']
                gaussian_w = aux['w']
            else:
                gaussian_h, gaussian_w = None, None
            spatial_rf = create_gaussian_rf(
                xy=gaussian_xy,
                h=gaussian_h,
                w=gaussian_w)
            spatial_weights += spatial_rf
        spatial_sparse = tf.reduce_mean(
            bottom * spatial_weights, reduction_indices=[1, 2])
        output = tf.matmul(spatial_sparse, channel_weights)
        return self, output


def get_conv_var(
        self,
        filter_size,
        in_channels,
        out_channels,
        name,
        init_type='xavier'):
    """Prepare convolutional kernel weights."""
    if init_type == 'xavier':
        weight_init = [
            [filter_size, filter_size, in_channels, out_channels],
            tf.contrib.layers.xavier_initializer_conv2d(uniform=False)]
    else:
        weight_init = tf.truncated_normal(
            [filter_size, filter_size, in_channels, out_channels],
            0.0, 0.001)
    bias_init = tf.truncated_normal([out_channels], .0, .001)
    self, filters = get_var(
        self=self,
        initial_value=weight_init,
        name=name,
        idx=0,
        var_name=name + "_filters")
    self, biases = get_var(
        self=self,
        initial_value=bias_init,
        name=name,
        idx=1,
        var_name=name + "_biases")
    return self, filters, biases


def get_fc_var(
        self,
        in_size,
        out_size,
        name,
        init_type='xavier'):
    """Prepare fully connected weights."""
    if init_type == 'xavier':
        weight_init = [
            [in_size, out_size],
            tf.contrib.layers.xavier_initializer(uniform=False)]
    else:
        weight_init = tf.truncated_normal(
            [in_size, out_size], 0.0, 0.001)
    bias_init = tf.truncated_normal([out_size], .0, .001)
    self, weights = get_var(
        self=self,
        initial_value=weight_init,
        name=name,
        idx=0,
        var_name=name + "_weights")
    self, biases = get_var(
        self=self,
        initial_value=bias_init,
        name=name,
        idx=1,
        var_name=name + "_biases")
    return self, weights, biases


def get_var(
        self,
        initial_value,
        name,
        idx,
        var_name,
        in_size=None,
        out_size=None):
    """Handle variable loading if necessary."""
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
    return self, var
