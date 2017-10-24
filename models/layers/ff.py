"""Functions for handling feedforward and pooling operations."""
import numpy as np
import tensorflow as tf
from ops import initialization
from models.layers.activations import activations
from models.layers.normalizations import normalizations
from models.layers.ff_functions import ff as ff_fun
from models.layers import pool


class ff(object):
    """Wrapper class for network filter operations."""

    def __getitem__(self, name):
        """Get attribute from class."""
        return getattr(self, name)

    def __contains__(self, name):
        """Check if class contains attribute."""
        return hasattr(self, name)

    def __init__(self, kwargs=None):
        """Global variables for ff functions."""
        self.update_params(kwargs)
        self.pool_class = pool.pool()

    def update_params(self, kwargs):
        """Update the class attributes with kwargs."""
        if kwargs is not None:
            for k, v in kwargs.iteritems():
                setattr(self, k, v)

    def dog(
            self,
            context,
            act,
            in_channels,
            out_channels,
            filter_size,
            name,
            it_dict):
        """Add a Difference of Gaussians layer."""
        context, act = dog_layer(
            self=context,
            bottom=act,
            layer_weights=out_channels,
            name=name)
        return context, act

    def conv(
            self,
            context,
            act,
            in_channels,
            out_channels,
            filter_size,
            name,
            it_dict):
        """Add a 2D convolution layer."""
        context, act = conv_layer(
            self=context,
            bottom=act,
            in_channels=in_channels,
            out_channels=out_channels,
            name=name,
            filter_size=filter_size)
        return context, act

    def conv3d(
            self,
            context,
            act,
            in_channels,
            out_channels,
            filter_size,
            name,
            it_dict):
        """Add a 3D convolution layer."""
        context, act = conv_3d_layer(
            self=context,
            bottom=act,
            in_channels=in_channels,
            out_channels=out_channels,
            name=name,
            filter_size=filter_size)
        return context, act

    def sep_conv(
            self,
            context,
            act,
            in_channels,
            out_channels,
            filter_size,
            name,
            it_dict):
        """Add a separable 2D convolution layer."""
        context, act = sep_conv_layer(
            self=context,
            bottom=act,
            in_channels=in_channels,
            out_channels=out_channels,
            name=name,
            filter_size=filter_size)
        return context, act

    def fc(
            self,
            context,
            act,
            in_channels,
            out_channels,
            filter_size,
            name,
            it_dict):
        """Add a fully-connected layer."""
        context, act = fc_layer(
            self=context,
            bottom=act,
            in_channels=in_channels,
            out_channels=out_channels,
            name=name)
        return context, act

    def sparse_pool(
            self,
            context,
            act,
            in_channels,
            out_channels,
            filter_size,
            name,
            it_dict):
        """Add a sparse pooling layer."""
        context, act = sparse_pool_layer(
            self=context,
            bottom=act,
            in_channels=in_channels,
            out_channels=out_channels,
            aux=it_dict,
            name=name)
        return context, act

    def res(
            self,
            context,
            act,
            in_channels,
            out_channels,
            filter_size,
            name,
            it_dict):
        """Add a residual layer."""
        context, act = resnet_layer(
            self=context,
            bottom=act,
            aux=it_dict['aux'],
            layer_weights=it_dict['weights'],
            name=name)
        return context, act

    def _pass(
            self,
            context,
            act,
            in_channels,
            out_channels,
            filter_size,
            name,
            it_dict):
        """Skip a filter operation on this layer."""
        return context, act

    def pool(
            self,
            context,
            act,
            in_channels,
            out_channels,
            filter_size,
            name,
            it_dict):
        """Wrapper for 2d pool. TODO: add op flexibility."""
        if filter_size is None:
            filter_size = [1, 2, 2, 1]
        stride_size = it_dict.get('stride', [1, 2, 2, 1])
        if not isinstance(filter_size, list):
            filter_size = [1, filter_size, filter_size, 1]
        if not isinstance(stride_size, list):
            filter_size = [1, stride_size, stride_size, 1]
        if 'aux' in it_dict and 'pool_type' in it_dict['aux']:
            pool_type = it_dict['aux']['pool_type']
        else:
            pool_type = 'max'

        context, act = self.pool_class.interpret_2dpool(
            context=context,
            bottom=act,
            name=name,
            filter_size=filter_size,
            stride_size=stride_size,
            pool_type=pool_type
        )
        return context, act

    def pool3d(
            self,
            context,
            act,
            in_channels,
            out_channels,
            filter_size,
            name,
            it_dict):
        """Wrapper for 3d pool. TODO: add op flexibility."""
        if filter_size is None:
            filter_size = [1, 2, 2, 1]
        stride_size = it_dict.get('stride', [1, 2, 2, 1])
        if not isinstance(filter_size, list):
            filter_size = [1, filter_size, filter_size, 1]
        if not isinstance(stride_size, list):
            filter_size = [1, stride_size, stride_size, 1]
        if 'aux' in it_dict and 'pool_type' in it_dict['aux']:
            pool_type = it_dict['aux']['pool_type']
        else:
            pool_type = 'max'

        context, act = self.pool_class.max_pool_3d(
            context=context,
            bottom=act,
            name=name,
            filter_size=filter_size,
            stride_size=stride_size,
            pool_type=pool_type
        )
        return context, act

    def vgg16(
            self,
            context,
            act,
            in_channels,
            out_channels,
            filter_size,
            name,
            it_dict):
        """Wrapper for loading an imagnet initialized VGG16."""
        context, act = ff_fun.vgg16(
            self=context,
            bottom=act,
            aux=it_dict['aux'],
            layer_weights=it_dict['weights'],
            name=name)
        return context, act


def gather_value_layer(
        self,
        bottom,
        aux,
        name):
    """Gather a value from a location in an activity tensor."""
    assert aux is not None,\
        'Gather op needs an aux dict with h/w coordinates.'
    assert 'h' in aux.keys() and 'w' in aux.keys(),\
        'Gather op dict needs h/w key value pairs'
    h = aux['h']
    w = aux['w']
    out = tf.squeeze(bottom[:, h, w, :])
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
        aux=None,
        combination=None):  # tf.multiply
    """Residual layer."""
    ln = '%s_branch' % name
    rlayer = tf.identity(bottom)
    if aux is not None:
        if 'activation' in aux.keys():
            activation = aux['activation']
        if 'normalization' in aux.keys():
            normalization = aux['normalization']
        if 'combination' in aux.keys():
            if aux['combination'] == 'add':
                combination = tf.add
            elif aux['combination'] == 'prod':
                combination = tf.multiply
            elif aux['combination'] == 'add_prod':
                combination = lambda x, y: tf.concat(
                        tf.add(x, y),
                        tf.multiply(x, y)
                    )
        else:
            combination = tf.add
    if normalization is not None:
        if normalization is not 'batch':
            raise RuntimeError(
                'Normalization not yet implemented for non-batchnorm.')
        nm = normalizations({'training': self.training})[normalization]
    else:
        nm = lambda x: x
    if activation is not None:
        ac = activations()[activation]
    else:
        ac = lambda x: x
    for idx, lw in enumerate(layer_weights):
        ln = '%s_%s' % (name, idx)
        self, rlayer = conv_layer(
            self=self,
            bottom=rlayer,
            in_channels=int(rlayer.get_shape()[-1]),
            out_channels=lw,
            name=ln)
        rlayer = nm(ac(rlayer), None, None, None)
        if isinstance(rlayer, tuple):
            rlayer = rlayer[0]
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


def sep_conv_layer(
        self,
        bottom,
        out_channels,
        name,
        in_channels=None,
        filter_size=3,
        stride=[1, 1, 1, 1],
        padding='SAME',
        multiplier=4,
        aux=None):
    """2D convolutional layer."""
    if aux is not None and 'multiplier' in aux:
        multiplier = aux['multiplier']
    with tf.variable_scope(name):
        if in_channels is None:
            in_channels = int(bottom.get_shape()[-1])
        self, dfilt, _ = get_conv_var(
            self=self,
            filter_size=filter_size,
            in_channels=1 * multiplier,
            out_channels=out_channels,
            name='d_%s' % name)
        self, pfilt, conv_biases = get_conv_var(
            self=self,
            filter_size=filter_size,
            in_channels=1 * multiplier,
            out_channels=out_channels,
            name='d_%s' % name)
        conv = tf.nn.separable_conv2d(
            input=bottom,
            depthwise_filter=dfilt,
            pointwise_filt=pfilt,
            strides=stride,
            padding=padding)
        bias = tf.nn.bias_add(conv, conv_biases)
        return self, bias


def conv_3d_layer(
        self,
        bottom,
        out_channels,
        name,
        in_channels=None,
        filter_size=3,
        stride=[1, 1, 1, 1, 1],
        padding='SAME',
        aux=None):
    """NOT YET IMPLEMENTED: 3D convolutional layer."""
    with tf.variable_scope(name):
        if in_channels is None:
            in_channels = int(bottom.get_shape()[-1])
        self, filt, conv_biases = get_conv_var(
            self=self,
            filter_size=filter_size,
            in_channels=in_channels,
            out_channels=out_channels,
            name=name,
            kernel_dimensionality=3)
        if aux is not None and 'dilation' in aux.keys():
            conv = tf.nn.convolution(
                input=bottom,
                filter=filt,
                strides=stride,
                padding=padding,
                dilation_rate=aux['dilation'])
        else:
            conv = tf.nn.conv3d(
                bottom,
                filt,
                stride,
                padding=padding)
        bias = tf.nn.bias_add(conv, conv_biases)
        return self, bias


def st_resnet_layer(
        self,
        bottom,
        layer_weights,
        name,
        aux=None,
        combination=None):  # tf.multiply
    """Spatiotemporal residual layer."""
    ln = '%s_branch' % name
    rlayer = tf.identity(bottom)
    if aux is not None:
        if 'activation' in aux.keys():
            activation = aux['activation']
        if 'normalization' in aux.keys():
            normalization = aux['normalization']
        if 'combination' in aux.keys():
            if aux['combination'] == 'add':
                combination = tf.add
            elif aux['combination'] == 'prod':
                combination = tf.multiply
            elif aux['combination'] == 'add_prod':
                combination = lambda x, y: tf.concat(
                        tf.add(x, y),
                        tf.multiply(x, y)
                    )
        else:
            combination = tf.add
    if normalization is not None:
        if normalization is not 'batch':
            raise RuntimeError(
                'Normalization not yet implemented for non-batchnorm.')
        nm = normalizations({'training': self.training})[normalization]
    else:
        nm = lambda x: x
    if activation is not None:
        ac = activations()[activation]
    else:
        ac = lambda x: x
    for idx, lw in enumerate(layer_weights):
        ln = '%s_%s' % (name, idx)
        self, rlayer = conv_3d_layer(
            self=self,
            bottom=rlayer,
            in_channels=int(rlayer.get_shape()[-1]),
            out_channels=lw,
            name=ln)
        rlayer = nm(ac(rlayer), None, None, None)
        if isinstance(rlayer, tuple):
            rlayer = rlayer[0]
    return self, combination(rlayer, bottom)


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
        import ipdb;ipdb.set_trace()
        raise NotImplementedError

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
                k = aux['k']
            else:
                gaussian_h, gaussian_w, k = None, None, None
            spatial_rf = create_gaussian_rf(
                xy=gaussian_xy,
                h=gaussian_h,
                w=gaussian_w,
                k=k)
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
        init_type='xavier',
        kernel_dimensionality=2):
    """Prepare convolutional kernel weights."""
    kernel = [filter_size] * kernel_dimensionality
    if init_type == 'xavier':
        weight_init = [
            kernel + [in_channels, out_channels],
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


def pool_ff_interpreter(
        self,
        it_neuron_op,
        act,
        it_name,
        it_dict):
    """Wrapper for FF and pooling functions. DEPRECIATED."""
    if it_neuron_op == 'pool':  # TODO create wrapper for FF ops.
        self, act = pool.max_pool(
            self=self,
            bottom=act,
            name=it_name
        )
    elif it_neuron_op == 'pool3d':  # TODO create wrapper for FF ops.
        self, act = pool.max_pool_3d(
            self=self,
            bottom=act,
            name=it_name
        )
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
            filter_size=it_dict['filter_size'][0],
            aux=it_dict['aux']
        )
    elif it_neuron_op == 'residual_conv3d':
        pass
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
            aux=it_dict['aux'],
            layer_weights=it_dict['weights'],
            name=it_name)
    elif it_neuron_op == 'gather':
        self, act = gather_value_layer(
            self=self,
            bottom=act,
            aux=it_dict['aux'],
            name=it_name)
    elif it_neuron_op == 'pass':
        pass
    else:
        raise RuntimeError(
            'Your specified operation %s is not implemented' % it_neuron_op)
    return self, act
