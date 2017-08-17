import numpy as np
import tensorflow as tf
from models.layers import ff
from models.layers import pool
from models.layers.activations import activations
from models.layers.normalizations import normalizations
from models.layers.regularizations import regularizations
from ops.eRF_calculator import eRF_calculator


eRF = eRF_calculator()


class model_class(object):
    """Default model class that is generated with a layer_structure dict."""
    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)

    def __init__(self, mean, training, output_size, **kwargs):
        """Set model to trainable/not and pass the mean values."""
        self.var_dict = {}
        self.data_dict = None
        self.regularizations = {}
        self.training = training
        self.mean = mean
        self.output_size = output_size
        self.share_vars = ['training', 'output_size']
        self.layer_vars = {k: self[k] for k in self.share_vars}

    def build(
            self,
            data,
            layer_structure,
            output_layer_structure=None,
            output_size=None,
            tower_name='cnn'):
        """Main model creation method."""
        # data -= (self.mean[None, :, :, :]).astype(np.float32)  # H/W/C mean
        input_data = tf.identity(data, name="lrp_input")

        # Calculate eRF info for main tower
        tower_eRFs = eRF.calculate(
            layer_structure,
            data,
            verbose=True)
        features, layer_summary = create_conv_tower(
            self=self,
            act=input_data,
            layer_structure=layer_structure,
            tower_name=tower_name,
            eRFs=tower_eRFs,
            layer_summary=None)
        if output_layer_structure is None:
            assert self.output_size is not None, 'Give model an output shape.'
            output_layer_structure = self.default_output_layer()
        output_eRFs = eRF.calculate(
            output_layer_structure,
            features,
            r_i=tower_eRFs.items()[-1][1],
            verbose=True)
        output, layer_summary = create_conv_tower(
            self=self,
            act=features,
            layer_structure=output_layer_structure,
            tower_name='output',
            eRFs=output_eRFs,
            layer_summary=layer_summary)
        self.output = tf.identity(output, name='lrp_output')
        self.data_dict = None
        return output, layer_summary

    def default_output_layer(self):
        return [
            {
                'layers': ['fc'],
                'names': ['output'],
                'flatten': [True],
                'flatten_target': ['pre'],
                'weights': [self.output_size]
            }
        ]

    def save_npy(self, sess, npy_path="./saved_weights.npy"):
        """Default method: Save your model's weights to a numpy."""
        assert isinstance(sess, tf.Session)

        data_dict = {}
        num_files = 0

        for (name, idx), var in self.var_dict.items():
            # print(var.get_shape())
            var_out = sess.run(var)
            if name not in data_dict.keys():
                data_dict[name] = {}
            data_dict[name][idx] = var_out
        np.save('%s%s' % (npy_path, str(num_files)), data_dict)
        print 'Weights saved to: %s' % npy_path
        return npy_path

    def get_var_count(self):
        """Default method: Count your variables."""
        count = 0
        for v in self.var_dict.values():
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count


def update_summary(layer_summary, op_name):
    """Produce a summary of the structure of a CNN used in an experiment."""
    if layer_summary is None:
        bottom_name = 'Input'
        layer_summary = []
    else:
        bottom_name = layer_summary[len(layer_summary) - 1].split(
            'Operation: ')[-1].strip('\n')
    layer_summary += ['Bottom: %s | Operation: %s\n' % (
        bottom_name, op_name[0])]
    return layer_summary


def flatten_op(self, it_dict, act, layer_summary, eRFs, target):
    """Wrapper for a flatten operation in a graph."""
    tshape = [int(x) for x in act.get_shape()]
    if 'flatten_target' in it_dict.keys() and \
            it_dict['flatten_target'][0] == target and \
            len(tshape) >= 4:
        rows = tshape[0]
        cols = np.prod(tshape[1:])
        act = tf.reshape(act, [rows, cols])
        layer_summary = update_summary(
            layer_summary=layer_summary,
            op_name=['flattened'])
    return act, layer_summary


def wd_op(self, it_dict, act, layer_summary, reg_mod, eRFs, target):
    """Wrapper for a weight decay operation in a graph."""
    it_name = it_dict['names'][0]
    if 'wd_target' in it_dict.keys() and \
            it_dict['wd_type'][0] is not None and \
            it_dict['wd_target'][0] == target:
        wd_type = it_dict['wd_type'][0]
        self.regularizations['%s_%s' % (
            it_name, target)] = reg_mod[
                wd_type](act)
        layer_summary = update_summary(
            layer_summary=layer_summary,
            op_name=wd_type)
    return act, layer_summary


def dropout_op(self, it_dict, act, layer_summary, reg_mod, eRFs, target):
    """Wrapper for a dropout operation in a graph."""
    if 'dropout_target' in it_dict.keys() and \
            it_dict['dropout_target'][0] == target:
        dropout_prop = it_dict['dropout'][0]
        act = reg_mod.dropout(act, keep_prob=dropout_prop)
        layer_summary = update_summary(
            layer_summary=layer_summary,
            op_name=['dropout_%s' % dropout_prop])
    return act, layer_summary


def activ_op(self, it_dict, act, layer_summary, activ_mod, eRFs, target):
    """Wrapper for an activation operation in a graph."""
    if 'activation_target' in it_dict.keys() and \
            it_dict['activation_target'][0] == target:
        activation = it_dict['activation'][0]
        act = activ_mod[activation](act)
        layer_summary = update_summary(
            layer_summary=layer_summary,
            op_name=activation)
    return act, layer_summary


def norm_op(self, it_dict, act, layer_summary, norm_mod, eRFs, target):
    """Wrapper for a normalization operation in a graph."""
    if 'normalization_target' in it_dict.keys() and \
            it_dict['normalization_target'][0] == target:
        normalization = it_dict['normalization'][0]
        if len(it_dict['names']) > 1:
            raise RuntimeError('TODO: Fix implementation for multiple names.')
        act = norm_mod[normalization](act, eRFs[it_dict['names'][0]])
        layer_summary = update_summary(
            layer_summary=layer_summary,
            op_name=normalization)
    return act, layer_summary


def create_conv_tower(
        self,
        act,
        layer_structure,
        tower_name,
        eRFs=None,
        layer_summary=None):
    """
    Construct a feedforward neural model tower.
    Inputs:::
    act: a tensor to be fed into the model.
    layer_structure: a list of dictionaries that
    specify model layers. Note that not all operations are commutative
    (e.g. act fun -> dropout -> normalization).
    tower_name: name of the tower's variable scope.
    """
    activ_mod = activations(self.layer_vars)
    norm_mod = normalizations(self.layer_vars)
    reg_mod = regularizations(self.layer_vars)
    with tf.variable_scope(tower_name):
        for it_dict in layer_structure:
            it_name = it_dict['names'][0]
            it_neuron_op = it_dict['layers'][0]
            act, layer_summary = flatten_op(
                self,
                it_dict,
                act,
                layer_summary,
                eRFs=eRFs,
                target='pre')
            act, layer_summary = wd_op(
                self,
                it_dict,
                act, layer_summary,
                reg_mod,
                eRFs=eRFs,
                target='pre')
            act, layer_summary = dropout_op(
                self,
                it_dict,
                act,
                layer_summary,
                reg_mod,
                eRFs=eRFs,
                target='pre')
            act, layer_summary = activ_op(
                self,
                it_dict,
                act,
                layer_summary,
                activ_mod,
                eRFs=eRFs,
                target='pre')
            act, layer_summary = norm_op(
                self,
                it_dict,
                act,
                layer_summary,
                norm_mod,
                eRFs=eRFs,
                target='pre')
            if it_neuron_op == 'pool':  # TODO create wrapper for FF ops.
                act = pool.max_pool(
                    bottom=act,
                    name=it_name)
            elif it_neuron_op == 'conv':
                act = ff.conv_layer(
                    self=self,
                    bottom=act,
                    in_channels=int(act.get_shape()[-1]),
                    out_channels=it_dict['weights'][0],
                    name=it_name,
                    filter_size=it_dict['filter_size'][0]
                )
            elif it_neuron_op == 'fc':
                act = ff.fc_layer(
                    self=self,
                    bottom=act,
                    in_channels=int(act.get_shape()[-1]),
                    out_channels=it_dict['weights'][0],
                    name=it_name)
            elif it_neuron_op == 'res':
                act = ff.resnet_layer(
                    self=self,
                    bottom=act,
                    layer_weights=it_dict['weights'],
                    name=it_name)
            layer_summary = update_summary(
                layer_summary=layer_summary,
                op_name=it_dict['layers'])
            act, layer_summary = norm_op(
                self,
                it_dict,
                act,
                layer_summary,
                norm_mod,
                eRFs=eRFs,
                target='post')
            act, layer_summary = activ_op(
                self,
                it_dict,
                act,
                layer_summary,
                activ_mod,
                eRFs=eRFs,
                target='post')
            act, layer_summary = dropout_op(
                self,
                it_dict,
                act,
                layer_summary,
                reg_mod,
                eRFs=eRFs,
                target='post')
            act, layer_summary = wd_op(
                self,
                it_dict,
                act, layer_summary,
                reg_mod,
                eRFs=eRFs,
                target='post')
            act, layer_summary = flatten_op(
                self,
                it_dict,
                act,
                layer_summary,
                eRFs=eRFs,
                target='post')
            setattr(self, it_name, act)
            print 'Added layer: %s' % it_name
    return act, layer_summary
