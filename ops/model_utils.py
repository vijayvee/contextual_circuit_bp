import numpy as np
import tensorflow as tf
from models import layers as lmod
from models.layers.activations import activations
from models.layers.normalizations import normalizations


class model_class(object):
    """Default model class that is generated with a layer_structure dict."""
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

    def model(self, data, output_size, layer_structure, tower_name):
        """Main model creation method."""
        data -= self.mean[None, :, :, :]  # Assuming H/W/C mean
        features = create_conv_tower(self, data, layer_structure, tower_name)
        if output_layer_structure is None:
            output_layer_structure = self.default_output_layer()
        create_conv_tower(self, features, output_layer_structure, 'output')

    def default_output_layer(self):
        return {
            'layers': 'fc',
            'names': 'output',
            'filter_size': self.output_size
        }

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


def create_conv_tower(self, act, layer_structure, tower_name):
    """
    Construct a feedforward neural model tower.
    Inputs:::
    act: a tensor to be fed into the model.
    layer_structure: a list of dictionaries that specify model layers. Note that
    not all operations are commutative (e.g. act fun -> dropout -> normalization).
    tower_name: name of the tower's variable scope.
    """
    print 'Creating tower: %s' % tower_name
    activ_mod = activations(self.layer_vars**)
    norm_mod = normalizations(self.layer_vars**)
    reg_mod = regularizations(self.layer_vars**)
    with tf.variable_scope(tower_name):
        for layer in layer_structure:
            keys = layer.keys()
            for vals in zip(*(d[k] for k in keys)):
                it_dict = dict(zip(keys, vals))
                if 'wd_target' in it_dict.keys() and it_dict['wd_target'] == 'pre':
                    self.regularizations['%s_%s' % (
                        it_dict['names'], it_dict['wd_target'])] = reg_mod[it_dict['wd_type']](act)
                if 'dropout_target' in it_dict.keys() and it_dict['dropout_target'] == 'pre':
                    act = reg_mod.dropout(act, keep_prob=it_dict['dropout'])
                if 'activation_target' in it_dict.keys() and it_dict['activation_target'] == 'pre':
                    act = activ_mod[it_dict['activation']](act)
                if 'normalization_target' in it_dict.keys() and it_dict['normalization_target'] == 'pre':
                    act = norm_mod[it_dict['normalization']](act) 
                if it_dict['layers'] == 'pool':
                    act = lmod.pool.max_pool(
                        bottom=act,
                        name=it_dict['names'])
                elif it_dict['layers'] == 'conv':
                    act = lmod.ff.conv_layer(
                        self=self,
                        bottom=act,
                        in_channels=int(act.get_shape()[-1]),
                        out_channels=it_dict['weights'],
                        name=it_dict['names'],
                        filter_size=it_dict['filter_size']
                    )
                elif it_dict['layers'] == 'fc':
                    act = lmod.ff.fc_layer(
                        self=self,
                        bottom=act,
                        in_channels=int(act.get_shape()[-1]),
                        out_channels=it_dict['weights'],
                        name=it_dict['names'])
                    )
                elif it_dict['layers'] == 'res':
                    act = lmod.ff.resnet_layer(
                        self=self,
                        bottom=act,
                        layer_weights=it_dict['weights'],
                        name=it_dict['names'])
                if 'wd_target' in it_dict.keys() and it_dict['wd_target'] == 'post':
                    self.regularizations['%s_%s' % (
                        it_dict['names'], it_dict['wd_target'])] = reg_mod[it_dict['wd_type']](act)
                if 'dropout_target' in it_dict.keys() and it_dict['dropout_target'] == 'post':
                    act = reg_mod.dropout(act, keep_prob=it_dict['dropout'])
                if 'activation_target' in it_dict.keys() and it_dict['activation_target'] == 'post':
                    act = activ_mod[it_dict['activation']](act)
                if 'normalization_target' in it_dict.keys() and it_dict['normalization_target'] == 'post':
                    act = norm_mod[it_dict['normalization']](act)
                setattr(self, it_dict['names'], act)
                print 'Added layer: %s' % na
    return act

