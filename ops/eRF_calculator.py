import numpy as np
from collections import OrderedDict


class eRF_calculator(object):
    def __init__(self):
        """eRF calculation initialization."""
        self.default_kernel = 2  # For max pooling
        self.default_stride = 1
        self.default_padding = lambda k: [ik // 2 for ik in k]

    def calculate(self, network, image, verbose=True):
        """Routine for calculating effective RF size."""
        network_params = self.extract_params(network)
        imsize = int(image.get_shape()[1])
        if verbose:
            print '-------Net summary-------'
        currentLayer = {
            'n_i': imsize,
            'j_i': 1,
            'r_i': 1,
            'start_i': 0.5
        }
        # self.easy_calculate(network_params)
        self.printLayer(currentLayer, 'input image')
        layer_infos = []
        for k, v in network_params.iteritems():
            currentLayer = self.outFromIn(v, currentLayer)
            layer_infos += [currentLayer]
            if verbose:
                self.printLayer(currentLayer, k)
        return {k: v['r_i'] for k, v in zip(
            network_params.keys(), layer_infos)}

    def easy_calculate(self, layers):
        rfs = [1]
        count = 0
        for l in layers.iteritems():
            rfs += [rfs[count] + ((l[1]['kernel'] - 1) * l[1]['stride'])]
            count += 1
        return {k: v for k, v in zip(layers.keys(), rfs)}

    def interpret_padding(self, padding_string, kernel_size):
        """Translate TF padding type + kernel_size."""
        if padding_string == 'FULL':
            return kernel_size - 1
        elif padding_string == 'SAME':
            return kernel_size // 2

    def extract_params(self, network):
        """Extract the filter size, stride, and padding from each layer."""
        params = {}
        for l in network:
            K = l['filter_size']
            K = [k if k is not None else self.default_kernel for k in K]
            layer_len = len(K)

            if 'stride' in l:
                S = l['stride']
            else:
                S = np.repeat(self.default_stride, layer_len)
            if 'padding' in l:
                P = l['padding']
                P = [self.interpret_padding(
                        p,
                        k) for p, k in zip(P, K)]
            else:
                P = self.default_padding(K)

            for idx, (k, s, p) in enumerate(zip(K, S, P)):
                params[l['names'][idx]] = {
                    'kernel': k,
                    'stride': s,
                    'padding': p
                }
        return OrderedDict(sorted(params.items(), key=lambda t: t[0]))

    def outFromIn(self, conv, layer):
        """Calculate effective RF for a layer.
        Assume the two dimensions are the same
        Each conv is a dict with:
         - k_i: kernel size
         - s_i: stride
         - p_i: padding (if padding is uneven, right padding is higher than
            left padding; "SAME" option in tensorflow)
        Each layer is a dict with:
         - n_i: number of feature (data layer has n_1 = imagesize )
         - j_i: distance (projected to image pixel distance) between center
            of two adjacent features
         - r_i: receptive field of a feature in layer i
         - start_i: position of the first feature's receptive field in layer i
            (idx start from 0, negative means the center fall into padding)
        """
        n_in = layer['n_i']
        j_in = layer['j_i']
        r_in = layer['r_i']
        start_in = layer['start_i']
        k = conv['kernel']
        s = conv['stride']
        p = conv['padding']
        n_out = np.floor((n_in - k + 2*p)/s) + 1
        actualP = (n_out-1)*s - n_in + k
        pR = np.ceil(actualP/2)
        pL = np.floor(actualP/2)
        j_out = j_in * s
        r_out = r_in + (k - 1)*j_in
        start_out = start_in + ((k-1)/2 - pL)*j_in
        return {
            'n_i': n_out,
            'j_i': j_out,
            'r_i': r_out,
            'start_i': start_out
        }

    def printLayer(self, layer, layer_name):
        print '%s:' % layer_name
        print '\t n features: %s \n \t jump: %s \n \t receptive size: %s \n \t start: %s ' % (
            layer['n_i'], layer['j_i'], layer['r_i'], layer['start_i'])
