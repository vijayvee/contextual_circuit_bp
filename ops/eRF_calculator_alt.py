import numpy as np
from collections import OrderedDict


class eRF_calculator(object):
    def __init__(self):
        """eRF calculation initialization."""
        self.default_kernel = 2  # For max pooling
        self.default_stride = 1
        self.default_padding = lambda k: [ik // 2 for ik in k]

    def calculate(self, network, image, verbose=True, r_i=None):
        """Routine for calculating effective RF size."""
        try:
            network_params = self.extract_params(network)
            imsize = int(image.get_shape()[1])
            if verbose:
                print '-------Net summary-------'
            keys = []
            rf_dict = []
            for k in network_params.keys():
                keys += [k]
                p = self.outFromIn(imsize, keys, network_params)
                rf = self.inFromOut(keys, network_params)
                rf_dict += {
                    'n_i': p[0],
                    'j_i': p[1],
                    'r_i': rf,
                    'layer': k,
                }

            # self.easy_calculate(network_params)
            if verbose:
                [self.printLayer(x) for x in rf_dict]
            return rf_dict
        except:
            print 'Could not derive eRFs.'
            return None

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

    def outFromIn(self, insize, keys, net):
        """Activity sizes per layer."""
        totstride = 1
        for key in keys:
            it_layer = net[key]
            outsize = (
                insize - it_layer['kernel'] +
                2 * it_layer['padding']
                ) / it_layer['stride'] + 1
            insize = outsize
            totstride = totstride * it_layer['stride']
        return outsize, totstride

    def inFromOut(self, keys, net):
        """Calulate effective RF sizes per layer."""
        outsize = 1
        for key in reversed(keys):
            it_layer = net[key]
            outsize = ((outsize - 1) * it_layer['stride']) + it_layer['kernel']
        RFsize = outsize
        return RFsize

    def printLayer(self, layer):
        print '%s:' % layer['layer']
        print '\t n features: %s \n \t jump: %s \n \t receptive size: %s \n' % (
            layer['n_i'], layer['j_i'], layer['r_i'])
