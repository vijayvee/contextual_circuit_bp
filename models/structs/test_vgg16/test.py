

layer_structure = [
    {
        'layers': ['conv', 'conv', 'pool'],
        'weights': [64, 64, None],
        'names': ['conv1_1', 'conv1_2', 'pool_1'],
        'filter_size': [3, 3, None],
        'activation': ['relu', 'relu', None],
        'activation_target': ['post', 'post', None],
    },
    {
        'layers': ['conv', 'conv', 'pool'],
        'weights': [128, 128, None],
        'names': ['conv2_1', 'conv2_2', 'pool_2'],
        'filter_size': [3, 3, None],
        'activation': ['relu', 'relu', None],
        'activation_target': ['post', 'post', None],
    },
    {
        'layers': ['conv', 'conv', 'conv', 'pool'],
        'weights': [256, 256, 256, None],
        'names': ['conv3_1', 'conv3_2', 'pool_3'],
        'filter_size': [3, 3, 3, None],
        'activation': ['relu', 'relu', 'relu', None],
        'activation_target': ['post', 'post', 'post', None],
    },
    {
        'layers': ['conv', 'conv', 'conv', 'pool'],
        'weights': [512, 512, 512, None],
        'names': ['conv4_1', 'conv4_2', 'pool_4'],
        'filter_size': [3, 3, 3, None],
        'activation': ['relu', 'relu', 'relu', None],
        'activation_target': ['post', 'post', 'post', None],
    },
    {
        'layers': ['conv', 'conv', 'conv', 'pool'],
        'weights': [512, 512, 512, None],
        'names': ['conv5_1', 'conv5_2', 'pool_5'],
        'filter_size': [3, 3, 3, None],
        'activation': ['relu', 'relu', 'relu', None],
        'activation_target': ['post', 'post', 'post', None],
    },
    {
        'layers': ['fc', 'fc'],
        'weights': [4096, 4096],
        'names': ['fc6', 'fc7'],
        'activation': ['relu', 'relu'],
        'activation_target': ['post', 'post'],
        'flatten': [True, False],
        'flatten_target': ['pre', None],
    },
]
