"""2D convolutional model for Allen data."""

layer_structure = [
    {
        'layers': ['sepgru2d'],
        'weights': [12],
        'filter_size': [11],
        'names': ['sepgru2d1'],
        'activation': ['selu'],
        'activation_target': ['post'],
    },
    {
        'layers': ['pool'],
        'weights': [None],
        'names': ['pool2'],
        'filter_size': [None]
    },
    {
        'layers': ['conv'],
        'weights': [24],
        'filter_size': [3],
        'names': ['conv3'],
        'activation': ['selu'],
        'activation_target': ['post'],
    }
]

output_structure = [
    {
        'flatten': [True],
        'flatten_target': ['pre'],
        'layers': ['fc'],
        'weights': [2],
        'names': ['fc4'],
    }
]
