"""2D convolutional model for Allen data."""

layer_structure = [
    {
        'layers': ['lstm2d'],
        'weights': [32],
        'names': ['lstm1'],
        'activation': ['selu'],
        'activation_target': ['post'],
    },
    {
        'layers': ['pool'],
        'weights': [None],
        'names': ['pool2'],
        'filter_size': [None],
        'flatten': [True],
        'flatten_target': ['post']
    },
]

output_structure = [
    {
        'layers': ['fc'],
        'weights': [1],
        'names': ['fc3'],
    }
]
