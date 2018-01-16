"""2D convolutional model for Allen data."""

layer_structure = [
    {
        'layers': ['lstm1d'],
        'weights': [32],
        'names': ['lstm1'],
        'activation': ['selu'],
        'activation_target': ['post'],
    },
    {
        'layers': ['pool1d'],
        'weights': [None],
        'names': ['pool2'],
        'filter_size': [None]
    },
]

output_structure = [
    {
        'layers': ['fc'],
        'weights': [1],
        'names': ['fc3'],
    }
]
