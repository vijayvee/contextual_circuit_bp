"""2D convolutional model for Allen data."""

layer_structure = [
    {
        'layers': ['conv'],
        'weights': [32],
        'names': ['conv1'],
        'filter_size': [3],
        'activation': ['selu'],
        'activation_target': ['post'],
    },
    {
        'layers': ['conv'],
        'weights': [32],
        'names': ['conv2'],
        'filter_size': [3],
        'activation': ['selu'],
        'activation_target': ['post'],
    },
    {
        'layers': ['pool'],
        'weights': [None],
        'names': ['pool3'],
        'filter_size': [None]
    },
    {
        'layers': ['conv'],
        'weights': [32],
        'names': ['conv4'],
        'filter_size': [3],
        'activation': ['selu'],
        'activation_target': ['post'],
    }
]

output_structure = [
    {
        'layers': ['conv'],
        'weights': [1],
        'names': ['fc_5'],
        'filter_size': [1],
    }
]
