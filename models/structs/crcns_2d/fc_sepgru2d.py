"""2D convolutional model for Allen data."""

layer_structure = [
    {
        'layers': ['sepgru2d'],
        'weights': [12],
        'filter_size': [7],
        'names': ['sepgru2d1'],
        'activation': ['selu'],
        'activation_target': ['post'],
    },
    {
        'layers': ['conv'],
        'weights': [24],
        'filter_size': [3],
        'names': ['conv2'],
        'activation': ['selu'],
        'activation_target': ['post'],
    },
]

