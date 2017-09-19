"""3D convolutional model for Allen data."""

layer_structure = [
    {
        'layers': ['conv3d'],
        'weights': [64],
        'names': ['conv1_1'],
        'filter_size': [7],
        'activation': ['relu'],
        'normalization_target': ['post'],
    },
    {
        'layers': ['pool'],
        'weights': [None],
        'names': ['pool1'],
        'filter_size': [None]
    }
]
