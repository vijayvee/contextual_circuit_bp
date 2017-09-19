"""DoG model for Allen data."""

layer_structure = [
    {
        'layers': ['DoG'],
        'weights': [9],
        'names': ['conv1_1'],
        'filter_size': [7],
        'activation': ['logistic'],
        'activation_target': ['post'],
    },
    {
        'layers': ['pool'],
        'weights': [None],
        'names': ['pool1'],
        'filter_size': [None]
    }
]
