"""DoG model for Allen data."""

layer_structure = [
    {
        'layers': ['DoG'],
        'weights': [9],
        'names': ['dog1_1'],
        'filter_size': [7],
        'activation': ['logistic'],
        'activation_target': ['post'],
    },
    {
        'layers': ['fc'],
        'weights': [64],
        'names': ['fc2'],
        'flatten': [True],
        'flatten_target': ['pre'],
    }
]
