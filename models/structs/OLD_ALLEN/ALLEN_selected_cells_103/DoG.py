"""DoG model for Allen data."""

layer_structure = [
    {
        'layers': ['DoG'],
        'weights': [9],
        'names': ['dog1'],
        'filter_size': [5],
        'activation': ['selu'],
        'activation_target': ['post'],
        'dropout': [0.5],
        'dropout_target': ['post']
    },
    {
        'layers': ['fc'],
        'weights': [64],
        'filter_size': [1],
        'names': ['fc2'],
        'flatten': [True],
        'flatten_target': ['pre'],
        'activation': ['selu'],
        'activation_target': ['post']
    },
]

output_structure = [
    {
        'layers': ['fc'],
        'weights': [103],  # Output size
        'names': ['fc3'],
    }
]
