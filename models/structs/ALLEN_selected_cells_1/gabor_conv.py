"""DoG model for Allen data."""

layer_structure = [
    {
        'layers': ['gabor_conv'],
        'weights': [64],
        'names': ['gabor1_1'],
        'filter_size': [5],
        'activation': ['selu'],
        'activation_target': ['post']
    },
    {
        'layers': ['fc'],
        'weights': [64],
        'names': ['fc1'],
        'flatten': [True],
        'flatten_target': ['pre'],
        'dropout': [0.5],
        'dropout_target': ['post'],
        'activation': ['selu'],
        'activation_target': ['post'],
        'regularization_type': ['l2'],
        'regularization_target': ['post'],
        'regularization_strength': [5e-7]
    }
]

output_structure = [
    {
        'layers': ['fc'],
        'weights': [1],  # Output size
        'names': ['fc2'],
        'regularization_type': ['l2'],
        'regularization_target': ['post'],
        'regularization_strength': [5e-7]
    }
]
