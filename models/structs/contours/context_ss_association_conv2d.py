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
        'normalization': ['contextual_ss'],
        'normalization_target': ['pre'],
        'normalization_aux': {
            'timesteps': 3,
            'association_field': True,
            'regularization_targets': {  # Modulate sparsity
                'q_t': {
                   'regularization_type': 'l1',
                   'regularization_strength': 0.01
                },
                't_t': {
                    'regularization_type': 'l1',
                    'regularization_strength': 0.1
                },
                'p_t': {
                    'regularization_type': 'orthogonal',
                    'regularization_strength': 0.1
                },
            }
        }
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
