"""2D convolutional model (1-layer) for Allen data."""

layer_structure = [
    {
        'layers': ['conv'],
        'weights': [32],
        'names': ['conv1'],
        'filter_size': [12],
        'normalization': ['contextual'],
        'normalization_target': ['post'],
        'normalization_aux': {
            'timesteps': 3,
            'association_field': True,
            'full_far_eCRF': True,
            'exclude_CRF': False,
            'regularization_targets': {  # Modulate sparsity
                'q_t': {
                   'regularization_type': 'orthogonal',
                   'regularization_strength': 0.01
                },
                't_t': {
                    'regularization_type': 'orthogonal',
                    'regularization_strength': 0.5
                },
                'p_t': {
                    'regularization_type': 'orthogonal',
                    'regularization_strength': 1
                    },
                }
            },
        }
]

output_structure = [
    {
        'layers': ['conv'],
        'weights': [1],
        'names': ['fc4'],
        'filter_size': [1],
        'activation': ['sigmoid'],
        'activation_target': ['post']
    }
]
