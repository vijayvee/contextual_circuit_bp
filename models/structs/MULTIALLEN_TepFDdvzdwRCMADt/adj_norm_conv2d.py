"""2D convolutional model for Allen data."""

layer_structure = [
    {
        'layers': ['conv'],
        'weights': [64],
        'names': ['conv1_1'],
        'filter_size': [5],
        'activation': ['selu'],
        'activation_target': ['post'],
        'dropout': [0.5],
        'dropout_target': ['post'],
        'normalization': ['contextual_vector_separable_random'],
        'normalization_target': ['post'],
        'normalization_aux': {
            'timesteps': 5,
            'regularization_targets': {  # Modulate sparsity
                'q_t': {
                   'regularization_type': 'l1',
                   'regularization_strength': 0.1
                },
                't_t': {
                    'regularization_type': 'l1',
                    'regularization_strength': 0.01
                },
                'p_t': {
                    'regularization_type': 'l1',
                    'regularization_strength': 0.1
                },
            }
        },
    }
]

output_structure = [
    {
        'layers': ['sparse_pool'],
        'weights': [103],  # Output size
        'names': ['sp2'],
    }
]
