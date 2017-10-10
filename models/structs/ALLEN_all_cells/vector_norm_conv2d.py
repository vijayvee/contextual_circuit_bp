"""2D convolutional model for Allen data."""
"""2D convolutional model for Allen data."""

layer_structure = [
    {
        'layers': ['conv'],
        'weights': [64],
        'names': ['conv1_1'],
        'filter_size': [7],
        'activation': ['selu'],
        'activation_target': ['post'],
        'dropout': [0.5],
        'dropout_target': ['post']
    },
    {
        'layers': ['pool'],
        'weights': [None],
        'names': ['pool1'],
        'filter_size': [2],
        'normalization': ['contextual_alt_learned_transition_learned_connectivity_vector_modulation'],
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
    }
]

output_structure = [
    {
        'layers': ['fc'],
        'weights': [103],  # Output size
        'names': ['fc3'],
    }
]
