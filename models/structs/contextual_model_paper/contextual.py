"""2D convolutional model for Allen data."""

layer_structure = [
    {
        'layers': ['pass'],
        'names': ['contextual'],
        'normalization': ['contextual_alt_learned_transition_learned_connectivity_scalar_modulation'],
        'normalization_target': ['post'],
        'normalization_aux': {
            'timesteps': 10,
            'regularization_targets': {  # Modulate sparsity
                'q_t': {
                    'regularization_type': 'l1',
                    'regularization_strength': 0.1
                },
                't_t': {
                    'regularization_type': 'l1',
                    'regularization_strength': 0.5
                },
                'p_t': {
                    'regularization_type': 'l1',
                    'regularization_strength': 0.5
                },
            }
        },
    }
]

output_structure = [
    {
        'layers': ['gather'],
        'aux': {
            'x': 25,
            'y': 25
        },  # Output size
        'names': ['gather'],
    }
]
