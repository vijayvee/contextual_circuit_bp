"""2D convolutional model for Allen data."""

layer_structure = [
    {
        'layers': ['pass'],
        'names': ['contextual'],
        'hardcoded_erfs': {
            'SRF': 3,
            'CRF_excitation': 3, 
            'CRF_inhibition': 3,
            'SSN': 9,
            'SSF': 29
        },
        'normalization': ['contextual_alt_learned_transition_learned_connectivity_scalar_modulation'],
        'normalization_target': ['pre'],
        'normalization_aux': {
            'timesteps': 10,
            'regularization_targets': {  # Modulate sparsity
                'q_t': {
                   'regularization_type': 'l1',
                   'regularization_strength': 0.0001
                },
                't_t': {
                    'regularization_type': 'l1',
                    'regularization_strength': 0.001
                },
                'p_t': {
                    'regularization_type': 'l1',
                    'regularization_strength': 0.001
                },
            }
        },
    }
]

output_structure = [
    {
        'layers': ['gather'],
        'aux': {
            'h': 25,
            'w': 25
        },  # Output size
        'names': ['gather'],
    }
]
