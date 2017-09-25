"""2D convolutional model for Allen data."""

layer_structure = [
    {
        'layers': ['pass'],
        'names': ['contextual'],
        'hardcoded_erfs': {
            'SRF': 1,
            'CRF_excitation': 1, 
            'CRF_inhibition': 1,
            'SSN': 9,
            'SSF': 29
        },
        'normalization': ['contextual_alt_learned_transition_learned_connectivity_scalar_modulation'],
        'normalization_target': ['pre'],
        'normalization_aux': {
            'timesteps': 5,
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
            'h': 25,
            'w': 25
        },  # Output size
        'names': ['gather'],
    }
]
