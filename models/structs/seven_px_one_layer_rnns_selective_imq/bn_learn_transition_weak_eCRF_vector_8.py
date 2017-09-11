

layer_structure = [
    {
        'layers': ['conv'],
        'weights': [64],
        'names': ['conv1_1'],
        'filter_size': [3],
        'normalization': ['bn_contextual_frozen_connectivity_learned_transition_weak_eCRF_vector_modulation'],
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
                    'regularization_strength': 0.5
                    },
                'p_t': {
                    'regularization_type': 'l1',
                    'regularization_strength': 0.5
                    },
            }
        },
    },
    {
        'layers': ['pool'],
        'weights': [None],
        'names': ['pool2'],
        'filter_size': [None]
    }
]
