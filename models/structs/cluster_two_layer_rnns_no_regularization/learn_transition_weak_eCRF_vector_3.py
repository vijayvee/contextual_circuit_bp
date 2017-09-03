

layer_structure = [
    {
        'layers': ['conv'],
        'weights': [64],
        'names': ['conv1_1'],
        'filter_size': [3],
        'normalization': ['contextual_frozen_connectivity_learned_transition_weak_eCRF_vector_modulation'],
        'normalization_target': ['post'],
        'normalization_aux': {
            'timesteps': 3,
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
                    'regularization_type': 'l1',
                    'regularization_strength': 0.1
                    },
            }
        },
    },
    {
        'layers': ['conv'],
        'weights': [64],
        'names': ['conv2_1'],
        'filter_size': [3],
    },
    {
        'layers': ['pool'],
        'weights': [None],
        'names': ['pool2'],
        'filter_size': [None]
    }
]
