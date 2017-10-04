

layer_structure = [
    {
        'layers': ['conv'],
        'weights': [64],
        'names': ['conv1_1'],
        'activation': ['relu'],
        'activation_target': ['post'],
        'filter_size': [3],
    },
    {
        'layers': ['res'],
        'weights': [64, 64],
        'names': ['res1_2'],
        'aux': {
            'activation': 'relu',
            'normalization': 'batch'
        },
        'filter_size': [3],
    },
    {
        'layers': ['res'],
        'weights': [64, 64],
        'names': ['res1_3'],
        'aux': {
            'activation': 'relu',
            'normalization': 'batch'
        },
        'filter_size': [3],
    },
    {
        'layers': ['pool'],
        'weights': [None],
        'names': ['pool1'],
    },
    {
        'layers': ['conv'],
        'weights': [64],
        'names': ['conv2_1'],
        'filter_size': [3],
        'normalization': ['contextual_alt_learned_transition_learned_connectivity_vector_modulation'],
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
        'weights': [256],
        'names': ['fc3'],
        'flatten': [True],
        'flatten_target': ['pre'],
    },
]
