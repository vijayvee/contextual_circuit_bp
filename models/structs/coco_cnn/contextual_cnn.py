

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
        'activation': ['relu'],
        'activation_target': ['post'],
        'filter_size': [3],
        'normalization': ['contextual_alt_learned_transition_learned_connectivity_vector_modulation'],
        'normalization_target': ['pre'],
        'normalization_aux': {
            'timesteps': 5,
            'regularization_targets': {  # Modulate sparsity
                'q_t': {
                   'regularization_type': 'l1',
                   'regularization_strength': 0.01
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
        'layers': ['res'],
        'weights': [64, 64],
        'names': ['res2_2'],
        'aux': {
            'activation': 'relu',
            'normalization': 'batch'
        },
        'filter_size': [3],
    },
    {
        'layers': ['res'],
        'weights': [64, 64],
        'names': ['res2_3'],
        'aux': {
            'activation': 'relu',
            'normalization': 'batch'
        },
        'filter_size': [3],
    },
    {
        'layers': ['pool'],
        'weights': [None],
        'names': ['pool2'],
    },

    {
        'layers': ['conv'],
        'weights': [128],
        'names': ['conv3_1'],
        'activation': ['relu'],
        'activation_target': ['post'],
        'filter_size': [3],
    },
    {
        'layers': ['res'],
        'weights': [128, 128],
        'names': ['res3_2'],
        'aux': {
            'activation': 'relu',
            'normalization': 'batch'
        },
        'filter_size': [3],
    },
    {
        'layers': ['res'],
        'weights': [128, 128],
        'names': ['res3_3'],
        'aux': {
            'activation': 'relu',
            'normalization': 'batch'
        },
        'filter_size': [3],
    },
    {
        'layers': ['pool'],
        'weights': [None],
        'names': ['pool3'],
    },
    {
        'layers': ['fc'],
        'weights': [512],
        'names': ['fc4'],
        'flatten': [True],
        'flatten_target': ['pre'],
    },
]