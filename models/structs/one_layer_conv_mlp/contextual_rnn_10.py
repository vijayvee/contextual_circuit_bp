

layer_structure = [
    {
        'layers': ['conv'],
        'weights': [64],
        'names': ['conv1_1'],
        'filter_size': [3],
        'normalization': ['contextual_rnn'],
        'normalization_target': ['post'],
        'normalization_aux': {
            'timesteps': 10,
            'regularization_type': 'l1',
            'regularization_strength': 1e-4,
        },
        'activation': ['relu'],
        'activation_target': ['post'],
        'wd_target': ['pre'],
        'wd_type': [None],
        'wd_strength': [1e-5]
    },
    {
        'layers': ['pool'],
        'weights': [None],
        'names': ['pool1'],
        'filter_size': [None]
    },
]
