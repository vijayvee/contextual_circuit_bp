

layer_structure = [
    {
        'layers': ['conv'],
        'weights': [64],
        'names': ['conv1_1'],
        'filter_size': [3],
        'normalization': ['contextual_rnn'],
        'normalization_target': ['post'],
        'normalization_aux': {
            'timesteps': 4,
            'regularization_type': 'l2',
            'regularization_strength': 1e-3,
        },
        # 'activation': ['relu'],
        # 'activation_target': ['post'],
    },
    {
        'layers': ['pool'],
        'weights': [None],
        'names': ['pool1'],
        'filter_size': [None]
    },
]
