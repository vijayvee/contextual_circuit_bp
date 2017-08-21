

layer_structure = [
    {
        'layers': ['conv'],
        'weights': [64],
        'names': ['conv1_1'],
        'filter_size': [3],
        'normalization': ['contextual_rnn'],
        'normalization_target': ['post'],
        'normalization_aux': {
            'timesteps': 5,
        },
        # 'activation': ['relu'],
        # 'activation_target': ['post'],
        'regularization_target': ['post'],
        'regularization_type': ['l1'],
        'regularization_strength': [1e-3],
        'regularization_activities_or_weights': 'activities',
    },
    {
        'layers': ['pool'],
        'weights': [None],
        'names': ['pool1'],
        'filter_size': [None]
    },
]
