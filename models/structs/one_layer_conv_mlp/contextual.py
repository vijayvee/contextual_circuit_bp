

layer_structure = [
    {
        'layers': ['conv'],
        'weights': [64],
        'names': ['conv1_1'],
        'filter_size': [3],
        'normalization': ['contextual'],
        'normalization_target': ['post'],
        'normalization_aux': {
            'timesteps': 5,
            'regularization_': 'l1',
            'regulatization_stregth': 1e-4,
        },
        'activation': ['relu'],
        'activation_target': ['post'],
        'wd_target': ['pre'],
        'wd_type': [None],
    },
    {
        'layers': ['pool'],
        'weights': [None],
        'names': ['pool1'],
        'filter_size': [None]
    }
]
