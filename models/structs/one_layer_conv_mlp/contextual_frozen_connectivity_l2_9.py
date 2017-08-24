

layer_structure = [
    {
        'layers': ['conv'],
        'weights': [64],
        'names': ['conv1_1'],
        'filter_size': [3],
        'normalization': ['contextual_frozen_connectivity'],
        'normalization_target': ['post'],
        'normalization_aux': {
            'timesteps': 9,
            'regularization_type': 'l2',
            'regularization_strength': 1e-5,
        },
    },
    {
        'layers': ['pool'],
        'weights': [None],
        'names': ['pool1'],
        'filter_size': [None]
    }
]
