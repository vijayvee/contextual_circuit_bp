

layer_structure = [
    {
        'layers': ['conv'],
        'weights': [64],
        'names': ['conv1_1'],
        'filter_size': [9],
        'normalization': ['contextual'],
        'normalization_target': ['post'],
        'wd_target': 'pre',
        'wd_type': None,
    },
    {
        'layers': ['pool'],
        'weights': [None],
        'names': ['pool1'],
        'filter_size': [None]
    }
]
