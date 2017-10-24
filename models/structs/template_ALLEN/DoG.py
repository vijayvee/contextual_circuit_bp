"""DoG model for Allen data."""

layer_structure = [
    {
        'layers': ['DoG'],
        'weights': [9],
        'names': ['dog1'],
        'filter_size': [5],
        'activation': ['selu'],
        'activation_target': ['post'],
        'dropout': [0.5],
        'dropout_target': ['post']
    },
]

output_structure = [
    {
        'layers': ['sparse_pool'],
        'weights': ['OUTPUT_SIZE'],  # Output size
        'aux': {
            'h': 'H_PIX',
            'w': 'W_PIX',
            'k': 'SIGMA'
        },
        'names': ['sp2'],
    }
]
