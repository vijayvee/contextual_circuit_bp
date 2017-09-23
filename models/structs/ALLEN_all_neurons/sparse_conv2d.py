"""2D convolutional model for Allen data."""

layer_structure = [
    {
        'layers': ['conv'],
        'weights': [64],
        'names': ['conv1_1'],
        'filter_size': [7],
        'activation': ['logistic'],
        'activation_target': ['post']
    },
    {
        'layers': ['pool'],
        'weights': [None],
        'names': ['pool1'],
        'filter_size': [None]
    }
]

output_structure = [
    {
        'layers': ['sparse_pool'],
        'weights': [1],  # Output size
        'names': ['sp2'],
    }
]

