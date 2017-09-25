"""2D convolutional model for Allen data."""

layer_structure = [
    {
        'layers': ['pass'],
        'names': ['contextual'],
        'hardcoded_erfs': {
            'SRF': 1,
            'CRF_excitation': 1, 
            'CRF_inhibition': 1,
            'SSN': 9,
            'SSF': 29
        },
        'normalization': ['batch'],
        'normalization_target': ['pre'],
    }
]

output_structure = [
    {
        'layers': ['gather'],
        'aux': {
            'h': 25,
            'w': 25
        },  # Output size
        'names': ['gather'],
    }
]
