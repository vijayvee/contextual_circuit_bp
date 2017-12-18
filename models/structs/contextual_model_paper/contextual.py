"""2D convolutional model for Allen data."""

layer_structure = [
    {
        'layers': ['_pass'],
        'names': ['contextual'],
        'hardcoded_erfs': {
            'SRF': 5,
            'CRF_excitation': 5,
            'CRF_inhibition': 5,
            'SSN': 10,
            'SSF': 27.5
        },
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
    },
    {
        'layers': ['fc'],
        'weights': [1],
        'names': ['fc1'],
    }
]
