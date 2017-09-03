

layer_structure = [
    {
        'layers': ['conv'],
        'weights': [64],
        'names': ['conv1_1'],
        'filter_size': [3],
        'normalization': ['contextual_frozen_connectivity_learned_transition_vector_weak_eCRF_vector_modulation'],
        'normalization_target': ['post'],
        'normalization_aux': {
            'timesteps': 5,
        },
    },
    {
        'layers': ['pool'],
        'weights': [None],
        'names': ['pool1'],
        'filter_size': [None]
    }
]
