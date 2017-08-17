import os
import numpy as np


class experiments():
    """Class for experiments."""
    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)

    def globals(self):
        """Globals."""
        return {
            'batch_size': 32,  # Train/val batch size.
            'data_augmentations': [None],  # Random_crop, etc.
            'epochs': 4,
            'shuffle': True,  # Shuffle data.
            'validation_iters': 250,  # How often to evaluate validation.
            'num_validation_evals': 100,  # How many validation batches.
            'top_n_validation': 0,  # Set to 0 to save all checkpoints.
            'early_stop': False  # Stop training if the loss stops improving.
        }

    def add_globals(self, exp):
        for k, v in self.globals().iteritems():
            exp[k] = v
        return exp

    def two_layer_conv_mlp(self):
        """Each key in experiment_dict must be manually added to the schema."""
        model_folder = 'two_layer_conv_mlp'
        exp = {
            'experiment_name': [model_folder],
            'lr': [1e-4],
            'loss_function': ['cce'],
            'optimizer': ['adam'],
            'wd_type': [None],  # [None, 'l1', 'l2'],
            'wd_penalty': [0.005],
            'model_struct': [
                # os.path.join(model_folder, 'divisive'),
                # os.path.join(model_folder, 'batch'),
                # os.path.join(model_folder, 'layer'),
                # os.path.join(model_folder, 'lrn'),
                os.path.join(model_folder, 'contextual'),
                os.path.join(model_folder, 'contextual_rnn'),
                # os.path.join(model_folder, 'contextual_selu'),
                # os.path.join(model_folder, 'contextual_rnn_selu'),
            ],
            'dataset': ['cifar_10']
        }
        return self.add_globals(exp)  # Add globals to the experiment

    def test_conv(self):
        """Each key in experiment_dict must be manually added to the schema."""
        model_folder = 'test_conv'
        exp = {
            'experiment_name': [model_folder],
            'lr': list(np.logspace(-5, -1, 5, base=10)),
            'loss_function': ['cce'],
            'optimizer': ['adam'],
            'wd_type': [None],
            'wd_penalty': [None],
            'model_struct': [
                os.path.join(model_folder, 'test'),
            ],
            'dataset': ['mnist', 'cifar_10']
        }
        return self.add_globals(exp)  # Add globals to the experiment

    def test_fc(self):
        """Each key in experiment_dict must be manually added to the schema."""
        model_folder = 'test_fc'
        exp = {
            'experiment_name': [model_folder],
            'lr': list(np.logspace(-5, -1, 5, base=10)),
            'loss_function': ['cce'],
            'optimizer': ['sgd'],
            'wd_type': [None],
            'wd_penalty': [None],
            'model_struct': [
                os.path.join(model_folder, 'test'),
            ],
            'dataset': ['mnist', 'cifar_10']
        }
        return self.add_globals(exp)  # Add globals to the experiment

    def test_res(self):
        """Each key in experiment_dict must be manually added to the schema."""
        model_folder = 'test_res'
        exp = {
            'experiment_name': [model_folder],
            'lr': list(np.logspace(-5, -1, 5, base=10)),
            'loss_function': ['cce'],
            'optimizer': ['adam'],
            'wd_type': [None],
            'wd_penalty': [None],
            'model_struct': [
                os.path.join(model_folder, 'test'),
            ],
            'dataset': ['mnist', 'cifar_10']
        }
        return self.add_globals(exp)  # Add globals to the experiment
