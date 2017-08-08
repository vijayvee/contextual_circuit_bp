import os


class experiments():
    """Class for experiments."""
    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)

    def globals(self):
        """Globals."""
        return {
            'batch_size': 32,
            'data_augmentations': [None],
            'epochs': 100,
            'shuffle': True
        }

    def add_globals(self, exp):
        for k, v in self.globals():
            exp[k] = v
        return exp

    def one_layer_conv_mlp(self):
        """Each key in experiment_dict must be manually added to the schema."""
        model_foler = 'one_layer_conv_mlp'
        exp = {
            'experiment_name': [model_foler],
            'lr': [1e-3],  # np.logspace(-5, -2, 4, base=10),
            'loss_function': ['cce'],
            'optimizer': ['adam'],
            'wd_type': [None, 'l1', 'l2'],
            'wd_penalty': [0.005],
            'model_struct': [
                os.path.join(model_foler, 'divisive'),
                os.path.join(model_foler, 'batch'),
                os.path.join(model_foler, 'layer'),
                os.path.join(model_foler, 'lrn'),
                os.path.join(model_foler, 'contextual'),
            ],
            'dataset': ['mnist', 'cifar']
        }
        exp = self.add_globals(exp)
        return exp
