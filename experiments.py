"""Class to specify all DNN experiments."""
import os
import numpy as np


class experiments():
    """Class for experiments."""

    def __getitem__(self, name):
        """Method for addressing class methods."""
        return getattr(self, name)

    def __contains__(self, name):
        """Method for checking class contents."""
        return hasattr(self, name)

    def globals(self):
        """Global variables for all experiments."""
        return {
            'batch_size': 64,  # Train/val batch size.
            'data_augmentations': [
                [
                    'random_crop',
                    'left_right'
                ]
            ],  # TODO: document all data augmentations.
            'epochs': 200,
            'shuffle': True,  # Shuffle data.
            'validation_iters': 5000,  # How often to evaluate validation.
            'num_validation_evals': 100,  # How many validation batches.
            'top_n_validation': 0,  # Set to 0 to save all checkpoints.
            'early_stop': False,  # Stop training if the loss stops improving.
            'save_weights': False,  # Save model weights on validation evals.
            'optimizer_constraints': None  # A {var name: bound} dictionary.
        }

    def add_globals(self, exp):
        """Add attributes to this class."""
        for k, v in self.globals().iteritems():
            exp[k] = v
        return exp

    def perceptual_iq_hp_optimization(self):
        """Each key in experiment_dict must be manually added to the schema.

        If using grid-search -- put hps in lists.
        If using hp-optim, do not use lists except for domains.
        """
        model_folder = 'perceptual_iq_hp_optimization'
        exp = {
            'experiment_name': model_folder,
            'hp_optim': 'gpyopt',
            'hp_multiple': 10,
            'lr': 1e-4,
            'lr_domain': [1e-1, 1e-5],
            'loss_function': None,  # Leave as None to use dataset default
            'optimizer': 'adam',
            'regularization_type': None,  # [None, 'l1', 'l2'],
            'regularization_strength': 1e-5,
            'regularization_strength_domain': [1e-1, 1e-7],
            # 'timesteps': True,
            'model_struct': [
                os.path.join(model_folder, 'divisive_1l'),
                os.path.join(model_folder, 'layer_1l'),
                os.path.join(model_folder, 'divisive_2l'),
                os.path.join(model_folder, 'layer_2l'),
            ],
            'dataset': 'ChallengeDB_release'
        }
        return self.add_globals(exp)  # Add globals to the experiment

    def two_layer_conv_mlp(self):
        """Each key in experiment_dict must be manually added to the schema."""
        model_folder = 'two_layer_conv_mlp'
        exp = {
            'experiment_name': [model_folder],
            'lr': [1e-5],
            'loss_function': ['cce'],
            'optimizer': ['adam'],
            'regularization_type': [None],  # [None, 'l1', 'l2'],
            'regularization_strength': [0.005],
            'model_struct': [
                os.path.join(model_folder, 'divisive'),
                os.path.join(model_folder, 'batch'),
                os.path.join(model_folder, 'layer'),
                os.path.join(model_folder, 'lrn'),
                os.path.join(model_folder, 'contextual'),
                os.path.join(model_folder, 'contextual_rnn'),
                os.path.join(model_folder, 'contextual_rnn_no_relu'),
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
            'regularization_type': [None],
            'regularization_strength': [None],
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
            'regularization_type': [None],
            'regularization_strength': [None],
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
            'regularization_type': [None],
            'regularization_strength': [None],
            'model_struct': [
                os.path.join(model_folder, 'test'),
            ],
            'dataset': ['mnist', 'cifar_10']
        }
        return self.add_globals(exp)  # Add globals to the experiment

    def coco_cnn(self):
        """Each key in experiment_dict must be manually added to the schema."""
        model_folder = 'coco_cnn'
        exp = {
            'experiment_name': [model_folder],
            'lr': [3e-4],
            'loss_function': ['sigmoid'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'cnn'),
                os.path.join(model_folder, 'contextual_cnn')
            ],
            'dataset': ['coco_2014']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['epochs'] = 200
        exp['batch_size'] = 16  # Train/val batch size.
        exp['save_weights'] = True
        return exp

    def challengedb_cnns(self):
        """Each key in experiment_dict must be manually added to the schema."""
        model_folder = 'challengedb_cnns'
        exp = {
            'experiment_name': [model_folder],
            'lr': [1e-3],
            'loss_function': ['l2'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'cnn'),
                os.path.join(model_folder, 'contextual_cnn')
            ],
            'dataset': ['ChallengeDB_release']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['epochs'] = 200
        exp['batch_size'] = 8  # Train/val batch size.
        exp['save_weights'] = True
        return exp

    def contextual_model_paper(self):
        """Each key in experiment_dict must be manually added to the schema."""
        model_folder = 'contextual_model_paper'
        exp = {
            'experiment_name': [model_folder],
            'lr': [5e-3],
            'loss_function': ['l2'],
            'optimizer': ['adam'],
            'q_t': [1e-3, 1e-1],
            'p_t': [1e-2, 1e-1, 1],
            't_t': [1e-2, 1e-1, 1],
            'timesteps': [5, 10],
            'model_struct': [
                # os.path.join(model_folder, 'divisive_paper_rfs'),
                os.path.join(model_folder, 'contextual_paper_rfs'),
                # os.path.join(model_folder, 'divisive'),
                # os.path.join(model_folder, 'contextual'),
            ],
            'dataset': ['contextual_model_multi_stimuli']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = [[None]]
        exp['epochs'] = 1000
        exp['save_weights'] = True
        return exp

    def ALLEN_selected_cells_103(self):
        """Each key in experiment_dict must be manually added to the schema."""
        model_folder = 'ALLEN_selected_cells_103'
        exp = {
            'experiment_name': [model_folder],
            'lr': [1e-3],
            'loss_function': ['l2'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'adj_norm_conv2d'),
                os.path.join(model_folder, 'scalar_norm_conv2d'),
                os.path.join(model_folder, 'vector_norm_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['ALLEN_selected_cells_103']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = [['resize']]
        exp['epochs'] = 100
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def ALLEN_random_cells_103(self):
        """Each key in experiment_dict must be manually added to the schema."""
        model_folder = 'ALLEN_random_cells_103'
        exp = {
            'experiment_name': [model_folder],
            'lr': [1e-3],
            'loss_function': ['l2'],
            'optimizer': ['hessian'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['ALLEN_random_cells_103']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = [['resize']]
        exp['epochs'] = 100
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['optimizer_constraints'] = {'dog1': (0, 10)}
        return exp

    def ALLEN_selected_cells_1(self):
        """Each key in experiment_dict must be manually added to the schema."""
        model_folder = 'ALLEN_selected_cells_1'
        exp = {
            'experiment_name': [model_folder],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            # 'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            # 'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['ALLEN_selected_cells_1']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = [['resize']]
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def ALLEN_ss_cells_1_movies(self):
        """Each key in experiment_dict must be manually added to the schema."""
        model_folder = 'ALLEN_ss_cells_1_movies'
        exp = {
            'experiment_name': [model_folder],
            'lr': [1e-3],
            'loss_function': ['l2'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['ALLEN_ss_cells_1_movies']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = [['resize']]
        exp['epochs'] = 100
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 32  # Train/val batch size.
        return exp


    def MULTIALLEN_DNRdcpbFcNLDNHia(self):
        """MULTIALLEN_DNRdcpbFcNLDNHia multi-experiment creation."""
        model_folder = 'MULTIALLEN_DNRdcpbFcNLDNHia'
        exp = {
            'experiment_name': ['MULTIALLEN_DNRdcpbFcNLDNHia'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_DNRdcpbFcNLDNHia']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_rLnfEgeZlXmljGSq(self):
        """MULTIALLEN_rLnfEgeZlXmljGSq multi-experiment creation."""
        model_folder = 'MULTIALLEN_rLnfEgeZlXmljGSq'
        exp = {
            'experiment_name': ['MULTIALLEN_rLnfEgeZlXmljGSq'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_rLnfEgeZlXmljGSq']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_duAMEUsyebESITwT(self):
        """MULTIALLEN_duAMEUsyebESITwT multi-experiment creation."""
        model_folder = 'MULTIALLEN_duAMEUsyebESITwT'
        exp = {
            'experiment_name': ['MULTIALLEN_duAMEUsyebESITwT'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_duAMEUsyebESITwT']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_IgFRlGkiZvgoHCqd(self):
        """MULTIALLEN_IgFRlGkiZvgoHCqd multi-experiment creation."""
        model_folder = 'MULTIALLEN_IgFRlGkiZvgoHCqd'
        exp = {
            'experiment_name': ['MULTIALLEN_IgFRlGkiZvgoHCqd'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_IgFRlGkiZvgoHCqd']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_vSmqNivLIjrhKNDs(self):
        """MULTIALLEN_vSmqNivLIjrhKNDs multi-experiment creation."""
        model_folder = 'MULTIALLEN_vSmqNivLIjrhKNDs'
        exp = {
            'experiment_name': ['MULTIALLEN_vSmqNivLIjrhKNDs'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_vSmqNivLIjrhKNDs']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ZAdxgRhFlmJhBIPh(self):
        """MULTIALLEN_ZAdxgRhFlmJhBIPh multi-experiment creation."""
        model_folder = 'MULTIALLEN_ZAdxgRhFlmJhBIPh'
        exp = {
            'experiment_name': ['MULTIALLEN_ZAdxgRhFlmJhBIPh'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ZAdxgRhFlmJhBIPh']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_YUclsyCzBJVNEPgR(self):
        """MULTIALLEN_YUclsyCzBJVNEPgR multi-experiment creation."""
        model_folder = 'MULTIALLEN_YUclsyCzBJVNEPgR'
        exp = {
            'experiment_name': ['MULTIALLEN_YUclsyCzBJVNEPgR'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_YUclsyCzBJVNEPgR']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_uyXpVwDUaFfEdTwP(self):
        """MULTIALLEN_uyXpVwDUaFfEdTwP multi-experiment creation."""
        model_folder = 'MULTIALLEN_uyXpVwDUaFfEdTwP'
        exp = {
            'experiment_name': ['MULTIALLEN_uyXpVwDUaFfEdTwP'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_uyXpVwDUaFfEdTwP']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_lMyCZtPlMtHesmAr(self):
        """MULTIALLEN_lMyCZtPlMtHesmAr multi-experiment creation."""
        model_folder = 'MULTIALLEN_lMyCZtPlMtHesmAr'
        exp = {
            'experiment_name': ['MULTIALLEN_lMyCZtPlMtHesmAr'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_lMyCZtPlMtHesmAr']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_OUQXrsAaqcDVjnXC(self):
        """MULTIALLEN_OUQXrsAaqcDVjnXC multi-experiment creation."""
        model_folder = 'MULTIALLEN_OUQXrsAaqcDVjnXC'
        exp = {
            'experiment_name': ['MULTIALLEN_OUQXrsAaqcDVjnXC'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_OUQXrsAaqcDVjnXC']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_UpLUvnwltrqzwXtA(self):
        """MULTIALLEN_UpLUvnwltrqzwXtA multi-experiment creation."""
        model_folder = 'MULTIALLEN_UpLUvnwltrqzwXtA'
        exp = {
            'experiment_name': ['MULTIALLEN_UpLUvnwltrqzwXtA'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_UpLUvnwltrqzwXtA']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_PthZFUbgHKmVaVSL(self):
        """MULTIALLEN_PthZFUbgHKmVaVSL multi-experiment creation."""
        model_folder = 'MULTIALLEN_PthZFUbgHKmVaVSL'
        exp = {
            'experiment_name': ['MULTIALLEN_PthZFUbgHKmVaVSL'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_PthZFUbgHKmVaVSL']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_GPDqbjlogJnHCYrF(self):
        """MULTIALLEN_GPDqbjlogJnHCYrF multi-experiment creation."""
        model_folder = 'MULTIALLEN_GPDqbjlogJnHCYrF'
        exp = {
            'experiment_name': ['MULTIALLEN_GPDqbjlogJnHCYrF'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_GPDqbjlogJnHCYrF']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_LGBWspAmVaNxpZqG(self):
        """MULTIALLEN_LGBWspAmVaNxpZqG multi-experiment creation."""
        model_folder = 'MULTIALLEN_LGBWspAmVaNxpZqG'
        exp = {
            'experiment_name': ['MULTIALLEN_LGBWspAmVaNxpZqG'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_LGBWspAmVaNxpZqG']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_VbOcJnJyjdWOZcCO(self):
        """MULTIALLEN_VbOcJnJyjdWOZcCO multi-experiment creation."""
        model_folder = 'MULTIALLEN_VbOcJnJyjdWOZcCO'
        exp = {
            'experiment_name': ['MULTIALLEN_VbOcJnJyjdWOZcCO'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_VbOcJnJyjdWOZcCO']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_IhuMYMMjTEoqpFYc(self):
        """MULTIALLEN_IhuMYMMjTEoqpFYc multi-experiment creation."""
        model_folder = 'MULTIALLEN_IhuMYMMjTEoqpFYc'
        exp = {
            'experiment_name': ['MULTIALLEN_IhuMYMMjTEoqpFYc'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_IhuMYMMjTEoqpFYc']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_UHTTnmWqRCbMpFPm(self):
        """MULTIALLEN_UHTTnmWqRCbMpFPm multi-experiment creation."""
        model_folder = 'MULTIALLEN_UHTTnmWqRCbMpFPm'
        exp = {
            'experiment_name': ['MULTIALLEN_UHTTnmWqRCbMpFPm'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_UHTTnmWqRCbMpFPm']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_GLMENHexyrDXAsAp(self):
        """MULTIALLEN_GLMENHexyrDXAsAp multi-experiment creation."""
        model_folder = 'MULTIALLEN_GLMENHexyrDXAsAp'
        exp = {
            'experiment_name': ['MULTIALLEN_GLMENHexyrDXAsAp'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_GLMENHexyrDXAsAp']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_coWCmyojaLsFowVU(self):
        """MULTIALLEN_coWCmyojaLsFowVU multi-experiment creation."""
        model_folder = 'MULTIALLEN_coWCmyojaLsFowVU'
        exp = {
            'experiment_name': ['MULTIALLEN_coWCmyojaLsFowVU'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_coWCmyojaLsFowVU']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_vUkeodjmXeiXxttP(self):
        """MULTIALLEN_vUkeodjmXeiXxttP multi-experiment creation."""
        model_folder = 'MULTIALLEN_vUkeodjmXeiXxttP'
        exp = {
            'experiment_name': ['MULTIALLEN_vUkeodjmXeiXxttP'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_vUkeodjmXeiXxttP']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_svKZlLseszJwrqlN(self):
        """MULTIALLEN_svKZlLseszJwrqlN multi-experiment creation."""
        model_folder = 'MULTIALLEN_svKZlLseszJwrqlN'
        exp = {
            'experiment_name': ['MULTIALLEN_svKZlLseszJwrqlN'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_svKZlLseszJwrqlN']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ONjgmUWDmISHKMPN(self):
        """MULTIALLEN_ONjgmUWDmISHKMPN multi-experiment creation."""
        model_folder = 'MULTIALLEN_ONjgmUWDmISHKMPN'
        exp = {
            'experiment_name': ['MULTIALLEN_ONjgmUWDmISHKMPN'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ONjgmUWDmISHKMPN']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_IRTgulckCelhTdzE(self):
        """MULTIALLEN_IRTgulckCelhTdzE multi-experiment creation."""
        model_folder = 'MULTIALLEN_IRTgulckCelhTdzE'
        exp = {
            'experiment_name': ['MULTIALLEN_IRTgulckCelhTdzE'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_IRTgulckCelhTdzE']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_nQLhFtIEhEblhwai(self):
        """MULTIALLEN_nQLhFtIEhEblhwai multi-experiment creation."""
        model_folder = 'MULTIALLEN_nQLhFtIEhEblhwai'
        exp = {
            'experiment_name': ['MULTIALLEN_nQLhFtIEhEblhwai'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_nQLhFtIEhEblhwai']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_SvjOhJTIOqjHZRcj(self):
        """MULTIALLEN_SvjOhJTIOqjHZRcj multi-experiment creation."""
        model_folder = 'MULTIALLEN_SvjOhJTIOqjHZRcj'
        exp = {
            'experiment_name': ['MULTIALLEN_SvjOhJTIOqjHZRcj'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_SvjOhJTIOqjHZRcj']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_xODaeSsmfFjTUnYS(self):
        """MULTIALLEN_xODaeSsmfFjTUnYS multi-experiment creation."""
        model_folder = 'MULTIALLEN_xODaeSsmfFjTUnYS'
        exp = {
            'experiment_name': ['MULTIALLEN_xODaeSsmfFjTUnYS'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_xODaeSsmfFjTUnYS']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_VxkSgBxfecvkQUhv(self):
        """MULTIALLEN_VxkSgBxfecvkQUhv multi-experiment creation."""
        model_folder = 'MULTIALLEN_VxkSgBxfecvkQUhv'
        exp = {
            'experiment_name': ['MULTIALLEN_VxkSgBxfecvkQUhv'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_VxkSgBxfecvkQUhv']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_TrnHykCutIVNIHyV(self):
        """MULTIALLEN_TrnHykCutIVNIHyV multi-experiment creation."""
        model_folder = 'MULTIALLEN_TrnHykCutIVNIHyV'
        exp = {
            'experiment_name': ['MULTIALLEN_TrnHykCutIVNIHyV'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_TrnHykCutIVNIHyV']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_kZRdmtPwGjfeFbpN(self):
        """MULTIALLEN_kZRdmtPwGjfeFbpN multi-experiment creation."""
        model_folder = 'MULTIALLEN_kZRdmtPwGjfeFbpN'
        exp = {
            'experiment_name': ['MULTIALLEN_kZRdmtPwGjfeFbpN'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_kZRdmtPwGjfeFbpN']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_kqOwsGHvsthozmbq(self):
        """MULTIALLEN_kqOwsGHvsthozmbq multi-experiment creation."""
        model_folder = 'MULTIALLEN_kqOwsGHvsthozmbq'
        exp = {
            'experiment_name': ['MULTIALLEN_kqOwsGHvsthozmbq'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_kqOwsGHvsthozmbq']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_jbphELfgciFcuOhD(self):
        """MULTIALLEN_jbphELfgciFcuOhD multi-experiment creation."""
        model_folder = 'MULTIALLEN_jbphELfgciFcuOhD'
        exp = {
            'experiment_name': ['MULTIALLEN_jbphELfgciFcuOhD'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_jbphELfgciFcuOhD']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_MGVkbeANjQhpcODR(self):
        """MULTIALLEN_MGVkbeANjQhpcODR multi-experiment creation."""
        model_folder = 'MULTIALLEN_MGVkbeANjQhpcODR'
        exp = {
            'experiment_name': ['MULTIALLEN_MGVkbeANjQhpcODR'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_MGVkbeANjQhpcODR']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_UUqCzfcHYwppjrVi(self):
        """MULTIALLEN_UUqCzfcHYwppjrVi multi-experiment creation."""
        model_folder = 'MULTIALLEN_UUqCzfcHYwppjrVi'
        exp = {
            'experiment_name': ['MULTIALLEN_UUqCzfcHYwppjrVi'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_UUqCzfcHYwppjrVi']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_zceZasfUtJbMiPun(self):
        """MULTIALLEN_zceZasfUtJbMiPun multi-experiment creation."""
        model_folder = 'MULTIALLEN_zceZasfUtJbMiPun'
        exp = {
            'experiment_name': ['MULTIALLEN_zceZasfUtJbMiPun'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_zceZasfUtJbMiPun']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_fAOiVqAOtkvPZXMD(self):
        """MULTIALLEN_fAOiVqAOtkvPZXMD multi-experiment creation."""
        model_folder = 'MULTIALLEN_fAOiVqAOtkvPZXMD'
        exp = {
            'experiment_name': ['MULTIALLEN_fAOiVqAOtkvPZXMD'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_fAOiVqAOtkvPZXMD']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_cIeKrCPganlDEuPg(self):
        """MULTIALLEN_cIeKrCPganlDEuPg multi-experiment creation."""
        model_folder = 'MULTIALLEN_cIeKrCPganlDEuPg'
        exp = {
            'experiment_name': ['MULTIALLEN_cIeKrCPganlDEuPg'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_cIeKrCPganlDEuPg']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_OOSQHcEVXPkbFCQZ(self):
        """MULTIALLEN_OOSQHcEVXPkbFCQZ multi-experiment creation."""
        model_folder = 'MULTIALLEN_OOSQHcEVXPkbFCQZ'
        exp = {
            'experiment_name': ['MULTIALLEN_OOSQHcEVXPkbFCQZ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_OOSQHcEVXPkbFCQZ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ooCXAdEuTWdyjgoV(self):
        """MULTIALLEN_ooCXAdEuTWdyjgoV multi-experiment creation."""
        model_folder = 'MULTIALLEN_ooCXAdEuTWdyjgoV'
        exp = {
            'experiment_name': ['MULTIALLEN_ooCXAdEuTWdyjgoV'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ooCXAdEuTWdyjgoV']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_WPBMIlIhHuDEOcUo(self):
        """MULTIALLEN_WPBMIlIhHuDEOcUo multi-experiment creation."""
        model_folder = 'MULTIALLEN_WPBMIlIhHuDEOcUo'
        exp = {
            'experiment_name': ['MULTIALLEN_WPBMIlIhHuDEOcUo'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_WPBMIlIhHuDEOcUo']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_eukNXHUxXVPRbmNd(self):
        """MULTIALLEN_eukNXHUxXVPRbmNd multi-experiment creation."""
        model_folder = 'MULTIALLEN_eukNXHUxXVPRbmNd'
        exp = {
            'experiment_name': ['MULTIALLEN_eukNXHUxXVPRbmNd'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_eukNXHUxXVPRbmNd']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ysJAqlcmWzVWjpdJ(self):
        """MULTIALLEN_ysJAqlcmWzVWjpdJ multi-experiment creation."""
        model_folder = 'MULTIALLEN_ysJAqlcmWzVWjpdJ'
        exp = {
            'experiment_name': ['MULTIALLEN_ysJAqlcmWzVWjpdJ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ysJAqlcmWzVWjpdJ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_irXTNpEzvcKXNaKd(self):
        """MULTIALLEN_irXTNpEzvcKXNaKd multi-experiment creation."""
        model_folder = 'MULTIALLEN_irXTNpEzvcKXNaKd'
        exp = {
            'experiment_name': ['MULTIALLEN_irXTNpEzvcKXNaKd'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_irXTNpEzvcKXNaKd']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_dHWxFDOPvCHKBpdh(self):
        """MULTIALLEN_dHWxFDOPvCHKBpdh multi-experiment creation."""
        model_folder = 'MULTIALLEN_dHWxFDOPvCHKBpdh'
        exp = {
            'experiment_name': ['MULTIALLEN_dHWxFDOPvCHKBpdh'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_dHWxFDOPvCHKBpdh']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_MKRcywqCJcrIxmIM(self):
        """MULTIALLEN_MKRcywqCJcrIxmIM multi-experiment creation."""
        model_folder = 'MULTIALLEN_MKRcywqCJcrIxmIM'
        exp = {
            'experiment_name': ['MULTIALLEN_MKRcywqCJcrIxmIM'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_MKRcywqCJcrIxmIM']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_qEAklUmrYPqcqejc(self):
        """MULTIALLEN_qEAklUmrYPqcqejc multi-experiment creation."""
        model_folder = 'MULTIALLEN_qEAklUmrYPqcqejc'
        exp = {
            'experiment_name': ['MULTIALLEN_qEAklUmrYPqcqejc'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_qEAklUmrYPqcqejc']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_rflLulNbeVguOeDi(self):
        """MULTIALLEN_rflLulNbeVguOeDi multi-experiment creation."""
        model_folder = 'MULTIALLEN_rflLulNbeVguOeDi'
        exp = {
            'experiment_name': ['MULTIALLEN_rflLulNbeVguOeDi'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_rflLulNbeVguOeDi']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_siCKYeuAZAnWfPUd(self):
        """MULTIALLEN_siCKYeuAZAnWfPUd multi-experiment creation."""
        model_folder = 'MULTIALLEN_siCKYeuAZAnWfPUd'
        exp = {
            'experiment_name': ['MULTIALLEN_siCKYeuAZAnWfPUd'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_siCKYeuAZAnWfPUd']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_NKrXaWNFrdFEbBmy(self):
        """MULTIALLEN_NKrXaWNFrdFEbBmy multi-experiment creation."""
        model_folder = 'MULTIALLEN_NKrXaWNFrdFEbBmy'
        exp = {
            'experiment_name': ['MULTIALLEN_NKrXaWNFrdFEbBmy'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_NKrXaWNFrdFEbBmy']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_HJQirgnYjHchjEBW(self):
        """MULTIALLEN_HJQirgnYjHchjEBW multi-experiment creation."""
        model_folder = 'MULTIALLEN_HJQirgnYjHchjEBW'
        exp = {
            'experiment_name': ['MULTIALLEN_HJQirgnYjHchjEBW'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_HJQirgnYjHchjEBW']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_CASxoPcYDwyzEDCg(self):
        """MULTIALLEN_CASxoPcYDwyzEDCg multi-experiment creation."""
        model_folder = 'MULTIALLEN_CASxoPcYDwyzEDCg'
        exp = {
            'experiment_name': ['MULTIALLEN_CASxoPcYDwyzEDCg'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_CASxoPcYDwyzEDCg']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_uwdzWmDOGKSXvWDU(self):
        """MULTIALLEN_uwdzWmDOGKSXvWDU multi-experiment creation."""
        model_folder = 'MULTIALLEN_uwdzWmDOGKSXvWDU'
        exp = {
            'experiment_name': ['MULTIALLEN_uwdzWmDOGKSXvWDU'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_uwdzWmDOGKSXvWDU']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_UvlPccxrEXGJlSvT(self):
        """MULTIALLEN_UvlPccxrEXGJlSvT multi-experiment creation."""
        model_folder = 'MULTIALLEN_UvlPccxrEXGJlSvT'
        exp = {
            'experiment_name': ['MULTIALLEN_UvlPccxrEXGJlSvT'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_UvlPccxrEXGJlSvT']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_thIQYvDlFoOEFPBd(self):
        """MULTIALLEN_thIQYvDlFoOEFPBd multi-experiment creation."""
        model_folder = 'MULTIALLEN_thIQYvDlFoOEFPBd'
        exp = {
            'experiment_name': ['MULTIALLEN_thIQYvDlFoOEFPBd'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_thIQYvDlFoOEFPBd']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_wZftIDuNjTWFSAny(self):
        """MULTIALLEN_wZftIDuNjTWFSAny multi-experiment creation."""
        model_folder = 'MULTIALLEN_wZftIDuNjTWFSAny'
        exp = {
            'experiment_name': ['MULTIALLEN_wZftIDuNjTWFSAny'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_wZftIDuNjTWFSAny']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_dSMkSdyfcjcraUQR(self):
        """MULTIALLEN_dSMkSdyfcjcraUQR multi-experiment creation."""
        model_folder = 'MULTIALLEN_dSMkSdyfcjcraUQR'
        exp = {
            'experiment_name': ['MULTIALLEN_dSMkSdyfcjcraUQR'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_dSMkSdyfcjcraUQR']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_mbWzABCjKdCABxLB(self):
        """MULTIALLEN_mbWzABCjKdCABxLB multi-experiment creation."""
        model_folder = 'MULTIALLEN_mbWzABCjKdCABxLB'
        exp = {
            'experiment_name': ['MULTIALLEN_mbWzABCjKdCABxLB'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_mbWzABCjKdCABxLB']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_gfsQYpHmbUbbYaEi(self):
        """MULTIALLEN_gfsQYpHmbUbbYaEi multi-experiment creation."""
        model_folder = 'MULTIALLEN_gfsQYpHmbUbbYaEi'
        exp = {
            'experiment_name': ['MULTIALLEN_gfsQYpHmbUbbYaEi'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_gfsQYpHmbUbbYaEi']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_VhCYZirSKOHXVwcD(self):
        """MULTIALLEN_VhCYZirSKOHXVwcD multi-experiment creation."""
        model_folder = 'MULTIALLEN_VhCYZirSKOHXVwcD'
        exp = {
            'experiment_name': ['MULTIALLEN_VhCYZirSKOHXVwcD'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_VhCYZirSKOHXVwcD']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_zmQbUcETKerYZlsF(self):
        """MULTIALLEN_zmQbUcETKerYZlsF multi-experiment creation."""
        model_folder = 'MULTIALLEN_zmQbUcETKerYZlsF'
        exp = {
            'experiment_name': ['MULTIALLEN_zmQbUcETKerYZlsF'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_zmQbUcETKerYZlsF']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_vLIsuNDfbNMOQVgT(self):
        """MULTIALLEN_vLIsuNDfbNMOQVgT multi-experiment creation."""
        model_folder = 'MULTIALLEN_vLIsuNDfbNMOQVgT'
        exp = {
            'experiment_name': ['MULTIALLEN_vLIsuNDfbNMOQVgT'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_vLIsuNDfbNMOQVgT']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_zFEnueBiKtvIyFFl(self):
        """MULTIALLEN_zFEnueBiKtvIyFFl multi-experiment creation."""
        model_folder = 'MULTIALLEN_zFEnueBiKtvIyFFl'
        exp = {
            'experiment_name': ['MULTIALLEN_zFEnueBiKtvIyFFl'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_zFEnueBiKtvIyFFl']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_LeASqjKhJjVkpcfQ(self):
        """MULTIALLEN_LeASqjKhJjVkpcfQ multi-experiment creation."""
        model_folder = 'MULTIALLEN_LeASqjKhJjVkpcfQ'
        exp = {
            'experiment_name': ['MULTIALLEN_LeASqjKhJjVkpcfQ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_LeASqjKhJjVkpcfQ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_vYyZQfWDeZAMAnWy(self):
        """MULTIALLEN_vYyZQfWDeZAMAnWy multi-experiment creation."""
        model_folder = 'MULTIALLEN_vYyZQfWDeZAMAnWy'
        exp = {
            'experiment_name': ['MULTIALLEN_vYyZQfWDeZAMAnWy'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_vYyZQfWDeZAMAnWy']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_fqcjPZemJlKJRAil(self):
        """MULTIALLEN_fqcjPZemJlKJRAil multi-experiment creation."""
        model_folder = 'MULTIALLEN_fqcjPZemJlKJRAil'
        exp = {
            'experiment_name': ['MULTIALLEN_fqcjPZemJlKJRAil'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_fqcjPZemJlKJRAil']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ueupBpeEreSAizzf(self):
        """MULTIALLEN_ueupBpeEreSAizzf multi-experiment creation."""
        model_folder = 'MULTIALLEN_ueupBpeEreSAizzf'
        exp = {
            'experiment_name': ['MULTIALLEN_ueupBpeEreSAizzf'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ueupBpeEreSAizzf']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_LLckRrmczKqrmBqi(self):
        """MULTIALLEN_LLckRrmczKqrmBqi multi-experiment creation."""
        model_folder = 'MULTIALLEN_LLckRrmczKqrmBqi'
        exp = {
            'experiment_name': ['MULTIALLEN_LLckRrmczKqrmBqi'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_LLckRrmczKqrmBqi']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_pAzkhPhQcLPlXbXQ(self):
        """MULTIALLEN_pAzkhPhQcLPlXbXQ multi-experiment creation."""
        model_folder = 'MULTIALLEN_pAzkhPhQcLPlXbXQ'
        exp = {
            'experiment_name': ['MULTIALLEN_pAzkhPhQcLPlXbXQ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_pAzkhPhQcLPlXbXQ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_kJtCeZbJnkQAIdyI(self):
        """MULTIALLEN_kJtCeZbJnkQAIdyI multi-experiment creation."""
        model_folder = 'MULTIALLEN_kJtCeZbJnkQAIdyI'
        exp = {
            'experiment_name': ['MULTIALLEN_kJtCeZbJnkQAIdyI'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_kJtCeZbJnkQAIdyI']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_uKcvhOerYEtwkwzG(self):
        """MULTIALLEN_uKcvhOerYEtwkwzG multi-experiment creation."""
        model_folder = 'MULTIALLEN_uKcvhOerYEtwkwzG'
        exp = {
            'experiment_name': ['MULTIALLEN_uKcvhOerYEtwkwzG'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_uKcvhOerYEtwkwzG']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_eEMMRfZvhbMvmroM(self):
        """MULTIALLEN_eEMMRfZvhbMvmroM multi-experiment creation."""
        model_folder = 'MULTIALLEN_eEMMRfZvhbMvmroM'
        exp = {
            'experiment_name': ['MULTIALLEN_eEMMRfZvhbMvmroM'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_eEMMRfZvhbMvmroM']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_mtLrEbVSJhhhehum(self):
        """MULTIALLEN_mtLrEbVSJhhhehum multi-experiment creation."""
        model_folder = 'MULTIALLEN_mtLrEbVSJhhhehum'
        exp = {
            'experiment_name': ['MULTIALLEN_mtLrEbVSJhhhehum'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_mtLrEbVSJhhhehum']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_TgwRxgHRUcIgsdCp(self):
        """MULTIALLEN_TgwRxgHRUcIgsdCp multi-experiment creation."""
        model_folder = 'MULTIALLEN_TgwRxgHRUcIgsdCp'
        exp = {
            'experiment_name': ['MULTIALLEN_TgwRxgHRUcIgsdCp'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_TgwRxgHRUcIgsdCp']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_CZWFAnKgHGqDPtRM(self):
        """MULTIALLEN_CZWFAnKgHGqDPtRM multi-experiment creation."""
        model_folder = 'MULTIALLEN_CZWFAnKgHGqDPtRM'
        exp = {
            'experiment_name': ['MULTIALLEN_CZWFAnKgHGqDPtRM'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_CZWFAnKgHGqDPtRM']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_aWmshIOimJrnEzDo(self):
        """MULTIALLEN_aWmshIOimJrnEzDo multi-experiment creation."""
        model_folder = 'MULTIALLEN_aWmshIOimJrnEzDo'
        exp = {
            'experiment_name': ['MULTIALLEN_aWmshIOimJrnEzDo'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_aWmshIOimJrnEzDo']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ybMFcOhMPiWPABZS(self):
        """MULTIALLEN_ybMFcOhMPiWPABZS multi-experiment creation."""
        model_folder = 'MULTIALLEN_ybMFcOhMPiWPABZS'
        exp = {
            'experiment_name': ['MULTIALLEN_ybMFcOhMPiWPABZS'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ybMFcOhMPiWPABZS']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_OgPFNOsKucyxFvSe(self):
        """MULTIALLEN_OgPFNOsKucyxFvSe multi-experiment creation."""
        model_folder = 'MULTIALLEN_OgPFNOsKucyxFvSe'
        exp = {
            'experiment_name': ['MULTIALLEN_OgPFNOsKucyxFvSe'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_OgPFNOsKucyxFvSe']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ugBsAtyuBYbePbdy(self):
        """MULTIALLEN_ugBsAtyuBYbePbdy multi-experiment creation."""
        model_folder = 'MULTIALLEN_ugBsAtyuBYbePbdy'
        exp = {
            'experiment_name': ['MULTIALLEN_ugBsAtyuBYbePbdy'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ugBsAtyuBYbePbdy']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_xRkzJivpByOkJZRM(self):
        """MULTIALLEN_xRkzJivpByOkJZRM multi-experiment creation."""
        model_folder = 'MULTIALLEN_xRkzJivpByOkJZRM'
        exp = {
            'experiment_name': ['MULTIALLEN_xRkzJivpByOkJZRM'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_xRkzJivpByOkJZRM']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_wUnbQkhaeIBrHPNx(self):
        """MULTIALLEN_wUnbQkhaeIBrHPNx multi-experiment creation."""
        model_folder = 'MULTIALLEN_wUnbQkhaeIBrHPNx'
        exp = {
            'experiment_name': ['MULTIALLEN_wUnbQkhaeIBrHPNx'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_wUnbQkhaeIBrHPNx']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_rIDENkIDTfSBmAJQ(self):
        """MULTIALLEN_rIDENkIDTfSBmAJQ multi-experiment creation."""
        model_folder = 'MULTIALLEN_rIDENkIDTfSBmAJQ'
        exp = {
            'experiment_name': ['MULTIALLEN_rIDENkIDTfSBmAJQ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_rIDENkIDTfSBmAJQ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_rYcvOMAXaHLuiCWb(self):
        """MULTIALLEN_rYcvOMAXaHLuiCWb multi-experiment creation."""
        model_folder = 'MULTIALLEN_rYcvOMAXaHLuiCWb'
        exp = {
            'experiment_name': ['MULTIALLEN_rYcvOMAXaHLuiCWb'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_rYcvOMAXaHLuiCWb']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_QYsTrhUZAbpDJyji(self):
        """MULTIALLEN_QYsTrhUZAbpDJyji multi-experiment creation."""
        model_folder = 'MULTIALLEN_QYsTrhUZAbpDJyji'
        exp = {
            'experiment_name': ['MULTIALLEN_QYsTrhUZAbpDJyji'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_QYsTrhUZAbpDJyji']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_zKEZRZzXMAyCSMqj(self):
        """MULTIALLEN_zKEZRZzXMAyCSMqj multi-experiment creation."""
        model_folder = 'MULTIALLEN_zKEZRZzXMAyCSMqj'
        exp = {
            'experiment_name': ['MULTIALLEN_zKEZRZzXMAyCSMqj'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_zKEZRZzXMAyCSMqj']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_scFupPNcXIRgBFMo(self):
        """MULTIALLEN_scFupPNcXIRgBFMo multi-experiment creation."""
        model_folder = 'MULTIALLEN_scFupPNcXIRgBFMo'
        exp = {
            'experiment_name': ['MULTIALLEN_scFupPNcXIRgBFMo'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_scFupPNcXIRgBFMo']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_SCeXGbOYeSBvlkqR(self):
        """MULTIALLEN_SCeXGbOYeSBvlkqR multi-experiment creation."""
        model_folder = 'MULTIALLEN_SCeXGbOYeSBvlkqR'
        exp = {
            'experiment_name': ['MULTIALLEN_SCeXGbOYeSBvlkqR'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_SCeXGbOYeSBvlkqR']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_KBFSMNzwabcZysqA(self):
        """MULTIALLEN_KBFSMNzwabcZysqA multi-experiment creation."""
        model_folder = 'MULTIALLEN_KBFSMNzwabcZysqA'
        exp = {
            'experiment_name': ['MULTIALLEN_KBFSMNzwabcZysqA'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_KBFSMNzwabcZysqA']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_XnciuniypzTyvlLk(self):
        """MULTIALLEN_XnciuniypzTyvlLk multi-experiment creation."""
        model_folder = 'MULTIALLEN_XnciuniypzTyvlLk'
        exp = {
            'experiment_name': ['MULTIALLEN_XnciuniypzTyvlLk'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_XnciuniypzTyvlLk']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_bbUSDPiONJFVGqhC(self):
        """MULTIALLEN_bbUSDPiONJFVGqhC multi-experiment creation."""
        model_folder = 'MULTIALLEN_bbUSDPiONJFVGqhC'
        exp = {
            'experiment_name': ['MULTIALLEN_bbUSDPiONJFVGqhC'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_bbUSDPiONJFVGqhC']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_rdOowUkNeRCexZMS(self):
        """MULTIALLEN_rdOowUkNeRCexZMS multi-experiment creation."""
        model_folder = 'MULTIALLEN_rdOowUkNeRCexZMS'
        exp = {
            'experiment_name': ['MULTIALLEN_rdOowUkNeRCexZMS'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_rdOowUkNeRCexZMS']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ncXhtuZxGAoXMCzq(self):
        """MULTIALLEN_ncXhtuZxGAoXMCzq multi-experiment creation."""
        model_folder = 'MULTIALLEN_ncXhtuZxGAoXMCzq'
        exp = {
            'experiment_name': ['MULTIALLEN_ncXhtuZxGAoXMCzq'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ncXhtuZxGAoXMCzq']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_HiiNlPjfANgsiTiC(self):
        """MULTIALLEN_HiiNlPjfANgsiTiC multi-experiment creation."""
        model_folder = 'MULTIALLEN_HiiNlPjfANgsiTiC'
        exp = {
            'experiment_name': ['MULTIALLEN_HiiNlPjfANgsiTiC'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_HiiNlPjfANgsiTiC']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_tdDchKTwdjbLsBMU(self):
        """MULTIALLEN_tdDchKTwdjbLsBMU multi-experiment creation."""
        model_folder = 'MULTIALLEN_tdDchKTwdjbLsBMU'
        exp = {
            'experiment_name': ['MULTIALLEN_tdDchKTwdjbLsBMU'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_tdDchKTwdjbLsBMU']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_uCsbSMPslIbjoxLa(self):
        """MULTIALLEN_uCsbSMPslIbjoxLa multi-experiment creation."""
        model_folder = 'MULTIALLEN_uCsbSMPslIbjoxLa'
        exp = {
            'experiment_name': ['MULTIALLEN_uCsbSMPslIbjoxLa'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_uCsbSMPslIbjoxLa']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_MTJyMvjimgsSLvsX(self):
        """MULTIALLEN_MTJyMvjimgsSLvsX multi-experiment creation."""
        model_folder = 'MULTIALLEN_MTJyMvjimgsSLvsX'
        exp = {
            'experiment_name': ['MULTIALLEN_MTJyMvjimgsSLvsX'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_MTJyMvjimgsSLvsX']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_BIreUiSjmPDJJpsA(self):
        """MULTIALLEN_BIreUiSjmPDJJpsA multi-experiment creation."""
        model_folder = 'MULTIALLEN_BIreUiSjmPDJJpsA'
        exp = {
            'experiment_name': ['MULTIALLEN_BIreUiSjmPDJJpsA'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_BIreUiSjmPDJJpsA']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_GHgSGeRSIMoHmrQb(self):
        """MULTIALLEN_GHgSGeRSIMoHmrQb multi-experiment creation."""
        model_folder = 'MULTIALLEN_GHgSGeRSIMoHmrQb'
        exp = {
            'experiment_name': ['MULTIALLEN_GHgSGeRSIMoHmrQb'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_GHgSGeRSIMoHmrQb']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_wRbgPlyBPCWddtEc(self):
        """MULTIALLEN_wRbgPlyBPCWddtEc multi-experiment creation."""
        model_folder = 'MULTIALLEN_wRbgPlyBPCWddtEc'
        exp = {
            'experiment_name': ['MULTIALLEN_wRbgPlyBPCWddtEc'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_wRbgPlyBPCWddtEc']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_OXOKXIghHExCynQu(self):
        """MULTIALLEN_OXOKXIghHExCynQu multi-experiment creation."""
        model_folder = 'MULTIALLEN_OXOKXIghHExCynQu'
        exp = {
            'experiment_name': ['MULTIALLEN_OXOKXIghHExCynQu'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_OXOKXIghHExCynQu']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_KmebHpNYNifMDrAX(self):
        """MULTIALLEN_KmebHpNYNifMDrAX multi-experiment creation."""
        model_folder = 'MULTIALLEN_KmebHpNYNifMDrAX'
        exp = {
            'experiment_name': ['MULTIALLEN_KmebHpNYNifMDrAX'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_KmebHpNYNifMDrAX']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_tBDwVFJUspAQcKrt(self):
        """MULTIALLEN_tBDwVFJUspAQcKrt multi-experiment creation."""
        model_folder = 'MULTIALLEN_tBDwVFJUspAQcKrt'
        exp = {
            'experiment_name': ['MULTIALLEN_tBDwVFJUspAQcKrt'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_tBDwVFJUspAQcKrt']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_XRNtCIKXbXvNgTIH(self):
        """MULTIALLEN_XRNtCIKXbXvNgTIH multi-experiment creation."""
        model_folder = 'MULTIALLEN_XRNtCIKXbXvNgTIH'
        exp = {
            'experiment_name': ['MULTIALLEN_XRNtCIKXbXvNgTIH'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_XRNtCIKXbXvNgTIH']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_WODcwgNAnaqqKnaB(self):
        """MULTIALLEN_WODcwgNAnaqqKnaB multi-experiment creation."""
        model_folder = 'MULTIALLEN_WODcwgNAnaqqKnaB'
        exp = {
            'experiment_name': ['MULTIALLEN_WODcwgNAnaqqKnaB'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_WODcwgNAnaqqKnaB']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_roZyzRBVqQZfPpuL(self):
        """MULTIALLEN_roZyzRBVqQZfPpuL multi-experiment creation."""
        model_folder = 'MULTIALLEN_roZyzRBVqQZfPpuL'
        exp = {
            'experiment_name': ['MULTIALLEN_roZyzRBVqQZfPpuL'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_roZyzRBVqQZfPpuL']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_GSmoHJmZvTlhZMmw(self):
        """MULTIALLEN_GSmoHJmZvTlhZMmw multi-experiment creation."""
        model_folder = 'MULTIALLEN_GSmoHJmZvTlhZMmw'
        exp = {
            'experiment_name': ['MULTIALLEN_GSmoHJmZvTlhZMmw'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_GSmoHJmZvTlhZMmw']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_pXTWDcVvGqYGtrNh(self):
        """MULTIALLEN_pXTWDcVvGqYGtrNh multi-experiment creation."""
        model_folder = 'MULTIALLEN_pXTWDcVvGqYGtrNh'
        exp = {
            'experiment_name': ['MULTIALLEN_pXTWDcVvGqYGtrNh'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_pXTWDcVvGqYGtrNh']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_fQMoNYQAHzYEOOin(self):
        """MULTIALLEN_fQMoNYQAHzYEOOin multi-experiment creation."""
        model_folder = 'MULTIALLEN_fQMoNYQAHzYEOOin'
        exp = {
            'experiment_name': ['MULTIALLEN_fQMoNYQAHzYEOOin'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_fQMoNYQAHzYEOOin']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_kgXAHHGzYNDbJZSo(self):
        """MULTIALLEN_kgXAHHGzYNDbJZSo multi-experiment creation."""
        model_folder = 'MULTIALLEN_kgXAHHGzYNDbJZSo'
        exp = {
            'experiment_name': ['MULTIALLEN_kgXAHHGzYNDbJZSo'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_kgXAHHGzYNDbJZSo']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ueBPcafkNedtOsqW(self):
        """MULTIALLEN_ueBPcafkNedtOsqW multi-experiment creation."""
        model_folder = 'MULTIALLEN_ueBPcafkNedtOsqW'
        exp = {
            'experiment_name': ['MULTIALLEN_ueBPcafkNedtOsqW'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ueBPcafkNedtOsqW']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_bSVZkvICjvaWOPFu(self):
        """MULTIALLEN_bSVZkvICjvaWOPFu multi-experiment creation."""
        model_folder = 'MULTIALLEN_bSVZkvICjvaWOPFu'
        exp = {
            'experiment_name': ['MULTIALLEN_bSVZkvICjvaWOPFu'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_bSVZkvICjvaWOPFu']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_kHEbOYxJxJFNhvLd(self):
        """MULTIALLEN_kHEbOYxJxJFNhvLd multi-experiment creation."""
        model_folder = 'MULTIALLEN_kHEbOYxJxJFNhvLd'
        exp = {
            'experiment_name': ['MULTIALLEN_kHEbOYxJxJFNhvLd'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_kHEbOYxJxJFNhvLd']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_MoLqnDEGgJaXgfHL(self):
        """MULTIALLEN_MoLqnDEGgJaXgfHL multi-experiment creation."""
        model_folder = 'MULTIALLEN_MoLqnDEGgJaXgfHL'
        exp = {
            'experiment_name': ['MULTIALLEN_MoLqnDEGgJaXgfHL'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_MoLqnDEGgJaXgfHL']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_JtqDdkSiiOiYiVgk(self):
        """MULTIALLEN_JtqDdkSiiOiYiVgk multi-experiment creation."""
        model_folder = 'MULTIALLEN_JtqDdkSiiOiYiVgk'
        exp = {
            'experiment_name': ['MULTIALLEN_JtqDdkSiiOiYiVgk'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_JtqDdkSiiOiYiVgk']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_AYjGNfICjXfzDZIi(self):
        """MULTIALLEN_AYjGNfICjXfzDZIi multi-experiment creation."""
        model_folder = 'MULTIALLEN_AYjGNfICjXfzDZIi'
        exp = {
            'experiment_name': ['MULTIALLEN_AYjGNfICjXfzDZIi'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_AYjGNfICjXfzDZIi']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_lhUEgvNkOKWYPCpB(self):
        """MULTIALLEN_lhUEgvNkOKWYPCpB multi-experiment creation."""
        model_folder = 'MULTIALLEN_lhUEgvNkOKWYPCpB'
        exp = {
            'experiment_name': ['MULTIALLEN_lhUEgvNkOKWYPCpB'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_lhUEgvNkOKWYPCpB']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_kxXNYzDlcifoYHpN(self):
        """MULTIALLEN_kxXNYzDlcifoYHpN multi-experiment creation."""
        model_folder = 'MULTIALLEN_kxXNYzDlcifoYHpN'
        exp = {
            'experiment_name': ['MULTIALLEN_kxXNYzDlcifoYHpN'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_kxXNYzDlcifoYHpN']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ualOykOMZtfwtzAL(self):
        """MULTIALLEN_ualOykOMZtfwtzAL multi-experiment creation."""
        model_folder = 'MULTIALLEN_ualOykOMZtfwtzAL'
        exp = {
            'experiment_name': ['MULTIALLEN_ualOykOMZtfwtzAL'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ualOykOMZtfwtzAL']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_RVHfpypjtMQkAWdz(self):
        """MULTIALLEN_RVHfpypjtMQkAWdz multi-experiment creation."""
        model_folder = 'MULTIALLEN_RVHfpypjtMQkAWdz'
        exp = {
            'experiment_name': ['MULTIALLEN_RVHfpypjtMQkAWdz'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_RVHfpypjtMQkAWdz']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_pSwCnLXxEVbWnjTB(self):
        """MULTIALLEN_pSwCnLXxEVbWnjTB multi-experiment creation."""
        model_folder = 'MULTIALLEN_pSwCnLXxEVbWnjTB'
        exp = {
            'experiment_name': ['MULTIALLEN_pSwCnLXxEVbWnjTB'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_pSwCnLXxEVbWnjTB']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_oQzvmEUWPgzvQJUZ(self):
        """MULTIALLEN_oQzvmEUWPgzvQJUZ multi-experiment creation."""
        model_folder = 'MULTIALLEN_oQzvmEUWPgzvQJUZ'
        exp = {
            'experiment_name': ['MULTIALLEN_oQzvmEUWPgzvQJUZ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_oQzvmEUWPgzvQJUZ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_eilHUWTUIvzIMjfe(self):
        """MULTIALLEN_eilHUWTUIvzIMjfe multi-experiment creation."""
        model_folder = 'MULTIALLEN_eilHUWTUIvzIMjfe'
        exp = {
            'experiment_name': ['MULTIALLEN_eilHUWTUIvzIMjfe'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_eilHUWTUIvzIMjfe']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_pbluQvdcrtwECgZh(self):
        """MULTIALLEN_pbluQvdcrtwECgZh multi-experiment creation."""
        model_folder = 'MULTIALLEN_pbluQvdcrtwECgZh'
        exp = {
            'experiment_name': ['MULTIALLEN_pbluQvdcrtwECgZh'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_pbluQvdcrtwECgZh']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_oBoNDkoQFNRnYRUA(self):
        """MULTIALLEN_oBoNDkoQFNRnYRUA multi-experiment creation."""
        model_folder = 'MULTIALLEN_oBoNDkoQFNRnYRUA'
        exp = {
            'experiment_name': ['MULTIALLEN_oBoNDkoQFNRnYRUA'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_oBoNDkoQFNRnYRUA']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_YtfzrcsDrDVQzjDP(self):
        """MULTIALLEN_YtfzrcsDrDVQzjDP multi-experiment creation."""
        model_folder = 'MULTIALLEN_YtfzrcsDrDVQzjDP'
        exp = {
            'experiment_name': ['MULTIALLEN_YtfzrcsDrDVQzjDP'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_YtfzrcsDrDVQzjDP']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_LlBfSguIhfUCFYrW(self):
        """MULTIALLEN_LlBfSguIhfUCFYrW multi-experiment creation."""
        model_folder = 'MULTIALLEN_LlBfSguIhfUCFYrW'
        exp = {
            'experiment_name': ['MULTIALLEN_LlBfSguIhfUCFYrW'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_LlBfSguIhfUCFYrW']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_RDRlNrbYJYRhofwq(self):
        """MULTIALLEN_RDRlNrbYJYRhofwq multi-experiment creation."""
        model_folder = 'MULTIALLEN_RDRlNrbYJYRhofwq'
        exp = {
            'experiment_name': ['MULTIALLEN_RDRlNrbYJYRhofwq'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_RDRlNrbYJYRhofwq']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_BNfVDVzKGhVMnTOm(self):
        """MULTIALLEN_BNfVDVzKGhVMnTOm multi-experiment creation."""
        model_folder = 'MULTIALLEN_BNfVDVzKGhVMnTOm'
        exp = {
            'experiment_name': ['MULTIALLEN_BNfVDVzKGhVMnTOm'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_BNfVDVzKGhVMnTOm']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_VmHrLrUQtJmwIGca(self):
        """MULTIALLEN_VmHrLrUQtJmwIGca multi-experiment creation."""
        model_folder = 'MULTIALLEN_VmHrLrUQtJmwIGca'
        exp = {
            'experiment_name': ['MULTIALLEN_VmHrLrUQtJmwIGca'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_VmHrLrUQtJmwIGca']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_wQSejuBcFRdrkYsb(self):
        """MULTIALLEN_wQSejuBcFRdrkYsb multi-experiment creation."""
        model_folder = 'MULTIALLEN_wQSejuBcFRdrkYsb'
        exp = {
            'experiment_name': ['MULTIALLEN_wQSejuBcFRdrkYsb'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_wQSejuBcFRdrkYsb']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_bxlbxhPnoAXrUGFN(self):
        """MULTIALLEN_bxlbxhPnoAXrUGFN multi-experiment creation."""
        model_folder = 'MULTIALLEN_bxlbxhPnoAXrUGFN'
        exp = {
            'experiment_name': ['MULTIALLEN_bxlbxhPnoAXrUGFN'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_bxlbxhPnoAXrUGFN']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_MFZfafIuLMPOeigy(self):
        """MULTIALLEN_MFZfafIuLMPOeigy multi-experiment creation."""
        model_folder = 'MULTIALLEN_MFZfafIuLMPOeigy'
        exp = {
            'experiment_name': ['MULTIALLEN_MFZfafIuLMPOeigy'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_MFZfafIuLMPOeigy']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_fDPNnCJssDscyhBv(self):
        """MULTIALLEN_fDPNnCJssDscyhBv multi-experiment creation."""
        model_folder = 'MULTIALLEN_fDPNnCJssDscyhBv'
        exp = {
            'experiment_name': ['MULTIALLEN_fDPNnCJssDscyhBv'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_fDPNnCJssDscyhBv']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ORTVrfcdRyNxqsMS(self):
        """MULTIALLEN_ORTVrfcdRyNxqsMS multi-experiment creation."""
        model_folder = 'MULTIALLEN_ORTVrfcdRyNxqsMS'
        exp = {
            'experiment_name': ['MULTIALLEN_ORTVrfcdRyNxqsMS'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ORTVrfcdRyNxqsMS']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_wabJsZmKtTBvZMpY(self):
        """MULTIALLEN_wabJsZmKtTBvZMpY multi-experiment creation."""
        model_folder = 'MULTIALLEN_wabJsZmKtTBvZMpY'
        exp = {
            'experiment_name': ['MULTIALLEN_wabJsZmKtTBvZMpY'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_wabJsZmKtTBvZMpY']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_OGaYdMpqEMZLuhtv(self):
        """MULTIALLEN_OGaYdMpqEMZLuhtv multi-experiment creation."""
        model_folder = 'MULTIALLEN_OGaYdMpqEMZLuhtv'
        exp = {
            'experiment_name': ['MULTIALLEN_OGaYdMpqEMZLuhtv'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_OGaYdMpqEMZLuhtv']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_mlcvdEAbMAExqkFV(self):
        """MULTIALLEN_mlcvdEAbMAExqkFV multi-experiment creation."""
        model_folder = 'MULTIALLEN_mlcvdEAbMAExqkFV'
        exp = {
            'experiment_name': ['MULTIALLEN_mlcvdEAbMAExqkFV'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_mlcvdEAbMAExqkFV']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_GovTVtKIOwgnjeGd(self):
        """MULTIALLEN_GovTVtKIOwgnjeGd multi-experiment creation."""
        model_folder = 'MULTIALLEN_GovTVtKIOwgnjeGd'
        exp = {
            'experiment_name': ['MULTIALLEN_GovTVtKIOwgnjeGd'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_GovTVtKIOwgnjeGd']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_HjFYRccgGAuczIsu(self):
        """MULTIALLEN_HjFYRccgGAuczIsu multi-experiment creation."""
        model_folder = 'MULTIALLEN_HjFYRccgGAuczIsu'
        exp = {
            'experiment_name': ['MULTIALLEN_HjFYRccgGAuczIsu'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_HjFYRccgGAuczIsu']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_DqNoLMDlIdRKpSJq(self):
        """MULTIALLEN_DqNoLMDlIdRKpSJq multi-experiment creation."""
        model_folder = 'MULTIALLEN_DqNoLMDlIdRKpSJq'
        exp = {
            'experiment_name': ['MULTIALLEN_DqNoLMDlIdRKpSJq'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_DqNoLMDlIdRKpSJq']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_IMrFUSOHImvUzzLC(self):
        """MULTIALLEN_IMrFUSOHImvUzzLC multi-experiment creation."""
        model_folder = 'MULTIALLEN_IMrFUSOHImvUzzLC'
        exp = {
            'experiment_name': ['MULTIALLEN_IMrFUSOHImvUzzLC'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_IMrFUSOHImvUzzLC']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_yHCWdVDMQoYICnkx(self):
        """MULTIALLEN_yHCWdVDMQoYICnkx multi-experiment creation."""
        model_folder = 'MULTIALLEN_yHCWdVDMQoYICnkx'
        exp = {
            'experiment_name': ['MULTIALLEN_yHCWdVDMQoYICnkx'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_yHCWdVDMQoYICnkx']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_rORIxDomIekMWcyD(self):
        """MULTIALLEN_rORIxDomIekMWcyD multi-experiment creation."""
        model_folder = 'MULTIALLEN_rORIxDomIekMWcyD'
        exp = {
            'experiment_name': ['MULTIALLEN_rORIxDomIekMWcyD'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_rORIxDomIekMWcyD']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_cPrPsaXvdEkHAHcK(self):
        """MULTIALLEN_cPrPsaXvdEkHAHcK multi-experiment creation."""
        model_folder = 'MULTIALLEN_cPrPsaXvdEkHAHcK'
        exp = {
            'experiment_name': ['MULTIALLEN_cPrPsaXvdEkHAHcK'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_cPrPsaXvdEkHAHcK']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_LSywsPydZdzwxwOy(self):
        """MULTIALLEN_LSywsPydZdzwxwOy multi-experiment creation."""
        model_folder = 'MULTIALLEN_LSywsPydZdzwxwOy'
        exp = {
            'experiment_name': ['MULTIALLEN_LSywsPydZdzwxwOy'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_LSywsPydZdzwxwOy']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_rnhUPhEXgUqfMHUo(self):
        """MULTIALLEN_rnhUPhEXgUqfMHUo multi-experiment creation."""
        model_folder = 'MULTIALLEN_rnhUPhEXgUqfMHUo'
        exp = {
            'experiment_name': ['MULTIALLEN_rnhUPhEXgUqfMHUo'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_rnhUPhEXgUqfMHUo']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_gxDxkAlRTnBxekKQ(self):
        """MULTIALLEN_gxDxkAlRTnBxekKQ multi-experiment creation."""
        model_folder = 'MULTIALLEN_gxDxkAlRTnBxekKQ'
        exp = {
            'experiment_name': ['MULTIALLEN_gxDxkAlRTnBxekKQ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_gxDxkAlRTnBxekKQ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_pqtmXZZAICInIoki(self):
        """MULTIALLEN_pqtmXZZAICInIoki multi-experiment creation."""
        model_folder = 'MULTIALLEN_pqtmXZZAICInIoki'
        exp = {
            'experiment_name': ['MULTIALLEN_pqtmXZZAICInIoki'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_pqtmXZZAICInIoki']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_LyxaJXgHOKQYbjxr(self):
        """MULTIALLEN_LyxaJXgHOKQYbjxr multi-experiment creation."""
        model_folder = 'MULTIALLEN_LyxaJXgHOKQYbjxr'
        exp = {
            'experiment_name': ['MULTIALLEN_LyxaJXgHOKQYbjxr'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_LyxaJXgHOKQYbjxr']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_WXQYJVNEpQBxphub(self):
        """MULTIALLEN_WXQYJVNEpQBxphub multi-experiment creation."""
        model_folder = 'MULTIALLEN_WXQYJVNEpQBxphub'
        exp = {
            'experiment_name': ['MULTIALLEN_WXQYJVNEpQBxphub'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_WXQYJVNEpQBxphub']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_RZeuvDVLHuRMoAqu(self):
        """MULTIALLEN_RZeuvDVLHuRMoAqu multi-experiment creation."""
        model_folder = 'MULTIALLEN_RZeuvDVLHuRMoAqu'
        exp = {
            'experiment_name': ['MULTIALLEN_RZeuvDVLHuRMoAqu'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_RZeuvDVLHuRMoAqu']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_dKaDheDPZvrDAlLa(self):
        """MULTIALLEN_dKaDheDPZvrDAlLa multi-experiment creation."""
        model_folder = 'MULTIALLEN_dKaDheDPZvrDAlLa'
        exp = {
            'experiment_name': ['MULTIALLEN_dKaDheDPZvrDAlLa'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_dKaDheDPZvrDAlLa']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_KFlatKBickeiUznY(self):
        """MULTIALLEN_KFlatKBickeiUznY multi-experiment creation."""
        model_folder = 'MULTIALLEN_KFlatKBickeiUznY'
        exp = {
            'experiment_name': ['MULTIALLEN_KFlatKBickeiUznY'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_KFlatKBickeiUznY']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_EDYSsoRSAALZgoyO(self):
        """MULTIALLEN_EDYSsoRSAALZgoyO multi-experiment creation."""
        model_folder = 'MULTIALLEN_EDYSsoRSAALZgoyO'
        exp = {
            'experiment_name': ['MULTIALLEN_EDYSsoRSAALZgoyO'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_EDYSsoRSAALZgoyO']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_nVZxxEGLWEbhAvaL(self):
        """MULTIALLEN_nVZxxEGLWEbhAvaL multi-experiment creation."""
        model_folder = 'MULTIALLEN_nVZxxEGLWEbhAvaL'
        exp = {
            'experiment_name': ['MULTIALLEN_nVZxxEGLWEbhAvaL'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_nVZxxEGLWEbhAvaL']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ZsqLIETfadDHaYna(self):
        """MULTIALLEN_ZsqLIETfadDHaYna multi-experiment creation."""
        model_folder = 'MULTIALLEN_ZsqLIETfadDHaYna'
        exp = {
            'experiment_name': ['MULTIALLEN_ZsqLIETfadDHaYna'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ZsqLIETfadDHaYna']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_MMhNKLavwdQaFMiR(self):
        """MULTIALLEN_MMhNKLavwdQaFMiR multi-experiment creation."""
        model_folder = 'MULTIALLEN_MMhNKLavwdQaFMiR'
        exp = {
            'experiment_name': ['MULTIALLEN_MMhNKLavwdQaFMiR'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_MMhNKLavwdQaFMiR']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_fpEdGjvPAaMxiTYu(self):
        """MULTIALLEN_fpEdGjvPAaMxiTYu multi-experiment creation."""
        model_folder = 'MULTIALLEN_fpEdGjvPAaMxiTYu'
        exp = {
            'experiment_name': ['MULTIALLEN_fpEdGjvPAaMxiTYu'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_fpEdGjvPAaMxiTYu']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_GTMyFHaiiLNySwug(self):
        """MULTIALLEN_GTMyFHaiiLNySwug multi-experiment creation."""
        model_folder = 'MULTIALLEN_GTMyFHaiiLNySwug'
        exp = {
            'experiment_name': ['MULTIALLEN_GTMyFHaiiLNySwug'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_GTMyFHaiiLNySwug']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_wXAMkdRfBYuYNGqO(self):
        """MULTIALLEN_wXAMkdRfBYuYNGqO multi-experiment creation."""
        model_folder = 'MULTIALLEN_wXAMkdRfBYuYNGqO'
        exp = {
            'experiment_name': ['MULTIALLEN_wXAMkdRfBYuYNGqO'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_wXAMkdRfBYuYNGqO']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_zNjwIadGaRWKIHDY(self):
        """MULTIALLEN_zNjwIadGaRWKIHDY multi-experiment creation."""
        model_folder = 'MULTIALLEN_zNjwIadGaRWKIHDY'
        exp = {
            'experiment_name': ['MULTIALLEN_zNjwIadGaRWKIHDY'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_zNjwIadGaRWKIHDY']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_evnyZkNckBJMLcys(self):
        """MULTIALLEN_evnyZkNckBJMLcys multi-experiment creation."""
        model_folder = 'MULTIALLEN_evnyZkNckBJMLcys'
        exp = {
            'experiment_name': ['MULTIALLEN_evnyZkNckBJMLcys'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_evnyZkNckBJMLcys']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_JkWgEWpADSkHjVir(self):
        """MULTIALLEN_JkWgEWpADSkHjVir multi-experiment creation."""
        model_folder = 'MULTIALLEN_JkWgEWpADSkHjVir'
        exp = {
            'experiment_name': ['MULTIALLEN_JkWgEWpADSkHjVir'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_JkWgEWpADSkHjVir']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_VyhRHIYXOZgkKgeW(self):
        """MULTIALLEN_VyhRHIYXOZgkKgeW multi-experiment creation."""
        model_folder = 'MULTIALLEN_VyhRHIYXOZgkKgeW'
        exp = {
            'experiment_name': ['MULTIALLEN_VyhRHIYXOZgkKgeW'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_VyhRHIYXOZgkKgeW']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_THnluCYeyGBoubDg(self):
        """MULTIALLEN_THnluCYeyGBoubDg multi-experiment creation."""
        model_folder = 'MULTIALLEN_THnluCYeyGBoubDg'
        exp = {
            'experiment_name': ['MULTIALLEN_THnluCYeyGBoubDg'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_THnluCYeyGBoubDg']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_qqFbQXyFQOMDNbQb(self):
        """MULTIALLEN_qqFbQXyFQOMDNbQb multi-experiment creation."""
        model_folder = 'MULTIALLEN_qqFbQXyFQOMDNbQb'
        exp = {
            'experiment_name': ['MULTIALLEN_qqFbQXyFQOMDNbQb'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_qqFbQXyFQOMDNbQb']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_knidzhbsEyDdvGkR(self):
        """MULTIALLEN_knidzhbsEyDdvGkR multi-experiment creation."""
        model_folder = 'MULTIALLEN_knidzhbsEyDdvGkR'
        exp = {
            'experiment_name': ['MULTIALLEN_knidzhbsEyDdvGkR'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_knidzhbsEyDdvGkR']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_viyxjTTGhCnYcxqT(self):
        """MULTIALLEN_viyxjTTGhCnYcxqT multi-experiment creation."""
        model_folder = 'MULTIALLEN_viyxjTTGhCnYcxqT'
        exp = {
            'experiment_name': ['MULTIALLEN_viyxjTTGhCnYcxqT'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_viyxjTTGhCnYcxqT']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_yXMOfjYKprUiXBkR(self):
        """MULTIALLEN_yXMOfjYKprUiXBkR multi-experiment creation."""
        model_folder = 'MULTIALLEN_yXMOfjYKprUiXBkR'
        exp = {
            'experiment_name': ['MULTIALLEN_yXMOfjYKprUiXBkR'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_yXMOfjYKprUiXBkR']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_mpEejbynhlxdLgtI(self):
        """MULTIALLEN_mpEejbynhlxdLgtI multi-experiment creation."""
        model_folder = 'MULTIALLEN_mpEejbynhlxdLgtI'
        exp = {
            'experiment_name': ['MULTIALLEN_mpEejbynhlxdLgtI'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_mpEejbynhlxdLgtI']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_KKezSOlseQZlpRhZ(self):
        """MULTIALLEN_KKezSOlseQZlpRhZ multi-experiment creation."""
        model_folder = 'MULTIALLEN_KKezSOlseQZlpRhZ'
        exp = {
            'experiment_name': ['MULTIALLEN_KKezSOlseQZlpRhZ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_KKezSOlseQZlpRhZ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_FnBvgCimzGpskVYm(self):
        """MULTIALLEN_FnBvgCimzGpskVYm multi-experiment creation."""
        model_folder = 'MULTIALLEN_FnBvgCimzGpskVYm'
        exp = {
            'experiment_name': ['MULTIALLEN_FnBvgCimzGpskVYm'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_FnBvgCimzGpskVYm']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_wMnzbqYzQInxtDJh(self):
        """MULTIALLEN_wMnzbqYzQInxtDJh multi-experiment creation."""
        model_folder = 'MULTIALLEN_wMnzbqYzQInxtDJh'
        exp = {
            'experiment_name': ['MULTIALLEN_wMnzbqYzQInxtDJh'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_wMnzbqYzQInxtDJh']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_rRJbogSBnhBgudZj(self):
        """MULTIALLEN_rRJbogSBnhBgudZj multi-experiment creation."""
        model_folder = 'MULTIALLEN_rRJbogSBnhBgudZj'
        exp = {
            'experiment_name': ['MULTIALLEN_rRJbogSBnhBgudZj'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_rRJbogSBnhBgudZj']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_tOUfbfeTIsNpBXwp(self):
        """MULTIALLEN_tOUfbfeTIsNpBXwp multi-experiment creation."""
        model_folder = 'MULTIALLEN_tOUfbfeTIsNpBXwp'
        exp = {
            'experiment_name': ['MULTIALLEN_tOUfbfeTIsNpBXwp'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_tOUfbfeTIsNpBXwp']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ewAZsyoojcwelnVX(self):
        """MULTIALLEN_ewAZsyoojcwelnVX multi-experiment creation."""
        model_folder = 'MULTIALLEN_ewAZsyoojcwelnVX'
        exp = {
            'experiment_name': ['MULTIALLEN_ewAZsyoojcwelnVX'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ewAZsyoojcwelnVX']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_pUOdkRIqQoqEsMfd(self):
        """MULTIALLEN_pUOdkRIqQoqEsMfd multi-experiment creation."""
        model_folder = 'MULTIALLEN_pUOdkRIqQoqEsMfd'
        exp = {
            'experiment_name': ['MULTIALLEN_pUOdkRIqQoqEsMfd'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_pUOdkRIqQoqEsMfd']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_UIDmnAzKulbbKNNI(self):
        """MULTIALLEN_UIDmnAzKulbbKNNI multi-experiment creation."""
        model_folder = 'MULTIALLEN_UIDmnAzKulbbKNNI'
        exp = {
            'experiment_name': ['MULTIALLEN_UIDmnAzKulbbKNNI'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_UIDmnAzKulbbKNNI']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_dzwHrndxpReneNdW(self):
        """MULTIALLEN_dzwHrndxpReneNdW multi-experiment creation."""
        model_folder = 'MULTIALLEN_dzwHrndxpReneNdW'
        exp = {
            'experiment_name': ['MULTIALLEN_dzwHrndxpReneNdW'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_dzwHrndxpReneNdW']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_XbxlQAdTofkOfYFx(self):
        """MULTIALLEN_XbxlQAdTofkOfYFx multi-experiment creation."""
        model_folder = 'MULTIALLEN_XbxlQAdTofkOfYFx'
        exp = {
            'experiment_name': ['MULTIALLEN_XbxlQAdTofkOfYFx'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_XbxlQAdTofkOfYFx']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_XIOVKspzQutphFuz(self):
        """MULTIALLEN_XIOVKspzQutphFuz multi-experiment creation."""
        model_folder = 'MULTIALLEN_XIOVKspzQutphFuz'
        exp = {
            'experiment_name': ['MULTIALLEN_XIOVKspzQutphFuz'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_XIOVKspzQutphFuz']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_SJJsyWgHazOWcaDC(self):
        """MULTIALLEN_SJJsyWgHazOWcaDC multi-experiment creation."""
        model_folder = 'MULTIALLEN_SJJsyWgHazOWcaDC'
        exp = {
            'experiment_name': ['MULTIALLEN_SJJsyWgHazOWcaDC'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_SJJsyWgHazOWcaDC']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_SqfKTXBvbkxKzfic(self):
        """MULTIALLEN_SqfKTXBvbkxKzfic multi-experiment creation."""
        model_folder = 'MULTIALLEN_SqfKTXBvbkxKzfic'
        exp = {
            'experiment_name': ['MULTIALLEN_SqfKTXBvbkxKzfic'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_SqfKTXBvbkxKzfic']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_vBNKdorCAQnSuiCb(self):
        """MULTIALLEN_vBNKdorCAQnSuiCb multi-experiment creation."""
        model_folder = 'MULTIALLEN_vBNKdorCAQnSuiCb'
        exp = {
            'experiment_name': ['MULTIALLEN_vBNKdorCAQnSuiCb'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_vBNKdorCAQnSuiCb']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_MUIfVrBfIbekPjaR(self):
        """MULTIALLEN_MUIfVrBfIbekPjaR multi-experiment creation."""
        model_folder = 'MULTIALLEN_MUIfVrBfIbekPjaR'
        exp = {
            'experiment_name': ['MULTIALLEN_MUIfVrBfIbekPjaR'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_MUIfVrBfIbekPjaR']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ixpTPfeLmCQIbmVY(self):
        """MULTIALLEN_ixpTPfeLmCQIbmVY multi-experiment creation."""
        model_folder = 'MULTIALLEN_ixpTPfeLmCQIbmVY'
        exp = {
            'experiment_name': ['MULTIALLEN_ixpTPfeLmCQIbmVY'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ixpTPfeLmCQIbmVY']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_RsBLwRqUOYghtcMM(self):
        """MULTIALLEN_RsBLwRqUOYghtcMM multi-experiment creation."""
        model_folder = 'MULTIALLEN_RsBLwRqUOYghtcMM'
        exp = {
            'experiment_name': ['MULTIALLEN_RsBLwRqUOYghtcMM'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_RsBLwRqUOYghtcMM']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_BnnnMRqyYnhuFnpT(self):
        """MULTIALLEN_BnnnMRqyYnhuFnpT multi-experiment creation."""
        model_folder = 'MULTIALLEN_BnnnMRqyYnhuFnpT'
        exp = {
            'experiment_name': ['MULTIALLEN_BnnnMRqyYnhuFnpT'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_BnnnMRqyYnhuFnpT']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_RsHCJRecfjuYwlWn(self):
        """MULTIALLEN_RsHCJRecfjuYwlWn multi-experiment creation."""
        model_folder = 'MULTIALLEN_RsHCJRecfjuYwlWn'
        exp = {
            'experiment_name': ['MULTIALLEN_RsHCJRecfjuYwlWn'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_RsHCJRecfjuYwlWn']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ybPDitAuLmchIsKE(self):
        """MULTIALLEN_ybPDitAuLmchIsKE multi-experiment creation."""
        model_folder = 'MULTIALLEN_ybPDitAuLmchIsKE'
        exp = {
            'experiment_name': ['MULTIALLEN_ybPDitAuLmchIsKE'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ybPDitAuLmchIsKE']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_UhmiklCCMnlXubsN(self):
        """MULTIALLEN_UhmiklCCMnlXubsN multi-experiment creation."""
        model_folder = 'MULTIALLEN_UhmiklCCMnlXubsN'
        exp = {
            'experiment_name': ['MULTIALLEN_UhmiklCCMnlXubsN'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_UhmiklCCMnlXubsN']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_bZMRrFHgUmZQEAll(self):
        """MULTIALLEN_bZMRrFHgUmZQEAll multi-experiment creation."""
        model_folder = 'MULTIALLEN_bZMRrFHgUmZQEAll'
        exp = {
            'experiment_name': ['MULTIALLEN_bZMRrFHgUmZQEAll'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_bZMRrFHgUmZQEAll']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_nYDqRUdlQxSCuRhs(self):
        """MULTIALLEN_nYDqRUdlQxSCuRhs multi-experiment creation."""
        model_folder = 'MULTIALLEN_nYDqRUdlQxSCuRhs'
        exp = {
            'experiment_name': ['MULTIALLEN_nYDqRUdlQxSCuRhs'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_nYDqRUdlQxSCuRhs']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_GJGxwupDYuvzEWvH(self):
        """MULTIALLEN_GJGxwupDYuvzEWvH multi-experiment creation."""
        model_folder = 'MULTIALLEN_GJGxwupDYuvzEWvH'
        exp = {
            'experiment_name': ['MULTIALLEN_GJGxwupDYuvzEWvH'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_GJGxwupDYuvzEWvH']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_FQvCSVFqMemrRndI(self):
        """MULTIALLEN_FQvCSVFqMemrRndI multi-experiment creation."""
        model_folder = 'MULTIALLEN_FQvCSVFqMemrRndI'
        exp = {
            'experiment_name': ['MULTIALLEN_FQvCSVFqMemrRndI'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_FQvCSVFqMemrRndI']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_XGkNQOReKEEAjHUK(self):
        """MULTIALLEN_XGkNQOReKEEAjHUK multi-experiment creation."""
        model_folder = 'MULTIALLEN_XGkNQOReKEEAjHUK'
        exp = {
            'experiment_name': ['MULTIALLEN_XGkNQOReKEEAjHUK'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_XGkNQOReKEEAjHUK']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_zBJEwjpIXZVonTDV(self):
        """MULTIALLEN_zBJEwjpIXZVonTDV multi-experiment creation."""
        model_folder = 'MULTIALLEN_zBJEwjpIXZVonTDV'
        exp = {
            'experiment_name': ['MULTIALLEN_zBJEwjpIXZVonTDV'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_zBJEwjpIXZVonTDV']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_kNnWPcwRmXDZaYjQ(self):
        """MULTIALLEN_kNnWPcwRmXDZaYjQ multi-experiment creation."""
        model_folder = 'MULTIALLEN_kNnWPcwRmXDZaYjQ'
        exp = {
            'experiment_name': ['MULTIALLEN_kNnWPcwRmXDZaYjQ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_kNnWPcwRmXDZaYjQ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_EJbPygCDtdRBoham(self):
        """MULTIALLEN_EJbPygCDtdRBoham multi-experiment creation."""
        model_folder = 'MULTIALLEN_EJbPygCDtdRBoham'
        exp = {
            'experiment_name': ['MULTIALLEN_EJbPygCDtdRBoham'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_EJbPygCDtdRBoham']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_amoqklRCebvfdObt(self):
        """MULTIALLEN_amoqklRCebvfdObt multi-experiment creation."""
        model_folder = 'MULTIALLEN_amoqklRCebvfdObt'
        exp = {
            'experiment_name': ['MULTIALLEN_amoqklRCebvfdObt'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_amoqklRCebvfdObt']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_EIColqbKVdjGxLIf(self):
        """MULTIALLEN_EIColqbKVdjGxLIf multi-experiment creation."""
        model_folder = 'MULTIALLEN_EIColqbKVdjGxLIf'
        exp = {
            'experiment_name': ['MULTIALLEN_EIColqbKVdjGxLIf'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_EIColqbKVdjGxLIf']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_YvBlAJpZUablZFnQ(self):
        """MULTIALLEN_YvBlAJpZUablZFnQ multi-experiment creation."""
        model_folder = 'MULTIALLEN_YvBlAJpZUablZFnQ'
        exp = {
            'experiment_name': ['MULTIALLEN_YvBlAJpZUablZFnQ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_YvBlAJpZUablZFnQ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_mPzNxussTPYfsxvT(self):
        """MULTIALLEN_mPzNxussTPYfsxvT multi-experiment creation."""
        model_folder = 'MULTIALLEN_mPzNxussTPYfsxvT'
        exp = {
            'experiment_name': ['MULTIALLEN_mPzNxussTPYfsxvT'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_mPzNxussTPYfsxvT']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ObyQGVnsPGDatmfe(self):
        """MULTIALLEN_ObyQGVnsPGDatmfe multi-experiment creation."""
        model_folder = 'MULTIALLEN_ObyQGVnsPGDatmfe'
        exp = {
            'experiment_name': ['MULTIALLEN_ObyQGVnsPGDatmfe'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ObyQGVnsPGDatmfe']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_zurcehrDJWaIRubm(self):
        """MULTIALLEN_zurcehrDJWaIRubm multi-experiment creation."""
        model_folder = 'MULTIALLEN_zurcehrDJWaIRubm'
        exp = {
            'experiment_name': ['MULTIALLEN_zurcehrDJWaIRubm'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_zurcehrDJWaIRubm']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_nDnLkCiHDjizNyTd(self):
        """MULTIALLEN_nDnLkCiHDjizNyTd multi-experiment creation."""
        model_folder = 'MULTIALLEN_nDnLkCiHDjizNyTd'
        exp = {
            'experiment_name': ['MULTIALLEN_nDnLkCiHDjizNyTd'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_nDnLkCiHDjizNyTd']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_mqNAxTsuNimOdRTQ(self):
        """MULTIALLEN_mqNAxTsuNimOdRTQ multi-experiment creation."""
        model_folder = 'MULTIALLEN_mqNAxTsuNimOdRTQ'
        exp = {
            'experiment_name': ['MULTIALLEN_mqNAxTsuNimOdRTQ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_mqNAxTsuNimOdRTQ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_YEXfaCDLacCJfDdm(self):
        """MULTIALLEN_YEXfaCDLacCJfDdm multi-experiment creation."""
        model_folder = 'MULTIALLEN_YEXfaCDLacCJfDdm'
        exp = {
            'experiment_name': ['MULTIALLEN_YEXfaCDLacCJfDdm'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_YEXfaCDLacCJfDdm']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_vSWMpFepOPKAPCae(self):
        """MULTIALLEN_vSWMpFepOPKAPCae multi-experiment creation."""
        model_folder = 'MULTIALLEN_vSWMpFepOPKAPCae'
        exp = {
            'experiment_name': ['MULTIALLEN_vSWMpFepOPKAPCae'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_vSWMpFepOPKAPCae']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_nBKvwnCPHuwbLLHJ(self):
        """MULTIALLEN_nBKvwnCPHuwbLLHJ multi-experiment creation."""
        model_folder = 'MULTIALLEN_nBKvwnCPHuwbLLHJ'
        exp = {
            'experiment_name': ['MULTIALLEN_nBKvwnCPHuwbLLHJ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_nBKvwnCPHuwbLLHJ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_DBonoSUWFAOWxfYp(self):
        """MULTIALLEN_DBonoSUWFAOWxfYp multi-experiment creation."""
        model_folder = 'MULTIALLEN_DBonoSUWFAOWxfYp'
        exp = {
            'experiment_name': ['MULTIALLEN_DBonoSUWFAOWxfYp'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_DBonoSUWFAOWxfYp']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_hKqdzIGUbsjZsIgP(self):
        """MULTIALLEN_hKqdzIGUbsjZsIgP multi-experiment creation."""
        model_folder = 'MULTIALLEN_hKqdzIGUbsjZsIgP'
        exp = {
            'experiment_name': ['MULTIALLEN_hKqdzIGUbsjZsIgP'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_hKqdzIGUbsjZsIgP']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_fpjiKzmJIzGtNuQe(self):
        """MULTIALLEN_fpjiKzmJIzGtNuQe multi-experiment creation."""
        model_folder = 'MULTIALLEN_fpjiKzmJIzGtNuQe'
        exp = {
            'experiment_name': ['MULTIALLEN_fpjiKzmJIzGtNuQe'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_fpjiKzmJIzGtNuQe']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_NJXmKbbanGDRMYSc(self):
        """MULTIALLEN_NJXmKbbanGDRMYSc multi-experiment creation."""
        model_folder = 'MULTIALLEN_NJXmKbbanGDRMYSc'
        exp = {
            'experiment_name': ['MULTIALLEN_NJXmKbbanGDRMYSc'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_NJXmKbbanGDRMYSc']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ugsDNYwgBaNrYYYj(self):
        """MULTIALLEN_ugsDNYwgBaNrYYYj multi-experiment creation."""
        model_folder = 'MULTIALLEN_ugsDNYwgBaNrYYYj'
        exp = {
            'experiment_name': ['MULTIALLEN_ugsDNYwgBaNrYYYj'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ugsDNYwgBaNrYYYj']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_FtQubjMhRVnhBSKR(self):
        """MULTIALLEN_FtQubjMhRVnhBSKR multi-experiment creation."""
        model_folder = 'MULTIALLEN_FtQubjMhRVnhBSKR'
        exp = {
            'experiment_name': ['MULTIALLEN_FtQubjMhRVnhBSKR'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_FtQubjMhRVnhBSKR']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_nDxRzRbJAygDkmle(self):
        """MULTIALLEN_nDxRzRbJAygDkmle multi-experiment creation."""
        model_folder = 'MULTIALLEN_nDxRzRbJAygDkmle'
        exp = {
            'experiment_name': ['MULTIALLEN_nDxRzRbJAygDkmle'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_nDxRzRbJAygDkmle']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_iNerUmDuOSNWqndD(self):
        """MULTIALLEN_iNerUmDuOSNWqndD multi-experiment creation."""
        model_folder = 'MULTIALLEN_iNerUmDuOSNWqndD'
        exp = {
            'experiment_name': ['MULTIALLEN_iNerUmDuOSNWqndD'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_iNerUmDuOSNWqndD']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_hEwKDlKwdUpfGwDi(self):
        """MULTIALLEN_hEwKDlKwdUpfGwDi multi-experiment creation."""
        model_folder = 'MULTIALLEN_hEwKDlKwdUpfGwDi'
        exp = {
            'experiment_name': ['MULTIALLEN_hEwKDlKwdUpfGwDi'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_hEwKDlKwdUpfGwDi']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_HVMQXUxeWMAwazHO(self):
        """MULTIALLEN_HVMQXUxeWMAwazHO multi-experiment creation."""
        model_folder = 'MULTIALLEN_HVMQXUxeWMAwazHO'
        exp = {
            'experiment_name': ['MULTIALLEN_HVMQXUxeWMAwazHO'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_HVMQXUxeWMAwazHO']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ESRSxZsBPFlHsUDG(self):
        """MULTIALLEN_ESRSxZsBPFlHsUDG multi-experiment creation."""
        model_folder = 'MULTIALLEN_ESRSxZsBPFlHsUDG'
        exp = {
            'experiment_name': ['MULTIALLEN_ESRSxZsBPFlHsUDG'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ESRSxZsBPFlHsUDG']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ZygYfvZDGdIqQmRQ(self):
        """MULTIALLEN_ZygYfvZDGdIqQmRQ multi-experiment creation."""
        model_folder = 'MULTIALLEN_ZygYfvZDGdIqQmRQ'
        exp = {
            'experiment_name': ['MULTIALLEN_ZygYfvZDGdIqQmRQ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ZygYfvZDGdIqQmRQ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_bseEBpMCnVcICEHS(self):
        """MULTIALLEN_bseEBpMCnVcICEHS multi-experiment creation."""
        model_folder = 'MULTIALLEN_bseEBpMCnVcICEHS'
        exp = {
            'experiment_name': ['MULTIALLEN_bseEBpMCnVcICEHS'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_bseEBpMCnVcICEHS']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_XMsLwgEzUeNRnYUB(self):
        """MULTIALLEN_XMsLwgEzUeNRnYUB multi-experiment creation."""
        model_folder = 'MULTIALLEN_XMsLwgEzUeNRnYUB'
        exp = {
            'experiment_name': ['MULTIALLEN_XMsLwgEzUeNRnYUB'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_XMsLwgEzUeNRnYUB']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_dmzvmDPONGzSBPKw(self):
        """MULTIALLEN_dmzvmDPONGzSBPKw multi-experiment creation."""
        model_folder = 'MULTIALLEN_dmzvmDPONGzSBPKw'
        exp = {
            'experiment_name': ['MULTIALLEN_dmzvmDPONGzSBPKw'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_dmzvmDPONGzSBPKw']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_BWGuJbvOEGRCOqSy(self):
        """MULTIALLEN_BWGuJbvOEGRCOqSy multi-experiment creation."""
        model_folder = 'MULTIALLEN_BWGuJbvOEGRCOqSy'
        exp = {
            'experiment_name': ['MULTIALLEN_BWGuJbvOEGRCOqSy'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_BWGuJbvOEGRCOqSy']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_kLFxuTlgtmqnxpgF(self):
        """MULTIALLEN_kLFxuTlgtmqnxpgF multi-experiment creation."""
        model_folder = 'MULTIALLEN_kLFxuTlgtmqnxpgF'
        exp = {
            'experiment_name': ['MULTIALLEN_kLFxuTlgtmqnxpgF'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_kLFxuTlgtmqnxpgF']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_DGtYgJTmyhguRHYN(self):
        """MULTIALLEN_DGtYgJTmyhguRHYN multi-experiment creation."""
        model_folder = 'MULTIALLEN_DGtYgJTmyhguRHYN'
        exp = {
            'experiment_name': ['MULTIALLEN_DGtYgJTmyhguRHYN'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_DGtYgJTmyhguRHYN']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_RGNmqAywoAahvisW(self):
        """MULTIALLEN_RGNmqAywoAahvisW multi-experiment creation."""
        model_folder = 'MULTIALLEN_RGNmqAywoAahvisW'
        exp = {
            'experiment_name': ['MULTIALLEN_RGNmqAywoAahvisW'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_RGNmqAywoAahvisW']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_NgnVvSZMeWZptmqR(self):
        """MULTIALLEN_NgnVvSZMeWZptmqR multi-experiment creation."""
        model_folder = 'MULTIALLEN_NgnVvSZMeWZptmqR'
        exp = {
            'experiment_name': ['MULTIALLEN_NgnVvSZMeWZptmqR'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_NgnVvSZMeWZptmqR']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_YDZcdIZcVAoylwds(self):
        """MULTIALLEN_YDZcdIZcVAoylwds multi-experiment creation."""
        model_folder = 'MULTIALLEN_YDZcdIZcVAoylwds'
        exp = {
            'experiment_name': ['MULTIALLEN_YDZcdIZcVAoylwds'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_YDZcdIZcVAoylwds']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_LzIQKEDxZExXBkwb(self):
        """MULTIALLEN_LzIQKEDxZExXBkwb multi-experiment creation."""
        model_folder = 'MULTIALLEN_LzIQKEDxZExXBkwb'
        exp = {
            'experiment_name': ['MULTIALLEN_LzIQKEDxZExXBkwb'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_LzIQKEDxZExXBkwb']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_FhJlBNEJKphVoZRO(self):
        """MULTIALLEN_FhJlBNEJKphVoZRO multi-experiment creation."""
        model_folder = 'MULTIALLEN_FhJlBNEJKphVoZRO'
        exp = {
            'experiment_name': ['MULTIALLEN_FhJlBNEJKphVoZRO'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_FhJlBNEJKphVoZRO']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_PKFKJFdcgaMnLzQU(self):
        """MULTIALLEN_PKFKJFdcgaMnLzQU multi-experiment creation."""
        model_folder = 'MULTIALLEN_PKFKJFdcgaMnLzQU'
        exp = {
            'experiment_name': ['MULTIALLEN_PKFKJFdcgaMnLzQU'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_PKFKJFdcgaMnLzQU']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_LoEjTpWLcAGpRTdJ(self):
        """MULTIALLEN_LoEjTpWLcAGpRTdJ multi-experiment creation."""
        model_folder = 'MULTIALLEN_LoEjTpWLcAGpRTdJ'
        exp = {
            'experiment_name': ['MULTIALLEN_LoEjTpWLcAGpRTdJ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_LoEjTpWLcAGpRTdJ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_KvloUSWtgAJKemma(self):
        """MULTIALLEN_KvloUSWtgAJKemma multi-experiment creation."""
        model_folder = 'MULTIALLEN_KvloUSWtgAJKemma'
        exp = {
            'experiment_name': ['MULTIALLEN_KvloUSWtgAJKemma'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_KvloUSWtgAJKemma']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_XJeylRJXayDRuUhT(self):
        """MULTIALLEN_XJeylRJXayDRuUhT multi-experiment creation."""
        model_folder = 'MULTIALLEN_XJeylRJXayDRuUhT'
        exp = {
            'experiment_name': ['MULTIALLEN_XJeylRJXayDRuUhT'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_XJeylRJXayDRuUhT']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_FExEIsKsuiwSHxWO(self):
        """MULTIALLEN_FExEIsKsuiwSHxWO multi-experiment creation."""
        model_folder = 'MULTIALLEN_FExEIsKsuiwSHxWO'
        exp = {
            'experiment_name': ['MULTIALLEN_FExEIsKsuiwSHxWO'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_FExEIsKsuiwSHxWO']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_XKtScfcubHpHIbZM(self):
        """MULTIALLEN_XKtScfcubHpHIbZM multi-experiment creation."""
        model_folder = 'MULTIALLEN_XKtScfcubHpHIbZM'
        exp = {
            'experiment_name': ['MULTIALLEN_XKtScfcubHpHIbZM'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_XKtScfcubHpHIbZM']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_nxgzozPFEzpVyheB(self):
        """MULTIALLEN_nxgzozPFEzpVyheB multi-experiment creation."""
        model_folder = 'MULTIALLEN_nxgzozPFEzpVyheB'
        exp = {
            'experiment_name': ['MULTIALLEN_nxgzozPFEzpVyheB'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_nxgzozPFEzpVyheB']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_FjyXcaCOEbaKKtpZ(self):
        """MULTIALLEN_FjyXcaCOEbaKKtpZ multi-experiment creation."""
        model_folder = 'MULTIALLEN_FjyXcaCOEbaKKtpZ'
        exp = {
            'experiment_name': ['MULTIALLEN_FjyXcaCOEbaKKtpZ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_FjyXcaCOEbaKKtpZ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_vmxXvPoVMwXrlrFq(self):
        """MULTIALLEN_vmxXvPoVMwXrlrFq multi-experiment creation."""
        model_folder = 'MULTIALLEN_vmxXvPoVMwXrlrFq'
        exp = {
            'experiment_name': ['MULTIALLEN_vmxXvPoVMwXrlrFq'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_vmxXvPoVMwXrlrFq']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_DenqZjDIsexzOFKo(self):
        """MULTIALLEN_DenqZjDIsexzOFKo multi-experiment creation."""
        model_folder = 'MULTIALLEN_DenqZjDIsexzOFKo'
        exp = {
            'experiment_name': ['MULTIALLEN_DenqZjDIsexzOFKo'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_DenqZjDIsexzOFKo']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ByRnGmuTLJNoszVF(self):
        """MULTIALLEN_ByRnGmuTLJNoszVF multi-experiment creation."""
        model_folder = 'MULTIALLEN_ByRnGmuTLJNoszVF'
        exp = {
            'experiment_name': ['MULTIALLEN_ByRnGmuTLJNoszVF'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ByRnGmuTLJNoszVF']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ChlCPUzHilhfixIv(self):
        """MULTIALLEN_ChlCPUzHilhfixIv multi-experiment creation."""
        model_folder = 'MULTIALLEN_ChlCPUzHilhfixIv'
        exp = {
            'experiment_name': ['MULTIALLEN_ChlCPUzHilhfixIv'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ChlCPUzHilhfixIv']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_FIPfjSfUvTDrdwDd(self):
        """MULTIALLEN_FIPfjSfUvTDrdwDd multi-experiment creation."""
        model_folder = 'MULTIALLEN_FIPfjSfUvTDrdwDd'
        exp = {
            'experiment_name': ['MULTIALLEN_FIPfjSfUvTDrdwDd'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_FIPfjSfUvTDrdwDd']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_FGvBDzSwRNZYhtar(self):
        """MULTIALLEN_FGvBDzSwRNZYhtar multi-experiment creation."""
        model_folder = 'MULTIALLEN_FGvBDzSwRNZYhtar'
        exp = {
            'experiment_name': ['MULTIALLEN_FGvBDzSwRNZYhtar'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_FGvBDzSwRNZYhtar']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_gTMriHVYsVbWcJqo(self):
        """MULTIALLEN_gTMriHVYsVbWcJqo multi-experiment creation."""
        model_folder = 'MULTIALLEN_gTMriHVYsVbWcJqo'
        exp = {
            'experiment_name': ['MULTIALLEN_gTMriHVYsVbWcJqo'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_gTMriHVYsVbWcJqo']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_WigeVpVlmzoJHqNE(self):
        """MULTIALLEN_WigeVpVlmzoJHqNE multi-experiment creation."""
        model_folder = 'MULTIALLEN_WigeVpVlmzoJHqNE'
        exp = {
            'experiment_name': ['MULTIALLEN_WigeVpVlmzoJHqNE'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_WigeVpVlmzoJHqNE']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_CRaJlHxLNltfEiOU(self):
        """MULTIALLEN_CRaJlHxLNltfEiOU multi-experiment creation."""
        model_folder = 'MULTIALLEN_CRaJlHxLNltfEiOU'
        exp = {
            'experiment_name': ['MULTIALLEN_CRaJlHxLNltfEiOU'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_CRaJlHxLNltfEiOU']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_XItVvGMXoWumTKeD(self):
        """MULTIALLEN_XItVvGMXoWumTKeD multi-experiment creation."""
        model_folder = 'MULTIALLEN_XItVvGMXoWumTKeD'
        exp = {
            'experiment_name': ['MULTIALLEN_XItVvGMXoWumTKeD'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_XItVvGMXoWumTKeD']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_RAAqyubhdPMKWzgs(self):
        """MULTIALLEN_RAAqyubhdPMKWzgs multi-experiment creation."""
        model_folder = 'MULTIALLEN_RAAqyubhdPMKWzgs'
        exp = {
            'experiment_name': ['MULTIALLEN_RAAqyubhdPMKWzgs'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_RAAqyubhdPMKWzgs']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_RlSvNURmBUiCewgg(self):
        """MULTIALLEN_RlSvNURmBUiCewgg multi-experiment creation."""
        model_folder = 'MULTIALLEN_RlSvNURmBUiCewgg'
        exp = {
            'experiment_name': ['MULTIALLEN_RlSvNURmBUiCewgg'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_RlSvNURmBUiCewgg']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_qlmQcvvcowiBaHZM(self):
        """MULTIALLEN_qlmQcvvcowiBaHZM multi-experiment creation."""
        model_folder = 'MULTIALLEN_qlmQcvvcowiBaHZM'
        exp = {
            'experiment_name': ['MULTIALLEN_qlmQcvvcowiBaHZM'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_qlmQcvvcowiBaHZM']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_SsDtTtaBaGDSeiKD(self):
        """MULTIALLEN_SsDtTtaBaGDSeiKD multi-experiment creation."""
        model_folder = 'MULTIALLEN_SsDtTtaBaGDSeiKD'
        exp = {
            'experiment_name': ['MULTIALLEN_SsDtTtaBaGDSeiKD'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_SsDtTtaBaGDSeiKD']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_heMCNtTvKFUJUENE(self):
        """MULTIALLEN_heMCNtTvKFUJUENE multi-experiment creation."""
        model_folder = 'MULTIALLEN_heMCNtTvKFUJUENE'
        exp = {
            'experiment_name': ['MULTIALLEN_heMCNtTvKFUJUENE'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_heMCNtTvKFUJUENE']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_VkdcbNZdhdvjiVCF(self):
        """MULTIALLEN_VkdcbNZdhdvjiVCF multi-experiment creation."""
        model_folder = 'MULTIALLEN_VkdcbNZdhdvjiVCF'
        exp = {
            'experiment_name': ['MULTIALLEN_VkdcbNZdhdvjiVCF'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_VkdcbNZdhdvjiVCF']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_QcjxaryYcqzZZHDm(self):
        """MULTIALLEN_QcjxaryYcqzZZHDm multi-experiment creation."""
        model_folder = 'MULTIALLEN_QcjxaryYcqzZZHDm'
        exp = {
            'experiment_name': ['MULTIALLEN_QcjxaryYcqzZZHDm'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_QcjxaryYcqzZZHDm']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_RTmHCfsXeHdzfdod(self):
        """MULTIALLEN_RTmHCfsXeHdzfdod multi-experiment creation."""
        model_folder = 'MULTIALLEN_RTmHCfsXeHdzfdod'
        exp = {
            'experiment_name': ['MULTIALLEN_RTmHCfsXeHdzfdod'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_RTmHCfsXeHdzfdod']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_DTkiBFdPHPviOnvV(self):
        """MULTIALLEN_DTkiBFdPHPviOnvV multi-experiment creation."""
        model_folder = 'MULTIALLEN_DTkiBFdPHPviOnvV'
        exp = {
            'experiment_name': ['MULTIALLEN_DTkiBFdPHPviOnvV'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_DTkiBFdPHPviOnvV']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_nqaePseIUKySbbfW(self):
        """MULTIALLEN_nqaePseIUKySbbfW multi-experiment creation."""
        model_folder = 'MULTIALLEN_nqaePseIUKySbbfW'
        exp = {
            'experiment_name': ['MULTIALLEN_nqaePseIUKySbbfW'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_nqaePseIUKySbbfW']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_OBWoBTvZUYlybzTg(self):
        """MULTIALLEN_OBWoBTvZUYlybzTg multi-experiment creation."""
        model_folder = 'MULTIALLEN_OBWoBTvZUYlybzTg'
        exp = {
            'experiment_name': ['MULTIALLEN_OBWoBTvZUYlybzTg'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_OBWoBTvZUYlybzTg']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_MdhjIcJDajfTQFrE(self):
        """MULTIALLEN_MdhjIcJDajfTQFrE multi-experiment creation."""
        model_folder = 'MULTIALLEN_MdhjIcJDajfTQFrE'
        exp = {
            'experiment_name': ['MULTIALLEN_MdhjIcJDajfTQFrE'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_MdhjIcJDajfTQFrE']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_wutWWzwsunrOgkzV(self):
        """MULTIALLEN_wutWWzwsunrOgkzV multi-experiment creation."""
        model_folder = 'MULTIALLEN_wutWWzwsunrOgkzV'
        exp = {
            'experiment_name': ['MULTIALLEN_wutWWzwsunrOgkzV'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_wutWWzwsunrOgkzV']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_gSBJZQtYaeDVeihO(self):
        """MULTIALLEN_gSBJZQtYaeDVeihO multi-experiment creation."""
        model_folder = 'MULTIALLEN_gSBJZQtYaeDVeihO'
        exp = {
            'experiment_name': ['MULTIALLEN_gSBJZQtYaeDVeihO'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_gSBJZQtYaeDVeihO']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_GFGHiKtUtMbawnsB(self):
        """MULTIALLEN_GFGHiKtUtMbawnsB multi-experiment creation."""
        model_folder = 'MULTIALLEN_GFGHiKtUtMbawnsB'
        exp = {
            'experiment_name': ['MULTIALLEN_GFGHiKtUtMbawnsB'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_GFGHiKtUtMbawnsB']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_QbOqwuACkWTpKzua(self):
        """MULTIALLEN_QbOqwuACkWTpKzua multi-experiment creation."""
        model_folder = 'MULTIALLEN_QbOqwuACkWTpKzua'
        exp = {
            'experiment_name': ['MULTIALLEN_QbOqwuACkWTpKzua'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_QbOqwuACkWTpKzua']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_YUUTJfIVKnkJFtiM(self):
        """MULTIALLEN_YUUTJfIVKnkJFtiM multi-experiment creation."""
        model_folder = 'MULTIALLEN_YUUTJfIVKnkJFtiM'
        exp = {
            'experiment_name': ['MULTIALLEN_YUUTJfIVKnkJFtiM'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_YUUTJfIVKnkJFtiM']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_qMUZVzDFLDGCvWvb(self):
        """MULTIALLEN_qMUZVzDFLDGCvWvb multi-experiment creation."""
        model_folder = 'MULTIALLEN_qMUZVzDFLDGCvWvb'
        exp = {
            'experiment_name': ['MULTIALLEN_qMUZVzDFLDGCvWvb'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_qMUZVzDFLDGCvWvb']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_VhjmMSOKvIdZkTaj(self):
        """MULTIALLEN_VhjmMSOKvIdZkTaj multi-experiment creation."""
        model_folder = 'MULTIALLEN_VhjmMSOKvIdZkTaj'
        exp = {
            'experiment_name': ['MULTIALLEN_VhjmMSOKvIdZkTaj'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_VhjmMSOKvIdZkTaj']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_FuwzDytajSuLNXrP(self):
        """MULTIALLEN_FuwzDytajSuLNXrP multi-experiment creation."""
        model_folder = 'MULTIALLEN_FuwzDytajSuLNXrP'
        exp = {
            'experiment_name': ['MULTIALLEN_FuwzDytajSuLNXrP'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_FuwzDytajSuLNXrP']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_kONhLzQVCTXwBKUv(self):
        """MULTIALLEN_kONhLzQVCTXwBKUv multi-experiment creation."""
        model_folder = 'MULTIALLEN_kONhLzQVCTXwBKUv'
        exp = {
            'experiment_name': ['MULTIALLEN_kONhLzQVCTXwBKUv'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_kONhLzQVCTXwBKUv']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_GIHxCyGOwFhcmrgb(self):
        """MULTIALLEN_GIHxCyGOwFhcmrgb multi-experiment creation."""
        model_folder = 'MULTIALLEN_GIHxCyGOwFhcmrgb'
        exp = {
            'experiment_name': ['MULTIALLEN_GIHxCyGOwFhcmrgb'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_GIHxCyGOwFhcmrgb']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_uWHuWCofhoYHtDFy(self):
        """MULTIALLEN_uWHuWCofhoYHtDFy multi-experiment creation."""
        model_folder = 'MULTIALLEN_uWHuWCofhoYHtDFy'
        exp = {
            'experiment_name': ['MULTIALLEN_uWHuWCofhoYHtDFy'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_uWHuWCofhoYHtDFy']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_dxjnUcNHKMwzuGSs(self):
        """MULTIALLEN_dxjnUcNHKMwzuGSs multi-experiment creation."""
        model_folder = 'MULTIALLEN_dxjnUcNHKMwzuGSs'
        exp = {
            'experiment_name': ['MULTIALLEN_dxjnUcNHKMwzuGSs'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_dxjnUcNHKMwzuGSs']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_YMJjOSDlxzaXvqEs(self):
        """MULTIALLEN_YMJjOSDlxzaXvqEs multi-experiment creation."""
        model_folder = 'MULTIALLEN_YMJjOSDlxzaXvqEs'
        exp = {
            'experiment_name': ['MULTIALLEN_YMJjOSDlxzaXvqEs'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_YMJjOSDlxzaXvqEs']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ZMQrZfBxAUiPsXbK(self):
        """MULTIALLEN_ZMQrZfBxAUiPsXbK multi-experiment creation."""
        model_folder = 'MULTIALLEN_ZMQrZfBxAUiPsXbK'
        exp = {
            'experiment_name': ['MULTIALLEN_ZMQrZfBxAUiPsXbK'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ZMQrZfBxAUiPsXbK']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ljSLJtLCzCfEazaC(self):
        """MULTIALLEN_ljSLJtLCzCfEazaC multi-experiment creation."""
        model_folder = 'MULTIALLEN_ljSLJtLCzCfEazaC'
        exp = {
            'experiment_name': ['MULTIALLEN_ljSLJtLCzCfEazaC'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ljSLJtLCzCfEazaC']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_OjpezUOdlrGEPkug(self):
        """MULTIALLEN_OjpezUOdlrGEPkug multi-experiment creation."""
        model_folder = 'MULTIALLEN_OjpezUOdlrGEPkug'
        exp = {
            'experiment_name': ['MULTIALLEN_OjpezUOdlrGEPkug'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_OjpezUOdlrGEPkug']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ajHPnztvvsPaxZSH(self):
        """MULTIALLEN_ajHPnztvvsPaxZSH multi-experiment creation."""
        model_folder = 'MULTIALLEN_ajHPnztvvsPaxZSH'
        exp = {
            'experiment_name': ['MULTIALLEN_ajHPnztvvsPaxZSH'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ajHPnztvvsPaxZSH']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_mYqDcWAYzzVSTaNf(self):
        """MULTIALLEN_mYqDcWAYzzVSTaNf multi-experiment creation."""
        model_folder = 'MULTIALLEN_mYqDcWAYzzVSTaNf'
        exp = {
            'experiment_name': ['MULTIALLEN_mYqDcWAYzzVSTaNf'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_mYqDcWAYzzVSTaNf']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_sOhyWtTTyEyBaGgk(self):
        """MULTIALLEN_sOhyWtTTyEyBaGgk multi-experiment creation."""
        model_folder = 'MULTIALLEN_sOhyWtTTyEyBaGgk'
        exp = {
            'experiment_name': ['MULTIALLEN_sOhyWtTTyEyBaGgk'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_sOhyWtTTyEyBaGgk']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_DwRbzYokamKIXCCz(self):
        """MULTIALLEN_DwRbzYokamKIXCCz multi-experiment creation."""
        model_folder = 'MULTIALLEN_DwRbzYokamKIXCCz'
        exp = {
            'experiment_name': ['MULTIALLEN_DwRbzYokamKIXCCz'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_DwRbzYokamKIXCCz']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_udUcLKojZkScmBwu(self):
        """MULTIALLEN_udUcLKojZkScmBwu multi-experiment creation."""
        model_folder = 'MULTIALLEN_udUcLKojZkScmBwu'
        exp = {
            'experiment_name': ['MULTIALLEN_udUcLKojZkScmBwu'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_udUcLKojZkScmBwu']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_YtqfFWQNGTbyFouj(self):
        """MULTIALLEN_YtqfFWQNGTbyFouj multi-experiment creation."""
        model_folder = 'MULTIALLEN_YtqfFWQNGTbyFouj'
        exp = {
            'experiment_name': ['MULTIALLEN_YtqfFWQNGTbyFouj'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_YtqfFWQNGTbyFouj']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_RDuOHcDjrZniBoas(self):
        """MULTIALLEN_RDuOHcDjrZniBoas multi-experiment creation."""
        model_folder = 'MULTIALLEN_RDuOHcDjrZniBoas'
        exp = {
            'experiment_name': ['MULTIALLEN_RDuOHcDjrZniBoas'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_RDuOHcDjrZniBoas']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_MbwoSkKMoHOIlvvT(self):
        """MULTIALLEN_MbwoSkKMoHOIlvvT multi-experiment creation."""
        model_folder = 'MULTIALLEN_MbwoSkKMoHOIlvvT'
        exp = {
            'experiment_name': ['MULTIALLEN_MbwoSkKMoHOIlvvT'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_MbwoSkKMoHOIlvvT']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_INlXQCzpgWoxgTJc(self):
        """MULTIALLEN_INlXQCzpgWoxgTJc multi-experiment creation."""
        model_folder = 'MULTIALLEN_INlXQCzpgWoxgTJc'
        exp = {
            'experiment_name': ['MULTIALLEN_INlXQCzpgWoxgTJc'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_INlXQCzpgWoxgTJc']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_tDmuHELxCleOPUCp(self):
        """MULTIALLEN_tDmuHELxCleOPUCp multi-experiment creation."""
        model_folder = 'MULTIALLEN_tDmuHELxCleOPUCp'
        exp = {
            'experiment_name': ['MULTIALLEN_tDmuHELxCleOPUCp'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_tDmuHELxCleOPUCp']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_URxnPmVhybrbysUQ(self):
        """MULTIALLEN_URxnPmVhybrbysUQ multi-experiment creation."""
        model_folder = 'MULTIALLEN_URxnPmVhybrbysUQ'
        exp = {
            'experiment_name': ['MULTIALLEN_URxnPmVhybrbysUQ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_URxnPmVhybrbysUQ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_pYdDsFGFpQwngMwv(self):
        """MULTIALLEN_pYdDsFGFpQwngMwv multi-experiment creation."""
        model_folder = 'MULTIALLEN_pYdDsFGFpQwngMwv'
        exp = {
            'experiment_name': ['MULTIALLEN_pYdDsFGFpQwngMwv'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_pYdDsFGFpQwngMwv']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_yTBzXKFuWzJCmqwt(self):
        """MULTIALLEN_yTBzXKFuWzJCmqwt multi-experiment creation."""
        model_folder = 'MULTIALLEN_yTBzXKFuWzJCmqwt'
        exp = {
            'experiment_name': ['MULTIALLEN_yTBzXKFuWzJCmqwt'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_yTBzXKFuWzJCmqwt']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_sbkKYOIAHleEyCCG(self):
        """MULTIALLEN_sbkKYOIAHleEyCCG multi-experiment creation."""
        model_folder = 'MULTIALLEN_sbkKYOIAHleEyCCG'
        exp = {
            'experiment_name': ['MULTIALLEN_sbkKYOIAHleEyCCG'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_sbkKYOIAHleEyCCG']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_karPCGVQjLiMEGiw(self):
        """MULTIALLEN_karPCGVQjLiMEGiw multi-experiment creation."""
        model_folder = 'MULTIALLEN_karPCGVQjLiMEGiw'
        exp = {
            'experiment_name': ['MULTIALLEN_karPCGVQjLiMEGiw'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_karPCGVQjLiMEGiw']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_NmkxtSYlOYsBMhEo(self):
        """MULTIALLEN_NmkxtSYlOYsBMhEo multi-experiment creation."""
        model_folder = 'MULTIALLEN_NmkxtSYlOYsBMhEo'
        exp = {
            'experiment_name': ['MULTIALLEN_NmkxtSYlOYsBMhEo'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_NmkxtSYlOYsBMhEo']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_GojHTwHONAnigFkU(self):
        """MULTIALLEN_GojHTwHONAnigFkU multi-experiment creation."""
        model_folder = 'MULTIALLEN_GojHTwHONAnigFkU'
        exp = {
            'experiment_name': ['MULTIALLEN_GojHTwHONAnigFkU'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_GojHTwHONAnigFkU']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_rjdplaxvAQhxfdCa(self):
        """MULTIALLEN_rjdplaxvAQhxfdCa multi-experiment creation."""
        model_folder = 'MULTIALLEN_rjdplaxvAQhxfdCa'
        exp = {
            'experiment_name': ['MULTIALLEN_rjdplaxvAQhxfdCa'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_rjdplaxvAQhxfdCa']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_BWfdYQzixzoLrOTQ(self):
        """MULTIALLEN_BWfdYQzixzoLrOTQ multi-experiment creation."""
        model_folder = 'MULTIALLEN_BWfdYQzixzoLrOTQ'
        exp = {
            'experiment_name': ['MULTIALLEN_BWfdYQzixzoLrOTQ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_BWfdYQzixzoLrOTQ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_niarhHxtyBUxoShQ(self):
        """MULTIALLEN_niarhHxtyBUxoShQ multi-experiment creation."""
        model_folder = 'MULTIALLEN_niarhHxtyBUxoShQ'
        exp = {
            'experiment_name': ['MULTIALLEN_niarhHxtyBUxoShQ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_niarhHxtyBUxoShQ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_TEplAqfsMMsWJMIX(self):
        """MULTIALLEN_TEplAqfsMMsWJMIX multi-experiment creation."""
        model_folder = 'MULTIALLEN_TEplAqfsMMsWJMIX'
        exp = {
            'experiment_name': ['MULTIALLEN_TEplAqfsMMsWJMIX'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_TEplAqfsMMsWJMIX']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_yEuBRUYknXbLwpCN(self):
        """MULTIALLEN_yEuBRUYknXbLwpCN multi-experiment creation."""
        model_folder = 'MULTIALLEN_yEuBRUYknXbLwpCN'
        exp = {
            'experiment_name': ['MULTIALLEN_yEuBRUYknXbLwpCN'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_yEuBRUYknXbLwpCN']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_aTVpBsSepZZVWRll(self):
        """MULTIALLEN_aTVpBsSepZZVWRll multi-experiment creation."""
        model_folder = 'MULTIALLEN_aTVpBsSepZZVWRll'
        exp = {
            'experiment_name': ['MULTIALLEN_aTVpBsSepZZVWRll'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_aTVpBsSepZZVWRll']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_cfiCyBRHgyHovAwA(self):
        """MULTIALLEN_cfiCyBRHgyHovAwA multi-experiment creation."""
        model_folder = 'MULTIALLEN_cfiCyBRHgyHovAwA'
        exp = {
            'experiment_name': ['MULTIALLEN_cfiCyBRHgyHovAwA'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_cfiCyBRHgyHovAwA']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_trGbutQIiUrahZRp(self):
        """MULTIALLEN_trGbutQIiUrahZRp multi-experiment creation."""
        model_folder = 'MULTIALLEN_trGbutQIiUrahZRp'
        exp = {
            'experiment_name': ['MULTIALLEN_trGbutQIiUrahZRp'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_trGbutQIiUrahZRp']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_uudeJexLQQjjmyRD(self):
        """MULTIALLEN_uudeJexLQQjjmyRD multi-experiment creation."""
        model_folder = 'MULTIALLEN_uudeJexLQQjjmyRD'
        exp = {
            'experiment_name': ['MULTIALLEN_uudeJexLQQjjmyRD'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_uudeJexLQQjjmyRD']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_UjvWLgouFKRcVuqW(self):
        """MULTIALLEN_UjvWLgouFKRcVuqW multi-experiment creation."""
        model_folder = 'MULTIALLEN_UjvWLgouFKRcVuqW'
        exp = {
            'experiment_name': ['MULTIALLEN_UjvWLgouFKRcVuqW'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_UjvWLgouFKRcVuqW']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_wMLbXswSaeNHmFLT(self):
        """MULTIALLEN_wMLbXswSaeNHmFLT multi-experiment creation."""
        model_folder = 'MULTIALLEN_wMLbXswSaeNHmFLT'
        exp = {
            'experiment_name': ['MULTIALLEN_wMLbXswSaeNHmFLT'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_wMLbXswSaeNHmFLT']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_kJWAdSLKWbbDetyc(self):
        """MULTIALLEN_kJWAdSLKWbbDetyc multi-experiment creation."""
        model_folder = 'MULTIALLEN_kJWAdSLKWbbDetyc'
        exp = {
            'experiment_name': ['MULTIALLEN_kJWAdSLKWbbDetyc'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_kJWAdSLKWbbDetyc']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_KLbhgyUjKsjDAFfy(self):
        """MULTIALLEN_KLbhgyUjKsjDAFfy multi-experiment creation."""
        model_folder = 'MULTIALLEN_KLbhgyUjKsjDAFfy'
        exp = {
            'experiment_name': ['MULTIALLEN_KLbhgyUjKsjDAFfy'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_KLbhgyUjKsjDAFfy']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_pzOtXSiMbEGIMwbn(self):
        """MULTIALLEN_pzOtXSiMbEGIMwbn multi-experiment creation."""
        model_folder = 'MULTIALLEN_pzOtXSiMbEGIMwbn'
        exp = {
            'experiment_name': ['MULTIALLEN_pzOtXSiMbEGIMwbn'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_pzOtXSiMbEGIMwbn']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_zkmxnIEUtXlDnDjl(self):
        """MULTIALLEN_zkmxnIEUtXlDnDjl multi-experiment creation."""
        model_folder = 'MULTIALLEN_zkmxnIEUtXlDnDjl'
        exp = {
            'experiment_name': ['MULTIALLEN_zkmxnIEUtXlDnDjl'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_zkmxnIEUtXlDnDjl']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_CCDPBMfuwWfvlwXe(self):
        """MULTIALLEN_CCDPBMfuwWfvlwXe multi-experiment creation."""
        model_folder = 'MULTIALLEN_CCDPBMfuwWfvlwXe'
        exp = {
            'experiment_name': ['MULTIALLEN_CCDPBMfuwWfvlwXe'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_CCDPBMfuwWfvlwXe']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_XLwvgpVJeluotOAx(self):
        """MULTIALLEN_XLwvgpVJeluotOAx multi-experiment creation."""
        model_folder = 'MULTIALLEN_XLwvgpVJeluotOAx'
        exp = {
            'experiment_name': ['MULTIALLEN_XLwvgpVJeluotOAx'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_XLwvgpVJeluotOAx']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_LIgkOxAtJSzEFtho(self):
        """MULTIALLEN_LIgkOxAtJSzEFtho multi-experiment creation."""
        model_folder = 'MULTIALLEN_LIgkOxAtJSzEFtho'
        exp = {
            'experiment_name': ['MULTIALLEN_LIgkOxAtJSzEFtho'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_LIgkOxAtJSzEFtho']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_DjCJSbBkwTgMXbbP(self):
        """MULTIALLEN_DjCJSbBkwTgMXbbP multi-experiment creation."""
        model_folder = 'MULTIALLEN_DjCJSbBkwTgMXbbP'
        exp = {
            'experiment_name': ['MULTIALLEN_DjCJSbBkwTgMXbbP'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_DjCJSbBkwTgMXbbP']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_AitknUsEfgXvQBab(self):
        """MULTIALLEN_AitknUsEfgXvQBab multi-experiment creation."""
        model_folder = 'MULTIALLEN_AitknUsEfgXvQBab'
        exp = {
            'experiment_name': ['MULTIALLEN_AitknUsEfgXvQBab'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_AitknUsEfgXvQBab']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_YijfFcBZFKEThoEH(self):
        """MULTIALLEN_YijfFcBZFKEThoEH multi-experiment creation."""
        model_folder = 'MULTIALLEN_YijfFcBZFKEThoEH'
        exp = {
            'experiment_name': ['MULTIALLEN_YijfFcBZFKEThoEH'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_YijfFcBZFKEThoEH']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_MaqCvmPCcUwttYwr(self):
        """MULTIALLEN_MaqCvmPCcUwttYwr multi-experiment creation."""
        model_folder = 'MULTIALLEN_MaqCvmPCcUwttYwr'
        exp = {
            'experiment_name': ['MULTIALLEN_MaqCvmPCcUwttYwr'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_MaqCvmPCcUwttYwr']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_HodPntPYStWeEcXe(self):
        """MULTIALLEN_HodPntPYStWeEcXe multi-experiment creation."""
        model_folder = 'MULTIALLEN_HodPntPYStWeEcXe'
        exp = {
            'experiment_name': ['MULTIALLEN_HodPntPYStWeEcXe'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_HodPntPYStWeEcXe']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_wrskOnKcMQRVHwmy(self):
        """MULTIALLEN_wrskOnKcMQRVHwmy multi-experiment creation."""
        model_folder = 'MULTIALLEN_wrskOnKcMQRVHwmy'
        exp = {
            'experiment_name': ['MULTIALLEN_wrskOnKcMQRVHwmy'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_wrskOnKcMQRVHwmy']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_edLzXMKNSLqKEmSr(self):
        """MULTIALLEN_edLzXMKNSLqKEmSr multi-experiment creation."""
        model_folder = 'MULTIALLEN_edLzXMKNSLqKEmSr'
        exp = {
            'experiment_name': ['MULTIALLEN_edLzXMKNSLqKEmSr'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_edLzXMKNSLqKEmSr']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_TzDoJxQahcBsFQTO(self):
        """MULTIALLEN_TzDoJxQahcBsFQTO multi-experiment creation."""
        model_folder = 'MULTIALLEN_TzDoJxQahcBsFQTO'
        exp = {
            'experiment_name': ['MULTIALLEN_TzDoJxQahcBsFQTO'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_TzDoJxQahcBsFQTO']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ZLSltLHYQbrNDPuH(self):
        """MULTIALLEN_ZLSltLHYQbrNDPuH multi-experiment creation."""
        model_folder = 'MULTIALLEN_ZLSltLHYQbrNDPuH'
        exp = {
            'experiment_name': ['MULTIALLEN_ZLSltLHYQbrNDPuH'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ZLSltLHYQbrNDPuH']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_dDgYtaUaRektvtxI(self):
        """MULTIALLEN_dDgYtaUaRektvtxI multi-experiment creation."""
        model_folder = 'MULTIALLEN_dDgYtaUaRektvtxI'
        exp = {
            'experiment_name': ['MULTIALLEN_dDgYtaUaRektvtxI'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_dDgYtaUaRektvtxI']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_vbeLFbFJetunthXo(self):
        """MULTIALLEN_vbeLFbFJetunthXo multi-experiment creation."""
        model_folder = 'MULTIALLEN_vbeLFbFJetunthXo'
        exp = {
            'experiment_name': ['MULTIALLEN_vbeLFbFJetunthXo'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_vbeLFbFJetunthXo']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_nzrmvheslowihHLN(self):
        """MULTIALLEN_nzrmvheslowihHLN multi-experiment creation."""
        model_folder = 'MULTIALLEN_nzrmvheslowihHLN'
        exp = {
            'experiment_name': ['MULTIALLEN_nzrmvheslowihHLN'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_nzrmvheslowihHLN']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_RPWQinlLdaqgnTjE(self):
        """MULTIALLEN_RPWQinlLdaqgnTjE multi-experiment creation."""
        model_folder = 'MULTIALLEN_RPWQinlLdaqgnTjE'
        exp = {
            'experiment_name': ['MULTIALLEN_RPWQinlLdaqgnTjE'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_RPWQinlLdaqgnTjE']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_tozNBnIjYENWMAhb(self):
        """MULTIALLEN_tozNBnIjYENWMAhb multi-experiment creation."""
        model_folder = 'MULTIALLEN_tozNBnIjYENWMAhb'
        exp = {
            'experiment_name': ['MULTIALLEN_tozNBnIjYENWMAhb'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_tozNBnIjYENWMAhb']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_mJeGvnMcgQWVQsUI(self):
        """MULTIALLEN_mJeGvnMcgQWVQsUI multi-experiment creation."""
        model_folder = 'MULTIALLEN_mJeGvnMcgQWVQsUI'
        exp = {
            'experiment_name': ['MULTIALLEN_mJeGvnMcgQWVQsUI'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_mJeGvnMcgQWVQsUI']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_auHHSPeDARWRFAwd(self):
        """MULTIALLEN_auHHSPeDARWRFAwd multi-experiment creation."""
        model_folder = 'MULTIALLEN_auHHSPeDARWRFAwd'
        exp = {
            'experiment_name': ['MULTIALLEN_auHHSPeDARWRFAwd'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_auHHSPeDARWRFAwd']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_XiajxKgZXEOlwGTC(self):
        """MULTIALLEN_XiajxKgZXEOlwGTC multi-experiment creation."""
        model_folder = 'MULTIALLEN_XiajxKgZXEOlwGTC'
        exp = {
            'experiment_name': ['MULTIALLEN_XiajxKgZXEOlwGTC'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_XiajxKgZXEOlwGTC']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_JsnQmsElmqNEYBiP(self):
        """MULTIALLEN_JsnQmsElmqNEYBiP multi-experiment creation."""
        model_folder = 'MULTIALLEN_JsnQmsElmqNEYBiP'
        exp = {
            'experiment_name': ['MULTIALLEN_JsnQmsElmqNEYBiP'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_JsnQmsElmqNEYBiP']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_sHDmykZpexvbHCJx(self):
        """MULTIALLEN_sHDmykZpexvbHCJx multi-experiment creation."""
        model_folder = 'MULTIALLEN_sHDmykZpexvbHCJx'
        exp = {
            'experiment_name': ['MULTIALLEN_sHDmykZpexvbHCJx'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_sHDmykZpexvbHCJx']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_SgjyyHPNkXtdsPhJ(self):
        """MULTIALLEN_SgjyyHPNkXtdsPhJ multi-experiment creation."""
        model_folder = 'MULTIALLEN_SgjyyHPNkXtdsPhJ'
        exp = {
            'experiment_name': ['MULTIALLEN_SgjyyHPNkXtdsPhJ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_SgjyyHPNkXtdsPhJ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_zmQRKgLdweBBZqdH(self):
        """MULTIALLEN_zmQRKgLdweBBZqdH multi-experiment creation."""
        model_folder = 'MULTIALLEN_zmQRKgLdweBBZqdH'
        exp = {
            'experiment_name': ['MULTIALLEN_zmQRKgLdweBBZqdH'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_zmQRKgLdweBBZqdH']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_XPVQXOcoZoJWhPUt(self):
        """MULTIALLEN_XPVQXOcoZoJWhPUt multi-experiment creation."""
        model_folder = 'MULTIALLEN_XPVQXOcoZoJWhPUt'
        exp = {
            'experiment_name': ['MULTIALLEN_XPVQXOcoZoJWhPUt'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_XPVQXOcoZoJWhPUt']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_JzfFCSRCIgayLOCo(self):
        """MULTIALLEN_JzfFCSRCIgayLOCo multi-experiment creation."""
        model_folder = 'MULTIALLEN_JzfFCSRCIgayLOCo'
        exp = {
            'experiment_name': ['MULTIALLEN_JzfFCSRCIgayLOCo'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_JzfFCSRCIgayLOCo']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_sOGHjyXgFdEhnbvr(self):
        """MULTIALLEN_sOGHjyXgFdEhnbvr multi-experiment creation."""
        model_folder = 'MULTIALLEN_sOGHjyXgFdEhnbvr'
        exp = {
            'experiment_name': ['MULTIALLEN_sOGHjyXgFdEhnbvr'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_sOGHjyXgFdEhnbvr']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_gvkbEuNVczjhpDaE(self):
        """MULTIALLEN_gvkbEuNVczjhpDaE multi-experiment creation."""
        model_folder = 'MULTIALLEN_gvkbEuNVczjhpDaE'
        exp = {
            'experiment_name': ['MULTIALLEN_gvkbEuNVczjhpDaE'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_gvkbEuNVczjhpDaE']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_vinNtvMQISZJqLkw(self):
        """MULTIALLEN_vinNtvMQISZJqLkw multi-experiment creation."""
        model_folder = 'MULTIALLEN_vinNtvMQISZJqLkw'
        exp = {
            'experiment_name': ['MULTIALLEN_vinNtvMQISZJqLkw'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_vinNtvMQISZJqLkw']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_pACKfTDziEFFpacz(self):
        """MULTIALLEN_pACKfTDziEFFpacz multi-experiment creation."""
        model_folder = 'MULTIALLEN_pACKfTDziEFFpacz'
        exp = {
            'experiment_name': ['MULTIALLEN_pACKfTDziEFFpacz'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_pACKfTDziEFFpacz']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_NdfGvJakxbTbUAtY(self):
        """MULTIALLEN_NdfGvJakxbTbUAtY multi-experiment creation."""
        model_folder = 'MULTIALLEN_NdfGvJakxbTbUAtY'
        exp = {
            'experiment_name': ['MULTIALLEN_NdfGvJakxbTbUAtY'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_NdfGvJakxbTbUAtY']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_BhXLlKwIdtHqHObU(self):
        """MULTIALLEN_BhXLlKwIdtHqHObU multi-experiment creation."""
        model_folder = 'MULTIALLEN_BhXLlKwIdtHqHObU'
        exp = {
            'experiment_name': ['MULTIALLEN_BhXLlKwIdtHqHObU'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_BhXLlKwIdtHqHObU']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_hvuQUbROVdUZkMKF(self):
        """MULTIALLEN_hvuQUbROVdUZkMKF multi-experiment creation."""
        model_folder = 'MULTIALLEN_hvuQUbROVdUZkMKF'
        exp = {
            'experiment_name': ['MULTIALLEN_hvuQUbROVdUZkMKF'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_hvuQUbROVdUZkMKF']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_GdSJbsJpPBLoDKVY(self):
        """MULTIALLEN_GdSJbsJpPBLoDKVY multi-experiment creation."""
        model_folder = 'MULTIALLEN_GdSJbsJpPBLoDKVY'
        exp = {
            'experiment_name': ['MULTIALLEN_GdSJbsJpPBLoDKVY'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_GdSJbsJpPBLoDKVY']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_lvGjrsNpguhdJDOl(self):
        """MULTIALLEN_lvGjrsNpguhdJDOl multi-experiment creation."""
        model_folder = 'MULTIALLEN_lvGjrsNpguhdJDOl'
        exp = {
            'experiment_name': ['MULTIALLEN_lvGjrsNpguhdJDOl'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_lvGjrsNpguhdJDOl']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_TlnxSBWhQLWZtWhs(self):
        """MULTIALLEN_TlnxSBWhQLWZtWhs multi-experiment creation."""
        model_folder = 'MULTIALLEN_TlnxSBWhQLWZtWhs'
        exp = {
            'experiment_name': ['MULTIALLEN_TlnxSBWhQLWZtWhs'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_TlnxSBWhQLWZtWhs']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_PitYjeWdVidTClWq(self):
        """MULTIALLEN_PitYjeWdVidTClWq multi-experiment creation."""
        model_folder = 'MULTIALLEN_PitYjeWdVidTClWq'
        exp = {
            'experiment_name': ['MULTIALLEN_PitYjeWdVidTClWq'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_PitYjeWdVidTClWq']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_VRwxSGpKFyPqlXuE(self):
        """MULTIALLEN_VRwxSGpKFyPqlXuE multi-experiment creation."""
        model_folder = 'MULTIALLEN_VRwxSGpKFyPqlXuE'
        exp = {
            'experiment_name': ['MULTIALLEN_VRwxSGpKFyPqlXuE'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_VRwxSGpKFyPqlXuE']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_utbmWQOtFPvAxFcU(self):
        """MULTIALLEN_utbmWQOtFPvAxFcU multi-experiment creation."""
        model_folder = 'MULTIALLEN_utbmWQOtFPvAxFcU'
        exp = {
            'experiment_name': ['MULTIALLEN_utbmWQOtFPvAxFcU'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_utbmWQOtFPvAxFcU']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_aQHAFoAtFYiVkCGI(self):
        """MULTIALLEN_aQHAFoAtFYiVkCGI multi-experiment creation."""
        model_folder = 'MULTIALLEN_aQHAFoAtFYiVkCGI'
        exp = {
            'experiment_name': ['MULTIALLEN_aQHAFoAtFYiVkCGI'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_aQHAFoAtFYiVkCGI']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_YaUfELtqdFIsLFlS(self):
        """MULTIALLEN_YaUfELtqdFIsLFlS multi-experiment creation."""
        model_folder = 'MULTIALLEN_YaUfELtqdFIsLFlS'
        exp = {
            'experiment_name': ['MULTIALLEN_YaUfELtqdFIsLFlS'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_YaUfELtqdFIsLFlS']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_PsIwruRoGhgWvgMc(self):
        """MULTIALLEN_PsIwruRoGhgWvgMc multi-experiment creation."""
        model_folder = 'MULTIALLEN_PsIwruRoGhgWvgMc'
        exp = {
            'experiment_name': ['MULTIALLEN_PsIwruRoGhgWvgMc'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_PsIwruRoGhgWvgMc']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_vCnixelsFSJWodlI(self):
        """MULTIALLEN_vCnixelsFSJWodlI multi-experiment creation."""
        model_folder = 'MULTIALLEN_vCnixelsFSJWodlI'
        exp = {
            'experiment_name': ['MULTIALLEN_vCnixelsFSJWodlI'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_vCnixelsFSJWodlI']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_FyLiICUJlUHrPZak(self):
        """MULTIALLEN_FyLiICUJlUHrPZak multi-experiment creation."""
        model_folder = 'MULTIALLEN_FyLiICUJlUHrPZak'
        exp = {
            'experiment_name': ['MULTIALLEN_FyLiICUJlUHrPZak'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_FyLiICUJlUHrPZak']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_uKLlmlGCYCcKMCsw(self):
        """MULTIALLEN_uKLlmlGCYCcKMCsw multi-experiment creation."""
        model_folder = 'MULTIALLEN_uKLlmlGCYCcKMCsw'
        exp = {
            'experiment_name': ['MULTIALLEN_uKLlmlGCYCcKMCsw'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_uKLlmlGCYCcKMCsw']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_VYfzLtdmdtqyTpia(self):
        """MULTIALLEN_VYfzLtdmdtqyTpia multi-experiment creation."""
        model_folder = 'MULTIALLEN_VYfzLtdmdtqyTpia'
        exp = {
            'experiment_name': ['MULTIALLEN_VYfzLtdmdtqyTpia'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_VYfzLtdmdtqyTpia']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_uaqmiuilWCJoupKt(self):
        """MULTIALLEN_uaqmiuilWCJoupKt multi-experiment creation."""
        model_folder = 'MULTIALLEN_uaqmiuilWCJoupKt'
        exp = {
            'experiment_name': ['MULTIALLEN_uaqmiuilWCJoupKt'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_uaqmiuilWCJoupKt']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_OlGEeYdOFCNuOGqM(self):
        """MULTIALLEN_OlGEeYdOFCNuOGqM multi-experiment creation."""
        model_folder = 'MULTIALLEN_OlGEeYdOFCNuOGqM'
        exp = {
            'experiment_name': ['MULTIALLEN_OlGEeYdOFCNuOGqM'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_OlGEeYdOFCNuOGqM']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_rlXdtIlrxnsRPAth(self):
        """MULTIALLEN_rlXdtIlrxnsRPAth multi-experiment creation."""
        model_folder = 'MULTIALLEN_rlXdtIlrxnsRPAth'
        exp = {
            'experiment_name': ['MULTIALLEN_rlXdtIlrxnsRPAth'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_rlXdtIlrxnsRPAth']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_zPBBTOhpCedVewCB(self):
        """MULTIALLEN_zPBBTOhpCedVewCB multi-experiment creation."""
        model_folder = 'MULTIALLEN_zPBBTOhpCedVewCB'
        exp = {
            'experiment_name': ['MULTIALLEN_zPBBTOhpCedVewCB'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_zPBBTOhpCedVewCB']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_FSdOnkqbRzjCViZo(self):
        """MULTIALLEN_FSdOnkqbRzjCViZo multi-experiment creation."""
        model_folder = 'MULTIALLEN_FSdOnkqbRzjCViZo'
        exp = {
            'experiment_name': ['MULTIALLEN_FSdOnkqbRzjCViZo'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_FSdOnkqbRzjCViZo']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_bfPBBNurKdPzedIK(self):
        """MULTIALLEN_bfPBBNurKdPzedIK multi-experiment creation."""
        model_folder = 'MULTIALLEN_bfPBBNurKdPzedIK'
        exp = {
            'experiment_name': ['MULTIALLEN_bfPBBNurKdPzedIK'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_bfPBBNurKdPzedIK']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_HsvtghEBOqCQGzio(self):
        """MULTIALLEN_HsvtghEBOqCQGzio multi-experiment creation."""
        model_folder = 'MULTIALLEN_HsvtghEBOqCQGzio'
        exp = {
            'experiment_name': ['MULTIALLEN_HsvtghEBOqCQGzio'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_HsvtghEBOqCQGzio']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ByWEfWsydBquNiBO(self):
        """MULTIALLEN_ByWEfWsydBquNiBO multi-experiment creation."""
        model_folder = 'MULTIALLEN_ByWEfWsydBquNiBO'
        exp = {
            'experiment_name': ['MULTIALLEN_ByWEfWsydBquNiBO'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ByWEfWsydBquNiBO']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_TtvpfPwUqTOmmqWq(self):
        """MULTIALLEN_TtvpfPwUqTOmmqWq multi-experiment creation."""
        model_folder = 'MULTIALLEN_TtvpfPwUqTOmmqWq'
        exp = {
            'experiment_name': ['MULTIALLEN_TtvpfPwUqTOmmqWq'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_TtvpfPwUqTOmmqWq']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_eUwnTncTEdbxrLYX(self):
        """MULTIALLEN_eUwnTncTEdbxrLYX multi-experiment creation."""
        model_folder = 'MULTIALLEN_eUwnTncTEdbxrLYX'
        exp = {
            'experiment_name': ['MULTIALLEN_eUwnTncTEdbxrLYX'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_eUwnTncTEdbxrLYX']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_gaEWyrnitBkhvMec(self):
        """MULTIALLEN_gaEWyrnitBkhvMec multi-experiment creation."""
        model_folder = 'MULTIALLEN_gaEWyrnitBkhvMec'
        exp = {
            'experiment_name': ['MULTIALLEN_gaEWyrnitBkhvMec'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_gaEWyrnitBkhvMec']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_bilkcRHCobtbFWhb(self):
        """MULTIALLEN_bilkcRHCobtbFWhb multi-experiment creation."""
        model_folder = 'MULTIALLEN_bilkcRHCobtbFWhb'
        exp = {
            'experiment_name': ['MULTIALLEN_bilkcRHCobtbFWhb'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_bilkcRHCobtbFWhb']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_BqEapGVfzueVJHbA(self):
        """MULTIALLEN_BqEapGVfzueVJHbA multi-experiment creation."""
        model_folder = 'MULTIALLEN_BqEapGVfzueVJHbA'
        exp = {
            'experiment_name': ['MULTIALLEN_BqEapGVfzueVJHbA'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_BqEapGVfzueVJHbA']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_gSWlmBNdrsbehAJS(self):
        """MULTIALLEN_gSWlmBNdrsbehAJS multi-experiment creation."""
        model_folder = 'MULTIALLEN_gSWlmBNdrsbehAJS'
        exp = {
            'experiment_name': ['MULTIALLEN_gSWlmBNdrsbehAJS'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_gSWlmBNdrsbehAJS']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_tceSQIUROdvWYYIa(self):
        """MULTIALLEN_tceSQIUROdvWYYIa multi-experiment creation."""
        model_folder = 'MULTIALLEN_tceSQIUROdvWYYIa'
        exp = {
            'experiment_name': ['MULTIALLEN_tceSQIUROdvWYYIa'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_tceSQIUROdvWYYIa']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_GVnENmqqxHXZgLRz(self):
        """MULTIALLEN_GVnENmqqxHXZgLRz multi-experiment creation."""
        model_folder = 'MULTIALLEN_GVnENmqqxHXZgLRz'
        exp = {
            'experiment_name': ['MULTIALLEN_GVnENmqqxHXZgLRz'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_GVnENmqqxHXZgLRz']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_OzgthDyMfIBYfXZv(self):
        """MULTIALLEN_OzgthDyMfIBYfXZv multi-experiment creation."""
        model_folder = 'MULTIALLEN_OzgthDyMfIBYfXZv'
        exp = {
            'experiment_name': ['MULTIALLEN_OzgthDyMfIBYfXZv'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_OzgthDyMfIBYfXZv']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_BYEvHfccJxPzKgJh(self):
        """MULTIALLEN_BYEvHfccJxPzKgJh multi-experiment creation."""
        model_folder = 'MULTIALLEN_BYEvHfccJxPzKgJh'
        exp = {
            'experiment_name': ['MULTIALLEN_BYEvHfccJxPzKgJh'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_BYEvHfccJxPzKgJh']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_nUscdTPwNCWdiWgY(self):
        """MULTIALLEN_nUscdTPwNCWdiWgY multi-experiment creation."""
        model_folder = 'MULTIALLEN_nUscdTPwNCWdiWgY'
        exp = {
            'experiment_name': ['MULTIALLEN_nUscdTPwNCWdiWgY'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_nUscdTPwNCWdiWgY']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_koSiiGqgjnidaugL(self):
        """MULTIALLEN_koSiiGqgjnidaugL multi-experiment creation."""
        model_folder = 'MULTIALLEN_koSiiGqgjnidaugL'
        exp = {
            'experiment_name': ['MULTIALLEN_koSiiGqgjnidaugL'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_koSiiGqgjnidaugL']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_vepCSbChazegwaTR(self):
        """MULTIALLEN_vepCSbChazegwaTR multi-experiment creation."""
        model_folder = 'MULTIALLEN_vepCSbChazegwaTR'
        exp = {
            'experiment_name': ['MULTIALLEN_vepCSbChazegwaTR'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_vepCSbChazegwaTR']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_aAEQFNittnjPDNrg(self):
        """MULTIALLEN_aAEQFNittnjPDNrg multi-experiment creation."""
        model_folder = 'MULTIALLEN_aAEQFNittnjPDNrg'
        exp = {
            'experiment_name': ['MULTIALLEN_aAEQFNittnjPDNrg'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_aAEQFNittnjPDNrg']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_pyeGwXlURZJUMMmR(self):
        """MULTIALLEN_pyeGwXlURZJUMMmR multi-experiment creation."""
        model_folder = 'MULTIALLEN_pyeGwXlURZJUMMmR'
        exp = {
            'experiment_name': ['MULTIALLEN_pyeGwXlURZJUMMmR'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_pyeGwXlURZJUMMmR']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_IJAbmcqvfWkLbvdI(self):
        """MULTIALLEN_IJAbmcqvfWkLbvdI multi-experiment creation."""
        model_folder = 'MULTIALLEN_IJAbmcqvfWkLbvdI'
        exp = {
            'experiment_name': ['MULTIALLEN_IJAbmcqvfWkLbvdI'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_IJAbmcqvfWkLbvdI']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_kQVwgntOsSCkOQNL(self):
        """MULTIALLEN_kQVwgntOsSCkOQNL multi-experiment creation."""
        model_folder = 'MULTIALLEN_kQVwgntOsSCkOQNL'
        exp = {
            'experiment_name': ['MULTIALLEN_kQVwgntOsSCkOQNL'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_kQVwgntOsSCkOQNL']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_cxaOtfpioWJSTvVF(self):
        """MULTIALLEN_cxaOtfpioWJSTvVF multi-experiment creation."""
        model_folder = 'MULTIALLEN_cxaOtfpioWJSTvVF'
        exp = {
            'experiment_name': ['MULTIALLEN_cxaOtfpioWJSTvVF'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_cxaOtfpioWJSTvVF']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_HBKiaILMgPXgdhaK(self):
        """MULTIALLEN_HBKiaILMgPXgdhaK multi-experiment creation."""
        model_folder = 'MULTIALLEN_HBKiaILMgPXgdhaK'
        exp = {
            'experiment_name': ['MULTIALLEN_HBKiaILMgPXgdhaK'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_HBKiaILMgPXgdhaK']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_VmjVvIrEKEaAoQuS(self):
        """MULTIALLEN_VmjVvIrEKEaAoQuS multi-experiment creation."""
        model_folder = 'MULTIALLEN_VmjVvIrEKEaAoQuS'
        exp = {
            'experiment_name': ['MULTIALLEN_VmjVvIrEKEaAoQuS'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_VmjVvIrEKEaAoQuS']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_xyvhcTOMvgUvVMjA(self):
        """MULTIALLEN_xyvhcTOMvgUvVMjA multi-experiment creation."""
        model_folder = 'MULTIALLEN_xyvhcTOMvgUvVMjA'
        exp = {
            'experiment_name': ['MULTIALLEN_xyvhcTOMvgUvVMjA'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_xyvhcTOMvgUvVMjA']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_UJRPeATnsuSvulkY(self):
        """MULTIALLEN_UJRPeATnsuSvulkY multi-experiment creation."""
        model_folder = 'MULTIALLEN_UJRPeATnsuSvulkY'
        exp = {
            'experiment_name': ['MULTIALLEN_UJRPeATnsuSvulkY'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_UJRPeATnsuSvulkY']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_LnYClccaRRdRQZtP(self):
        """MULTIALLEN_LnYClccaRRdRQZtP multi-experiment creation."""
        model_folder = 'MULTIALLEN_LnYClccaRRdRQZtP'
        exp = {
            'experiment_name': ['MULTIALLEN_LnYClccaRRdRQZtP'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_LnYClccaRRdRQZtP']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_eczebvFPFbQBRlDR(self):
        """MULTIALLEN_eczebvFPFbQBRlDR multi-experiment creation."""
        model_folder = 'MULTIALLEN_eczebvFPFbQBRlDR'
        exp = {
            'experiment_name': ['MULTIALLEN_eczebvFPFbQBRlDR'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_eczebvFPFbQBRlDR']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_PiKNtQmcvYRaACNe(self):
        """MULTIALLEN_PiKNtQmcvYRaACNe multi-experiment creation."""
        model_folder = 'MULTIALLEN_PiKNtQmcvYRaACNe'
        exp = {
            'experiment_name': ['MULTIALLEN_PiKNtQmcvYRaACNe'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_PiKNtQmcvYRaACNe']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_oNYYoZbiZzEyQbas(self):
        """MULTIALLEN_oNYYoZbiZzEyQbas multi-experiment creation."""
        model_folder = 'MULTIALLEN_oNYYoZbiZzEyQbas'
        exp = {
            'experiment_name': ['MULTIALLEN_oNYYoZbiZzEyQbas'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_oNYYoZbiZzEyQbas']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_KpyfIqhorwKoDZTS(self):
        """MULTIALLEN_KpyfIqhorwKoDZTS multi-experiment creation."""
        model_folder = 'MULTIALLEN_KpyfIqhorwKoDZTS'
        exp = {
            'experiment_name': ['MULTIALLEN_KpyfIqhorwKoDZTS'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_KpyfIqhorwKoDZTS']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_tasIkXrAopylMwuz(self):
        """MULTIALLEN_tasIkXrAopylMwuz multi-experiment creation."""
        model_folder = 'MULTIALLEN_tasIkXrAopylMwuz'
        exp = {
            'experiment_name': ['MULTIALLEN_tasIkXrAopylMwuz'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_tasIkXrAopylMwuz']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_dXmJZqIECSfFKgNW(self):
        """MULTIALLEN_dXmJZqIECSfFKgNW multi-experiment creation."""
        model_folder = 'MULTIALLEN_dXmJZqIECSfFKgNW'
        exp = {
            'experiment_name': ['MULTIALLEN_dXmJZqIECSfFKgNW'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_dXmJZqIECSfFKgNW']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_thdipHNHvHDGlMip(self):
        """MULTIALLEN_thdipHNHvHDGlMip multi-experiment creation."""
        model_folder = 'MULTIALLEN_thdipHNHvHDGlMip'
        exp = {
            'experiment_name': ['MULTIALLEN_thdipHNHvHDGlMip'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_thdipHNHvHDGlMip']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_XrNaunWsmLECiUeM(self):
        """MULTIALLEN_XrNaunWsmLECiUeM multi-experiment creation."""
        model_folder = 'MULTIALLEN_XrNaunWsmLECiUeM'
        exp = {
            'experiment_name': ['MULTIALLEN_XrNaunWsmLECiUeM'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_XrNaunWsmLECiUeM']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_IyuWpfukmSmyEDcz(self):
        """MULTIALLEN_IyuWpfukmSmyEDcz multi-experiment creation."""
        model_folder = 'MULTIALLEN_IyuWpfukmSmyEDcz'
        exp = {
            'experiment_name': ['MULTIALLEN_IyuWpfukmSmyEDcz'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_IyuWpfukmSmyEDcz']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_qDsjofcKSlUlnavj(self):
        """MULTIALLEN_qDsjofcKSlUlnavj multi-experiment creation."""
        model_folder = 'MULTIALLEN_qDsjofcKSlUlnavj'
        exp = {
            'experiment_name': ['MULTIALLEN_qDsjofcKSlUlnavj'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_qDsjofcKSlUlnavj']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_prAHBSvgRQCRkthZ(self):
        """MULTIALLEN_prAHBSvgRQCRkthZ multi-experiment creation."""
        model_folder = 'MULTIALLEN_prAHBSvgRQCRkthZ'
        exp = {
            'experiment_name': ['MULTIALLEN_prAHBSvgRQCRkthZ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_prAHBSvgRQCRkthZ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_LbcKGKHWfPZZNhaq(self):
        """MULTIALLEN_LbcKGKHWfPZZNhaq multi-experiment creation."""
        model_folder = 'MULTIALLEN_LbcKGKHWfPZZNhaq'
        exp = {
            'experiment_name': ['MULTIALLEN_LbcKGKHWfPZZNhaq'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_LbcKGKHWfPZZNhaq']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_atmFLkqSNDawvqhN(self):
        """MULTIALLEN_atmFLkqSNDawvqhN multi-experiment creation."""
        model_folder = 'MULTIALLEN_atmFLkqSNDawvqhN'
        exp = {
            'experiment_name': ['MULTIALLEN_atmFLkqSNDawvqhN'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_atmFLkqSNDawvqhN']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_weuMkhFEBUhoYtye(self):
        """MULTIALLEN_weuMkhFEBUhoYtye multi-experiment creation."""
        model_folder = 'MULTIALLEN_weuMkhFEBUhoYtye'
        exp = {
            'experiment_name': ['MULTIALLEN_weuMkhFEBUhoYtye'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_weuMkhFEBUhoYtye']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_mhbKEdzDLaLTfKOO(self):
        """MULTIALLEN_mhbKEdzDLaLTfKOO multi-experiment creation."""
        model_folder = 'MULTIALLEN_mhbKEdzDLaLTfKOO'
        exp = {
            'experiment_name': ['MULTIALLEN_mhbKEdzDLaLTfKOO'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_mhbKEdzDLaLTfKOO']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_piwTkQyqjKOcXqeB(self):
        """MULTIALLEN_piwTkQyqjKOcXqeB multi-experiment creation."""
        model_folder = 'MULTIALLEN_piwTkQyqjKOcXqeB'
        exp = {
            'experiment_name': ['MULTIALLEN_piwTkQyqjKOcXqeB'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_piwTkQyqjKOcXqeB']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_tKNGdgRwfZyiqMEQ(self):
        """MULTIALLEN_tKNGdgRwfZyiqMEQ multi-experiment creation."""
        model_folder = 'MULTIALLEN_tKNGdgRwfZyiqMEQ'
        exp = {
            'experiment_name': ['MULTIALLEN_tKNGdgRwfZyiqMEQ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_tKNGdgRwfZyiqMEQ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ZgJQiWhyTnvgcWDt(self):
        """MULTIALLEN_ZgJQiWhyTnvgcWDt multi-experiment creation."""
        model_folder = 'MULTIALLEN_ZgJQiWhyTnvgcWDt'
        exp = {
            'experiment_name': ['MULTIALLEN_ZgJQiWhyTnvgcWDt'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ZgJQiWhyTnvgcWDt']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_MnsNblFLnsuefabd(self):
        """MULTIALLEN_MnsNblFLnsuefabd multi-experiment creation."""
        model_folder = 'MULTIALLEN_MnsNblFLnsuefabd'
        exp = {
            'experiment_name': ['MULTIALLEN_MnsNblFLnsuefabd'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_MnsNblFLnsuefabd']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ibCxrBNXiGDQDizt(self):
        """MULTIALLEN_ibCxrBNXiGDQDizt multi-experiment creation."""
        model_folder = 'MULTIALLEN_ibCxrBNXiGDQDizt'
        exp = {
            'experiment_name': ['MULTIALLEN_ibCxrBNXiGDQDizt'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ibCxrBNXiGDQDizt']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_RQodICsruNQVDhYz(self):
        """MULTIALLEN_RQodICsruNQVDhYz multi-experiment creation."""
        model_folder = 'MULTIALLEN_RQodICsruNQVDhYz'
        exp = {
            'experiment_name': ['MULTIALLEN_RQodICsruNQVDhYz'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_RQodICsruNQVDhYz']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_FTskGoTbEmccWKoK(self):
        """MULTIALLEN_FTskGoTbEmccWKoK multi-experiment creation."""
        model_folder = 'MULTIALLEN_FTskGoTbEmccWKoK'
        exp = {
            'experiment_name': ['MULTIALLEN_FTskGoTbEmccWKoK'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_FTskGoTbEmccWKoK']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_WoOEmhsiGsvLlLkS(self):
        """MULTIALLEN_WoOEmhsiGsvLlLkS multi-experiment creation."""
        model_folder = 'MULTIALLEN_WoOEmhsiGsvLlLkS'
        exp = {
            'experiment_name': ['MULTIALLEN_WoOEmhsiGsvLlLkS'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_WoOEmhsiGsvLlLkS']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_TmyumkNzxmrAnWWv(self):
        """MULTIALLEN_TmyumkNzxmrAnWWv multi-experiment creation."""
        model_folder = 'MULTIALLEN_TmyumkNzxmrAnWWv'
        exp = {
            'experiment_name': ['MULTIALLEN_TmyumkNzxmrAnWWv'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_TmyumkNzxmrAnWWv']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_TsbyOpMQXMFmSkxT(self):
        """MULTIALLEN_TsbyOpMQXMFmSkxT multi-experiment creation."""
        model_folder = 'MULTIALLEN_TsbyOpMQXMFmSkxT'
        exp = {
            'experiment_name': ['MULTIALLEN_TsbyOpMQXMFmSkxT'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_TsbyOpMQXMFmSkxT']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_dbPtpagBfDcewNCJ(self):
        """MULTIALLEN_dbPtpagBfDcewNCJ multi-experiment creation."""
        model_folder = 'MULTIALLEN_dbPtpagBfDcewNCJ'
        exp = {
            'experiment_name': ['MULTIALLEN_dbPtpagBfDcewNCJ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_dbPtpagBfDcewNCJ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_pVsadQKjStnIZECI(self):
        """MULTIALLEN_pVsadQKjStnIZECI multi-experiment creation."""
        model_folder = 'MULTIALLEN_pVsadQKjStnIZECI'
        exp = {
            'experiment_name': ['MULTIALLEN_pVsadQKjStnIZECI'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_pVsadQKjStnIZECI']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_MBbTgWFIcTZhXvSE(self):
        """MULTIALLEN_MBbTgWFIcTZhXvSE multi-experiment creation."""
        model_folder = 'MULTIALLEN_MBbTgWFIcTZhXvSE'
        exp = {
            'experiment_name': ['MULTIALLEN_MBbTgWFIcTZhXvSE'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_MBbTgWFIcTZhXvSE']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_ySHpKxVhwoNnHuIv(self):
        """MULTIALLEN_ySHpKxVhwoNnHuIv multi-experiment creation."""
        model_folder = 'MULTIALLEN_ySHpKxVhwoNnHuIv'
        exp = {
            'experiment_name': ['MULTIALLEN_ySHpKxVhwoNnHuIv'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ySHpKxVhwoNnHuIv']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_nhASSSPnMfCHCVzf(self):
        """MULTIALLEN_nhASSSPnMfCHCVzf multi-experiment creation."""
        model_folder = 'MULTIALLEN_nhASSSPnMfCHCVzf'
        exp = {
            'experiment_name': ['MULTIALLEN_nhASSSPnMfCHCVzf'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_nhASSSPnMfCHCVzf']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_pQQagJMqIJhvaWKt(self):
        """MULTIALLEN_pQQagJMqIJhvaWKt multi-experiment creation."""
        model_folder = 'MULTIALLEN_pQQagJMqIJhvaWKt'
        exp = {
            'experiment_name': ['MULTIALLEN_pQQagJMqIJhvaWKt'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_pQQagJMqIJhvaWKt']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_rBPxhABaeBSLADDg(self):
        """MULTIALLEN_rBPxhABaeBSLADDg multi-experiment creation."""
        model_folder = 'MULTIALLEN_rBPxhABaeBSLADDg'
        exp = {
            'experiment_name': ['MULTIALLEN_rBPxhABaeBSLADDg'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_rBPxhABaeBSLADDg']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_unYWBloQoeHEPzap(self):
        """MULTIALLEN_unYWBloQoeHEPzap multi-experiment creation."""
        model_folder = 'MULTIALLEN_unYWBloQoeHEPzap'
        exp = {
            'experiment_name': ['MULTIALLEN_unYWBloQoeHEPzap'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_unYWBloQoeHEPzap']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_MkFhDJEmEMquFLaE(self):
        """MULTIALLEN_MkFhDJEmEMquFLaE multi-experiment creation."""
        model_folder = 'MULTIALLEN_MkFhDJEmEMquFLaE'
        exp = {
            'experiment_name': ['MULTIALLEN_MkFhDJEmEMquFLaE'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_MkFhDJEmEMquFLaE']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_GxNbaUfUGVctSxVr(self):
        """MULTIALLEN_GxNbaUfUGVctSxVr multi-experiment creation."""
        model_folder = 'MULTIALLEN_GxNbaUfUGVctSxVr'
        exp = {
            'experiment_name': ['MULTIALLEN_GxNbaUfUGVctSxVr'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_GxNbaUfUGVctSxVr']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_viRwjNcoUbxvucpC(self):
        """MULTIALLEN_viRwjNcoUbxvucpC multi-experiment creation."""
        model_folder = 'MULTIALLEN_viRwjNcoUbxvucpC'
        exp = {
            'experiment_name': ['MULTIALLEN_viRwjNcoUbxvucpC'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_viRwjNcoUbxvucpC']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_BksRoDSCYZrsWJoj(self):
        """MULTIALLEN_BksRoDSCYZrsWJoj multi-experiment creation."""
        model_folder = 'MULTIALLEN_BksRoDSCYZrsWJoj'
        exp = {
            'experiment_name': ['MULTIALLEN_BksRoDSCYZrsWJoj'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_BksRoDSCYZrsWJoj']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_svVrxDNkiANKxgYV(self):
        """MULTIALLEN_svVrxDNkiANKxgYV multi-experiment creation."""
        model_folder = 'MULTIALLEN_svVrxDNkiANKxgYV'
        exp = {
            'experiment_name': ['MULTIALLEN_svVrxDNkiANKxgYV'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_svVrxDNkiANKxgYV']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_hVEuxluQGDDGoKjj(self):
        """MULTIALLEN_hVEuxluQGDDGoKjj multi-experiment creation."""
        model_folder = 'MULTIALLEN_hVEuxluQGDDGoKjj'
        exp = {
            'experiment_name': ['MULTIALLEN_hVEuxluQGDDGoKjj'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_hVEuxluQGDDGoKjj']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_eoiCszUGaZrbvwFt(self):
        """MULTIALLEN_eoiCszUGaZrbvwFt multi-experiment creation."""
        model_folder = 'MULTIALLEN_eoiCszUGaZrbvwFt'
        exp = {
            'experiment_name': ['MULTIALLEN_eoiCszUGaZrbvwFt'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_eoiCszUGaZrbvwFt']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_SQAgAthmuLylThOi(self):
        """MULTIALLEN_SQAgAthmuLylThOi multi-experiment creation."""
        model_folder = 'MULTIALLEN_SQAgAthmuLylThOi'
        exp = {
            'experiment_name': ['MULTIALLEN_SQAgAthmuLylThOi'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_SQAgAthmuLylThOi']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_gWlKxQnMfUNXvljz(self):
        """MULTIALLEN_gWlKxQnMfUNXvljz multi-experiment creation."""
        model_folder = 'MULTIALLEN_gWlKxQnMfUNXvljz'
        exp = {
            'experiment_name': ['MULTIALLEN_gWlKxQnMfUNXvljz'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_gWlKxQnMfUNXvljz']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_TeOaaYIRhXYPErpe(self):
        """MULTIALLEN_TeOaaYIRhXYPErpe multi-experiment creation."""
        model_folder = 'MULTIALLEN_TeOaaYIRhXYPErpe'
        exp = {
            'experiment_name': ['MULTIALLEN_TeOaaYIRhXYPErpe'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_TeOaaYIRhXYPErpe']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_opgQypzzkvukzYcS(self):
        """MULTIALLEN_opgQypzzkvukzYcS multi-experiment creation."""
        model_folder = 'MULTIALLEN_opgQypzzkvukzYcS'
        exp = {
            'experiment_name': ['MULTIALLEN_opgQypzzkvukzYcS'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_opgQypzzkvukzYcS']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_sidkXitAqBkWUweG(self):
        """MULTIALLEN_sidkXitAqBkWUweG multi-experiment creation."""
        model_folder = 'MULTIALLEN_sidkXitAqBkWUweG'
        exp = {
            'experiment_name': ['MULTIALLEN_sidkXitAqBkWUweG'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_sidkXitAqBkWUweG']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_bGLsxYSEoUXGAuXo(self):
        """MULTIALLEN_bGLsxYSEoUXGAuXo multi-experiment creation."""
        model_folder = 'MULTIALLEN_bGLsxYSEoUXGAuXo'
        exp = {
            'experiment_name': ['MULTIALLEN_bGLsxYSEoUXGAuXo'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_bGLsxYSEoUXGAuXo']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_TzaZHbeaJaeRUiPg(self):
        """MULTIALLEN_TzaZHbeaJaeRUiPg multi-experiment creation."""
        model_folder = 'MULTIALLEN_TzaZHbeaJaeRUiPg'
        exp = {
            'experiment_name': ['MULTIALLEN_TzaZHbeaJaeRUiPg'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_TzaZHbeaJaeRUiPg']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_XsALjTkMCUwjKves(self):
        """MULTIALLEN_XsALjTkMCUwjKves multi-experiment creation."""
        model_folder = 'MULTIALLEN_XsALjTkMCUwjKves'
        exp = {
            'experiment_name': ['MULTIALLEN_XsALjTkMCUwjKves'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_XsALjTkMCUwjKves']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_EwlkQFglsOBWDsno(self):
        """MULTIALLEN_EwlkQFglsOBWDsno multi-experiment creation."""
        model_folder = 'MULTIALLEN_EwlkQFglsOBWDsno'
        exp = {
            'experiment_name': ['MULTIALLEN_EwlkQFglsOBWDsno'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_EwlkQFglsOBWDsno']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_woQZYwxwyjnuewQD(self):
        """MULTIALLEN_woQZYwxwyjnuewQD multi-experiment creation."""
        model_folder = 'MULTIALLEN_woQZYwxwyjnuewQD'
        exp = {
            'experiment_name': ['MULTIALLEN_woQZYwxwyjnuewQD'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_woQZYwxwyjnuewQD']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_biXBhSCYbJZFtPDA(self):
        """MULTIALLEN_biXBhSCYbJZFtPDA multi-experiment creation."""
        model_folder = 'MULTIALLEN_biXBhSCYbJZFtPDA'
        exp = {
            'experiment_name': ['MULTIALLEN_biXBhSCYbJZFtPDA'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_biXBhSCYbJZFtPDA']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_NqybpYjzsPXEjVHL(self):
        """MULTIALLEN_NqybpYjzsPXEjVHL multi-experiment creation."""
        model_folder = 'MULTIALLEN_NqybpYjzsPXEjVHL'
        exp = {
            'experiment_name': ['MULTIALLEN_NqybpYjzsPXEjVHL'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_NqybpYjzsPXEjVHL']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_eDTyJpoIJhTvwwWN(self):
        """MULTIALLEN_eDTyJpoIJhTvwwWN multi-experiment creation."""
        model_folder = 'MULTIALLEN_eDTyJpoIJhTvwwWN'
        exp = {
            'experiment_name': ['MULTIALLEN_eDTyJpoIJhTvwwWN'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_eDTyJpoIJhTvwwWN']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_cxWWXUuIAxemzCZd(self):
        """MULTIALLEN_cxWWXUuIAxemzCZd multi-experiment creation."""
        model_folder = 'MULTIALLEN_cxWWXUuIAxemzCZd'
        exp = {
            'experiment_name': ['MULTIALLEN_cxWWXUuIAxemzCZd'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_cxWWXUuIAxemzCZd']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_cLQTcZISTqJnxJtm(self):
        """MULTIALLEN_cLQTcZISTqJnxJtm multi-experiment creation."""
        model_folder = 'MULTIALLEN_cLQTcZISTqJnxJtm'
        exp = {
            'experiment_name': ['MULTIALLEN_cLQTcZISTqJnxJtm'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_cLQTcZISTqJnxJtm']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_QfXSiPgLhEJXJHET(self):
        """MULTIALLEN_QfXSiPgLhEJXJHET multi-experiment creation."""
        model_folder = 'MULTIALLEN_QfXSiPgLhEJXJHET'
        exp = {
            'experiment_name': ['MULTIALLEN_QfXSiPgLhEJXJHET'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_QfXSiPgLhEJXJHET']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_uNYEZyZLUqfsiAUE(self):
        """MULTIALLEN_uNYEZyZLUqfsiAUE multi-experiment creation."""
        model_folder = 'MULTIALLEN_uNYEZyZLUqfsiAUE'
        exp = {
            'experiment_name': ['MULTIALLEN_uNYEZyZLUqfsiAUE'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_uNYEZyZLUqfsiAUE']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_fqWQNUpCFbciAkFZ(self):
        """MULTIALLEN_fqWQNUpCFbciAkFZ multi-experiment creation."""
        model_folder = 'MULTIALLEN_fqWQNUpCFbciAkFZ'
        exp = {
            'experiment_name': ['MULTIALLEN_fqWQNUpCFbciAkFZ'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_fqWQNUpCFbciAkFZ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_sTwlRvuFzAwZfdwD(self):
        """MULTIALLEN_sTwlRvuFzAwZfdwD multi-experiment creation."""
        model_folder = 'MULTIALLEN_sTwlRvuFzAwZfdwD'
        exp = {
            'experiment_name': ['MULTIALLEN_sTwlRvuFzAwZfdwD'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_sTwlRvuFzAwZfdwD']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_brUGETyibqWfrhOi(self):
        """MULTIALLEN_brUGETyibqWfrhOi multi-experiment creation."""
        model_folder = 'MULTIALLEN_brUGETyibqWfrhOi'
        exp = {
            'experiment_name': ['MULTIALLEN_brUGETyibqWfrhOi'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_brUGETyibqWfrhOi']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_hMirIAcGiZXvAWMC(self):
        """MULTIALLEN_hMirIAcGiZXvAWMC multi-experiment creation."""
        model_folder = 'MULTIALLEN_hMirIAcGiZXvAWMC'
        exp = {
            'experiment_name': ['MULTIALLEN_hMirIAcGiZXvAWMC'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_hMirIAcGiZXvAWMC']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_oqPfziLFEcUTgRGM(self):
        """MULTIALLEN_oqPfziLFEcUTgRGM multi-experiment creation."""
        model_folder = 'MULTIALLEN_oqPfziLFEcUTgRGM'
        exp = {
            'experiment_name': ['MULTIALLEN_oqPfziLFEcUTgRGM'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_oqPfziLFEcUTgRGM']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_wRpmHkCxAMDMwERR(self):
        """MULTIALLEN_wRpmHkCxAMDMwERR multi-experiment creation."""
        model_folder = 'MULTIALLEN_wRpmHkCxAMDMwERR'
        exp = {
            'experiment_name': ['MULTIALLEN_wRpmHkCxAMDMwERR'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_wRpmHkCxAMDMwERR']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_rGAXpfCXrGquPswn(self):
        """MULTIALLEN_rGAXpfCXrGquPswn multi-experiment creation."""
        model_folder = 'MULTIALLEN_rGAXpfCXrGquPswn'
        exp = {
            'experiment_name': ['MULTIALLEN_rGAXpfCXrGquPswn'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_rGAXpfCXrGquPswn']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_XRSUsyWMkYefCUGM(self):
        """MULTIALLEN_XRSUsyWMkYefCUGM multi-experiment creation."""
        model_folder = 'MULTIALLEN_XRSUsyWMkYefCUGM'
        exp = {
            'experiment_name': ['MULTIALLEN_XRSUsyWMkYefCUGM'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_XRSUsyWMkYefCUGM']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_riaigzuhJiUKmheC(self):
        """MULTIALLEN_riaigzuhJiUKmheC multi-experiment creation."""
        model_folder = 'MULTIALLEN_riaigzuhJiUKmheC'
        exp = {
            'experiment_name': ['MULTIALLEN_riaigzuhJiUKmheC'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_riaigzuhJiUKmheC']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_PXTQVsrRirfVjVkt(self):
        """MULTIALLEN_PXTQVsrRirfVjVkt multi-experiment creation."""
        model_folder = 'MULTIALLEN_PXTQVsrRirfVjVkt'
        exp = {
            'experiment_name': ['MULTIALLEN_PXTQVsrRirfVjVkt'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_PXTQVsrRirfVjVkt']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_wtrNFVfufHCMpdSb(self):
        """MULTIALLEN_wtrNFVfufHCMpdSb multi-experiment creation."""
        model_folder = 'MULTIALLEN_wtrNFVfufHCMpdSb'
        exp = {
            'experiment_name': ['MULTIALLEN_wtrNFVfufHCMpdSb'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_wtrNFVfufHCMpdSb']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_zymZqihMQknNPpZt(self):
        """MULTIALLEN_zymZqihMQknNPpZt multi-experiment creation."""
        model_folder = 'MULTIALLEN_zymZqihMQknNPpZt'
        exp = {
            'experiment_name': ['MULTIALLEN_zymZqihMQknNPpZt'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_zymZqihMQknNPpZt']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_rcgaZWjApJKUvJND(self):
        """MULTIALLEN_rcgaZWjApJKUvJND multi-experiment creation."""
        model_folder = 'MULTIALLEN_rcgaZWjApJKUvJND'
        exp = {
            'experiment_name': ['MULTIALLEN_rcgaZWjApJKUvJND'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_rcgaZWjApJKUvJND']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_zxUHiiSwrDzTiLNY(self):
        """MULTIALLEN_zxUHiiSwrDzTiLNY multi-experiment creation."""
        model_folder = 'MULTIALLEN_zxUHiiSwrDzTiLNY'
        exp = {
            'experiment_name': ['MULTIALLEN_zxUHiiSwrDzTiLNY'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_zxUHiiSwrDzTiLNY']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_xGCvpWaHjcWWgZNP(self):
        """MULTIALLEN_xGCvpWaHjcWWgZNP multi-experiment creation."""
        model_folder = 'MULTIALLEN_xGCvpWaHjcWWgZNP'
        exp = {
            'experiment_name': ['MULTIALLEN_xGCvpWaHjcWWgZNP'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_xGCvpWaHjcWWgZNP']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp


    def MULTIALLEN_oNNkcKCdDwwJTkNt(self):
        """MULTIALLEN_oNNkcKCdDwwJTkNt multi-experiment creation."""
        model_folder = 'MULTIALLEN_oNNkcKCdDwwJTkNt'
        exp = {
            'experiment_name': ['MULTIALLEN_oNNkcKCdDwwJTkNt'],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_oNNkcKCdDwwJTkNt']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 150
        exp['num_validation_evals'] = 20
        exp['batch_size'] = 16  # Train/val batch size.
        exp['normalize_labels'] = 'zscore'
        return exp

