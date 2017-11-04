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
            'dataset': ['ALLEN_selected_cells_1']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = [['resize']]
        exp['epochs'] = 100
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 32  # Train/val batch size.
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


    def GXQsEIlMiBzOrqGd(self):
        """VISp_Cux2-CreERT2_175_7900_7100_0 multi-experiment creation."""
        model_folder = 'VISp_Cux2-CreERT2_175_7900_7100_0'
        exp = {
            'experiment_name': [VISp_Cux2-CreERT2_175_7900_7100_0],
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
            'dataset': ['VISp_Cux2-CreERT2_175_7900_7100_0']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = [['resize']]
        exp['epochs'] = 100
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 32  # Train/val batch size.
        return exp


    def MULTIALLEN_wtBNaqommlfvSMAP(self):
        """MULTIALLEN_wtBNaqommlfvSMAP multi-experiment creation."""
        model_folder = 'MULTIALLEN_wtBNaqommlfvSMAP'
        exp = {
            'experiment_name': ['MULTIALLEN_wtBNaqommlfvSMAP'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_wtBNaqommlfvSMAP']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_scLpPgPocQAHvIYH(self):
        """MULTIALLEN_scLpPgPocQAHvIYH multi-experiment creation."""
        model_folder = 'MULTIALLEN_scLpPgPocQAHvIYH'
        exp = {
            'experiment_name': ['MULTIALLEN_scLpPgPocQAHvIYH'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_scLpPgPocQAHvIYH']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_jBfozlwljjGBSIpM(self):
        """MULTIALLEN_jBfozlwljjGBSIpM multi-experiment creation."""
        model_folder = 'MULTIALLEN_jBfozlwljjGBSIpM'
        exp = {
            'experiment_name': ['MULTIALLEN_jBfozlwljjGBSIpM'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_jBfozlwljjGBSIpM']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_hpScdAGDZswSLUXW(self):
        """MULTIALLEN_hpScdAGDZswSLUXW multi-experiment creation."""
        model_folder = 'MULTIALLEN_hpScdAGDZswSLUXW'
        exp = {
            'experiment_name': ['MULTIALLEN_hpScdAGDZswSLUXW'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_hpScdAGDZswSLUXW']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_FPOaVUJEGsvjnyBb(self):
        """MULTIALLEN_FPOaVUJEGsvjnyBb multi-experiment creation."""
        model_folder = 'MULTIALLEN_FPOaVUJEGsvjnyBb'
        exp = {
            'experiment_name': ['MULTIALLEN_FPOaVUJEGsvjnyBb'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_FPOaVUJEGsvjnyBb']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_egJhfzShmKTOVThi(self):
        """MULTIALLEN_egJhfzShmKTOVThi multi-experiment creation."""
        model_folder = 'MULTIALLEN_egJhfzShmKTOVThi'
        exp = {
            'experiment_name': ['MULTIALLEN_egJhfzShmKTOVThi'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_egJhfzShmKTOVThi']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_OUzwQcmobwEyyzeh(self):
        """MULTIALLEN_OUzwQcmobwEyyzeh multi-experiment creation."""
        model_folder = 'MULTIALLEN_OUzwQcmobwEyyzeh'
        exp = {
            'experiment_name': ['MULTIALLEN_OUzwQcmobwEyyzeh'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_OUzwQcmobwEyyzeh']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_ukeYPiaXXwpjNSgB(self):
        """MULTIALLEN_ukeYPiaXXwpjNSgB multi-experiment creation."""
        model_folder = 'MULTIALLEN_ukeYPiaXXwpjNSgB'
        exp = {
            'experiment_name': ['MULTIALLEN_ukeYPiaXXwpjNSgB'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ukeYPiaXXwpjNSgB']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_YanAatUdKHechBLn(self):
        """MULTIALLEN_YanAatUdKHechBLn multi-experiment creation."""
        model_folder = 'MULTIALLEN_YanAatUdKHechBLn'
        exp = {
            'experiment_name': ['MULTIALLEN_YanAatUdKHechBLn'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_YanAatUdKHechBLn']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_OEArgKABKJwHiRyp(self):
        """MULTIALLEN_OEArgKABKJwHiRyp multi-experiment creation."""
        model_folder = 'MULTIALLEN_OEArgKABKJwHiRyp'
        exp = {
            'experiment_name': ['MULTIALLEN_OEArgKABKJwHiRyp'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_OEArgKABKJwHiRyp']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_EBWVMbOOcAikKKPz(self):
        """MULTIALLEN_EBWVMbOOcAikKKPz multi-experiment creation."""
        model_folder = 'MULTIALLEN_EBWVMbOOcAikKKPz'
        exp = {
            'experiment_name': ['MULTIALLEN_EBWVMbOOcAikKKPz'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_EBWVMbOOcAikKKPz']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_UYCHTNlRdPQFnDUZ(self):
        """MULTIALLEN_UYCHTNlRdPQFnDUZ multi-experiment creation."""
        model_folder = 'MULTIALLEN_UYCHTNlRdPQFnDUZ'
        exp = {
            'experiment_name': ['MULTIALLEN_UYCHTNlRdPQFnDUZ'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_UYCHTNlRdPQFnDUZ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_TAoiuOqwBVSIGvQF(self):
        """MULTIALLEN_TAoiuOqwBVSIGvQF multi-experiment creation."""
        model_folder = 'MULTIALLEN_TAoiuOqwBVSIGvQF'
        exp = {
            'experiment_name': ['MULTIALLEN_TAoiuOqwBVSIGvQF'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_TAoiuOqwBVSIGvQF']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_VgrvTIvtbugreIKH(self):
        """MULTIALLEN_VgrvTIvtbugreIKH multi-experiment creation."""
        model_folder = 'MULTIALLEN_VgrvTIvtbugreIKH'
        exp = {
            'experiment_name': ['MULTIALLEN_VgrvTIvtbugreIKH'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_VgrvTIvtbugreIKH']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_GwzPhUgdZFmmvOIL(self):
        """MULTIALLEN_GwzPhUgdZFmmvOIL multi-experiment creation."""
        model_folder = 'MULTIALLEN_GwzPhUgdZFmmvOIL'
        exp = {
            'experiment_name': ['MULTIALLEN_GwzPhUgdZFmmvOIL'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_GwzPhUgdZFmmvOIL']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_NCIoqbnHcADTLRGj(self):
        """MULTIALLEN_NCIoqbnHcADTLRGj multi-experiment creation."""
        model_folder = 'MULTIALLEN_NCIoqbnHcADTLRGj'
        exp = {
            'experiment_name': ['MULTIALLEN_NCIoqbnHcADTLRGj'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_NCIoqbnHcADTLRGj']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_xNGktpQNczhAXoUM(self):
        """MULTIALLEN_xNGktpQNczhAXoUM multi-experiment creation."""
        model_folder = 'MULTIALLEN_xNGktpQNczhAXoUM'
        exp = {
            'experiment_name': ['MULTIALLEN_xNGktpQNczhAXoUM'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_xNGktpQNczhAXoUM']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_cMVZxijgiBVDVwcS(self):
        """MULTIALLEN_cMVZxijgiBVDVwcS multi-experiment creation."""
        model_folder = 'MULTIALLEN_cMVZxijgiBVDVwcS'
        exp = {
            'experiment_name': ['MULTIALLEN_cMVZxijgiBVDVwcS'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_cMVZxijgiBVDVwcS']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_NQpsIrASeVUobRvs(self):
        """MULTIALLEN_NQpsIrASeVUobRvs multi-experiment creation."""
        model_folder = 'MULTIALLEN_NQpsIrASeVUobRvs'
        exp = {
            'experiment_name': ['MULTIALLEN_NQpsIrASeVUobRvs'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_NQpsIrASeVUobRvs']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_PEkmTsgXphqYIeDl(self):
        """MULTIALLEN_PEkmTsgXphqYIeDl multi-experiment creation."""
        model_folder = 'MULTIALLEN_PEkmTsgXphqYIeDl'
        exp = {
            'experiment_name': ['MULTIALLEN_PEkmTsgXphqYIeDl'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_PEkmTsgXphqYIeDl']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_KGNOozchMkmrGugP(self):
        """MULTIALLEN_KGNOozchMkmrGugP multi-experiment creation."""
        model_folder = 'MULTIALLEN_KGNOozchMkmrGugP'
        exp = {
            'experiment_name': ['MULTIALLEN_KGNOozchMkmrGugP'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_KGNOozchMkmrGugP']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_GuEIupTFxMxmIJwC(self):
        """MULTIALLEN_GuEIupTFxMxmIJwC multi-experiment creation."""
        model_folder = 'MULTIALLEN_GuEIupTFxMxmIJwC'
        exp = {
            'experiment_name': ['MULTIALLEN_GuEIupTFxMxmIJwC'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_GuEIupTFxMxmIJwC']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_cnHemTDjPaTNehUW(self):
        """MULTIALLEN_cnHemTDjPaTNehUW multi-experiment creation."""
        model_folder = 'MULTIALLEN_cnHemTDjPaTNehUW'
        exp = {
            'experiment_name': ['MULTIALLEN_cnHemTDjPaTNehUW'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_cnHemTDjPaTNehUW']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_gQfNubKVSXpQZNkf(self):
        """MULTIALLEN_gQfNubKVSXpQZNkf multi-experiment creation."""
        model_folder = 'MULTIALLEN_gQfNubKVSXpQZNkf'
        exp = {
            'experiment_name': ['MULTIALLEN_gQfNubKVSXpQZNkf'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_gQfNubKVSXpQZNkf']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_GBIzBBMXUmejkGyj(self):
        """MULTIALLEN_GBIzBBMXUmejkGyj multi-experiment creation."""
        model_folder = 'MULTIALLEN_GBIzBBMXUmejkGyj'
        exp = {
            'experiment_name': ['MULTIALLEN_GBIzBBMXUmejkGyj'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_GBIzBBMXUmejkGyj']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_zKSqaPHiHSdnWlaX(self):
        """MULTIALLEN_zKSqaPHiHSdnWlaX multi-experiment creation."""
        model_folder = 'MULTIALLEN_zKSqaPHiHSdnWlaX'
        exp = {
            'experiment_name': ['MULTIALLEN_zKSqaPHiHSdnWlaX'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_zKSqaPHiHSdnWlaX']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_gxxoWeyZknIjNBIs(self):
        """MULTIALLEN_gxxoWeyZknIjNBIs multi-experiment creation."""
        model_folder = 'MULTIALLEN_gxxoWeyZknIjNBIs'
        exp = {
            'experiment_name': ['MULTIALLEN_gxxoWeyZknIjNBIs'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_gxxoWeyZknIjNBIs']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_TkTDkBXLlltgprpn(self):
        """MULTIALLEN_TkTDkBXLlltgprpn multi-experiment creation."""
        model_folder = 'MULTIALLEN_TkTDkBXLlltgprpn'
        exp = {
            'experiment_name': ['MULTIALLEN_TkTDkBXLlltgprpn'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_TkTDkBXLlltgprpn']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_XRCnqEkjAZWRJDHF(self):
        """MULTIALLEN_XRCnqEkjAZWRJDHF multi-experiment creation."""
        model_folder = 'MULTIALLEN_XRCnqEkjAZWRJDHF'
        exp = {
            'experiment_name': ['MULTIALLEN_XRCnqEkjAZWRJDHF'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_XRCnqEkjAZWRJDHF']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_mGlrtcbGdkrnUUmL(self):
        """MULTIALLEN_mGlrtcbGdkrnUUmL multi-experiment creation."""
        model_folder = 'MULTIALLEN_mGlrtcbGdkrnUUmL'
        exp = {
            'experiment_name': ['MULTIALLEN_mGlrtcbGdkrnUUmL'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_mGlrtcbGdkrnUUmL']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_gWRZejZMtpMzmzdV(self):
        """MULTIALLEN_gWRZejZMtpMzmzdV multi-experiment creation."""
        model_folder = 'MULTIALLEN_gWRZejZMtpMzmzdV'
        exp = {
            'experiment_name': ['MULTIALLEN_gWRZejZMtpMzmzdV'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_gWRZejZMtpMzmzdV']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_lBCOZLqeTIpSQwZL(self):
        """MULTIALLEN_lBCOZLqeTIpSQwZL multi-experiment creation."""
        model_folder = 'MULTIALLEN_lBCOZLqeTIpSQwZL'
        exp = {
            'experiment_name': ['MULTIALLEN_lBCOZLqeTIpSQwZL'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_lBCOZLqeTIpSQwZL']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_fUEPwwYHdDYxrfdg(self):
        """MULTIALLEN_fUEPwwYHdDYxrfdg multi-experiment creation."""
        model_folder = 'MULTIALLEN_fUEPwwYHdDYxrfdg'
        exp = {
            'experiment_name': ['MULTIALLEN_fUEPwwYHdDYxrfdg'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_fUEPwwYHdDYxrfdg']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_PtBKztFhlYVgjJBX(self):
        """MULTIALLEN_PtBKztFhlYVgjJBX multi-experiment creation."""
        model_folder = 'MULTIALLEN_PtBKztFhlYVgjJBX'
        exp = {
            'experiment_name': ['MULTIALLEN_PtBKztFhlYVgjJBX'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_PtBKztFhlYVgjJBX']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_iEDwxiigPSZdrOxR(self):
        """MULTIALLEN_iEDwxiigPSZdrOxR multi-experiment creation."""
        model_folder = 'MULTIALLEN_iEDwxiigPSZdrOxR'
        exp = {
            'experiment_name': ['MULTIALLEN_iEDwxiigPSZdrOxR'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_iEDwxiigPSZdrOxR']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_ShoekTXlmttjnfiU(self):
        """MULTIALLEN_ShoekTXlmttjnfiU multi-experiment creation."""
        model_folder = 'MULTIALLEN_ShoekTXlmttjnfiU'
        exp = {
            'experiment_name': ['MULTIALLEN_ShoekTXlmttjnfiU'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ShoekTXlmttjnfiU']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_klRhVsLsNyrdVfjp(self):
        """MULTIALLEN_klRhVsLsNyrdVfjp multi-experiment creation."""
        model_folder = 'MULTIALLEN_klRhVsLsNyrdVfjp'
        exp = {
            'experiment_name': ['MULTIALLEN_klRhVsLsNyrdVfjp'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_klRhVsLsNyrdVfjp']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_AaTMxlNnYWJGmORs(self):
        """MULTIALLEN_AaTMxlNnYWJGmORs multi-experiment creation."""
        model_folder = 'MULTIALLEN_AaTMxlNnYWJGmORs'
        exp = {
            'experiment_name': ['MULTIALLEN_AaTMxlNnYWJGmORs'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_AaTMxlNnYWJGmORs']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_djRUIrTUxOSRDApX(self):
        """MULTIALLEN_djRUIrTUxOSRDApX multi-experiment creation."""
        model_folder = 'MULTIALLEN_djRUIrTUxOSRDApX'
        exp = {
            'experiment_name': ['MULTIALLEN_djRUIrTUxOSRDApX'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_djRUIrTUxOSRDApX']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_CKiEPdFobUCHkYZH(self):
        """MULTIALLEN_CKiEPdFobUCHkYZH multi-experiment creation."""
        model_folder = 'MULTIALLEN_CKiEPdFobUCHkYZH'
        exp = {
            'experiment_name': ['MULTIALLEN_CKiEPdFobUCHkYZH'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_CKiEPdFobUCHkYZH']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_VcKdFNuCafJeSljo(self):
        """MULTIALLEN_VcKdFNuCafJeSljo multi-experiment creation."""
        model_folder = 'MULTIALLEN_VcKdFNuCafJeSljo'
        exp = {
            'experiment_name': ['MULTIALLEN_VcKdFNuCafJeSljo'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_VcKdFNuCafJeSljo']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_HKAAGoeifBxvEdYp(self):
        """MULTIALLEN_HKAAGoeifBxvEdYp multi-experiment creation."""
        model_folder = 'MULTIALLEN_HKAAGoeifBxvEdYp'
        exp = {
            'experiment_name': ['MULTIALLEN_HKAAGoeifBxvEdYp'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_HKAAGoeifBxvEdYp']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_KZpBlOORHjpXOGUF(self):
        """MULTIALLEN_KZpBlOORHjpXOGUF multi-experiment creation."""
        model_folder = 'MULTIALLEN_KZpBlOORHjpXOGUF'
        exp = {
            'experiment_name': ['MULTIALLEN_KZpBlOORHjpXOGUF'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_KZpBlOORHjpXOGUF']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_docQPzaTVznZhKpc(self):
        """MULTIALLEN_docQPzaTVznZhKpc multi-experiment creation."""
        model_folder = 'MULTIALLEN_docQPzaTVznZhKpc'
        exp = {
            'experiment_name': ['MULTIALLEN_docQPzaTVznZhKpc'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_docQPzaTVznZhKpc']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_niDntatFxqDCxJNc(self):
        """MULTIALLEN_niDntatFxqDCxJNc multi-experiment creation."""
        model_folder = 'MULTIALLEN_niDntatFxqDCxJNc'
        exp = {
            'experiment_name': ['MULTIALLEN_niDntatFxqDCxJNc'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_niDntatFxqDCxJNc']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_dcTkoEWPThzTMjVq(self):
        """MULTIALLEN_dcTkoEWPThzTMjVq multi-experiment creation."""
        model_folder = 'MULTIALLEN_dcTkoEWPThzTMjVq'
        exp = {
            'experiment_name': ['MULTIALLEN_dcTkoEWPThzTMjVq'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_dcTkoEWPThzTMjVq']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_fKNvPleTfAMovZfp(self):
        """MULTIALLEN_fKNvPleTfAMovZfp multi-experiment creation."""
        model_folder = 'MULTIALLEN_fKNvPleTfAMovZfp'
        exp = {
            'experiment_name': ['MULTIALLEN_fKNvPleTfAMovZfp'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_fKNvPleTfAMovZfp']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_ANWGqPjvFtHRMZCP(self):
        """MULTIALLEN_ANWGqPjvFtHRMZCP multi-experiment creation."""
        model_folder = 'MULTIALLEN_ANWGqPjvFtHRMZCP'
        exp = {
            'experiment_name': ['MULTIALLEN_ANWGqPjvFtHRMZCP'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ANWGqPjvFtHRMZCP']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_LSHbcdsiyLodfSgy(self):
        """MULTIALLEN_LSHbcdsiyLodfSgy multi-experiment creation."""
        model_folder = 'MULTIALLEN_LSHbcdsiyLodfSgy'
        exp = {
            'experiment_name': ['MULTIALLEN_LSHbcdsiyLodfSgy'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_LSHbcdsiyLodfSgy']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_pSUgQubWErVfpGDx(self):
        """MULTIALLEN_pSUgQubWErVfpGDx multi-experiment creation."""
        model_folder = 'MULTIALLEN_pSUgQubWErVfpGDx'
        exp = {
            'experiment_name': ['MULTIALLEN_pSUgQubWErVfpGDx'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_pSUgQubWErVfpGDx']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_IjFhhZtPgVvUbjHp(self):
        """MULTIALLEN_IjFhhZtPgVvUbjHp multi-experiment creation."""
        model_folder = 'MULTIALLEN_IjFhhZtPgVvUbjHp'
        exp = {
            'experiment_name': ['MULTIALLEN_IjFhhZtPgVvUbjHp'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_IjFhhZtPgVvUbjHp']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_EGiJMrqzXzDBimhR(self):
        """MULTIALLEN_EGiJMrqzXzDBimhR multi-experiment creation."""
        model_folder = 'MULTIALLEN_EGiJMrqzXzDBimhR'
        exp = {
            'experiment_name': ['MULTIALLEN_EGiJMrqzXzDBimhR'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_EGiJMrqzXzDBimhR']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_DPqOhweffUaTJnNR(self):
        """MULTIALLEN_DPqOhweffUaTJnNR multi-experiment creation."""
        model_folder = 'MULTIALLEN_DPqOhweffUaTJnNR'
        exp = {
            'experiment_name': ['MULTIALLEN_DPqOhweffUaTJnNR'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_DPqOhweffUaTJnNR']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_TNPNWyguuEglvkGB(self):
        """MULTIALLEN_TNPNWyguuEglvkGB multi-experiment creation."""
        model_folder = 'MULTIALLEN_TNPNWyguuEglvkGB'
        exp = {
            'experiment_name': ['MULTIALLEN_TNPNWyguuEglvkGB'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_TNPNWyguuEglvkGB']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_jZxuhgRoeKfCsbZI(self):
        """MULTIALLEN_jZxuhgRoeKfCsbZI multi-experiment creation."""
        model_folder = 'MULTIALLEN_jZxuhgRoeKfCsbZI'
        exp = {
            'experiment_name': ['MULTIALLEN_jZxuhgRoeKfCsbZI'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_jZxuhgRoeKfCsbZI']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_ZDyfOAWlMtHLxBkH(self):
        """MULTIALLEN_ZDyfOAWlMtHLxBkH multi-experiment creation."""
        model_folder = 'MULTIALLEN_ZDyfOAWlMtHLxBkH'
        exp = {
            'experiment_name': ['MULTIALLEN_ZDyfOAWlMtHLxBkH'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ZDyfOAWlMtHLxBkH']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_lqpWtKoXpOzfYxJv(self):
        """MULTIALLEN_lqpWtKoXpOzfYxJv multi-experiment creation."""
        model_folder = 'MULTIALLEN_lqpWtKoXpOzfYxJv'
        exp = {
            'experiment_name': ['MULTIALLEN_lqpWtKoXpOzfYxJv'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_lqpWtKoXpOzfYxJv']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_lYSTvZZZjYNtvArG(self):
        """MULTIALLEN_lYSTvZZZjYNtvArG multi-experiment creation."""
        model_folder = 'MULTIALLEN_lYSTvZZZjYNtvArG'
        exp = {
            'experiment_name': ['MULTIALLEN_lYSTvZZZjYNtvArG'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_lYSTvZZZjYNtvArG']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_nIYmRFixuWNZczhS(self):
        """MULTIALLEN_nIYmRFixuWNZczhS multi-experiment creation."""
        model_folder = 'MULTIALLEN_nIYmRFixuWNZczhS'
        exp = {
            'experiment_name': ['MULTIALLEN_nIYmRFixuWNZczhS'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_nIYmRFixuWNZczhS']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_pGgnMEUQPPBXRqmu(self):
        """MULTIALLEN_pGgnMEUQPPBXRqmu multi-experiment creation."""
        model_folder = 'MULTIALLEN_pGgnMEUQPPBXRqmu'
        exp = {
            'experiment_name': ['MULTIALLEN_pGgnMEUQPPBXRqmu'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_pGgnMEUQPPBXRqmu']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_UEsHSOLoSnlsDtSQ(self):
        """MULTIALLEN_UEsHSOLoSnlsDtSQ multi-experiment creation."""
        model_folder = 'MULTIALLEN_UEsHSOLoSnlsDtSQ'
        exp = {
            'experiment_name': ['MULTIALLEN_UEsHSOLoSnlsDtSQ'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_UEsHSOLoSnlsDtSQ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_cDVqMssdADcJBEPy(self):
        """MULTIALLEN_cDVqMssdADcJBEPy multi-experiment creation."""
        model_folder = 'MULTIALLEN_cDVqMssdADcJBEPy'
        exp = {
            'experiment_name': ['MULTIALLEN_cDVqMssdADcJBEPy'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_cDVqMssdADcJBEPy']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_YLAByqOVrMfpRjdo(self):
        """MULTIALLEN_YLAByqOVrMfpRjdo multi-experiment creation."""
        model_folder = 'MULTIALLEN_YLAByqOVrMfpRjdo'
        exp = {
            'experiment_name': ['MULTIALLEN_YLAByqOVrMfpRjdo'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_YLAByqOVrMfpRjdo']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_FUlFBDbLyzsMRyne(self):
        """MULTIALLEN_FUlFBDbLyzsMRyne multi-experiment creation."""
        model_folder = 'MULTIALLEN_FUlFBDbLyzsMRyne'
        exp = {
            'experiment_name': ['MULTIALLEN_FUlFBDbLyzsMRyne'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_FUlFBDbLyzsMRyne']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_hxBhuctwShwFFYTs(self):
        """MULTIALLEN_hxBhuctwShwFFYTs multi-experiment creation."""
        model_folder = 'MULTIALLEN_hxBhuctwShwFFYTs'
        exp = {
            'experiment_name': ['MULTIALLEN_hxBhuctwShwFFYTs'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_hxBhuctwShwFFYTs']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_hMngeFNSsHvuXdUR(self):
        """MULTIALLEN_hMngeFNSsHvuXdUR multi-experiment creation."""
        model_folder = 'MULTIALLEN_hMngeFNSsHvuXdUR'
        exp = {
            'experiment_name': ['MULTIALLEN_hMngeFNSsHvuXdUR'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_hMngeFNSsHvuXdUR']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_PhivsWvQFfCElzpy(self):
        """MULTIALLEN_PhivsWvQFfCElzpy multi-experiment creation."""
        model_folder = 'MULTIALLEN_PhivsWvQFfCElzpy'
        exp = {
            'experiment_name': ['MULTIALLEN_PhivsWvQFfCElzpy'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_PhivsWvQFfCElzpy']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_MdRWnnCKFlVWgWku(self):
        """MULTIALLEN_MdRWnnCKFlVWgWku multi-experiment creation."""
        model_folder = 'MULTIALLEN_MdRWnnCKFlVWgWku'
        exp = {
            'experiment_name': ['MULTIALLEN_MdRWnnCKFlVWgWku'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_MdRWnnCKFlVWgWku']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_lPMaurrIozQxGjGA(self):
        """MULTIALLEN_lPMaurrIozQxGjGA multi-experiment creation."""
        model_folder = 'MULTIALLEN_lPMaurrIozQxGjGA'
        exp = {
            'experiment_name': ['MULTIALLEN_lPMaurrIozQxGjGA'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_lPMaurrIozQxGjGA']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_CIKBrVsRbdclTUEN(self):
        """MULTIALLEN_CIKBrVsRbdclTUEN multi-experiment creation."""
        model_folder = 'MULTIALLEN_CIKBrVsRbdclTUEN'
        exp = {
            'experiment_name': ['MULTIALLEN_CIKBrVsRbdclTUEN'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_CIKBrVsRbdclTUEN']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_TUkClYXjTqeIAeck(self):
        """MULTIALLEN_TUkClYXjTqeIAeck multi-experiment creation."""
        model_folder = 'MULTIALLEN_TUkClYXjTqeIAeck'
        exp = {
            'experiment_name': ['MULTIALLEN_TUkClYXjTqeIAeck'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_TUkClYXjTqeIAeck']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_HSJolkVFegpYjgzd(self):
        """MULTIALLEN_HSJolkVFegpYjgzd multi-experiment creation."""
        model_folder = 'MULTIALLEN_HSJolkVFegpYjgzd'
        exp = {
            'experiment_name': ['MULTIALLEN_HSJolkVFegpYjgzd'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_HSJolkVFegpYjgzd']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_yjNndKKbcXqcayCd(self):
        """MULTIALLEN_yjNndKKbcXqcayCd multi-experiment creation."""
        model_folder = 'MULTIALLEN_yjNndKKbcXqcayCd'
        exp = {
            'experiment_name': ['MULTIALLEN_yjNndKKbcXqcayCd'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_yjNndKKbcXqcayCd']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_AFWNyDLNMEjLLhbU(self):
        """MULTIALLEN_AFWNyDLNMEjLLhbU multi-experiment creation."""
        model_folder = 'MULTIALLEN_AFWNyDLNMEjLLhbU'
        exp = {
            'experiment_name': ['MULTIALLEN_AFWNyDLNMEjLLhbU'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_AFWNyDLNMEjLLhbU']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_MpzhnybSYqZyvjZK(self):
        """MULTIALLEN_MpzhnybSYqZyvjZK multi-experiment creation."""
        model_folder = 'MULTIALLEN_MpzhnybSYqZyvjZK'
        exp = {
            'experiment_name': ['MULTIALLEN_MpzhnybSYqZyvjZK'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_MpzhnybSYqZyvjZK']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_fRkvxAHUjeyoxAEA(self):
        """MULTIALLEN_fRkvxAHUjeyoxAEA multi-experiment creation."""
        model_folder = 'MULTIALLEN_fRkvxAHUjeyoxAEA'
        exp = {
            'experiment_name': ['MULTIALLEN_fRkvxAHUjeyoxAEA'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_fRkvxAHUjeyoxAEA']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_tAxbLMZkOcaghObN(self):
        """MULTIALLEN_tAxbLMZkOcaghObN multi-experiment creation."""
        model_folder = 'MULTIALLEN_tAxbLMZkOcaghObN'
        exp = {
            'experiment_name': ['MULTIALLEN_tAxbLMZkOcaghObN'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_tAxbLMZkOcaghObN']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_BCIxayNEgUCpUKRg(self):
        """MULTIALLEN_BCIxayNEgUCpUKRg multi-experiment creation."""
        model_folder = 'MULTIALLEN_BCIxayNEgUCpUKRg'
        exp = {
            'experiment_name': ['MULTIALLEN_BCIxayNEgUCpUKRg'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_BCIxayNEgUCpUKRg']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_baYvfaVPGJvRHbgV(self):
        """MULTIALLEN_baYvfaVPGJvRHbgV multi-experiment creation."""
        model_folder = 'MULTIALLEN_baYvfaVPGJvRHbgV'
        exp = {
            'experiment_name': ['MULTIALLEN_baYvfaVPGJvRHbgV'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_baYvfaVPGJvRHbgV']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_KIYNApnErgplsbDe(self):
        """MULTIALLEN_KIYNApnErgplsbDe multi-experiment creation."""
        model_folder = 'MULTIALLEN_KIYNApnErgplsbDe'
        exp = {
            'experiment_name': ['MULTIALLEN_KIYNApnErgplsbDe'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_KIYNApnErgplsbDe']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_sFxOzZfLNWZTepKJ(self):
        """MULTIALLEN_sFxOzZfLNWZTepKJ multi-experiment creation."""
        model_folder = 'MULTIALLEN_sFxOzZfLNWZTepKJ'
        exp = {
            'experiment_name': ['MULTIALLEN_sFxOzZfLNWZTepKJ'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_sFxOzZfLNWZTepKJ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_zvQaMjkCSHdsUeDL(self):
        """MULTIALLEN_zvQaMjkCSHdsUeDL multi-experiment creation."""
        model_folder = 'MULTIALLEN_zvQaMjkCSHdsUeDL'
        exp = {
            'experiment_name': ['MULTIALLEN_zvQaMjkCSHdsUeDL'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_zvQaMjkCSHdsUeDL']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_LVbaZGNVbYFKLEiU(self):
        """MULTIALLEN_LVbaZGNVbYFKLEiU multi-experiment creation."""
        model_folder = 'MULTIALLEN_LVbaZGNVbYFKLEiU'
        exp = {
            'experiment_name': ['MULTIALLEN_LVbaZGNVbYFKLEiU'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_LVbaZGNVbYFKLEiU']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_fStGJwprurXDviWU(self):
        """MULTIALLEN_fStGJwprurXDviWU multi-experiment creation."""
        model_folder = 'MULTIALLEN_fStGJwprurXDviWU'
        exp = {
            'experiment_name': ['MULTIALLEN_fStGJwprurXDviWU'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_fStGJwprurXDviWU']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_koaNcuztUTNPLOUR(self):
        """MULTIALLEN_koaNcuztUTNPLOUR multi-experiment creation."""
        model_folder = 'MULTIALLEN_koaNcuztUTNPLOUR'
        exp = {
            'experiment_name': ['MULTIALLEN_koaNcuztUTNPLOUR'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_koaNcuztUTNPLOUR']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_UPGzvpUADBQgMolz(self):
        """MULTIALLEN_UPGzvpUADBQgMolz multi-experiment creation."""
        model_folder = 'MULTIALLEN_UPGzvpUADBQgMolz'
        exp = {
            'experiment_name': ['MULTIALLEN_UPGzvpUADBQgMolz'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_UPGzvpUADBQgMolz']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_bemHhDoJhaExboSy(self):
        """MULTIALLEN_bemHhDoJhaExboSy multi-experiment creation."""
        model_folder = 'MULTIALLEN_bemHhDoJhaExboSy'
        exp = {
            'experiment_name': ['MULTIALLEN_bemHhDoJhaExboSy'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_bemHhDoJhaExboSy']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_aNeEtCJQpmBESmDw(self):
        """MULTIALLEN_aNeEtCJQpmBESmDw multi-experiment creation."""
        model_folder = 'MULTIALLEN_aNeEtCJQpmBESmDw'
        exp = {
            'experiment_name': ['MULTIALLEN_aNeEtCJQpmBESmDw'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_aNeEtCJQpmBESmDw']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_VqUzlKmgmukzzOMb(self):
        """MULTIALLEN_VqUzlKmgmukzzOMb multi-experiment creation."""
        model_folder = 'MULTIALLEN_VqUzlKmgmukzzOMb'
        exp = {
            'experiment_name': ['MULTIALLEN_VqUzlKmgmukzzOMb'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_VqUzlKmgmukzzOMb']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_UXvvZMXOwGdxezxw(self):
        """MULTIALLEN_UXvvZMXOwGdxezxw multi-experiment creation."""
        model_folder = 'MULTIALLEN_UXvvZMXOwGdxezxw'
        exp = {
            'experiment_name': ['MULTIALLEN_UXvvZMXOwGdxezxw'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_UXvvZMXOwGdxezxw']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_evPiohGRNVrOjWfO(self):
        """MULTIALLEN_evPiohGRNVrOjWfO multi-experiment creation."""
        model_folder = 'MULTIALLEN_evPiohGRNVrOjWfO'
        exp = {
            'experiment_name': ['MULTIALLEN_evPiohGRNVrOjWfO'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_evPiohGRNVrOjWfO']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_aPochQGMJOPdsvKn(self):
        """MULTIALLEN_aPochQGMJOPdsvKn multi-experiment creation."""
        model_folder = 'MULTIALLEN_aPochQGMJOPdsvKn'
        exp = {
            'experiment_name': ['MULTIALLEN_aPochQGMJOPdsvKn'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_aPochQGMJOPdsvKn']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_IkdUZUGjyvJqrMeJ(self):
        """MULTIALLEN_IkdUZUGjyvJqrMeJ multi-experiment creation."""
        model_folder = 'MULTIALLEN_IkdUZUGjyvJqrMeJ'
        exp = {
            'experiment_name': ['MULTIALLEN_IkdUZUGjyvJqrMeJ'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_IkdUZUGjyvJqrMeJ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_BCNypUXcqueFeqWR(self):
        """MULTIALLEN_BCNypUXcqueFeqWR multi-experiment creation."""
        model_folder = 'MULTIALLEN_BCNypUXcqueFeqWR'
        exp = {
            'experiment_name': ['MULTIALLEN_BCNypUXcqueFeqWR'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_BCNypUXcqueFeqWR']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_ezVGcrTwmFMHviLA(self):
        """MULTIALLEN_ezVGcrTwmFMHviLA multi-experiment creation."""
        model_folder = 'MULTIALLEN_ezVGcrTwmFMHviLA'
        exp = {
            'experiment_name': ['MULTIALLEN_ezVGcrTwmFMHviLA'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ezVGcrTwmFMHviLA']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_NAXKXnwBJUShAniG(self):
        """MULTIALLEN_NAXKXnwBJUShAniG multi-experiment creation."""
        model_folder = 'MULTIALLEN_NAXKXnwBJUShAniG'
        exp = {
            'experiment_name': ['MULTIALLEN_NAXKXnwBJUShAniG'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_NAXKXnwBJUShAniG']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_tbHouwLuXnHIXFTq(self):
        """MULTIALLEN_tbHouwLuXnHIXFTq multi-experiment creation."""
        model_folder = 'MULTIALLEN_tbHouwLuXnHIXFTq'
        exp = {
            'experiment_name': ['MULTIALLEN_tbHouwLuXnHIXFTq'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_tbHouwLuXnHIXFTq']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_HvZLviLsEbMRMFkL(self):
        """MULTIALLEN_HvZLviLsEbMRMFkL multi-experiment creation."""
        model_folder = 'MULTIALLEN_HvZLviLsEbMRMFkL'
        exp = {
            'experiment_name': ['MULTIALLEN_HvZLviLsEbMRMFkL'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_HvZLviLsEbMRMFkL']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_oYTYfPEiPZPDIirT(self):
        """MULTIALLEN_oYTYfPEiPZPDIirT multi-experiment creation."""
        model_folder = 'MULTIALLEN_oYTYfPEiPZPDIirT'
        exp = {
            'experiment_name': ['MULTIALLEN_oYTYfPEiPZPDIirT'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_oYTYfPEiPZPDIirT']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_CAnyAShvvEkNahUL(self):
        """MULTIALLEN_CAnyAShvvEkNahUL multi-experiment creation."""
        model_folder = 'MULTIALLEN_CAnyAShvvEkNahUL'
        exp = {
            'experiment_name': ['MULTIALLEN_CAnyAShvvEkNahUL'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_CAnyAShvvEkNahUL']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_hHkvQJnrikltUnCb(self):
        """MULTIALLEN_hHkvQJnrikltUnCb multi-experiment creation."""
        model_folder = 'MULTIALLEN_hHkvQJnrikltUnCb'
        exp = {
            'experiment_name': ['MULTIALLEN_hHkvQJnrikltUnCb'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_hHkvQJnrikltUnCb']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_yzmRcrrjmyWFdxPA(self):
        """MULTIALLEN_yzmRcrrjmyWFdxPA multi-experiment creation."""
        model_folder = 'MULTIALLEN_yzmRcrrjmyWFdxPA'
        exp = {
            'experiment_name': ['MULTIALLEN_yzmRcrrjmyWFdxPA'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_yzmRcrrjmyWFdxPA']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_LPQslwqKSAEHpbNb(self):
        """MULTIALLEN_LPQslwqKSAEHpbNb multi-experiment creation."""
        model_folder = 'MULTIALLEN_LPQslwqKSAEHpbNb'
        exp = {
            'experiment_name': ['MULTIALLEN_LPQslwqKSAEHpbNb'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_LPQslwqKSAEHpbNb']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_ufwhPbdLGKtMakQA(self):
        """MULTIALLEN_ufwhPbdLGKtMakQA multi-experiment creation."""
        model_folder = 'MULTIALLEN_ufwhPbdLGKtMakQA'
        exp = {
            'experiment_name': ['MULTIALLEN_ufwhPbdLGKtMakQA'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ufwhPbdLGKtMakQA']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_EVkQyBxJsUAnZMHy(self):
        """MULTIALLEN_EVkQyBxJsUAnZMHy multi-experiment creation."""
        model_folder = 'MULTIALLEN_EVkQyBxJsUAnZMHy'
        exp = {
            'experiment_name': ['MULTIALLEN_EVkQyBxJsUAnZMHy'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_EVkQyBxJsUAnZMHy']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_tngVmUITXhOzhDky(self):
        """MULTIALLEN_tngVmUITXhOzhDky multi-experiment creation."""
        model_folder = 'MULTIALLEN_tngVmUITXhOzhDky'
        exp = {
            'experiment_name': ['MULTIALLEN_tngVmUITXhOzhDky'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_tngVmUITXhOzhDky']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_NBcxDjkLNQurntAH(self):
        """MULTIALLEN_NBcxDjkLNQurntAH multi-experiment creation."""
        model_folder = 'MULTIALLEN_NBcxDjkLNQurntAH'
        exp = {
            'experiment_name': ['MULTIALLEN_NBcxDjkLNQurntAH'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_NBcxDjkLNQurntAH']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_GjljMmZjdpCEHEgl(self):
        """MULTIALLEN_GjljMmZjdpCEHEgl multi-experiment creation."""
        model_folder = 'MULTIALLEN_GjljMmZjdpCEHEgl'
        exp = {
            'experiment_name': ['MULTIALLEN_GjljMmZjdpCEHEgl'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_GjljMmZjdpCEHEgl']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_PiacIQuAnBxhfJRq(self):
        """MULTIALLEN_PiacIQuAnBxhfJRq multi-experiment creation."""
        model_folder = 'MULTIALLEN_PiacIQuAnBxhfJRq'
        exp = {
            'experiment_name': ['MULTIALLEN_PiacIQuAnBxhfJRq'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_PiacIQuAnBxhfJRq']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_xvIAmSdTaCthTdBk(self):
        """MULTIALLEN_xvIAmSdTaCthTdBk multi-experiment creation."""
        model_folder = 'MULTIALLEN_xvIAmSdTaCthTdBk'
        exp = {
            'experiment_name': ['MULTIALLEN_xvIAmSdTaCthTdBk'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_xvIAmSdTaCthTdBk']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_fHYcjhzOwqAGVoMa(self):
        """MULTIALLEN_fHYcjhzOwqAGVoMa multi-experiment creation."""
        model_folder = 'MULTIALLEN_fHYcjhzOwqAGVoMa'
        exp = {
            'experiment_name': ['MULTIALLEN_fHYcjhzOwqAGVoMa'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_fHYcjhzOwqAGVoMa']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_isNeIsMYamYspEKI(self):
        """MULTIALLEN_isNeIsMYamYspEKI multi-experiment creation."""
        model_folder = 'MULTIALLEN_isNeIsMYamYspEKI'
        exp = {
            'experiment_name': ['MULTIALLEN_isNeIsMYamYspEKI'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_isNeIsMYamYspEKI']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_MdICOPdbevfytQmN(self):
        """MULTIALLEN_MdICOPdbevfytQmN multi-experiment creation."""
        model_folder = 'MULTIALLEN_MdICOPdbevfytQmN'
        exp = {
            'experiment_name': ['MULTIALLEN_MdICOPdbevfytQmN'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_MdICOPdbevfytQmN']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_MvTuzhRQeSQosrTM(self):
        """MULTIALLEN_MvTuzhRQeSQosrTM multi-experiment creation."""
        model_folder = 'MULTIALLEN_MvTuzhRQeSQosrTM'
        exp = {
            'experiment_name': ['MULTIALLEN_MvTuzhRQeSQosrTM'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_MvTuzhRQeSQosrTM']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_UCYxWBSsFfYZnaFA(self):
        """MULTIALLEN_UCYxWBSsFfYZnaFA multi-experiment creation."""
        model_folder = 'MULTIALLEN_UCYxWBSsFfYZnaFA'
        exp = {
            'experiment_name': ['MULTIALLEN_UCYxWBSsFfYZnaFA'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_UCYxWBSsFfYZnaFA']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_bljBHRMTiVDSanBY(self):
        """MULTIALLEN_bljBHRMTiVDSanBY multi-experiment creation."""
        model_folder = 'MULTIALLEN_bljBHRMTiVDSanBY'
        exp = {
            'experiment_name': ['MULTIALLEN_bljBHRMTiVDSanBY'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_bljBHRMTiVDSanBY']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_LwCFvSOwYAIqFVkH(self):
        """MULTIALLEN_LwCFvSOwYAIqFVkH multi-experiment creation."""
        model_folder = 'MULTIALLEN_LwCFvSOwYAIqFVkH'
        exp = {
            'experiment_name': ['MULTIALLEN_LwCFvSOwYAIqFVkH'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_LwCFvSOwYAIqFVkH']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_xujNHVfuEtoSWRPf(self):
        """MULTIALLEN_xujNHVfuEtoSWRPf multi-experiment creation."""
        model_folder = 'MULTIALLEN_xujNHVfuEtoSWRPf'
        exp = {
            'experiment_name': ['MULTIALLEN_xujNHVfuEtoSWRPf'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_xujNHVfuEtoSWRPf']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_jNxWMQqIpBLUNTxE(self):
        """MULTIALLEN_jNxWMQqIpBLUNTxE multi-experiment creation."""
        model_folder = 'MULTIALLEN_jNxWMQqIpBLUNTxE'
        exp = {
            'experiment_name': ['MULTIALLEN_jNxWMQqIpBLUNTxE'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_jNxWMQqIpBLUNTxE']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_SnFOhUCOGeCSMOKB(self):
        """MULTIALLEN_SnFOhUCOGeCSMOKB multi-experiment creation."""
        model_folder = 'MULTIALLEN_SnFOhUCOGeCSMOKB'
        exp = {
            'experiment_name': ['MULTIALLEN_SnFOhUCOGeCSMOKB'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_SnFOhUCOGeCSMOKB']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_eNPlLSXbHvqUUKRT(self):
        """MULTIALLEN_eNPlLSXbHvqUUKRT multi-experiment creation."""
        model_folder = 'MULTIALLEN_eNPlLSXbHvqUUKRT'
        exp = {
            'experiment_name': ['MULTIALLEN_eNPlLSXbHvqUUKRT'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_eNPlLSXbHvqUUKRT']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_wvEMyLGtTXppezjh(self):
        """MULTIALLEN_wvEMyLGtTXppezjh multi-experiment creation."""
        model_folder = 'MULTIALLEN_wvEMyLGtTXppezjh'
        exp = {
            'experiment_name': ['MULTIALLEN_wvEMyLGtTXppezjh'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_wvEMyLGtTXppezjh']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_EFKSnKjYRgEMHcsS(self):
        """MULTIALLEN_EFKSnKjYRgEMHcsS multi-experiment creation."""
        model_folder = 'MULTIALLEN_EFKSnKjYRgEMHcsS'
        exp = {
            'experiment_name': ['MULTIALLEN_EFKSnKjYRgEMHcsS'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_EFKSnKjYRgEMHcsS']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_kspAyfAHZPpIMrCw(self):
        """MULTIALLEN_kspAyfAHZPpIMrCw multi-experiment creation."""
        model_folder = 'MULTIALLEN_kspAyfAHZPpIMrCw'
        exp = {
            'experiment_name': ['MULTIALLEN_kspAyfAHZPpIMrCw'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_kspAyfAHZPpIMrCw']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_lZMOgINZTwlrjpfg(self):
        """MULTIALLEN_lZMOgINZTwlrjpfg multi-experiment creation."""
        model_folder = 'MULTIALLEN_lZMOgINZTwlrjpfg'
        exp = {
            'experiment_name': ['MULTIALLEN_lZMOgINZTwlrjpfg'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_lZMOgINZTwlrjpfg']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_HTEyixRfrMoqHxjO(self):
        """MULTIALLEN_HTEyixRfrMoqHxjO multi-experiment creation."""
        model_folder = 'MULTIALLEN_HTEyixRfrMoqHxjO'
        exp = {
            'experiment_name': ['MULTIALLEN_HTEyixRfrMoqHxjO'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_HTEyixRfrMoqHxjO']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_DTpiROTYlrRJjpND(self):
        """MULTIALLEN_DTpiROTYlrRJjpND multi-experiment creation."""
        model_folder = 'MULTIALLEN_DTpiROTYlrRJjpND'
        exp = {
            'experiment_name': ['MULTIALLEN_DTpiROTYlrRJjpND'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_DTpiROTYlrRJjpND']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_ZuuZvHiqOAaPmGTZ(self):
        """MULTIALLEN_ZuuZvHiqOAaPmGTZ multi-experiment creation."""
        model_folder = 'MULTIALLEN_ZuuZvHiqOAaPmGTZ'
        exp = {
            'experiment_name': ['MULTIALLEN_ZuuZvHiqOAaPmGTZ'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ZuuZvHiqOAaPmGTZ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_ymAecHBXfAgvbNUk(self):
        """MULTIALLEN_ymAecHBXfAgvbNUk multi-experiment creation."""
        model_folder = 'MULTIALLEN_ymAecHBXfAgvbNUk'
        exp = {
            'experiment_name': ['MULTIALLEN_ymAecHBXfAgvbNUk'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_ymAecHBXfAgvbNUk']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_KWpeXEcAmqIidhtI(self):
        """MULTIALLEN_KWpeXEcAmqIidhtI multi-experiment creation."""
        model_folder = 'MULTIALLEN_KWpeXEcAmqIidhtI'
        exp = {
            'experiment_name': ['MULTIALLEN_KWpeXEcAmqIidhtI'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_KWpeXEcAmqIidhtI']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_xKmovdXDoXQYDzUL(self):
        """MULTIALLEN_xKmovdXDoXQYDzUL multi-experiment creation."""
        model_folder = 'MULTIALLEN_xKmovdXDoXQYDzUL'
        exp = {
            'experiment_name': ['MULTIALLEN_xKmovdXDoXQYDzUL'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_xKmovdXDoXQYDzUL']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_fAudspIgNpNULhLP(self):
        """MULTIALLEN_fAudspIgNpNULhLP multi-experiment creation."""
        model_folder = 'MULTIALLEN_fAudspIgNpNULhLP'
        exp = {
            'experiment_name': ['MULTIALLEN_fAudspIgNpNULhLP'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_fAudspIgNpNULhLP']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_fAoKFsksZcNTcRxY(self):
        """MULTIALLEN_fAoKFsksZcNTcRxY multi-experiment creation."""
        model_folder = 'MULTIALLEN_fAoKFsksZcNTcRxY'
        exp = {
            'experiment_name': ['MULTIALLEN_fAoKFsksZcNTcRxY'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_fAoKFsksZcNTcRxY']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_FlwnhEtdxIJyVPan(self):
        """MULTIALLEN_FlwnhEtdxIJyVPan multi-experiment creation."""
        model_folder = 'MULTIALLEN_FlwnhEtdxIJyVPan'
        exp = {
            'experiment_name': ['MULTIALLEN_FlwnhEtdxIJyVPan'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_FlwnhEtdxIJyVPan']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_PfZZJreInFKGalPR(self):
        """MULTIALLEN_PfZZJreInFKGalPR multi-experiment creation."""
        model_folder = 'MULTIALLEN_PfZZJreInFKGalPR'
        exp = {
            'experiment_name': ['MULTIALLEN_PfZZJreInFKGalPR'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_PfZZJreInFKGalPR']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_RKJKZUikmMJzdHYe(self):
        """MULTIALLEN_RKJKZUikmMJzdHYe multi-experiment creation."""
        model_folder = 'MULTIALLEN_RKJKZUikmMJzdHYe'
        exp = {
            'experiment_name': ['MULTIALLEN_RKJKZUikmMJzdHYe'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_RKJKZUikmMJzdHYe']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_UFPfffTfWQPsxIUI(self):
        """MULTIALLEN_UFPfffTfWQPsxIUI multi-experiment creation."""
        model_folder = 'MULTIALLEN_UFPfffTfWQPsxIUI'
        exp = {
            'experiment_name': ['MULTIALLEN_UFPfffTfWQPsxIUI'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_UFPfffTfWQPsxIUI']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_fQfnpzLGZNIhUsAw(self):
        """MULTIALLEN_fQfnpzLGZNIhUsAw multi-experiment creation."""
        model_folder = 'MULTIALLEN_fQfnpzLGZNIhUsAw'
        exp = {
            'experiment_name': ['MULTIALLEN_fQfnpzLGZNIhUsAw'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_fQfnpzLGZNIhUsAw']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_eMilRZNoaIlGPhFK(self):
        """MULTIALLEN_eMilRZNoaIlGPhFK multi-experiment creation."""
        model_folder = 'MULTIALLEN_eMilRZNoaIlGPhFK'
        exp = {
            'experiment_name': ['MULTIALLEN_eMilRZNoaIlGPhFK'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_eMilRZNoaIlGPhFK']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_kkpUgVHBTyjlulKm(self):
        """MULTIALLEN_kkpUgVHBTyjlulKm multi-experiment creation."""
        model_folder = 'MULTIALLEN_kkpUgVHBTyjlulKm'
        exp = {
            'experiment_name': ['MULTIALLEN_kkpUgVHBTyjlulKm'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_kkpUgVHBTyjlulKm']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_jmAFaDtopwisAVQa(self):
        """MULTIALLEN_jmAFaDtopwisAVQa multi-experiment creation."""
        model_folder = 'MULTIALLEN_jmAFaDtopwisAVQa'
        exp = {
            'experiment_name': ['MULTIALLEN_jmAFaDtopwisAVQa'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_jmAFaDtopwisAVQa']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_XASgHixQrslUzghg(self):
        """MULTIALLEN_XASgHixQrslUzghg multi-experiment creation."""
        model_folder = 'MULTIALLEN_XASgHixQrslUzghg'
        exp = {
            'experiment_name': ['MULTIALLEN_XASgHixQrslUzghg'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_XASgHixQrslUzghg']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_nkpagmdDQVaRJABI(self):
        """MULTIALLEN_nkpagmdDQVaRJABI multi-experiment creation."""
        model_folder = 'MULTIALLEN_nkpagmdDQVaRJABI'
        exp = {
            'experiment_name': ['MULTIALLEN_nkpagmdDQVaRJABI'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_nkpagmdDQVaRJABI']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_NBYLWgxdjtgxFuFJ(self):
        """MULTIALLEN_NBYLWgxdjtgxFuFJ multi-experiment creation."""
        model_folder = 'MULTIALLEN_NBYLWgxdjtgxFuFJ'
        exp = {
            'experiment_name': ['MULTIALLEN_NBYLWgxdjtgxFuFJ'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_NBYLWgxdjtgxFuFJ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_eGsIBpFzNwSmtHqt(self):
        """MULTIALLEN_eGsIBpFzNwSmtHqt multi-experiment creation."""
        model_folder = 'MULTIALLEN_eGsIBpFzNwSmtHqt'
        exp = {
            'experiment_name': ['MULTIALLEN_eGsIBpFzNwSmtHqt'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_eGsIBpFzNwSmtHqt']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_qcrgcwAJXsLFAkgk(self):
        """MULTIALLEN_qcrgcwAJXsLFAkgk multi-experiment creation."""
        model_folder = 'MULTIALLEN_qcrgcwAJXsLFAkgk'
        exp = {
            'experiment_name': ['MULTIALLEN_qcrgcwAJXsLFAkgk'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_qcrgcwAJXsLFAkgk']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_XgmcKtbzvsIlVDLj(self):
        """MULTIALLEN_XgmcKtbzvsIlVDLj multi-experiment creation."""
        model_folder = 'MULTIALLEN_XgmcKtbzvsIlVDLj'
        exp = {
            'experiment_name': ['MULTIALLEN_XgmcKtbzvsIlVDLj'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_XgmcKtbzvsIlVDLj']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_COeTwGNnfGGTMwDM(self):
        """MULTIALLEN_COeTwGNnfGGTMwDM multi-experiment creation."""
        model_folder = 'MULTIALLEN_COeTwGNnfGGTMwDM'
        exp = {
            'experiment_name': ['MULTIALLEN_COeTwGNnfGGTMwDM'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_COeTwGNnfGGTMwDM']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_VtLYZAJvUSNoCxdI(self):
        """MULTIALLEN_VtLYZAJvUSNoCxdI multi-experiment creation."""
        model_folder = 'MULTIALLEN_VtLYZAJvUSNoCxdI'
        exp = {
            'experiment_name': ['MULTIALLEN_VtLYZAJvUSNoCxdI'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_VtLYZAJvUSNoCxdI']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_PKHjQMaCyGNFcPeJ(self):
        """MULTIALLEN_PKHjQMaCyGNFcPeJ multi-experiment creation."""
        model_folder = 'MULTIALLEN_PKHjQMaCyGNFcPeJ'
        exp = {
            'experiment_name': ['MULTIALLEN_PKHjQMaCyGNFcPeJ'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_PKHjQMaCyGNFcPeJ']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_RWgfmkYJKIyIxsJG(self):
        """MULTIALLEN_RWgfmkYJKIyIxsJG multi-experiment creation."""
        model_folder = 'MULTIALLEN_RWgfmkYJKIyIxsJG'
        exp = {
            'experiment_name': ['MULTIALLEN_RWgfmkYJKIyIxsJG'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_RWgfmkYJKIyIxsJG']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_auvkJXeIdSEOGshz(self):
        """MULTIALLEN_auvkJXeIdSEOGshz multi-experiment creation."""
        model_folder = 'MULTIALLEN_auvkJXeIdSEOGshz'
        exp = {
            'experiment_name': ['MULTIALLEN_auvkJXeIdSEOGshz'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_auvkJXeIdSEOGshz']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_LQBgGqVoDECVwzmj(self):
        """MULTIALLEN_LQBgGqVoDECVwzmj multi-experiment creation."""
        model_folder = 'MULTIALLEN_LQBgGqVoDECVwzmj'
        exp = {
            'experiment_name': ['MULTIALLEN_LQBgGqVoDECVwzmj'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_LQBgGqVoDECVwzmj']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_kjNOnxgEgQaPTctM(self):
        """MULTIALLEN_kjNOnxgEgQaPTctM multi-experiment creation."""
        model_folder = 'MULTIALLEN_kjNOnxgEgQaPTctM'
        exp = {
            'experiment_name': ['MULTIALLEN_kjNOnxgEgQaPTctM'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_kjNOnxgEgQaPTctM']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_rApAcAKHTmogdbxc(self):
        """MULTIALLEN_rApAcAKHTmogdbxc multi-experiment creation."""
        model_folder = 'MULTIALLEN_rApAcAKHTmogdbxc'
        exp = {
            'experiment_name': ['MULTIALLEN_rApAcAKHTmogdbxc'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_rApAcAKHTmogdbxc']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_JmNNZXBzbXMwWAmm(self):
        """MULTIALLEN_JmNNZXBzbXMwWAmm multi-experiment creation."""
        model_folder = 'MULTIALLEN_JmNNZXBzbXMwWAmm'
        exp = {
            'experiment_name': ['MULTIALLEN_JmNNZXBzbXMwWAmm'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_JmNNZXBzbXMwWAmm']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_uCGqvRjiwlxtuqfU(self):
        """MULTIALLEN_uCGqvRjiwlxtuqfU multi-experiment creation."""
        model_folder = 'MULTIALLEN_uCGqvRjiwlxtuqfU'
        exp = {
            'experiment_name': ['MULTIALLEN_uCGqvRjiwlxtuqfU'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_uCGqvRjiwlxtuqfU']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_bOclxltvkumkxlmr(self):
        """MULTIALLEN_bOclxltvkumkxlmr multi-experiment creation."""
        model_folder = 'MULTIALLEN_bOclxltvkumkxlmr'
        exp = {
            'experiment_name': ['MULTIALLEN_bOclxltvkumkxlmr'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_bOclxltvkumkxlmr']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_fZJjjQrecDdChico(self):
        """MULTIALLEN_fZJjjQrecDdChico multi-experiment creation."""
        model_folder = 'MULTIALLEN_fZJjjQrecDdChico'
        exp = {
            'experiment_name': ['MULTIALLEN_fZJjjQrecDdChico'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_fZJjjQrecDdChico']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_SvtipPcoYttYjWtA(self):
        """MULTIALLEN_SvtipPcoYttYjWtA multi-experiment creation."""
        model_folder = 'MULTIALLEN_SvtipPcoYttYjWtA'
        exp = {
            'experiment_name': ['MULTIALLEN_SvtipPcoYttYjWtA'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_SvtipPcoYttYjWtA']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_otZgrODWNnSliblK(self):
        """MULTIALLEN_otZgrODWNnSliblK multi-experiment creation."""
        model_folder = 'MULTIALLEN_otZgrODWNnSliblK'
        exp = {
            'experiment_name': ['MULTIALLEN_otZgrODWNnSliblK'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_otZgrODWNnSliblK']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_nYRsfSXckItNAPTf(self):
        """MULTIALLEN_nYRsfSXckItNAPTf multi-experiment creation."""
        model_folder = 'MULTIALLEN_nYRsfSXckItNAPTf'
        exp = {
            'experiment_name': ['MULTIALLEN_nYRsfSXckItNAPTf'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_nYRsfSXckItNAPTf']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_kYWLrAAgoZUtAZcq(self):
        """MULTIALLEN_kYWLrAAgoZUtAZcq multi-experiment creation."""
        model_folder = 'MULTIALLEN_kYWLrAAgoZUtAZcq'
        exp = {
            'experiment_name': ['MULTIALLEN_kYWLrAAgoZUtAZcq'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_kYWLrAAgoZUtAZcq']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_VCMMMqNxXTWpzxlt(self):
        """MULTIALLEN_VCMMMqNxXTWpzxlt multi-experiment creation."""
        model_folder = 'MULTIALLEN_VCMMMqNxXTWpzxlt'
        exp = {
            'experiment_name': ['MULTIALLEN_VCMMMqNxXTWpzxlt'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_VCMMMqNxXTWpzxlt']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_SZrZfAbSxIMnRifW(self):
        """MULTIALLEN_SZrZfAbSxIMnRifW multi-experiment creation."""
        model_folder = 'MULTIALLEN_SZrZfAbSxIMnRifW'
        exp = {
            'experiment_name': ['MULTIALLEN_SZrZfAbSxIMnRifW'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_SZrZfAbSxIMnRifW']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_SZOsJMpzCJFsgfyo(self):
        """MULTIALLEN_SZOsJMpzCJFsgfyo multi-experiment creation."""
        model_folder = 'MULTIALLEN_SZOsJMpzCJFsgfyo'
        exp = {
            'experiment_name': ['MULTIALLEN_SZOsJMpzCJFsgfyo'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_SZOsJMpzCJFsgfyo']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_SgLdWGooXAkOYxWF(self):
        """MULTIALLEN_SgLdWGooXAkOYxWF multi-experiment creation."""
        model_folder = 'MULTIALLEN_SgLdWGooXAkOYxWF'
        exp = {
            'experiment_name': ['MULTIALLEN_SgLdWGooXAkOYxWF'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_SgLdWGooXAkOYxWF']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_YKgSSKDbimMsjJQC(self):
        """MULTIALLEN_YKgSSKDbimMsjJQC multi-experiment creation."""
        model_folder = 'MULTIALLEN_YKgSSKDbimMsjJQC'
        exp = {
            'experiment_name': ['MULTIALLEN_YKgSSKDbimMsjJQC'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_YKgSSKDbimMsjJQC']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_YwKEkeZsJfiUQKGC(self):
        """MULTIALLEN_YwKEkeZsJfiUQKGC multi-experiment creation."""
        model_folder = 'MULTIALLEN_YwKEkeZsJfiUQKGC'
        exp = {
            'experiment_name': ['MULTIALLEN_YwKEkeZsJfiUQKGC'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_YwKEkeZsJfiUQKGC']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp

    def MULTIALLEN_iyifjNMsWQIGFwCE(self):
        """MULTIALLEN_iyifjNMsWQIGFwCE multi-experiment creation."""
        model_folder = 'MULTIALLEN_iyifjNMsWQIGFwCE'
        exp = {
            'experiment_name': ['MULTIALLEN_iyifjNMsWQIGFwCE'],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'sep_conv2d'),
                os.path.join(model_folder, 'DoG'),
            ],
            'dataset': ['MULTIALLEN_iyifjNMsWQIGFwCE']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = ['resize']
        exp['epochs'] = 50
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 100
        exp['batch_size'] = 16  # Train/val batch size.
        return exp
