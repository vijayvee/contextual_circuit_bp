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
            'early_stop': False  # Stop training if the loss stops improving.
        }

    def add_globals(self, exp):
        """Add attributes to this class."""
        for k, v in self.globals().iteritems():
            exp[k] = v
        return exp

    def one_layer_conv_mlp(self):
        """Each key in experiment_dict must be manually added to the schema.
        Results 8/29/17: L2 regulariaztion sucks. No reg CM does pretty well.
        """
        model_folder = 'one_layer_conv_mlp'
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
                os.path.join(model_folder, 'contextual_div_norm_no_reg'),
                os.path.join(model_folder, 'contextual_frozen_connectivity_l2_1'),
                os.path.join(model_folder, 'contextual_frozen_connectivity_l2_2'),
                os.path.join(model_folder, 'contextual_frozen_connectivity_l2_3'),
                os.path.join(model_folder, 'contextual_frozen_connectivity_l2_4'),
                os.path.join(model_folder, 'contextual_frozen_connectivity_l2_5'),
                os.path.join(model_folder, 'contextual_frozen_connectivity_l2_6'),
                os.path.join(model_folder, 'contextual_frozen_connectivity_l2_7'),
                os.path.join(model_folder, 'contextual_frozen_connectivity_l2_8'),
                os.path.join(model_folder, 'contextual_frozen_connectivity_l2_9'),
                os.path.join(model_folder, 'contextual_frozen_connectivity_l2_10'),
            ],
            'dataset': ['cifar_100']
        }
        return self.add_globals(exp)  # Add globals to the experiment

    def one_layer_conv_mlp_reg_or_no(self):
        """Each key in experiment_dict must be manually added to the schema."""
        model_folder = 'one_layer_conv_mlp_reg_or_no'
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
                os.path.join(model_folder, 'contextual_div_norm'),
                os.path.join(model_folder, 'contextual_div_norm_no_reg'),
                os.path.join(model_folder, 'contextual_5'),
                os.path.join(model_folder, 'contextual_no_reg_5'),
                os.path.join(model_folder, 'contextual_frozen_connectivity_5'),
                os.path.join(model_folder, 'contextual_frozen_connectivity_no_reg_5'),
                os.path.join(model_folder, 'contextual_frozen_CRF_connectivity_5'),
                os.path.join(model_folder, 'contextual_frozen_CRF_connectivity_no_reg_5'),
                os.path.join(model_folder, 'contextual_frozen_eCRF_connectivity_5'),
                os.path.join(model_folder, 'contextual_frozen_eCRF_connectivity_no_reg_5')
            ],
            'dataset': ['cifar_100']
        }
        return self.add_globals(exp)  # Add globals to the experiment

    def one_layer_rnns(self):
        """Each key in experiment_dict must be manually added to the schema."""
        model_folder = 'one_layer_rnns'
        exp = {
            'experiment_name': [model_folder],
            'lr': [1e-3],
            'loss_function': ['cce'],
            'optimizer': ['adam'],
            'regularization_type': [None],  # [None, 'l1', 'l2'],
            'regularization_strength': [0.005],
            'model_struct': [
                os.path.join(model_folder, 'divisive'),
                os.path.join(model_folder, 'batch'),
                os.path.join(model_folder, 'layer'),
                os.path.join(model_folder, 'lrn'),
                os.path.join(model_folder, 'contextual_frozen_connectivity_learned_transition_untuned_eCRF'),
                os.path.join(model_folder, 'contextual_frozen_connectivity_learned_transition_untuned_eCRF_vector_modulation'),
                os.path.join(model_folder, 'contextual_frozen_connectivity_learned_transition_weak_eCRF_vector_modulation'),
                os.path.join(model_folder, 'contextual_frozen_connectivity_learned_transition_untuned_eCRF_10'),
            ],
            'dataset': ['cifar_100']
        }
        return self.add_globals(exp)  # Add globals to the experiment

    def cluster_one_layer_rnns(self):
        """Each key in experiment_dict must be manually added to the schema."""
        model_folder = 'cluster_one_layer_rnns'
        exp = {
            'experiment_name': [model_folder],
            'lr': [1e-4],
            'loss_function': [None],  # Leave as None to use dataset default
            'optimizer': ['adam'],
            'regularization_type': [None],  # [None, 'l1', 'l2'],
            'regularization_strength': [0.005],
            'model_struct': [
                os.path.join(model_folder, 'divisive'),
                os.path.join(model_folder, 'batch'),
                os.path.join(model_folder, 'layer'),
                os.path.join(model_folder, 'lrn'),
                os.path.join(model_folder, 'learn_transition_weak_eCRF_1'),
                #  os.path.join(model_folder, 'learn_transition_weak_eCRF_3'),
                #  os.path.join(model_folder, 'learn_transition_weak_eCRF_5'),
                os.path.join(model_folder, 'learn_transition_untuned_eCRF_1'),
                #  os.path.join(model_folder, 'learn_transition_untuned_eCRF_3'),
                #  os.path.join(model_folder, 'learn_transition_untuned_eCRF_5'),
                os.path.join(model_folder, 'learn_transition_untuned_eCRF_vector_1'),  # 
                os.path.join(model_folder, 'learn_transition_untuned_eCRF_vector_3'),  # 
                #  os.path.join(model_folder, 'learn_transition_untuned_eCRF_vector_5'),
                #  os.path.join(model_folder, 'learn_transition_scalar_weak_eCRF_scalar_1'),
                #  os.path.join(model_folder, 'learn_transition_scalar_weak_eCRF_scalar_3'),
                #  os.path.join(model_folder, 'learn_transition_scalar_weak_eCRF_scalar_5'),
                #  os.path.join(model_folder, 'learn_transition_scalar_weak_eCRF_scalar_no_reg'),
                #  os.path.join(model_folder, 'learn_transition_weak_eCRF_scalar_1'),
                #  os.path.join(model_folder, 'learn_transition_weak_eCRF_scalar_3'),
                os.path.join(model_folder, 'learn_transition_weak_eCRF_scalar_5'),  # 
                os.path.join(model_folder, 'learn_transition_weak_eCRF_scalar_no_reg'),
                os.path.join(model_folder, 'learn_transition_weak_eCRF_vector_1'),  # 
                os.path.join(model_folder, 'learn_transition_weak_eCRF_vector_3'),  # 
                os.path.join(model_folder, 'learn_transition_weak_eCRF_vector_5'),
                os.path.join(model_folder, 'learn_transition_weak_eCRF_vector_no_reg'),  # 
            ],
            'dataset': ['cifar_100']
        }
        return self.add_globals(exp)  # Add globals to the experiment

    def cluster_one_layer_rnns_selective(self):
        """Each key in experiment_dict must be manually added to the schema."""
        model_folder = 'cluster_one_layer_rnns'
        exp = {
            'experiment_name': [model_folder],
            'lr': [1e-4],
            'loss_function': [None],  # Leave as None to use dataset default
            'optimizer': ['adam'],
            'regularization_type': [None],  # [None, 'l1', 'l2'],
            'regularization_strength': [0.005],
            'model_struct': [
                os.path.join(model_folder, 'divisive'),
                os.path.join(model_folder, 'batch'),
                os.path.join(model_folder, 'layer'),
                os.path.join(model_folder, 'lrn'),
                os.path.join(model_folder, 'learn_transition_weak_eCRF_1'),
                os.path.join(model_folder, 'learn_transition_untuned_eCRF_1'),
                os.path.join(model_folder, 'learn_transition_untuned_eCRF_vector_1'),  # 
                os.path.join(model_folder, 'learn_transition_untuned_eCRF_vector_3'),  # 
                os.path.join(model_folder, 'learn_transition_weak_eCRF_scalar_5'),  # 
                os.path.join(model_folder, 'learn_transition_weak_eCRF_scalar_no_reg'),
                os.path.join(model_folder, 'learn_transition_weak_eCRF_vector_1'),  # 
                os.path.join(model_folder, 'learn_transition_weak_eCRF_vector_3'),  # 
                os.path.join(model_folder, 'learn_transition_weak_eCRF_vector_5'),
                os.path.join(model_folder, 'learn_transition_weak_eCRF_vector_no_reg'),  # 
            ],
            'dataset': ['cifar_100']
        }
        return self.add_globals(exp)  # Add globals to the experiment

    def cluster_two_layer_rnns_selective(self):
        """Each key in experiment_dict must be manually added to the schema."""
        model_folder = 'cluster_two_layer_rnns_selective'
        exp = {
            'experiment_name': [model_folder],
            'lr': [1e-4],
            'loss_function': [None],  # Leave as None to use dataset default
            'optimizer': ['adam'],
            'regularization_type': [None],  # [None, 'l1', 'l2'],
            'regularization_strength': [0.005],
            'model_struct': [
                os.path.join(model_folder, 'divisive'),
                os.path.join(model_folder, 'batch'),
                os.path.join(model_folder, 'layer'),
                os.path.join(model_folder, 'lrn'),
                os.path.join(model_folder, 'learn_transition_weak_eCRF_1'),
                os.path.join(model_folder, 'learn_transition_untuned_eCRF_1'),
                os.path.join(model_folder, 'learn_transition_untuned_eCRF_vector_1'),  # 
                os.path.join(model_folder, 'learn_transition_untuned_eCRF_vector_3'),  # 
                os.path.join(model_folder, 'learn_transition_weak_eCRF_scalar_5'),  # 
                os.path.join(model_folder, 'learn_transition_weak_eCRF_scalar_no_reg'),
                os.path.join(model_folder, 'learn_transition_weak_eCRF_vector_1'),  # 
                os.path.join(model_folder, 'learn_transition_weak_eCRF_vector_3'),  # 
                os.path.join(model_folder, 'learn_transition_weak_eCRF_vector_5'),
                os.path.join(model_folder, 'learn_transition_weak_eCRF_vector_no_reg'),  # 
            ],
            'dataset': ['cifar_100']
        }
        return self.add_globals(exp)  # Add globals to the experiment

    def seven_px_one_layer_rnns_selective(self):
        """Each key in experiment_dict must be manually added to the schema."""
        model_folder = 'seven_px_one_layer_rnns_selective'
        exp = {
            'experiment_name': [model_folder],
            'lr': [1e-4],
            'loss_function': [None],  # Leave as None to use dataset default
            'optimizer': ['adam'],
            'regularization_type': [None],  # [None, 'l1', 'l2'],
            'regularization_strength': [0.005],
            'model_struct': [
                os.path.join(model_folder, 'divisive'),
                os.path.join(model_folder, 'batch'),
                os.path.join(model_folder, 'layer'),
                os.path.join(model_folder, 'learn_transition_weak_eCRF_1'),
                os.path.join(model_folder, 'learn_transition_untuned_eCRF_1'),
                os.path.join(model_folder, 'learn_transition_untuned_eCRF_vector_1'),  # 
                os.path.join(model_folder, 'learn_transition_untuned_eCRF_vector_3'),  # 
                os.path.join(model_folder, 'learn_transition_weak_eCRF_scalar_5'),  # 
                os.path.join(model_folder, 'learn_transition_weak_eCRF_scalar_no_reg'),
                os.path.join(model_folder, 'learn_transition_weak_eCRF_vector_1'),  # 
                os.path.join(model_folder, 'learn_transition_weak_eCRF_vector_3'),  # 
                os.path.join(model_folder, 'learn_transition_weak_eCRF_vector_5'),
                os.path.join(model_folder, 'learn_transition_weak_eCRF_vector_no_reg'),  # 
            ],
            'dataset': ['cifar_100']
        }
        return self.add_globals(exp)  # Add globals to the experiment

    def seven_px_one_layer_rnns_selective_imq(self):
        """Each key in experiment_dict must be manually added to the schema."""
        model_folder = 'seven_px_one_layer_rnns_selective'
        exp = {
            'experiment_name': [model_folder],
            'lr': [1e-4],
            'loss_function': [None],  # Leave as None to use dataset default
            'optimizer': ['adam'],
            'regularization_type': [None],  # [None, 'l1', 'l2'],
            'regularization_strength': [0.005],
            'model_struct': [
                os.path.join(model_folder, 'divisive'),
                os.path.join(model_folder, 'batch'),
                os.path.join(model_folder, 'layer'),
                os.path.join(model_folder, 'learn_transition_untuned_eCRF_1'),
                os.path.join(model_folder, 'learn_transition_weak_eCRF_vector_1'), 
                os.path.join(model_folder, 'learn_transition_weak_eCRF_vector_3'),
                os.path.join(model_folder, 'learn_transition_weak_eCRF_vector_5'),
                os.path.join(model_folder, 'learn_transition_weak_eCRF_vector_8'),
                os.path.join(model_folder, 'bn_learn_transition_weak_eCRF_vector_1'),
                os.path.join(model_folder, 'bn_learn_transition_weak_eCRF_vector_3'),
                os.path.join(model_folder, 'bn_learn_transition_weak_eCRF_vector_5'),
                os.path.join(model_folder, 'bn_learn_transition_weak_eCRF_vector_8'),
            ],
            'dataset': ['ChallengeDB_release']
        }
        return self.add_globals(exp)  # Add globals to the experiment

    def cluster_two_layer_rnns_no_regularization(self):
        """Each key in experiment_dict must be manually added to the schema."""
        model_folder = 'cluster_two_layer_rnns_no_regularization'
        exp = {
            'experiment_name': [model_folder],
            'lr': [1e-4],
            'loss_function': [None],  # Leave as None to use dataset default
            'optimizer': ['adam'],
            'regularization_type': [None],  # [None, 'l1', 'l2'],
            'regularization_strength': [0.005],
            'model_struct': [
                # os.path.join(model_folder, 'divisive'),
                # os.path.join(model_folder, 'batch'),
                # os.path.join(model_folder, 'layer'),
                # os.path.join(model_folder, 'lrn'),
                os.path.join(model_folder, 'learn_transition_weak_eCRF_1'),
                os.path.join(model_folder, 'learn_transition_weak_eCRF_3'),
                os.path.join(model_folder, 'learn_transition_weak_eCRF_5'),
                os.path.join(model_folder, 'learn_transition_untuned_eCRF_1'),
                os.path.join(model_folder, 'learn_transition_untuned_eCRF_3'),
                os.path.join(model_folder, 'learn_transition_untuned_eCRF_5'),
                os.path.join(model_folder, 'learn_transition_untuned_eCRF_vector_1'),
                os.path.join(model_folder, 'learn_transition_untuned_eCRF_vector_3'),
                os.path.join(model_folder, 'learn_transition_untuned_eCRF_vector_5'),
                os.path.join(model_folder, 'learn_transition_scalar_weak_eCRF_scalar_1'),
                os.path.join(model_folder, 'learn_transition_scalar_weak_eCRF_scalar_3'),
                os.path.join(model_folder, 'learn_transition_scalar_weak_eCRF_scalar_5'),
                os.path.join(model_folder, 'learn_transition_scalar_weak_eCRF_scalar_no_reg'),
                os.path.join(model_folder, 'learn_transition_weak_eCRF_scalar_1'),
                os.path.join(model_folder, 'learn_transition_weak_eCRF_scalar_3'),
                os.path.join(model_folder, 'learn_transition_weak_eCRF_scalar_5'),
                os.path.join(model_folder, 'learn_transition_weak_eCRF_scalar_no_reg'),
                os.path.join(model_folder, 'learn_transition_weak_eCRF_vector_1'),
                os.path.join(model_folder, 'learn_transition_weak_eCRF_vector_3'),
                os.path.join(model_folder, 'learn_transition_weak_eCRF_vector_5'),
                os.path.join(model_folder, 'learn_transition_weak_eCRF_vector_no_reg'),
            ],
            'dataset': ['cifar_100']
        }
        return self.add_globals(exp)  # Add globals to the experiment

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

    def ALLEN_all_neurons(self):
        """Each key in experiment_dict must be manually added to the schema."""
        model_folder = 'ALLEN_all_neurons'
        exp = {
            'experiment_name': [model_folder],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                # os.path.join(model_folder, 'conv3d'),
                # os.path.join(model_folder, 'DoG'),
                # os.path.join(model_folder, 'sparse_conv2d')
            ],
            'dataset': ['ALLEN_all_neurons']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = [['resize']]
        return exp

    def ALLEN_selected_cells_1(self):
        """Each key in experiment_dict must be manually added to the schema."""
        model_folder = 'ALLEN_selected_cells_1'
        exp = {
            'experiment_name': [model_folder],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'conv2d'),
                os.path.join(model_folder, 'norm_conv2d'),
                os.path.join(model_folder, 'DoG'),
                os.path.join(model_folder, 'sparse_conv2d')
            ],
            'dataset': ['ALLEN_selected_cells_1']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = [['resize']]
        exp['epochs'] = 1000
        return exp

    def ALLEN_all_neurons_hp_conv(self):
        """Each key in experiment_dict must be manually added to the schema."""
        model_folder = 'ALLEN_all_neurons_hp_conv'
        exp = {
            'experiment_name': model_folder,
            'hp_optim': 'gpyopt',
            'hp_multiple': 10,
            'lr': 1e-4,
            'lr_domain': [1e-5, 1e-1],
            'loss_function': None,  # Leave as None to use dataset default
            'optimizer': 'adam',
            'regularization_type': 'l2',  # [None, 'l1', 'l2'],
            'regularization_strength': 1e-5,
            'regularization_strength_domain': [1e-7, 1e-1],
            # 'timesteps': True,
            'model_struct': os.path.join(model_folder, 'conv2d'),
            'dataset': 'ALLEN_all_neurons',
            'early_stop': True
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = [['resize']]
        return exp

    def ALLEN_all_neurons_hp_dog(self):
        """Each key in experiment_dict must be manually added to the schema."""
        model_folder = 'ALLEN_all_neurons_hp_dog'
        exp = {
            'experiment_name': model_folder,
            'hp_optim': 'gpyopt',
            'hp_multiple': 10,
            'lr': 1e-4,
            'lr_domain': [1e-5, 1e-1],
            'loss_function': None,  # Leave as None to use dataset default
            'optimizer': 'adam',
            'regularization_type': 'l2',  # [None, 'l1', 'l2'],
            'regularization_strength': 1e-5,
            'regularization_strength_domain': [1e-7, 1e-1],
            # 'timesteps': True,
            'model_struct': os.path.join(model_folder, 'dog'),
            'dataset': 'ALLEN_all_neurons'
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['early_stop'] = True
        exp['data_augmentations'] = [['resize']]
        exp['epochs'] = 1
        return exp

    def contextual_model_paper(self):
        """Each key in experiment_dict must be manually added to the schema."""
        model_folder = 'contextual_model_paper'
        exp = {
            'experiment_name': [model_folder],
            'lr': [3e-4],
            'loss_function': ['l2'],
            'optimizer': ['adam'],
            'model_struct': [
                os.path.join(model_folder, 'divisive_paper_rfs'),
                os.path.join(model_folder, 'contextual_paper_rfs'),
                os.path.join(model_folder, 'divisive'),
                os.path.join(model_folder, 'contextual'),
            ],
            'dataset': ['contextual_model_multi_stimuli']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = [[None]]
        exp['epochs'] = 1000
        exp['batch_size'] = 256
        return exp
