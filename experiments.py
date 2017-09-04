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
            'batch_size': 64,  # Train/val batch size.
            'data_augmentations': [['random_crop', 'left_right']],  # Random_crop, etc.
            'epochs': 200,
            'shuffle': True,  # Shuffle data.
            'validation_iters': 500,  # How often to evaluate validation.
            'num_validation_evals': 100,  # How many validation batches.
            'top_n_validation': 0,  # Set to 0 to save all checkpoints.
            'early_stop': False  # Stop training if the loss stops improving.
        }

    def add_globals(self, exp):
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

