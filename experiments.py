"""Class to specify all DNN experiments."""
import os


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
            'loss_weights': None,  # Weight your loss w/ a dictionary.
            'validation_iters': 5000,  # How often to evaluate validation.
            'num_validation_evals': 100,  # How many validation batches.
            'top_n_validation': 0,  # Set to 0 to save all checkpoints.
            'early_stop': False,  # Stop training if the loss stops improving.
            'save_weights': False,  # Save model weights on validation evals.
            'optimizer_constraints': None,  # A {var name: bound} dictionary.
            'resize_output': None  # Postproc resize of the output (FC models).
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

    def contextual_model_paper(self):
        """Each key in experiment_dict must be manually added to the schema."""
        model_folder = 'contextual_model_paper'
        exp = {
            'experiment_name': [model_folder],
            'lr': [5e-3],
            'loss_function': ['l2'],
            'optimizer': ['adam'],
            'q_t': [1e-1],  # [1e-3, 1e-1],
            'p_t': [1.],  # [1e-2, 1e-1, 1],
            't_t': [1.],  # [1e-2, 1e-1, 1],
            'timesteps': [5],
            'model_struct': [
                # os.path.join(model_folder, 'divisive_paper_rfs'),
                os.path.join(model_folder, 'contextual_paper_rfs'),
                # os.path.join(model_folder, 'contextual_ss_paper_rfs'),
                # os.path.join(model_folder, 'divisive'),
                # os.path.join(model_folder, 'contextual')
            ],
            'dataset': ['contextual_model_multi_stimuli']
            # 'dataset': ['contextual_model_stimuli']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = [[None]]
        exp['epochs'] = 500
        exp['batch_size'] = 1  # Train/val batch size.
        exp['save_weights'] = True
        return exp

    def contours(self):
        """Each key in experiment_dict must be manually added to the schema."""
        model_folder = 'contours'
        exp = {
            'experiment_name': [model_folder],
            'lr': [1e-3],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],  #, 'momentum','sgd'],
            # 'q_t': [1e-3, 1e-1],
            # 'p_t': [1e-2, 1e-1, 1],
            # 't_t': [1e-2, 1e-1, 1],
            'timesteps': [3],
            'model_struct': [
                # os.path.join(
                #     model_folder, 'context_conv2d'),
                #os.path.join(
                #    model_folder, 'context_association_conv2d'),
                # os.path.join(
                #     model_folder, 'context_association_l1_conv2d'),
                # os.path.join(
                #     model_folder, 'context_association_full_full_conv2d'),
                # os.path.join(
                #     model_folder, 'context_association_full_hole_conv2d'),
                # os.path.join(
                #     model_folder, 'context_association_crf_hole_conv2d'),
                #os.path.join(
                #    model_folder, 'context_association_l1_full_full_conv2d'),
                # os.path.join(
                #     model_folder, 'context_association_l1_full_hole_conv2d'),
                os.path.join(
                    model_folder, 'context_association_full_full_conv2d_single_conv'),
                # os.path.join(
                #     model_folder, 'context_association_l1_crf_hole_conv2d'),
                #os.path.join(
                #    model_folder, 'conv2d'),
            ],
            'dataset': ['BSDS500_2']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = [[
            'random_crop_image_label',
            # 'lr_flip_image_label',
            # 'ud_flip_image_label'
            ]]
        # exp['val_augmentations'] = [['center_crop_image_label']]
        exp['batch_size'] = 10  # Train/val batch size.
        exp['epochs'] = 1000
        exp['save_weights'] = True
        exp['validation_iters'] = 500
        exp['num_validation_evals'] = 10
        exp['resize_output'] = [[150, 240]]
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

    def ALLEN_st_selected_cells_1(self):
        """Each key in experiment_dict must be manually added to the schema."""
        model_folder = 'ALLEN_st_selected_cells_1'
        exp = {
            'experiment_name': [model_folder],
            'lr': [3e-4],
            'loss_function': ['pearson'],
            'optimizer': ['adam'],
            'regularization_type': ['l2'],  # [None, 'l1', 'l2'],
            'regularization_strength': [1e-7],
            'model_struct': [
                os.path.join(model_folder, 'complete_sep_conv3d'),
                os.path.join(model_folder, 'time_sep_conv3d'),
                os.path.join(model_folder, 'complete_sep_nl_conv3d'),
                os.path.join(model_folder, 'time_sep_nl_conv3d'),
                os.path.join(model_folder, 'conv3d'),
                os.path.join(model_folder, 'lstm2d'),
                os.path.join(model_folder, 'gru2d'),
                os.path.join(model_folder, 'rnn2d'),
                os.path.join(model_folder, 'sgru2d')
            ],
            'dataset': ['ALLEN_selected_cells_1']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['data_augmentations'] = [['resize']]
        exp['epochs'] = 50
        exp['validation_iters'] = 200
        exp['num_validation_evals'] = 225
        exp['batch_size'] = 10  # Train/val batch size.
        exp['save_weights'] = True
        return exp

    def crcns_1d(self):
        """Each key in experiment_dict must be manually added to the schema."""
        model_folder = 'crcns_1d'
        exp = {
            'experiment_name': [model_folder],
            'lr': [5e-3, 1e-3, 5e-4, 1e-4],
            'loss_function': ['cce'],
            'optimizer': ['nadam'],
            'model_struct': [
                os.path.join(model_folder, 'gru1d_1'),
                os.path.join(model_folder, 'gru1d_2'),
                os.path.join(model_folder, 'gru1d_3'),
                os.path.join(model_folder, 'gru1d_4'),
                os.path.join(model_folder, 'gru1d_5'),
                os.path.join(model_folder, 'gru1d_6'),
                os.path.join(model_folder, 'gru1d_7'),
                os.path.join(model_folder, 'gru1d_8'),
                os.path.join(model_folder, 'gru1d_9'),
                os.path.join(model_folder, 'gru1d_10'),
                os.path.join(model_folder, 'gru1d_11'),
                os.path.join(model_folder, 'gru1d_12'),
                os.path.join(model_folder, 'gru1d_13'),
                os.path.join(model_folder, 'gru1d_14'),
                os.path.join(model_folder, 'gru1d_15'),
                os.path.join(model_folder, 'gru1d_16'),
                os.path.join(model_folder, 'gru1d_17'),
                os.path.join(model_folder, 'gru1d_18'),
                os.path.join(model_folder, 'gru1d_19'),
                os.path.join(model_folder, 'gru1d_20'),
            ],
            'dataset': ['crcns_1d']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['epochs'] = 100
        exp['validation_iters'] = 200
        exp['num_validation_evals'] = 225
        exp['batch_size'] = 20  # Train/val batch size.
        exp['save_weights'] = True
        exp['data_augmentations'] = [
            [
                None
                # 'calculate_rate',
                # 'random_time_crop'
            ]
        ]
        return exp

    def crcns_2d(self):
        """Each key in experiment_dict must be manually added to the schema."""
        model_folder = 'crcns_2d'
        exp = {
            'experiment_name': [model_folder],
            'lr': [1e-5],
            'loss_function': ['cce'],
            'optimizer': ['nadam'],
            'model_struct': [
                # os.path.join(model_folder, 'sepgru2d'),
                os.path.join(model_folder, 'fc_sepgru2d'),
            ],
            'dataset': ['crcns_2d']
        }
        exp = self.add_globals(exp)  # Add globals to the experiment'
        exp['epochs'] = 50
        exp['validation_iters'] = 200
        exp['num_validation_evals'] = 225
        exp['batch_size'] = 16  # Train/val batch size.
        exp['save_weights'] = True
        exp['data_augmentations'] = [
            [
                # None
                # 'resize'
                # 'calculate_rate_time_crop',
                'left_right',
                # 'random_time_crop',
                'random_crop',
                'up_down'
                # 'rotate'
            ]
        ]
        return exp
