"""Project config file."""
import os
from utils import py_utils


class Config:
    """Config class with global project variables."""

    def __init__(self, **kwargs):
        """Global config file for normalization experiments."""
        self.data_root = '/Users/drewlinsley/Documents/contextual_model_psychophysic_data/cc_bp_data/image_datasets'
        self.project_directory = '/Users/drewlinsley/Documents/contextual_model_psychophysic_data/cc_bp_data/'
        self.tf_records = os.path.join(
            self.project_directory,
            'tf_records')
        self.checkpoints = os.path.join(
            self.project_directory,
            'checkpoints')
        self.summaries = os.path.join(
            self.project_directory,
            'summaries')
        self.experiment_evaluations = os.path.join(
            self.project_directory,
            'experiment_evaluations')
        self.condition_evaluations = os.path.join(
            self.project_directory,
            'condition_evaluations')
        self.visualizations = os.path.join(
            self.project_directory,
            'visualizations')
        self.plots = os.path.join(
            self.project_directory,
            'plots')
        self.log_dir = os.path.join(self.project_directory, 'logs')
        self.dataset_info = 'dataset_processing'  # local dataset classes

        # DB
        self.db_ssh_forward = False
        machine_name = os.uname()[1]
        if len(machine_name) == 12:  # Docker container
            self.db_ssh_forward = True

        # Create directories if they do not exist
        check_dirs = [
            self.tf_records,
            self.checkpoints,
            self.experiment_evaluations,
            self.condition_evaluations,
            self.visualizations,
            self.plots,
            self.log_dir,
            self.dataset_info
        ]
        [py_utils.make_dir(x) for x in check_dirs]

    def __getitem__(self, name):
        """Get item from class."""
        return getattr(self, name)

    def __contains__(self, name):
        """Check if class contains field."""
        return hasattr(self, name)
