import os
from utils import py_utils


class Config:
    def __init__(self, **kwargs):
        """Global config file for normalization experiments."""
        self.data_root = '/media/data_cifs/image_datasets/'
        self.project_directory = '/media/data_cifs/contextual_circuit/'
        self.tf_records = os.path.join(
            self.project_directory,
            'tf_records')
        self.checkpoints = os.path.join(
            self.project_directory,
            'checkpoints')
        self.summaries = os.path.join(
            self.project_directory,
            'summaries')
        self.evaluations = os.path.join(
            self.project_directory,
            'evaluations')
        self.visualizations = os.path.join(
            self.project_directory,
            'visualizations')
        self.log_dir = os.path.join(self.project_directory, 'logs')

        # DB
        self.db_ssh_forward = False

        # Create directories if they do not exist
        check_dirs = [
            self.tf_records,
            self.checkpoints,
            self.evaluations,
            self.visualizations,
            self.log_dir
        ]
        [py_utils.make_dir(x) for x in check_dirs]

    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)
