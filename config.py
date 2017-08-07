import os
from utils import py_utils


class Config:
    def __init__(self, **kwargs):
        """Global config file for normalization experiments."""
        self.data_root = '/media/data_cifs/image_datasets/'
        self.project_directory = '/media/data_cifs/contextual_circuit/'
        self.tf_records = os.path.join(self.project_directory, 'tf_records')

        # DB
        self.db_ssh_forward = False

        # Create directories if they do not exist
        check_dirs = [
            self.tf_records,
        ]
        [py_utils.make_dir(x) for x in check_dirs]
