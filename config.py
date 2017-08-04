import os
from ops import putils


class Config:
    def __init__(self, **kwargs):
        self.data_root = '/media/data_cifs/image_datasets/'
        self.project_directory = '/media/data_cifs/contextual_circuit/'
        self.tf_records = os.path.join(self.project_directory, 'tf_records')

        check_dirs = [
            self.tf_records,
        ]
        [putils.make_dir(x) for x in check_dirs]
