import numpy as np
from config import Config


class data_processing(object):
    """Template file for Allen neural data."""
    def __init__(self):
        """Init global variables for contextual circuit bp."""
        self.name = 'ALLEN_selected_cells_1'
        self.config = Config()
        self.output_size = [1, 1]
        self.im_size = (304, 608, 1)
        self.model_input_image_size = [76, 152, 1]
        self.meta = '/media/data_cifs/contextual_circuit/tf_records/ALLEN_selected_cells_1_meta.npy'
        self.default_loss_function = 'l2'
        self.score_metric = 'pearson'
        self.preprocess = ['resize']

        # Load vars from the meta file
        meta_data = np.load(self.meta).item()
        self.folds = meta_data['folds']
        self.tf_reader = meta_data['tf_reader']
        self.tf_dict = {k: v for k, v in meta_data['tf_dict'].iteritems() if k in meta_data['tf_reader'].keys()}