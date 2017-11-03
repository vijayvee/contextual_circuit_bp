import numpy as np
from config import Config


class data_processing(object):
    """Template file for Allen neural data."""
    def __init__(self):
        """Init global variables for contextual circuit bp."""
        self.name = 'VISp_Cux2-CreERT2_175_7900_7100_0'
        self.config = Config()
        self.output_size = [1, 1]
        self.im_size = (304, 608, 1)
        self.model_input_image_size = [152, 304, 1]
        self.meta = '/media/data_cifs/contextual_circuit/tf_records/VISp_Cux2-CreERT2_175_7900_7100_0_meta.npy'
        self.default_loss_function = 'pearson'
        self.score_metric = 'pearson'
        self.preprocess = ['resize']

        # Load vars from the meta file
        meta_data = np.load(self.meta).item()
        self.folds = meta_data['folds']
        self.tf_reader = meta_data['tf_reader']
        self.tf_dict = {
            k: v for k, v in meta_data['tf_dict'].iteritems()
            if k in meta_data['tf_reader'].keys()}
        if len(self.output_size) > 2:
            # We are doing 3D convolutions
            for k, v in self.tf_reader.iteritems():
                v['reshape'] = (self.output_size[0],) + tuple(v['reshape'])
                self.tf_reader[k] = v
