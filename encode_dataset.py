import os
import config
import dataset_processing
from ops import putils
from ops import data_to_tfrecords


def encode_dataset(dataset):
    config = Config()
    data_class = import_module(dataset)
    data_proc = data_class.data_processing()
    files, labels = data_proc.get_data()
    targets = data_proc.targets
    im_size = data_proc.im_size
    ds_name = os.path.join(config.tf_records, data_proc.name)
    data_to_tfrecords(
        files=files,
        labels=labels,
        targets=targets,
        ds_name=ds_name,
        im_size=data_proc.im_size)
            
