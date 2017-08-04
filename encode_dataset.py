import os
import dataset_processing
from config import Config
from ops import putils
from ops import data_to_tfrecords
from argparse import ArgumentParser
from ops.data_to_tfrecords import data_to_tfrecords

def encode_dataset(dataset):
    config = Config()
    data_class = putils.import_module(dataset)
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
        im_size=im_size)
            

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--dataset',
        dest='dataset',
        help='Name of the dataset.')
    args = parser.parse_args()
    encode_dataset(**vars(args))
