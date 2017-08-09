import os
import numpy as np


def flatten_list(l):
    """Flatten a list of lists."""
    return [val for sublist in l for val in sublist]


def import_module(dataset, model_dir='dataset_processing'):
    """Dynamically import a module."""
    return getattr(
        __import__(model_dir, fromlist=[dataset]), dataset)


def make_dir(d):
    """Make directory d if it does not exist."""
    if not os.path.exists(d):
        os.makedirs(d)


def save_npys(data, model_name, output_string):
    """Save key/values in data as numpys."""
    for k, v in data.iteritems():
        output = os.path.join(
            output_string,
            '%s_%s' % (model_name, k)
            )
        np.save(output, v)


def check_path(data_pointer, log, msg):
    if not os.path.exists(data_pointer):
        log.error(msg)
