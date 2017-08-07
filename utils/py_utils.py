import os


def flatten_list(l):
    return [val for sublist in l for val in sublist]


def import_module(dataset, model_dir='dataset_processing'):
    return getattr(
        __import__(model_dir, fromlist=[dataset]), dataset)


def make_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

