import argparse
import numpy as np
import itertools as it
from db import db
from db import credentials
from utils import logger as log


"""Each key in experiment_dict must be manually added to the schema."""
experiment_dict = {
    'lr': [1e-3],  # np.logspace(-5, -2, 4, base=10),
    'loss_function': ['cce'],
    'optimizer': ['adam'],
    'wd_layers': [None],
    'wd_penalty': [None],
    'model_struct': ['one_layer_conv_mlp'],
    'dataset': ['mnist', 'cifar']
}


def package_parameters(parameter_dict):
    """Derive combinations of experiment parameters."""
    keys_sorted = sorted(parameter_dict)
    values = list(it.product(*(parameter_dict[key] for key in keys_sorted)))
    combos = tuple({k: v for k, v in zip(keys_sorted, row)} for row in values)
    log.info('Derived %s combinations.' % len(combos))
    return combos


def main(reset_process, initialize_db):
    """Populate db with experiments to run."""
    if reset_process:
        db.reset_in_process()
        log.info('Reset experiment progress counter in DB.')
    if initialize_db:
        log.info('Initialized DB.')
        db.initialize_database()
        log.info('Adding new experiments.')
        config = credentials.postgresql_connection()
        exp_combos = package_parameters(experiment_dict)
        with db(config) as db_conn:
            db_conn.populate_db(exp_combos)
            db_conn.return_status('CREATE')
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset_process",
        dest="reset_process",
        action='store_true',
        help='Reset the in_process table.')
    parser.add_argument(
        "--initialize",
        dest="initialize_db",
        action='store_true',
        help='Recreate your database of experiments.')
    args = parser.parse_args()
    main(**vars(args))

