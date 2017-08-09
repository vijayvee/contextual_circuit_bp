import os
import argparse
import itertools as it
from db import db
from db import credentials
from utils import logger
from config import Config
from models.experiments import experiments


def package_parameters(parameter_dict, log):
    """Derive combinations of experiment parameters."""
    parameter_dict = {k: v for k, v in parameter_dict.iteritems() if isinstance(v, list)}
    keys_sorted = sorted(parameter_dict)
    values = list(it.product(*(parameter_dict[key] for key in keys_sorted)))
    combos = tuple({k: v for k, v in zip(keys_sorted, row)} for row in values)
    log.info('Derived %s combinations.' % len(combos))
    return combos


def main(reset_process, initialize_db, experiment_name):
    """Populate db with experiments to run."""
    main_config = Config()
    log = logger.get(os.path.join(main_config.log_dir, 'prepare_experiments'))
    if reset_process:
        db.reset_in_process()
        log.info('Reset experiment progress counter in DB.')
    if initialize_db:
        db.initialize_database()
        log.info('Initialized DB.')
        db_config = credentials.postgresql_connection()
        experiment_dict = experiments()[experiment_name]()
        exp_combos = package_parameters(experiment_dict, log)
        with db.db(db_config) as db_conn:
            db_conn.populate_db(exp_combos)
            db_conn.return_status('CREATE')
        log.info('Added new experiments.')


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
    parser.add_argument(
        "--experiment_name",
        dest="experiment_name",
        default='one_layer_conv_mlp',
        type=str,
        help='Experiment to add to the database.')
    args = parser.parse_args()
    main(**vars(args))
