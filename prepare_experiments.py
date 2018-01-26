import os
import json
import argparse
import config
import numpy as np
import pandas as pd
import itertools as it
from db import db
from db import credentials
from utils import logger, py_utils
from experiments import experiments


def protected_keys():
    """Keys for which lists are allowed."""
    return [
        'data_augmentations',
    ]


def prepare_hp_params(parameter_dict, pk, log=None):
    """Jsonify HP domains for online updates."""
    it_dict = {}
    for k, v in parameter_dict.iteritems():
        if '_domain' in k:
            if isinstance(v, np.ndarray):
                v = pd.Series(np.sort(v)).to_json(orient='values')
            elif isinstance(v, basestring):
                pass
            else:
                v = json.dumps(v)
        else:
            # Transform data from lists
            if isinstance(v, list):
                v = py_utils.flatten_list(v, log)
                if k not in pk:
                    v = v[0]
        it_dict[k] = v  # Handle special-case hp optim flags here.
    return it_dict


def online_hp_optim_parameters(parameter_dict, log, ms_key='model_struct'):
    """Experiment parameters in the case of hp_optimization algorithms."""
    model_structs = parameter_dict[ms_key]
    parameter_dict = {
        k: v for k, v in parameter_dict.iteritems() if k is not ms_key}
    assert 'aggregator' in parameter_dict.keys(),\
        'You must specify an aggregator as \'max\' or \'min\'.'
    pk = protected_keys()
    combos = []
    for ms in model_structs:
        it_dict = prepare_hp_params(
            parameter_dict=parameter_dict,
            pk=pk,
            log=log)
        it_dict[ms_key] = ms
        # Hard code hp_current_iteration=0
        it_dict['hp_current_iteration'] = 0
        combos += [it_dict]
    return combos


def package_parameters(parameter_dict, log):
    """Derive combinations of experiment parameters."""
    parameter_dict = {
        k: v for k, v in parameter_dict.iteritems() if isinstance(v, list)
    }
    keys_sorted = sorted(parameter_dict)
    values = list(it.product(*(parameter_dict[key] for key in keys_sorted)))
    combos = tuple({k: v for k, v in zip(keys_sorted, row)} for row in values)
    for c in combos:
        if c['optimizer'] != 'adam':
            c['lr'] = 5e-5
    log.info('Derived %s combinations.' % len(combos))
    return list(combos)


def main(reset_process, initialize_db, experiment_name, remove=None):
    """Populate db with experiments to run."""
    main_config = config.Config()
    log = logger.get(os.path.join(main_config.log_dir, 'prepare_experiments'))
    if reset_process:
        db.reset_in_process()
        log.info('Reset experiment progress counter in DB.')
    if initialize_db:
        db.initialize_database()
        log.info('Initialized DB.')
    if experiment_name is not None:  # TODO: add capability for bayesian opt.
        db_config = credentials.postgresql_connection()
        experiment_dict = experiments()[experiment_name]()
        if 'hp_optim' in experiment_dict.keys() and\
                experiment_dict['hp_optim'] is not None:
            exp_combos = online_hp_optim_parameters(experiment_dict, log)
            log.info('Preparing an hp-optimization experiment.')
        else:
            exp_combos = package_parameters(experiment_dict, log)
            log.info('Preparing a grid-search experiment.')
        with db.db(db_config) as db_conn:
            db_conn.populate_db(exp_combos)
            db_conn.return_status('CREATE')
        log.info('Added new experiments.')
    if remove is not None:
        db_config = credentials.postgresql_connection()
        with db.db(db_config) as db_conn:
            db_conn.remove_experiment(remove)
        log.info('Removed %s.' % remove)


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
        "--experiment",
        dest="experiment_name",
        default=None,
        type=str,
        help='Experiment to add to the database.')
    parser.add_argument(
        "--remove",
        dest="remove",
        default=None,
        type=str,
        help='Experiment to remove from the database.')
    args = parser.parse_args()
    main(**vars(args))
