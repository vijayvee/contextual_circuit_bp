import numpy as np
from utils import py_utils
try:
    import GPyOpt
except:
    print 'Could not install GPyOpt.'


def hp_optim_interpreter(
        performance_history,
        aggregator):
    """Pass hyperparameter and performance history to HP optimization alg.

    Parameters
    ----------
    performance_history : list of dictionaries
    performance_metric : str

    Returns
    -------
    hp_dict : dictionary of the next experiment's hyperparameters
    """
    exp_row = performance_history[-1]
    max_iteration = np.max(
        [x['hp_current_iteration'] for x in performance_history])
    exp_row['hp_current_iteration'] = max_iteration

    # Prepare hp domains
    hp_type = exp_row['hp_optim']
    domain = gather_domains(
        exp_params=exp_row,
        hp_type=hp_type)

    # Minimize or maximize
    if aggregator == 'max':
        mfun = lambda x: x * -1  # TODO: Make an optimizer interpreter
    else:
        mfun = lambda x: x

    # Pull performance into a list for Y
    experiment_performance = [
        mfun(p[aggregator]) for p in performance_history]

    # Derive next hyperparameters
    if hp_type == 'gpyopt':
        next_step = gpyopt_wrapper(
            X=performance_history,
            Y=experiment_performance,
            domain=domain)
        opted = [x['name'] for x in domain]
        next_params = dict(zip(opted, next_step[0]))
        for k, v in next_params.iteritems():
            # Update the experiment template with the next study's HPs
            exp_row[k] = v
        return exp_row
    else:
        raise RuntimeError('Hp-optimizer not implemented.')


def package_domain(hp_dict):
    """Package sql domain params into dict."""
    domains = {}
    for k, v in hp_dict.iteritems():
        if '_domain' in k:
            domains[k] = v
    return domains


def gather_domains(exp_params, hp_type):
    """Convert experiment params to a hp-optim ready domain dict.

    Parameters
    ----------
    exp_params : dictionary
    hp_type : str

    Returns
    -------
    dlist : list of dictionaries
    """
    hps = hp_opt_dict()
    dlist = []
    for k, v in hps.iteritems():
        if exp_params[k] is not None:
            # If we are searching this domain.
            if hp_type == 'gpyopt':
                dlist += [
                    {
                        'name': v,
                        'type': gpyopt_interpret_type(exp_params[v]),
                        'domain': py_utils.convert_to_tuple(exp_params[k])
                    }
                ]
            else:
                raise NotImplementedError
    return dlist


def gpyopt_interpret_type(v):
    """Interpret data type for gpyopt optimization."""
    if isinstance(v, float):
        return 'continuous'
    elif isinstance(v, int):
        return 'discrete'
    else:
        raise RuntimeError('Cannot handle non-numeric values')


def gpyopt_wrapper(
        X,
        Y,
        domain,
        f=None,
        bs=1,
        cores=1,
        evaluator_type='local_penalization',
        hp_type='bayesian'):
    """Wrapper for gpyopt optimization."""
    to_opt = [x['name'] for x in domain]

    # Preprocess X
    X_hist = np.asarray([[v[z] for z in to_opt] for v in X])

    # Preprocess Y
    Y_hist = np.asarray([[h] for h in Y])
    my_prob = GPyOpt.methods.BayesianOptimization(  # TODO: Check discrete domains.
        f=f,
        X=X_hist,
        Y=Y_hist,
        domain=domain,
        evaluator_type=evaluator_type,
        batch_size=bs,
        num_cores=cores)
    if hp_type == 'bayesian':
        return my_prob.suggested_sample
    else:
        raise RuntimeError('Gpyopt optimization not implemented.')


def hp_opt_dict():
    return {
        'regularization_type_domain': 'regularization_type',
        'regularization_strength_domain': 'regularization_strength',
        'optimizer_domain': 'optimizer',
        'lr_domain': 'lr',
        'timesteps_domain': 'timesteps',
        'u_t_domain': 'tuning_u',
        't_t_domain': 'tuning_t',
        'q_t_domain': 'tuning_q',
        'p_t_domain': 'tuning_p',
        'filter_size_domain': 'filter_size',
    }


def gpyopt_test():
    """gpyopt tester script."""
    # Function that takes in data and makes a suggested next parameter combo
    def next_hyps(x_dat, y_dat, vars_dom):
        my_prob = GPyOpt.methods.BayesianOptimization(
            f=None, X=x_dat, Y=y_dat, domain=vars_dom,
            evaluator_type='local_penalization', batch_size=1,
            num_cores=1)
        return my_prob.suggested_sample

    # Input parameter combos
    x_init = np.array([
        [1e-5, 1e-7],
        [1e-5, 1e-7]
        ])
    # Output of model evaluated at inputs
    y_init = np.array([
        [10000],
        [10002]
        ])
    # Details about the domain of variables. It seems like continuos variables
    # should be listed first for some reason
    bds = [
        {'name': 'var_1', 'type': 'continuous', 'domain': (1e-1, 1e-10)},
        {'name': 'var_2', 'type': 'continuous', 'domain': (1e-1, 1e-10)},
    ]

    # Display output
    print next_hyps(x_init, y_init, bds)


def inject_model_with_hps(layer_structure, exp_params):
    """Inject model description with hyperparameters from database."""
    # TODO Change API so this isn't hardcoded.
    for idx, layer in enumerate(layer_structure):
        if 'hp_optimize' in layer.keys():
            # If the layer has been designated as optimizable:
            if 'normalization_aux' in layer.keys():
                aux_dict = layer['normalization_aux']
                if 'u_t' in aux_dict and exp_params['u_t'] is not None:
                    aux_dict['u_t'] = exp_params['u_t']
                if 'q_t' in aux_dict and exp_params['q_t'] is not None:
                    aux_dict['q_t'] = exp_params['q_t']
                if 't_t' in aux_dict and exp_params['t_t'] is not None:
                    aux_dict['t_t'] = exp_params['t_t']
                if 'p_t' in aux_dict and exp_params['p_t'] is not None:
                    aux_dict['p_t'] = exp_params['p_t']
                if 'timesteps' in aux_dict and\
                        exp_params['timesteps'] is not None:
                    aux_dict['timesteps'] = int(exp_params['timesteps'])
                layer['normalization_aux'] = aux_dict
                layer_structure[idx] = layer
            if 'filter_size' in layer.keys() and\
                    exp_params['filter_size'] is not None:
                assert len(layer['filter_size']) == 1,\
                    'Only optimize a single layer of filters at once.'
                layer_structure[idx]['filter_size'] = [
                    exp_params['filter_size']]
    return layer_structure


if __name__ == '__main__':
    gpyopt_test()
