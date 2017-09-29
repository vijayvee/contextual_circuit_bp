import GPyOpt
import numpy as np
from utils import py_utils


def hp_optim_interpreter(hp_hist, performance_history, exp_params):
    """Pass hyperparameter and performance history to HP optimization alg."""
    if isinstance(exp_params, list):
        exp_params = exp_params[-1]
    hp_type = exp_params['hp_optim']
    domain = gather_domains(exp_params=exp_params, hp_type=hp_type)
    if hp_type == 'gpyopt':
        hp_hist_values = np.asarray([v.values() for v in hp_hist])
        next_step = gpyopt_wrapper(
            X=hp_hist_values,
            Y=np.asarray(performance_history),
            domain=domain)
        return {k: v for k, v in zip(hp_hist[0].keys(), next_step.ravel())}
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
    """Convert experiment params to a hp-optim ready domain dict."""
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
                raise RuntimeError('Hp-optimizer not implemented.')
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
    my_prob = GPyOpt.methods.BayesianOptimization(
        f=f,
        X=X,
        Y=Y,
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
        'tuning_u_domain': 'tuning_u',
        'tuning_t_domain': 'tuning_t',
        'tuning_q_domain': 'tuning_q',
        'tuning_p_domain': 'tuning_p',
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
            if 'timesteps' in aux_dict and exp_params['timesteps'] is not None:
                aux_dict['timesteps'] = int(exp_params['timesteps'])
            layer['normalization_aux'] = aux_dict
            layer_structure[idx] = layer
    return layer_structure


if __name__ == '__main__':
    gpyopt_test()
