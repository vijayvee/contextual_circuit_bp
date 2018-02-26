import os, sys
sys.path.append(os.path.abspath(os.path.join('..','..')))

def get_params():
    params = {}
    params['raw_input_size'] = [180, 180, 1]
    params['batch_size'] = 100
    params['num_categories'] = 2
    params['num_min_train_imgs'] = 2000000
    params['num_max_train_imgs'] = 20000000
    params['num_val_period_imgs'] = 5000 #50000
    params['num_val_imgs'] = 5000 #50000
    params['threshold_loss'] = 0.95

    params['learning_rate'] = 1e-4
    params['clip_gradient'] = True
    params['dropout_keep_prob'] = 0.8

    from instances import processor_instances

    params['model_obj'] = processor_instances.PSVRT_siamese_nopool
    # params['model_obj'] = prrcessor_instances.PSVRT_siamesenet
    # params['model_obj'] = processor_instances.PSVRT_multichannel
    params['model_name'] = 'model'
    params['model_init_args'] = {'num_categories': 2,
                                 'num_CP_layers': 7,
                                 'num_CP_features': 16, # 4
                                 'num_FC_layers': 4,
                                 'num_FC_features': 512,
                                 'initial_conv_rf_size': [2, 2],
                                 'interm_conv_rf_size': [2, 2],
                                 'pool_rf_size': [2, 2],
                                 'stride_size': [2, 2],
                                 'activation_type': 'relu',
                                 'trainable': True,
                                 'hamstring_factor': 1}

    from instances import psvrt_new
    params['train_data_obj'] = psvrt_new.n_sd_k
    params['train_data_init_args'] = {'item_size': [4,4],
                                      'box_extent': [90,90],
                                      'raw_input_size' : list(params['raw_input_size']),
                                      'n': 2,
                                      'k': 2,
                                      'num_item_pixel_values': 1,
                                      'organization': 'full',
                                      'display': False}

    params['val_data_obj'] = psvrt_new.n_sd_k
    params['val_data_init_args'] = params['train_data_init_args'].copy()

    params['save_ckpt_as'] = '/home/jk/PSVRT_test_result'
    params['save_learningcurve_as'] = '/home/jk/PSVRT_test_result'
    params['learningcurve_type'] = 'array'
    params['save_textsummary_as'] = '/home/jk/PSVRT_test_result'
    params['tb_logs_dir'] = '/home/jk/PSVRT_test_result'

    return params
