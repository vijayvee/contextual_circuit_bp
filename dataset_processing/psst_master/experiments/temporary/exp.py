import os
import sys
import time
import warnings

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

sys.path.append(os.path.abspath(os.path.join('..','..')))
from helpers import train_helpers
from experiments.organization_test import params

warnings.filterwarnings('ignore')
tf.logging.set_verbosity(tf.logging.ERROR)


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


if __name__ == '__main__':

    t = time.time()

    num_machines = int(sys.argv[1])
    i_machine = int(sys.argv[2])
    n_gpus = int(sys.argv[3])

    params = params.get_params()
    model_gpu_addresses = range(n_gpus)
    gradient_gpu_address = n_gpus

    box_extent_list = [[180,180]]

    item_size_list = [[5,5]]

    n_list = [2]
    k_list = [2]
    organization = 'raw'
    num_repeats = 1

    summary_dir = '/home/jk/PSVRT_test_result/'+'summary'
    ckpt_dir = '/home/jk/PSVRT_test_result/'+'ckpts'

    print(get_available_gpus())
    kth_job = 0
    
    for be, box_extent in enumerate(box_extent_list):
        for ps, item_size in enumerate(item_size_list):
            for ni, n in enumerate(n_list):
                for ki, k in enumerate(k_list):
                    for rep in range(num_repeats):
                        params['train_data_init_args']['item_size'] = item_size
                        params['train_data_init_args']['box_extent'] = box_extent
                        params['train_data_init_args']['n'] = n
                        params['train_data_init_args']['k'] = k
                        params['train_data_init_args']['organization'] = organization
                        if (organization == 'full') | (organization == 'obj'):
                            params['raw_input_size'][2] = n*k

                        params['val_data_init_args'] = params['train_data_init_args'].copy()

                        params['save_learningcurve_as'] = os.path.join(summary_dir,
                                                                       str(box_extent),
                                                                       str(n)+'x'+str(k),
                                                                       str(item_size),
                                                                       'lc'+str(rep)+'.npy')
                        params['save_textsummary_as'] = os.path.join(summary_dir,
                                                                     str(box_extent),
                                                                     str(n) + 'x' + str(k),
                                                                     str(item_size),
                                                                     'summary'+str(rep)+'.txt')
                        params['save_ckpt_as'] = os.path.join(ckpt_dir,
                                                              str(box_extent),
                                                              str(box_extent),
                                                              str(n) + 'x' + str(k),
                                                              str(item_size),
                                                              str(rep)+'checkpoint')
                        graph = tf.Graph()
                        with graph.as_default():
                            with tf.Session(graph=graph,
                                            config=tf.ConfigProto(allow_soft_placement=True,
                                                                  log_device_placement=True)) as session:
                                kth_job += 1
                                if np.mod(kth_job, num_machines) != i_machine and num_machines != i_machine:
                                    continue
                                elif np.mod(kth_job, num_machines) != 0 and num_machines == i_machine:
                                    continue

                                repeat = 'rep' + str(rep+1)
                                _, _, _, _, _, _, imgs_to_acquisition = train_helpers.fftrain(session=session,
                                                                                              model_gpu_addresses=model_gpu_addresses,
                                                                                              gradient_gpu_address=gradient_gpu_address,
                                                                                              **params)

    elapsed = time.time() - t

    print(imgs_to_acquisition)
    print('ELAPSED TIME : ', str(elapsed))
