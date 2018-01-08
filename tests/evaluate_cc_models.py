"""Evaluate a model for the Bush-Vannevar grant."""
import os
import main
from glob import glob


experiment_name = 'contours'
exp_pointer = os.path.join(
    '%smedia' % os.path.sep,
    'data_cifs',
    'contextual_circuit',
    'checkpoints',
    'contours_2018_01_07_10_52_30')  # 33/30/25
ckpt_pointers = glob(os.path.join(
    exp_pointer,
    '*ckpt*'))
ckpt_pointers = [x for x in ckpt_pointers if 'meta' in x]
ckpt_pointer = sorted(ckpt_pointers, key=lambda file: os.path.getctime(file))[-1].split('.meta')[0]
# ckpt_pointer = '/media/data_cifs/contextual_circuit/checkpoints/contours_2018_01_07_10_52_25/model_5000.ckpt-5000'
main.main(
    experiment_name=experiment_name,
    load_and_evaluate_ckpt=ckpt_pointer)

