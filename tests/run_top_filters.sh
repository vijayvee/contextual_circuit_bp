#!/bin/bash

# CUDA_VISIBLE_DEVICES=0 python tests/get_top_filters.py --experiment=760_cells_2017_11_05_16_36_55 --target_model=conv2d --target_layer=conv1_1 
# CUDA_VISIBLE_DEVICES=0 python tests/get_top_filters.py --experiment=760_cells_2017_11_05_16_36_55 --target_model=sep_conv2d --target_layer=sep_conv1_1
CUDA_VISIBLE_DEVICES=0 python tests/get_top_filters.py --experiment=760_cells_2017_11_05_16_36_55 --target_model=DoG --target_layer=dog1_1

# CUDA_VISIBLE_DEVICES=0 python tests/get_top_filters.py --experiment=760_cells_2017_11_05_16_36_55 --target_model=conv2d --target_layer=conv1_1 --recalc
# CUDA_VISIBLE_DEVICES=0 python tests/get_top_filters.py --experiment=760_cells_2017_11_05_16_36_55 --target_model=sep_conv2d --target_layer=sep_conv1_1 --recalc
CUDA_VISIBLE_DEVICES=0 python tests/get_top_filters.py --experiment=760_cells_2017_11_05_16_36_55 --target_model=DoG --target_layer=dog1_1 --recalc

