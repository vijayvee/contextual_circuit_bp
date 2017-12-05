#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python tests/best_loc_tfrecord_plot_filters.py --target_layer=conv1_1 --target_model=conv2d
CUDA_VISIBLE_DEVICES=0 python tests/best_loc_tfrecord_plot_filters.py --target_layer=sep_conv1_1 --target_model=sep_conv2d
CUDA_VISIBLE_DEVICES=0 python tests/best_loc_tfrecord_plot_filters.py --target_layer=dog1_1 --target_model=DoG

