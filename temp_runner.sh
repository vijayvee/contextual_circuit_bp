#!/usr/bin/env

START=1
END=30
for (( i=$START; i <= $END; ++i ))
do
    CUDA_VISIBLE_DEVICES=3 python main.py --experiment=one_layer_conv_mlp
done

