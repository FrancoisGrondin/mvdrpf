#!/bin/bash
dataset="/media/fgrondin/Scratch/tmp/features/features_test.txt"
model="ugru_1-512"
checkpoint_file="/media/fgrondin/Scratch/tmp/checkpoints/${model}/010.bin"
output_dir="/media/fgrondin/Scratch/tmp/results/${model}/"

python3 ml.py --dataset $dataset \
			  --model $model \
			  --action test \
			  --output_dir $output_dir \
			  --checkpoint_load $checkpoint_file