#!/bin/bash
dataset="/media/fgrondin/Scratch/mvdrpf/features/features_test.txt"
model="ugru_1-512_2ch"
checkpoint_file="/media/fgrondin/Scratch/mvdrpf/checkpoints/${model}/050.bin"
output_dir="/media/fgrondin/Scratch/mvdrpf/postfiltered/${model}/"

python3 ml.py --dataset $dataset \
			  --model $model \
			  --action test \
			  --output_dir $output_dir \
			  --checkpoint_load $checkpoint_file