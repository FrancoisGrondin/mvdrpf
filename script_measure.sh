#!/bin/bash
dataset="/media/fgrondin/Scratch/mvdrpf/features/features_test.txt"
#declare -a list_models=("ugru_1-128_1ch" "ugru_1-128_2ch" "ugru_1-256_1ch" "ugru_1-256_2ch" "ugru_1-512_1ch" "ugru_1-512_2ch")
#declare -a list_models=("ugru_2-128_1ch" "ugru_2-128_2ch" "ugru_2-256_1ch" "ugru_2-256_2ch" "ugru_2-512_1ch" "ugru_2-512_2ch")
#declare -a list_models=("ulstm_1-128_1ch" "ulstm_1-128_2ch" "ulstm_1-256_1ch" "ulstm_1-256_2ch" "ulstm_1-512_1ch" "ulstm_1-512_2ch")
declare -a list_models=("ulstm_2-128_1ch" "ulstm_2-128_2ch" "ulstm_2-256_1ch" "ulstm_2-256_2ch" "ulstm_2-512_1ch" "ulstm_2-512_2ch")

for model in "${list_models[@]}"
do

	checkpoints_dir="/media/fgrondin/Scratch/mvdrpf/checkpoints/${model}/"
	results_file="/media/fgrondin/Scratch/mvdrpf/results/${model}/metrics.txt"

	python3 ml.py --dataset $dataset \
				  --model $model \
				  --action measure \
				  --checkpoint_load "${checkpoints_dir}050.bin" > $results_file

done