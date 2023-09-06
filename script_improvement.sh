#!/bin/bash
dataset="/media/fgrondin/Scratch/mvdrpf/features/features_test.txt"
num_workers=16
batch_size=16

#declare -a list_models=("ugru_1-128_1ch")
#declare -a list_models=("ugru_1-128_2ch" "ugru_1-256_1ch" "ugru_1-256_2ch" "ugru_1-512_1ch" "ugru_1-512_2ch")
#declare -a list_models=("ugru_2-128_1ch" "ugru_2-128_2ch" "ugru_2-256_1ch" "ugru_2-256_2ch" "ugru_2-512_1ch" "ugru_2-512_2ch")

for model in "${list_models[@]}"
do

	checkpoints_dir="/media/fgrondin/Scratch/mvdrpf/checkpoints/${model}/"
	results_file="/media/fgrondin/Scratch/mvdrpf/results/${model}/improvements.txt"
	csv_file="/media/fgrondin/Scratch/mvdrpf/results/${model}/improvements.csv"

	python3 ml.py --dataset $dataset \
				  --model $model \
				  --action improvement \
				  --checkpoint_load "${checkpoints_dir}050.bin" >> $results_file

	cat $results_file | \
	sed --expression 's/PESQ_start = //g' | \
	sed --expression 's/PESQ_end = //g' | \
	sed --expression 's/STOI_start = //g' | \
	sed --expression 's/STOI_end = //g' | \
	sed --expression 's/SDR_start = //g' | \
	sed --expression 's/SDR_end = //g' | \
	sed --expression '1s/^/pesq_start,pesq_end,stoi_start,stoi_end,sdr_start,sdr_end\n/' > $csv_file

done