#!/bin/bash
dataset="/media/fgrondin/Scratch/mvdrpf/features/features_valid.txt"
num_workers=16
batch_size=16
#declare -a list_models=("ugru_1-128_1ch" "ugru_1-128_2ch" "ugru_1-256_1ch" "ugru_1-256_2ch" "ugru_1-512_1ch" "ugru_1-512_2ch")
#declare -a list_models=("ugru_2-128_1ch" "ugru_2-128_2ch" "ugru_2-256_1ch" "ugru_2-256_2ch" "ugru_2-512_1ch" "ugru_2-512_2ch")
#declare -a list_models=("ulstm_1-128_1ch" "ulstm_1-128_2ch" "ulstm_1-256_1ch" "ulstm_1-256_2ch" "ulstm_1-512_1ch" "ulstm_1-512_2ch")
declare -a list_models=("ulstm_2-128_1ch" "ulstm_2-128_2ch" "ulstm_2-256_1ch" "ulstm_2-256_2ch" "ulstm_2-512_1ch" "ulstm_2-512_2ch")

for model in "${list_models[@]}"
do

	checkpoints_dir="/media/fgrondin/Scratch/mvdrpf/checkpoints/${model}/"
	results_file="/media/fgrondin/Scratch/mvdrpf/results/${model}/loss.txt"
	csv_file="/media/fgrondin/Scratch/mvdrpf/results/${model}/loss.csv"

	for epoch in {001..050}
	do

		python3 ml.py --dataset $dataset \
					  --batch_size $batch_size \
					  --model $model \
					  --action eval \
					  --num_workers $num_workers \
					  --checkpoint_load "${checkpoints_dir}${epoch}.bin" >> $results_file
	done

	cat $results_file | \
	sed --expression 's/Eval loss = //g' | \
	nl -s "," -n "ln" -w 1 | \
	sed --expression '1s/^/epoch,loss\n/' > $csv_file

done