#!/bin/bash
dataset="/media/fgrondin/Scratch/mvdrpf/features/features_train.txt"
num_workers=16
batch_size=16
#declare -a list_models=("ugru_1-128_1ch" "ugru_1-128_2ch" "ugru_1-256_1ch" "ugru_1-256_2ch" "ugru_1-512_1ch" "ugru_1-512_2ch")
#declare -a list_models=("ugru_2-128_1ch" "ugru_2-128_2ch" "ugru_2-256_1ch" "ugru_2-256_2ch" "ugru_2-512_1ch" "ugru_2-512_2ch")
#declare -a list_models=("ulstm_1-128_1ch" "ulstm_1-128_2ch" "ulstm_1-256_1ch" "ulstm_1-256_2ch" "ulstm_1-512_1ch" "ulstm_1-512_2ch")
declare -a list_models=("ulstm_2-128_1ch" "ulstm_2-128_2ch" "ulstm_2-256_1ch" "ulstm_2-256_2ch" "ulstm_2-512_1ch" "ulstm_2-512_2ch")

for model in "${list_models[@]}"
do

	checkpoints_dir="/media/fgrondin/Scratch/mvdrpf/checkpoints/${model}/"

	for epoch in {001..050}
	do
		if [ $epoch -eq 1 ]
		then
			python3 ml.py --dataset $dataset \
						  --batch_size $batch_size \
						  --model $model \
						  --action train \
						  --num_workers $num_workers \
						  --num_epochs 1 \
						  --checkpoint_save "${checkpoints_dir}${epoch}.bin" 
		else
			python3 ml.py --dataset $dataset \
						  --batch_size $batch_size \
						  --model $model \
						  --action train \
						  --num_workers $num_workers \
						  --num_epochs 1 \
						  --checkpoint_load "${checkpoints_dir}${last_epoch}.bin" \
						  --checkpoint_save "${checkpoints_dir}${epoch}.bin"
		fi
		last_epoch=$epoch
	done

done