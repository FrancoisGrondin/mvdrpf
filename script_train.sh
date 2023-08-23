#!/bin/bash
dataset="/media/fgrondin/Scratch/tmp/features/features_train.txt"
num_workers=16
batch_size=16
num_epochs=5
model="ugru_1-512"
checkpoints_dir="/media/fgrondin/Scratch/tmp/checkpoints/${model}/"

for epoch in {001..100}
do
	if [ $epoch -eq 1 ]
	then
		python3 ml.py --dataset $dataset \
					  --batch_size $batch_size \
					  --model $model \
					  --action train \
					  --num_workers $num_workers \
					  --num_epochs $num_epochs \
					  --checkpoint_save "${checkpoints_dir}${epoch}.bin" 
	else
		python3 ml.py --dataset $dataset \
					  --batch_size $batch_size \
					  --model $model \
					  --action train \
					  --num_workers $num_workers \
					  --num_epochs $num_epochs \
					  --checkpoint_load "${checkpoints_dir}${last_epoch}.bin" \
					  --checkpoint_save "${checkpoints_dir}${epoch}.bin"
	fi
	last_epoch=$epoch
done