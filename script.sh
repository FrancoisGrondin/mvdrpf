#!/bin/bash
dataset="/media/fgrondin/Scratch/tmp/features/features_train.txt"
checkpoints_dir="/media/fgrondin/Scratch/tmp/checkpoints/"
num_workers=16
num_epochs=10
batch_size=16

python3 ml.py --dataset $dataset --batch_size $batch_size --action train --num_workers $num_workers --num_epochs $num_epochs --checkpoint_save "${checkpoints_dir}chkp010.bin"
python3 ml.py --dataset $dataset --batch_size $batch_size --action train --checkpoint_load "${checkpoints_dir}chkp010.bin" --num_workers $num_workers --num_epochs $num_epochs --checkpoint_save "${checkpoints_dir}chkp020.bin"
python3 ml.py --dataset $dataset --batch_size $batch_size --action train --checkpoint_load "${checkpoints_dir}chkp020.bin" --num_workers $num_workers --num_epochs $num_epochs --checkpoint_save "${checkpoints_dir}chkp030.bin"
python3 ml.py --dataset $dataset --batch_size $batch_size --action train --checkpoint_load "${checkpoints_dir}chkp030.bin" --num_workers $num_workers --num_epochs $num_epochs --checkpoint_save "${checkpoints_dir}chkp040.bin"
python3 ml.py --dataset $dataset --batch_size $batch_size --action train --checkpoint_load "${checkpoints_dir}chkp040.bin" --num_workers $num_workers --num_epochs $num_epochs --checkpoint_save "${checkpoints_dir}chkp050.bin"
python3 ml.py --dataset $dataset --batch_size $batch_size --action train --checkpoint_load "${checkpoints_dir}chkp050.bin" --num_workers $num_workers --num_epochs $num_epochs --checkpoint_save "${checkpoints_dir}chkp060.bin"
python3 ml.py --dataset $dataset --batch_size $batch_size --action train --checkpoint_load "${checkpoints_dir}chkp060.bin" --num_workers $num_workers --num_epochs $num_epochs --checkpoint_save "${checkpoints_dir}chkp070.bin"
python3 ml.py --dataset $dataset --batch_size $batch_size --action train --checkpoint_load "${checkpoints_dir}chkp070.bin" --num_workers $num_workers --num_epochs $num_epochs --checkpoint_save "${checkpoints_dir}chkp080.bin"
python3 ml.py --dataset $dataset --batch_size $batch_size --action train --checkpoint_load "${checkpoints_dir}chkp080.bin" --num_workers $num_workers --num_epochs $num_epochs --checkpoint_save "${checkpoints_dir}chkp090.bin"
python3 ml.py --dataset $dataset --batch_size $batch_size --action train --checkpoint_load "${checkpoints_dir}chkp090.bin" --num_workers $num_workers --num_epochs $num_epochs --checkpoint_save "${checkpoints_dir}chkp100.bin"