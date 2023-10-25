#!/bin/bash

# make sure to add export PYTHONPATH=${PYTHONPATH}:${HOME}/kissdsp if using local installation

if [ -z "$1" ]; then
    echo "Need base folder (where the speech text files are) or checkpoint file."
    exit 1
fi

if [[ -d "$1" ]]; then
    basedir=$1
    checkpoint=""
    if [ -z "$2" ]; then
        echo "Need model name. Possible model names:"
        echo "     ugru_1-128_1ch, ugru_1-128_2ch, ugru_1-256_1ch, ugru_1-256_2ch, ugru_1-512_1ch, ugru_1-512_2ch,"
        echo "     ugru_2-128_1ch, ugru_2-128_2ch, ugru_2-256_1ch, ugru_2-256_2ch, ugru_2-512_1ch, ugru_2-512_2ch," 
        echo "     ulstm_1-128_1ch, ulstm_1-128_2ch, ulstm_1-256_1ch, ulstm_1-256_2ch, ulstm_1-512_1ch, ulstm_1-512_2ch,"
        echo "     ulstm_2-128_1ch, ulstm_2-128_2ch, ulstm_2-256_1ch, ulstm_2-256_2ch, ulstm_2-512_1ch, ulstm_2-512_2ch"
        exit 1
    else
        model="$2"
    fi
elif [[ -f "$1" ]]; then
    checkpoint=$1
    filename=$(basename -- "$checkpoint")
    extension="${filename##*.}"
    
    if [ "$extension" != "bin" ]; then
        echo "Invalid checkpoint file: extension '.bin' required."
        exit 1
    fi
    basedir=$(dirname $1) #model folder
    model=$(basename $basedir) #getting model name from path
    basedir=$(dirname $basedir) #checkpoints folder
    basedir=$(dirname $basedir) #base folder
    
    echo "Assuming that base folder is located at $basedir"
else
    echo "Base folder nor checkpoint file found."
    exit 1
fi

echo "Training model: $model"

num_workers=16 #for testing: 4
batch_size=16
num_epochs=100 #for testing: 9
num_epochs_eval=5 #for testing: 3

dataset_train="$basedir/features/features_train.txt"
dataset_eval="$basedir/features/features_valid.txt"
checkpoints_dir="$basedir/checkpoints/${model}/"

if [ ! -d "$checkpoints_dir" ]; then
  mkdir -p "$checkpoints_dir"
fi


if [ -z "$checkpoint" ]; then
    python3 ml.py --dataset_train $dataset_train \
            --dataset_eval $dataset_eval \
            --batch_size $batch_size \
            --model $model \
            --action traineval \
            --num_workers $num_workers \
            --num_epochs $num_epochs \
            --num_epochs_eval $num_epochs_eval \
            --checkpoint_save_dir $checkpoints_dir
else
    python3 ml.py --dataset_train $dataset_train \
            --dataset_eval $dataset_eval \
            --batch_size $batch_size \
            --model $model \
            --action traineval \
            --num_workers $num_workers \
            --num_epochs $num_epochs \
            --num_epochs_eval $num_epochs_eval \
            --checkpoint_load $checkpoint \
            --checkpoint_save_dir $checkpoints_dir
fi


