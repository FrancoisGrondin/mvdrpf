#!/bin/bash

# make sure to add export PYTHONPATH=${PYTHONPATH}:${HOME}/kissdsp if using local installation

if [ -z "$1" ]; then
    echo "Need base folder (where the speech text files are) or checkpoint file."
    exit 1
fi

if [ -z "$2" ]; then
    echo "Need model name."
    exit 1
fi

if [ -z "$3" ]; then
    echo "Need checkpoint number."
    exit 1
fi

basedir="$1"
model="$2"

if [[ -d "$1" ]]; then
    output_dir="$basedir/postfiltered/$model/"
    
    if [[ ! -d "$output_dir" ]]; then
        mkdir -p "$output_dir"
    fi
else
    echo "Base folder not found."
    exit 1
fi

bin_zeropad=$(printf "%03d\n" "$3")
checkpoint_file="$basedir/checkpoints/${model}/$bin_zeropad.bin"
if [[ ! -f "$checkpoint_file" ]]; then
    echo "Invalid checkpoint bin file: $checkpoint_file"
    exit 1
fi

dataset="$basedir/features/features_test.txt"
if [[ ! -f "$dataset" ]]; then
    echo "Invalid test features text file: $dataset"
    exit 1
fi

python3 ml.py --dataset $dataset \
			  --model $model \
			  --action test \
			  --output_dir $output_dir \
			  --checkpoint_load $checkpoint_file
