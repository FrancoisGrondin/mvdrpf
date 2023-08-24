#!/bin/bash

# make sure to add export PYTHONPATH=${PYTHONPATH}:${HOME}/kissdsp if using local installation

if [ -z "$1" ]; then
        echo "Need base folder (where the speech text files are)."
        exit 1
fi

basedir=$1

trainnum=10000
validnum=2000
testnum=1000

train_start=1
train_end=$trainnum

valid_start=$(($train_end + 1))
valid_end=$(($valid_start + $validnum - 1))

test_start=$(($valid_end + 1))
test_end=$(($test_start + $testnum - 1))

#TODO: this still needs to be parallelized
echo "Creating training features..."
python3 features.py --speech_files $basedir/speech_train.txt --index_start $train_start --index_stop $train_end --output $basedir/data/train
find $basedir/data/train -name "*.wav" > $basedir/features/features_train.txt

echo "Creating validation features..."
python3 features.py --speech_files $basedir/speech_valid.txt --index_start $valid_start --index_stop $valid_end --output $basedir/data/valid
find $basedir/data/valid -name "*.wav" > $basedir/features/features_valid.txt

echo "Creating training features..."
python3 features.py --speech_files $basedir/speech_test.txt --index_start $test_start --index_stop $test_end --output $basedir/data/test
find $basedir/data/test -name "*.wav" > $basedir/features/features_test.txt

