#!/bin/bash

# make sure to add export PYTHONPATH=${PYTHONPATH}:${HOME}/kissdsp if using local installation

if [ -z "$1" ]; then
        echo "Need clean speech dataset folder."
        exit 1
fi

if [ -z "$2" ]; then
        echo "Need base folder."
        exit 1
fi

dnsdir=$1
basedir=$2

find "$dnsdir" -name "*.wav" > $basedir/speech_sorted.txt
shuf $basedir/speech_sorted.txt > $basedir/speech_shuffled.txt

speechnum=$(cat $basedir/speech_shuffled.txt | wc -l)

trainnum=$(echo "$speechnum * 0.7" | bc)
trainnum=$((speechnum * 7 / 10))
trainnum1=$((trainnum + 1))
validnum=$((trainnum + speechnum * 2 / 10))
validnum1=$((validnum + 1))
testnum=$((validnum + speechnum * 1 / 10))

sed -n "1,$trainnum p" $basedir/speech_shuffled.txt > $basedir/speech_train.txt
sed -n "$trainnum1,$validnum p" $basedir/speech_shuffled.txt > $basedir/speech_valid.txt
sed -n "$validnum1,$testnum p" $basedir/speech_shuffled.txt > $basedir/speech_test.txt

