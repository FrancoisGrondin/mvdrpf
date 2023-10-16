#!/bin/bash

# make sure to add export PYTHONPATH=${PYTHONPATH}:${HOME}/kissdsp if using local installation

# to run this script, the SteerNet git repository needs to be
# cloned somewhere locally (refered here as STEERNET_GITREPODIR)

# to facilitate the automatic import of steernet
# create a symbolic link inside STEERNET_GITREPODIR that
# directs it to the local "python" folder inside the repository
# and name it "steernet":
#     cd STEERNET_GITREPODIR
#     ln -s python ./steernet

if [ -z "$1" ]; then
        echo "Need base folder (where the speech text files are)."
        exit 1
fi

if [ -z "$2" ]; then
        echo "Need SteerNet repository base folder (where the 'trained' folder is, as well as a symbolic link to the 'python' folder named 'steernet')."
        exit 1
fi

basedir=$1
steernetdir=$2

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
python3 features_steernet.py --steernet_basedir $steernetdir --speech_files $basedir/speech_train.txt --index_start $train_start --index_stop $train_end --output $basedir/data/train
find $basedir/data/train -name "*.wav" > $basedir/features/features_train.txt

echo "Creating validation features..."
python3 features_steernet.py --steernet_basedir $steernetdir --speech_files $basedir/speech_valid.txt --index_start $valid_start --index_stop $valid_end --output $basedir/data/valid
find $basedir/data/valid -name "*.wav" > $basedir/features/features_valid.txt

echo "Creating training features..."
python3 features_steernet.py --steernet_basedir $steernetdir --speech_files $basedir/speech_test.txt --index_start $test_start --index_stop $test_end --output $basedir/data/test
find $basedir/data/test -name "*.wav" > $basedir/features/features_test.txt

