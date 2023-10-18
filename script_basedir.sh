#!/bin/bash

# this scripts creates the structure necessary to create the training corpus,
# as well as to train, evaluate and store results

if [ -z "$1" ]; then
        echo "Need base folder (where the speech text files are)."
        exit 1
fi

basedir=$1

if [ ! -d "$basedir"/data ]; then
  mkdir "$basedir"/data
  mkdir "$basedir"/data/test
  mkdir "$basedir"/data/train
  mkdir "$basedir"/data/valid
else
  if [ ! -d "$basedir"/data/test ]; then
    mkdir "$basedir"/data/test
  fi
  if [ ! -d "$basedir"/data/train ]; then
    mkdir "$basedir"/data/train
  fi
  if [ ! -d "$basedir"/data/valid ]; then
    mkdir "$basedir"/data/valid
  fi
fi

if [ ! -d "$basedir"/checkpoints ]; then
  mkdir "$basedir"/checkpoints
fi

if [ ! -d "$basedir"/features ]; then
  mkdir "$basedir"/features
fi

if [ ! -d "$basedir"/results ]; then
  mkdir "$basedir"/results
fi

