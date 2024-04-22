# Getting started

## Package requirements

The following packages are required:

```
torch
pesq
torchmetrics[audio]
matplotlib
numpy
kissdsp
progressbar
progressbar2
mir_eval
librosa
```

[comment]: <> (pypesq)
[comment]: <> (pystoi)

## Generating features

Using the DNS Challenge data, run the following command, where `<speech_directory>` contains the wav files with speech and `<speech_sorted_file>` will hold the list of speech files:

```
find <speech_directory> -name "*.wav" > <speech_sorted_file>
```

Shuffle the files and store result in a new file:

```
shuf <speech_sorted_file> > <speech_shuffled_file>
```

Get the number of speech files:

```
cat <speech_shuffled_file> | wc -l
```

From this number, separate in 3 sets, train, eval and test, to be stored in files `<speech_train_file>`, `<speech_eval_file>` and `<speech_test_file>`. For instance, with 1,000,000 elements, the following split (700,000/200,000/100,000) is a good choice:

```
sed -n '1,700000p' <speech_shuffled_file> > <speech_train_file>
sed -n '700001,900000p' <speech_shuffled_file> > <speech_eval_file>
sed -n '900001,1000000p' <speech_shuffled_file> > <speech_test_file>
```

For each list of speech files, the target and interference MVDR beamformed time-domain signals, along with the ideal time-domain signals (when no contamination by competing sources). For instance, to generate a set of 1000 samples from index 1 to 1000 for training, the following script can be used, where `<features_train_directory>` stands for the destination directory:

```
python3 features.py --list <speech_train_file> --index_start 1 --index_stop 1000 --output <features_train_directory>
```

The same idea applies to eval and test. This is convenient as multiple threads can be started to speed up features generation. For instance, each one of the following lines can be executed in different terminal:

```
python3 features.py --speech_files <speech_train_file> --index_start 1 --index_stop 1000 --output <features_train_directory>
python3 features.py --speech_files <speech_train_file> --index_start 1001 --index_stop 2000 --output <features_train_directory>
python3 features.py --speech_files <speech_train_file> --index_start 2001 --index_stop 3000 --output <features_train_directory>
python3 features.py --speech_files <speech_train_file> --index_start 3001 --index_stop 4000 --output <features_train_directory>
python3 features.py --speech_files <speech_train_file> --index_start 4001 --index_stop 5000 --output <features_train_directory>
python3 features.py --speech_files <speech_train_file> --index_start 5001 --index_stop 6000 --output <features_train_directory>
python3 features.py --speech_files <speech_train_file> --index_start 6001 --index_stop 7000 --output <features_train_directory>
python3 features.py --speech_files <speech_train_file> --index_start 7001 --index_stop 8000 --output <features_train_directory>
python3 features.py --speech_files <speech_train_file> --index_start 8001 --index_stop 9000 --output <features_train_directory>
python3 features.py --speech_files <speech_train_file> --index_start 9001 --index_stop 10000 --output <features_train_directory>
python3 features.py --speech_files <speech_eval_file> --index_start 10001 --index_stop 11000 --output <features_eval_directory>
python3 features.py --speech_files <speech_eval_file> --index_start 11001 --index_stop 12000 --output <features_eval_directory>
python3 features.py --speech_files <speech_test_file> --index_start 12001 --index_stop 13000 --output <features_test_directory>
```

Once this is done, we simply index features for all files and save them in files `<features_train_file>`, `<features_eval_file>` and `<features_test_file>` that will be used to load the content during training, eval or testing:

```
find <features_train_directory> -name "*.wav" > <features_train_file>
find <features_eval_directory> -name "*.wav" > <features_eval_file>
find <features_test_directory> -name "*.wav" > <features_test_file>
```

## Training

The model can be train as follows (for instance with batch size of 16, using 16 workers/cores, for 10 epochs):

```
python3 ml.py --dataset <features_train_file> --batch_size 16 --action train --num_workers 16 --num_epochs 10 --checkpoint_save <checkpoint_bin_file>
```

The trained parameters are saved in a binary file denoted by `<checkpoint_bin_file>`.

## Validation

It is possible to measure the loss of a specific model based on the saved checkpoint and using the validation set:

```
python3 ml.py --dataset <features_eval_file> --batch_size 16 --action eval --num_workers 16 --checkpoint_load <checkpoint_bin_file>
```

## Testing

The model can be used to generate masks and save the results to png figures on the disk in the directory `<output_png_directory>`. Producing time-domain waveforms with and without the estimated mask will be done soon.

```
python3 ml.py --dataset <features_test_file> --action test --checkpoint_load <checkpoint_bin_file> --output_dir <output_png_directory>
```

## Using the included scripts

The base directory in which all of the files are stored (training data, feature text files, checkpoints, results, etc.) should have the following structure:

```
$(base_dir)/
  features/
    features_test.txt
    features_train.txt
    features_valid.txt
  data/
    test/
      *.wav
    train/
      *.wav
    valid/
      *.wav
  checkpoints/
    model1/
      *.bin
    model2/
      *.bin
    ...
  results/
    model1/
      *.wav
    model2/
      *.wav
    ...
  speech_test.txt
  speech_train.txt
  speech_valid.txt
```

To create this structure, run:
```
bash script_basedir.sh $(base_dir)
```

We assume that the DNS Challenge data has been cloned, such that the clean folder is in `$(dns_clean)`.

To create the text files that bare the paths to the clean speech files (`speech_test.txt`, `speech_train.txt`, and `speech_valid.txt`), run:
```
bash script_data.sh $(dns_clean) $(base_dir)
```

To create the features files, run:
```
bash script_features.sh $(base_dir)
```

You are welcome to modify the following variables in `script_features.sh`: `trainnum` is the number of data points for training, `validnum` is the number of data points for evaluation/validation, and `testnum` is the number of data points for testing.

To train a model from scratch, with an evaluation at every given number of epochs, run:
```
bash script_traineval.sh $(base_dir) $(modelname)
```

Where `$(modelname)` is the name of the model to train. Valid choices are:
```
ugru_1-128_1ch, ugru_1-128_2ch, ugru_1-256_1ch, ugru_1-256_2ch, ugru_1-512_1ch, ugru_1-512_2ch,
ugru_2-128_1ch, ugru_2-128_2ch, ugru_2-256_1ch, ugru_2-256_2ch, ugru_2-512_1ch, ugru_2-512_2ch,
ulstm_1-128_1ch, ulstm_1-128_2ch, ulstm_1-256_1ch, ulstm_1-256_2ch, ulstm_1-512_1ch, ulstm_1-512_2ch,
ulstm_2-128_1ch, ulstm_2-128_2ch, ulstm_2-256_1ch, ulstm_2-256_2ch, ulstm_2-512_1ch, ulstm_2-512_2ch
```

You are welcome to modify the following variables in `script_traineval.sh`: `num_workers` is the number of threads to use while training and evaluating, `batch_size` is the isze of data points per batch, `num_epochs` is the total number of epochs to run (an epoch being running through all of the data points in `features_train.txt`), `num_epochs_eval` is the number of epochs to run after which an evaluation is carried out and a checkpoint is saved.

To train a model from a given checkpoint, with an evaluation at every given number of epochs, run:
```
bash script_traineval.sh $(path_to_checkpoint)
```

The path to the base directory is assumed from the checkpoint path, assuming the base directory structure is respected.

## Using SteerNet to estimate the covariance matrices

To use SteerNet, the whole process is very similar to that explained until now, the only difference is that the features are based on the SteerNet estimations instead of on the ideal estimations used by `script_features.sh`.

First, it is required to clone the SteerNet github repository locally in `$(steernet_dir)`:
```
cd <parent directory to $(steernet_dir)>
git clone https://github.com/FrancoisGrondin/steernet.git
```

The code that creates the SteerNet-based features imports the SteerNet python library automatically, depending of where it is located. To facilitate this, once the SteerNet github repository is cloned, create a symbolic link named `steernet` to point to the `python` folder inside it:
```
cd $(steernet_dir)
ln -s python ./steernet
```

Then, carry out the data creation process as established in the [SteerNet repository](https://github.com/FrancoisGrondin/steernet), using the included scripts in that repository. In summary:
1. Create the room impulse responses as explained in the README file located in `steernet/octave/rir` directory.
1. Download the librispeech corpus.
1. Prepare data for training as explained in the README file located in `steernet/python` directory.

You do not need to re-train the SteerNet; we will use the one located in `steernet/trained/blstm_epoch020.bin`.

Once the SteerNet data creation process is done, to create the SteerNet-based features used to train the MVDRpf model use `script_features_steernet.sh`:
```
bash script_features_steernet.sh $(base_dir) $(steernet_dir)
```

Similarly, you are also welcome to modify the `trainnum`, `validnum`, and `testnum` variables in `script_features_steernet.sh`.

The rest of the process follows the same as explained earlier:
```
bash script_traineval.sh $(base_dir)
```

or 
```
bash script_traineval.sh $(path_to_checkpoint)
```

