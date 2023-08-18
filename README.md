# Getting started

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

For each list of speech files, the spectrogram features can be generated. For instance, to generate a set of 1000 samples from index 1 to 1000 for training, the following script can be used, where `<features_train_directory>` stands for the destination directory:

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