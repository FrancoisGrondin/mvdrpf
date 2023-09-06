import argparse
import numpy as np

import torch
import matplotlib.pyplot as plt
import os

from dataset import SpexIrm
from model import URnn
from brain import Brain

def get_metrics_path(checkpoint_save_dir, model):
    if checkpoint_save_dir is None:
        metrics_csv_path = model+".csv"
    else:
        results_dir = os.path.dirname(os.path.dirname(os.path.dirname(checkpoint_save_dir)))
        metrics_csv_path = results_dir+"/results/"+model+".csv"
    print("Storing metrics in: "+metrics_csv_path)
    return metrics_csv_path

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train', type=str, choices=['train','eval','test','traineval','measure','improvement'], help='Action to perform.')
parser.add_argument('--dataset', default=None, type=str, help='Text file with list of all wave files. To be used with all actions except traineval.')
parser.add_argument('--dataset_train', default=None, type=str, help='Text file with list of all training wave files. To be used only with action traineval.')
parser.add_argument('--dataset_eval', default=None, type=str, help='Text file with list of all evaluation (validation) wave files. To be used only with action traineval.')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for training or validation.')
parser.add_argument('--shuffle', default=True, type=bool, help='Shuffle training samples.')
parser.add_argument('--num_workers', default=0, type=int, help='Number of workers.')
parser.add_argument('--num_epochs', default=1, type=int, help='Number of total epochs.')
parser.add_argument('--num_epochs_eval', default=1, type=int, help='Number of training epochs to do before doing an evaluation. To be used only with action traineval.')
parser.add_argument('--output_dir', default=None, type=str, help='Output directory to generate test results.')
parser.add_argument('--checkpoint_load', default=None, type=str, help='File with checkpoint to load from.')
parser.add_argument('--checkpoint_save', default=None, type=str, help='File with checkpoint to save to.')
parser.add_argument('--checkpoint_save_dir', default=None, type=str, help='Directory to save checkpoints to. To be used only with action traineval.')
parser.add_argument('--model', default='ugru_1-128_1ch', type=str, choices=['ugru_1-128_1ch', 'ugru_1-128_2ch', 'ugru_1-256_1ch', 'ugru_1-256_2ch', 'ugru_1-512_1ch', 'ugru_1-512_2ch', \
                                                                            'ugru_2-128_1ch', 'ugru_2-128_2ch', 'ugru_2-256_1ch', 'ugru_2-256_2ch', 'ugru_2-512_1ch', 'ugru_2-512_2ch', \
                                                                            'ulstm_1-128_1ch', 'ulstm_1-128_2ch', 'ulstm_1-256_1ch', 'ulstm_1-256_2ch', 'ulstm_1-512_1ch', 'ulstm_1-512_2ch', \
                                                                            'ulstm_2-128_1ch', 'ulstm_2-128_2ch', 'ulstm_2-256_1ch', 'ulstm_2-256_2ch', 'ulstm_2-512_1ch', 'ulstm_2-512_2ch' ])
args = parser.parse_args()

if args.model == 'ugru_1-128_1ch' or args.model == 'ugru_1-256_1ch' or args.model == 'ugru_1-512_1ch' or \
   args.model == 'ugru_2-128_1ch' or args.model == 'ugru_2-256_1ch' or args.model == 'ugru_2-512_1ch' or \
   args.model == 'ulstm_1-128_1ch' or args.model == 'ulstm_1-256_1ch' or args.model == 'ulstm_1-512_1ch' or \
   args.model == 'ulstm_2-128_1ch' or args.model == 'ulstm_2-256_1ch' or args.model == 'ulstm_2-512_1ch':
   channel_count = 1

if args.model == 'ugru_1-128_2ch' or args.model == 'ugru_1-256_2ch' or args.model == 'ugru_1-512_2ch' or \
   args.model == 'ugru_2-128_2ch' or args.model == 'ugru_2-256_2ch' or args.model == 'ugru_2-512_2ch' or \
   args.model == 'ulstm_1-128_2ch' or args.model == 'ulstm_1-256_2ch' or args.model == 'ulstm_1-512_2ch' or \
   args.model == 'ulstm_2-128_2ch' or args.model == 'ulstm_2-256_2ch' or args.model == 'ulstm_2-512_2ch':
   channel_count = 2

# Datasets
if args.action == 'traineval':
    dset = SpexIrm(path=args.dataset_train, beta=0.25, channel_count=channel_count)
    dset_eval = SpexIrm(path=args.dataset_eval, beta=0.25, channel_count=channel_count)
else:
    dset = SpexIrm(path=args.dataset, beta=0.25, channel_count=channel_count)

# Model
if args.model == 'ugru_1-128_1ch' or args.model == 'ugru_1-128_2ch':
    net = URnn(hidden_size=128, num_layers=1, type='gru')
if args.model == 'ugru_1-256_1ch' or args.model == 'ugru_1-256_2ch':
    net = URnn(hidden_size=256, num_layers=1, type='gru')
if args.model == 'ugru_1-512_1ch' or args.model == 'ugru_1-512_2ch':
    net = URnn(hidden_size=512, num_layers=1, type='gru')
if args.model == 'ugru_2-128_1ch' or args.model == 'ugru_2-128_2ch':
    net = URnn(hidden_size=128, num_layers=2, type='gru')
if args.model == 'ugru_2-256_1ch' or args.model == 'ugru_2-256_2ch':
    net = URnn(hidden_size=256, num_layers=2, type='gru')
if args.model == 'ugru_2-512_1ch' or args.model == 'ugru_2-512_2ch':
    net = URnn(hidden_size=512, num_layers=2, type='gru')
if args.model == 'ulstm_1-128_1ch' or args.model == 'ulstm_1-128_2ch':
    net = URnn(hidden_size=128, num_layers=1, type='lstm')
if args.model == 'ulstm_1-256_1ch' or args.model == 'ulstm_1-256_2ch':
    net = URnn(hidden_size=256, num_layers=1, type='lstm')
if args.model == 'ulstm_1-512_1ch' or args.model == 'ulstm_1-512_2ch':
    net = URnn(hidden_size=512, num_layers=1, type='lstm')
if args.model == 'ulstm_2-128_1ch' or args.model == 'ulstm_2-128_2ch':
    net = URnn(hidden_size=128, num_layers=2, type='lstm')
if args.model == 'ulstm_2-256_1ch' or args.model == 'ulstm_2-256_2ch':
    net = URnn(hidden_size=256, num_layers=2, type='lstm')
if args.model == 'ulstm_2-512_1ch' or args.model == 'ulstm_2-512_2ch':
    net = URnn(hidden_size=512, num_layers=2, type='lstm')


# Brain
if args.action == 'traineval':
    brn = Brain(net=net, 
                dset=dset,
                dset_eval=dset_eval)
else:
    brn = Brain(net=net, 
                dset=dset)

# Load from checkpoint if specified
if args.checkpoint_load is not None:
    brn.load_checkpoint(path=args.checkpoint_load)
    if args.action == 'traineval':
        metrics_csv_path = get_metrics_path(args.checkpoint_save_dir, args.model)

        checkpoint_num = int(os.path.basename(args.checkpoint_load).split(".bin")[0])
        print("Starting at the end of checkpoint "+str(checkpoint_num)+".")

        # remove all metrics in CSV file after checkpoint
        metrics_csv_file = open(metrics_csv_path,"r")
        metrics_csv_lines = metrics_csv_file.readlines()
        metrics_csv_file.close()
        
        for i in range(1,len(metrics_csv_lines)): #starting from index 1 to skip the CSV header
            this_checkpoint = int(metrics_csv_lines[i].split(",")[0])
            if this_checkpoint >= checkpoint_num:
                break
        i += 1
        
        if i < len(metrics_csv_lines):
            del metrics_csv_lines[i:]
            metrics_csv_file = open(metrics_csv_path,"w")
            metrics_csv_file.writelines(metrics_csv_lines)
            metrics_csv_file.close()
else:
    if args.action == 'traineval':
        metrics_csv_path = get_metrics_path(args.checkpoint_save_dir, args.model)

        checkpoint_num = 0
        print("Starting from scratch.")

        # create new CSV file with corresponding header
        metrics_csv_file = open(metrics_csv_path,"w")
        metrics_csv_file.write("checkpoint,epoch_start,epoch_end,loss_train,loss_eval,PESQ,STOI,SDR\n")
        metrics_csv_file.close()

# Training
if args.action == 'train':

    # Train
    loss = brn.train(batch_size=args.batch_size,
                     shuffle=args.shuffle,
                     num_workers=args.num_workers,
                     num_epochs=args.num_epochs)

    print("Train loss = %f" % loss)

    # Save to checkpoint if specified
    if args.checkpoint_save is not None:
        brn.save_checkpoint(path=args.checkpoint_save)

# Eval
if args.action == 'eval':

    # Eval
    loss = brn.eval(batch_size=args.batch_size,
                    shuffle=args.shuffle,
                    num_workers=args.num_workers)

    print("Eval loss = %f" % loss)

# Measure
if args.action == 'measure':

    # Measure performances
    [pesq_pf, pesq_no, stoi_pf, stoi_no, sdr_pf, sdr_no] = brn.measure(shuffle=args.shuffle)

    print("PESQ = (%f,%f), STOI = (%f,%f), SDR = (%f,%f)" % (pesq_no, pesq_pf, stoi_no, stoi_pf, sdr_no, sdr_pf))

# Improvement
if args.action == 'improvement':

    # Measure improvements for each sample
    [pesqs, stois, sdrs] = brn.improvement(shuffle=args.shuffle)

    for index in range(0, pesqs.shape[0]):
        print("PESQ_start = %f, PESQ_end = %f, STOI_start = %f, STOI_end = %f, SDR_start = %f, SDR_end = %f" % \
              (pesqs[index,0], pesqs[index,1], stois[index,0], stois[index,1], sdrs[index,0], sdrs[index,1]))

# Test
if args.action == 'test':

    # Test
    brn.test(directory=args.output_dir)

# Train-eval
if args.action == 'traineval':
    # This is where I would add a call to the brain 
    # class to a method that will alternate between training
    # and validation, and check how the loss changes.
    # We can probably add the option to use tensorboard here
    # to monitor the training, or simply write to a file
    # as it goes the loss function after each epoch. I would
    # save the model to a checkpoint after each epoch in this case.
    
    total_epochs = 0
    
    checkpoint_total = int(args.num_epochs/args.num_epochs_eval)
    
    for chckpoint in range(checkpoint_num, checkpoint_total):
        # Train
        print("<<<<<<")
        epoch_start = chckpoint*args.num_epochs_eval+1
        epoch_end = chckpoint*args.num_epochs_eval+args.num_epochs_eval
        print("--- Checkpoint "+str(chckpoint+1)+": training from epoch "+str(epoch_start)+ " to " +str(epoch_end)+"...")
        loss_train = brn.train(batch_size=args.batch_size,
                         shuffle=args.shuffle,
                         num_workers=args.num_workers,
                         num_epochs=args.num_epochs_eval)

        print("--- Checkpoint "+str(chckpoint+1)+": train loss = %f" % loss_train)

        # Save to checkpoint if specified
        if args.checkpoint_save_dir is not None:
            this_checkpoint_save = args.checkpoint_save_dir+f'{chckpoint+1:03}'+".bin"
            print("--- Checkpoint "+str(chckpoint+1)+": saving checkpoint to "+this_checkpoint_save)
            brn.save_checkpoint(path=this_checkpoint_save)

        # Eval
        print("--- Checkpoint "+str(chckpoint+1)+": evaluating checkpoint...")
        [loss_eval, pesq, stoi, sdr] = brn.eval(batch_size=args.batch_size,
                        shuffle=args.shuffle,
                        num_workers=args.num_workers)

        print("--- Checkpoint "+str(chckpoint+1)+": eval loss = %f, PESQ = %f, STOI = %f, SDR = %f" % (loss_eval, pesq, stoi, sdr))
        metrics_csv_file = open(metrics_csv_path,"a")
        metrics_csv_file.write(str(chckpoint+1)+","+str(epoch_start)+","+str(epoch_end)+","+str(loss_train)+","+str(loss_eval)+","+str(pesq.item())+","+str(stoi.item())+","+str(sdr)+"\n")
        metrics_csv_file.close()
        
        print(">>>>>>")
