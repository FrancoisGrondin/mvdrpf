import argparse
import numpy as np

import torch
import matplotlib.pyplot as plt
import os

from dataset import SpexIrm
from model import BGru, UGru
from brain import Brain

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train', type=str, choices=['train','eval','test','traineval'], help='Action to perform.')
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
parser.add_argument('--model', default='bgru_1-128', type=str, choices=['bgru_1-128', 'ugru_1-128', 'ugru_1-512', 'ugru_2-512', 'ugru_2-1024'])
args = parser.parse_args()

# Datasets
if args.action == 'traineval':
    dset = SpexIrm(path=args.dataset_train, beta=0.25)
    dset_eval = SpexIrm(path=args.dataset_eval, beta=0.25)
else:
    dset = SpexIrm(path=args.dataset, beta=0.25)

# Model
if args.model == 'bgru_1-128':
    net = BGru(hidden_size=128, num_layers=1)
if args.model == 'ugru_1-128':
    net = UGru(hidden_size=128, num_layers=1)
if args.model == 'ugru_1-512':
    net = UGru(hidden_size=512, num_layers=1)
if args.model == 'ugru_2-512':
    net = UGru(hidden_size=512, num_layers=2)
if args.model == 'ugru_2-1024':
    net = UGru(hidden_size=1024, num_layers=2)

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
        checkpoint_num = int(os.path.basename(args.checkpoint_load).split(".bin")[0])
        print("Starting at the end of checkpoint "+str(checkpoint_num)+".")
else:
    if args.action == 'traineval':
        checkpoint_num = 0
        print("Starting from scratch.")


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
    [loss, pesq, stoi, sdr] = brn.eval(batch_size=args.batch_size,
                    shuffle=args.shuffle,
                    num_workers=args.num_workers)

    print("Eval loss = %f" % loss)
    print("     PESQ = %f" % pesq)
    print("     STOI = %f" % stoi)
    print("     SDR  = %f" % sdr)

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
    
    for epoch in range(checkpoint_num, checkpoint_total):
        # Train
        print("<<<<<<")
        print("--- Checkpoint "+str(epoch+1)+": training from epoch "+str(epoch*args.num_epochs_eval+1)+ " to " +str(epoch*args.num_epochs_eval+args.num_epochs_eval)+"...")
        loss = brn.train(batch_size=args.batch_size,
                         shuffle=args.shuffle,
                         num_workers=args.num_workers,
                         num_epochs=args.num_epochs_eval)

        print("--- Checkpoint "+str(epoch+1)+": train loss = %f" % loss)

        # Save to checkpoint if specified
        if args.checkpoint_save_dir is not None:
            this_checkpoint_save = args.checkpoint_save_dir+f'{epoch+1:03}'+".bin"
            print("--- Checkpoint "+str(epoch+1)+": saving checkpoint to "+this_checkpoint_save)
            brn.save_checkpoint(path=this_checkpoint_save)

        # Eval
        print("--- Checkpoint "+str(epoch+1)+": evaluating checkpoint...")
        [loss, pesq, stoi, sdr] = brn.eval(batch_size=args.batch_size,
                        shuffle=args.shuffle,
                        num_workers=args.num_workers)

        print("--- Checkpoint "+str(epoch+1)+": eval loss = %f, PESQ = %f, STOI = %f, SDR = %f" % (loss, pesq, stoi, sdr))
        print(">>>>>>")
