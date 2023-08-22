import argparse
import numpy as np

import torch
import matplotlib.pyplot as plt

from dataset import SpexIrm
from model import BGru, UGru
from brain import Brain

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train', type=str, choices=['train','eval','test'], help='Action to perform.')
parser.add_argument('--dataset', default=None, type=str, help='Text file with list of all wave files.')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for training or validation.')
parser.add_argument('--shuffle', default=True, type=bool, help='Shuffle training samples.')
parser.add_argument('--num_workers', default=0, type=int, help='Number of workers.')
parser.add_argument('--num_epochs', default=1, type=int, help='Number of epochs.')
parser.add_argument('--output_dir', default=None, type=str, help='Output directory to generate test results.')
parser.add_argument('--checkpoint_load', default=None, type=str, help='File with checkpoint to load from.')
parser.add_argument('--checkpoint_save', default=None, type=str, help='File with checkpoint to save to.')
parser.add_argument('--model', default='bgru_1-128', type=str, choices=['bgru_1-128', 'ugru_1-128', 'ugru_1-512', 'ugru_2-512'])
args = parser.parse_args()

# Datasets
dset = SpexIrm(path=args.dataset)

# Model
if args.model == 'bgru_1-128':
    net = BGru(hidden_size=128, num_layers=1)
if args.model == 'ugru_1-128':
    net = UGru(hidden_size=128, num_layers=1)
if args.model == 'ugru_1-512':
    net = UGru(hidden_size=512, num_layers=1)
if args.model == 'ugru_2-512':
    net = UGru(hidden_size=512, num_layers=2)

# Brain
brn = Brain(net=net, 
            dset=dset)

# Load from checkpoint if specified
if args.checkpoint_load is not None:
    brn.load_checkpoint(path=args.checkpoint_load)

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

# Test
if args.action == 'test':

    # Test
    brn.test(directory=args.output_dir)