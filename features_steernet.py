import argparse
import numpy as np
import random as rnd
import librosa as lr
import os
import sys

import kissdsp.beamformer as bf
import kissdsp.filterbank as fb
import kissdsp.io as io
import kissdsp.masking as mk
import kissdsp.mixing as mx
import kissdsp.micarray as ma
import kissdsp.reverb as rb
import kissdsp.spatial as sp
import kissdsp.visualize as vz

import matplotlib.pyplot as plt

import torch

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--steernet_basedir', default='', type=str, help='Directory of SteerNet repository.')
    parser.add_argument('--index_start', default=1, type=int, help='Index of first feature to be generated.')
    parser.add_argument('--index_stop', default=1, type=int, help='Index of last feature to be generated.')
    parser.add_argument('--output', default='', type=str, help='Output directory to store features.')
    args = parser.parse_args()

    # Make sure SteerNet base directory is provided and exists
    if args.steernet_basedir == "":
        print("SteerNet base directory is required.")
        exit()
    elif not os.path.isdir(args.steernet_basedir):
        print("SteerNet base directory does not exist: "+args.steernet_basedir)
        exit()
    
    # Add SteerNet to syspath and import it
    sys.path.append(args.steernet_basedir)
    from steernet.model.blstm import Blstm
    from steernet.dataset.array_ext import Array
    import steernet.beam as beam
    net = Blstm(file_json=args.steernet_basedir+"/python/json/features.json")
    net.load_state_dict(torch.load(args.steernet_basedir+"/trained/blstm_epoch020.bin"))
    
    dataset = Array(file_meta=args.steernet_basedir+"/python/audio_meta.txt", file_json=args.steernet_basedir+"/python/json/features.json")
    
    index_max = len(dataset)
    print(index_max)

    # Get list of files
    f = open(args.speech_files, 'r')
    paths = f.readlines()

    # Generate feature for each index
    index = args.index_start
    while index <= args.index_stop:

        Xs, Ns, Ys, YYs = dataset[index]
        
        M = beam.mask(YYs, net);
        TTs, IIs = beam.cov(Ys, M)
        ZsTarget = beam.gev(Ys, TTs, IIs)
        ZsInterf = beam.gev(Ys, IIs, TTs)

        Cs = Xs[0,0,:,:]

        XsIdeal = np.transpose(Cs)
        XsGevTarget = np.transpose(ZsTarget)
        XsGevInterf = np.transpose(ZsInterf)

        xsIdeal = np.expand_dims(lr.core.istft(XsIdeal), 0)
        xsGevTarget = np.expand_dims(lr.core.istft(XsGevTarget), 0)
        xsGevInterf = np.expand_dims(lr.core.istft(XsGevInterf), 0)
        
        # Span full range
        vol_norm = max([ np.amax(np.abs(xsGevTarget)),
                         np.amax(np.abs(xsGevInterf)),
                         np.amax(np.abs(xsIdeal)) ])
        xsGevTarget /= np.amax(np.abs(xsGevTarget))
        xsGevInterf /= np.amax(np.abs(xsGevInterf))
        xsIdeal /= np.amax(np.abs(xsIdeal))

        # Apply new volume
        volume = np.random.uniform(low=-20, high=0)
        xsGevTarget = mx.gain(xsGevTarget, np.asarray([volume]))
        xsGevInterf = mx.gain(xsGevInterf, np.asarray([volume]))
        xsIdeal = mx.gain(xsIdeal, np.asarray([volume]))

        # Concatenate to a single multi-channel audio signal
        ys = np.concatenate([ xsGevTarget, xsGevInterf, xsIdeal ], axis=0)
        
        # Save to file
        output_path = os.path.join(args.output, '%010d.wav' % index)
        io.write(ys, output_path)
        print("[%u/%u]: %s" % (index-args.index_start+1, args.index_stop-args.index_start+1, output_path))
        index += 1


if __name__ == "__main__":

    main()
