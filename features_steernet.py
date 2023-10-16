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
    parser.add_argument('--speech_files', default='', type=str, help='File with list of wave files for speech.')
    parser.add_argument('--wave_duration', default=30, type=float, help='Wave file duration in second.')
    parser.add_argument('--silence_duration', default=0.5, type=float, help='Silence duration in second.')
    parser.add_argument('--sample_rate', default=16000, type=int, help='Sample rate (in samples/sec).')
    parser.add_argument('--number_sources', default=2, type=int, help='Number of speech sources.')
    parser.add_argument('--number_trials', default=10, type=int, help='Number of trials to fit speech signals in long sequence.')
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
    import steernet.beam as beam
    net = Blstm(file_json=args.steernet_basedir+"/python/json/features.json")
    net.load_state_dict(torch.load(args.steernet_basedir+"/trained/blstm_epoch020.bin"))

    # Get list of files
    f = open(args.speech_files, 'r')
    paths = f.readlines()

    # Generate feature for each index
    for index in range(args.index_start, args.index_stop+1):

        # Create speech sources
        cs = []
        for source_index in range(0, args.number_sources):
            cs.append(concatenate_waves(files=paths, 
                                        wave_duration=args.wave_duration,
                                        silence_duration=args.silence_duration,
                                        sample_rate=args.sample_rate,
                                        number_trials=args.number_trials))

        # Generate room            
        room = configure_room(size=rnd.choice(['small', 'medium', 'large']), 
                              micarray=rnd.choice(['respeaker_usb', 'respeaker_core', 'matrix_creator', 'matrix_voice', 'minidsp_uma']), 
                              number_sources=args.number_sources, 
                              rotate=True)

        # Create RIRs
        hs = rb.rir(room, sample_rate=args.sample_rate)

        # Split RIRs in target and interf
        hs_target = hs[[0], :, :]
        hs_interf = hs[1:, :, :]

        # Create target and interf sources
        snrs = np.random.uniform(low=-10, high=+10, size=(args.number_sources,))
        ss = np.zeros((len(cs), cs[0].shape[1]), dtype=np.float32)
        for source_index in range(0, args.number_sources):
            ss[source_index, :] = mx.pwr(cs[source_index], snrs[[source_index]])

        # Apply reverb
        xs_target = rb.conv(hs_target, ss[[0],:])
        xs_target += np.random.normal(size=xs_target.shape) * 1e-4
        xs_interf = rb.conv(hs_interf, ss[1:,:])
        xs_interf += np.random.normal(size=xs_interf.shape) * 1e-4
        xs = xs_target + xs_interf

        # STFT of microphones and ideal target
        Xs_target = fb.stft(xs_target)
        Xs = fb.stft(xs)

        # Create microphone pairs, required by steernet, steered at target
        tdoas = rb.tdoa(room)
        Ps_target, PPs_target = create_pairs(xs,tdoas,0)

        # Apply SteerNet to get covariance matrices from target and interference
        M_target = beam.mask(PPs_target, net);
        TTs_target, IIs_target = beam.cov(Ps_target, M_target)

        # Compute mvdr weights from SteerNet covariance matrices
        ws_target = bf.mvdr(TTs_target, IIs_target)
        ws_interf = bf.mvdr(IIs_target, TTs_target)

        # Perform beamforming
        Ys_target = bf.beam(Xs, ws_target)
        Ys_interf = bf.beam(Xs, ws_interf)
        Ys_ideal = bf.beam(Xs_target, ws_target)

        # Go back to time-domain
        ys_target = fb.istft(Ys_target)
        ys_interf = fb.istft(Ys_interf)
        ys_ideal = fb.istft(Ys_ideal)

        # Span full range
        vol_norm = max([ np.amax(np.abs(ys_target)),
                         np.amax(np.abs(ys_interf)),
                         np.amax(np.abs(ys_ideal)) ])
        ys_target /= vol_norm
        ys_interf /= vol_norm
        ys_ideal /= vol_norm

        # Apply new volume
        volume = np.random.uniform(low=-20, high=0)
        ys_target = mx.gain(ys_target, np.asarray([volume]))
        ys_interf = mx.gain(ys_interf, np.asarray([volume]))
        ys_ideal = mx.gain(ys_ideal, np.asarray([volume]))

        # Concatenate to a single multi-channel audio signal
        ys = np.concatenate([ ys_target, ys_interf, ys_ideal ], axis=0)

        # Save to file
        output_path = os.path.join(args.output, '%010d.wav' % index)
        io.write(ys, output_path)
        print("[%u/%u]: %s" % (index-args.index_start+1, args.index_stop-args.index_start+1, output_path))


def concatenate_waves(files, wave_duration, silence_duration, sample_rate, number_trials):

    wave_samples = int(wave_duration * sample_rate)
    silence_samples = int(silence_duration * sample_rate)

    s = np.zeros((1,wave_samples), dtype=np.float32)
    wave_samples_count = 0
    
    for trial in range(0, number_trials):

        while True:
        
            this_path = rnd.choice(files).rstrip()
            segment, _ = lr.load(path=this_path,
                                 sr=sample_rate,
                                 dtype=np.float32)

            segment = np.expand_dims(segment, axis=0)
            wave_samples_total = wave_samples_count + segment.shape[1]

            if wave_samples_count > 0:
                if wave_samples_total > wave_samples:
                    break
            else:
                if wave_samples_total > wave_samples:
                    segment = np.resize(segment,(1,wave_samples))
                    wave_samples_total = wave_samples

            segment_0dB = mx.pwr(segment, levels=np.asarray([0.0]))

            s[0, wave_samples_count:wave_samples_total] = segment_0dB
            wave_samples_count += segment.shape[1] + silence_samples

    return s

def configure_room(size, micarray, number_sources, margin=0.5, rotate=True):

    # Setup room dimensions

    if size == 'small':
        width = np.random.uniform(low=1, high=10, size=(1,))
        length = np.random.uniform(low=1, high=10, size=(1,))

    if size == 'medium':
        width = np.random.uniform(low=10, high=30, size=(1,))
        length = np.random.uniform(low=10, high=30, size=(1,))

    if size == 'large':
        width = np.random.uniform(low=30, high=50, size=(1,))
        length = np.random.uniform(low=30, high=50, size=(1,))

    height = np.random.uniform(low=2, high=5, size=(1,))

    box = np.concatenate([width, length, height], axis=0)

    # Choose microphone array (and rotate)

    if micarray == 'respeaker_usb':
        mics = ma.respeaker_usb()

    if micarray == 'respeaker_core':
        mics = ma.respeaker_core()

    if micarray == 'matrix_creator':
        mics = ma.matrix_creator()

    if micarray == 'matrix_voice':
        mics = ma.matrix_voice()

    if micarray == 'minidsp_uma':
        mics = ma.minidsp_uma()

    if rotate == True:

        theta_x = np.random.uniform(low=0, high=2*np.pi)
        theta_y = np.random.uniform(low=0, high=2*np.pi)
        theta_z = np.random.uniform(low=0, high=2*np.pi)

        Rx = np.asarray([[1, 0, 0],
                         [0, np.cos(theta_x), -np.sin(theta_x)],
                         [0, np.sin(theta_x), np.cos(theta_x)]])
        Ry = np.asarray([[np.cos(theta_y), 0, np.sin(theta_y)],
                         [0, 1, 0],
                         [-np.sin(theta_y), 0, np.cos(theta_y)]])
        Rz = np.asarray([[np.cos(theta_z), -np.sin(theta_z), 0],
                         [np.sin(theta_z), np.cos(theta_z), 0],
                         [0, 0, 1]])

        mics = (Rx @ Rz @ Rz @ mics.T).T
    
    # Define absorption

    absorption = np.random.uniform(low=0.2, high=0.8)
    alphas = absorption * np.ones(6, dtype=np.float32)

    # Position microphone array

    origin = np.concatenate([np.random.uniform(low=margin, high=box[0]-margin, size=(1,)),
                             np.random.uniform(low=margin, high=box[1]-margin, size=(1,)),
                             np.random.uniform(low=margin, high=box[2]-margin, size=(1,))], axis=0)

    # Position sources

    while True:

        srcs = np.concatenate([np.random.uniform(low=margin, high=box[0]-margin, size=(number_sources, 1)),
                               np.random.uniform(low=margin, high=box[1]-margin, size=(number_sources, 1)),
                               np.random.uniform(low=margin, high=box[2]-margin, size=(number_sources, 1))], axis=1)

        # Make sure distance is less or equal to 3 meters
        max_dist = np.amax(np.sqrt(np.sum((srcs - np.expand_dims(origin, axis=0)) ** 2, axis=1)))

        if max_dist < 3.0:
            break

    # Speed of sound
    c = np.random.uniform(low=330.0, high=355.0)
 
    # Create room
    return rb.room(mics, box, srcs, origin, alphas, c)

def create_pairs(ys,tdoas,src_tosteer):
    frameSize = 512
    hopSize = 128
    epsilon = 1E-20

    nMics = ys.shape[0]
    nSamples = ys.shape[1]

    Ys = []

    for iMic in range(0, nMics):
      Ys.append(np.expand_dims(np.transpose(lr.core.stft(ys[iMic,:], n_fft=frameSize, hop_length=hopSize)), axis=0))

    YYs = []

    k = np.transpose(np.expand_dims(np.arange(0, frameSize/2+1), axis=1))
    f = np.transpose(np.ones((1, Ys[0].shape[0]), dtype=np.float32))

    for iMic1 in range(0, nMics):
      
      for iMic2 in range(iMic1+1, nMics):
        tau = tdoas[src_tosteer,iMic1] - tdoas[src_tosteer,iMic2]
        A = np.exp(-1j*2*np.pi*tau*k*f/frameSize)
        YY = A * Ys[iMic1] * np.conj(Ys[iMic2])
        YY2 = np.zeros((Ys[0].shape[0], Ys[0].shape[1], Ys[0].shape[2], 2), dtype=np.float32)
        YY2[:,:,:,0] = np.log(np.abs(YY)**2 + epsilon) - np.log(epsilon)
        YY2[:,:,:,1] = np.angle(YY)
        YYs.append(YY2)

    Ys = np.concatenate(Ys, axis=0)
    YYs = np.concatenate(YYs, axis=0)

    return Ys, YYs

if __name__ == "__main__":

    main()
