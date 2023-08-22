import argparse
import numpy as np
import random as rnd
import librosa as lr
import os

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

def main():

    parser = argparse.ArgumentParser()
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

        # STFT
        Xs_target = fb.stft(xs_target)
        Xs_interf = fb.stft(xs_interf)
        Xs = fb.stft(xs)

        # Covariance matrices
        XXs_target = sp.scm(sp.xspec(Xs_target))
        XXs_interf = sp.scm(sp.xspec(Xs_interf))
        XXs = sp.scm(sp.xspec(Xs))

        # Compute mvdr weights
        ws_target = bf.mvdr(XXs_target, XXs_interf)
        ws_interf = bf.mvdr(XXs_interf, XXs_target)

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
        print(output_path)


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

    srcs = np.concatenate([np.random.uniform(low=margin, high=box[0]-margin, size=(number_sources, 1)),
                           np.random.uniform(low=margin, high=box[1]-margin, size=(number_sources, 1)),
                           np.random.uniform(low=margin, high=box[2]-margin, size=(number_sources, 1))], axis=1)

    # Speed of sound
    c = np.random.uniform(low=330.0, high=355.0)
 
    # Create room
    return rb.room(mics, box, srcs, origin, alphas, c)


if __name__ == "__main__":

    main()
