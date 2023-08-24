### Independent Components Analysis
###
### This program requires a working installation of:
###
### On Mac:
###     1. portaudio: On Mac: brew install portaudio
###     2. sounddevice: pip install sounddevice
###
### On windows:
###      pip install pyaudio sounddevice
###

from scipy.io import wavfile
import numpy as np

Fs = 11025

def normalize(dat):
    return 0.99 * dat / np.max(np.abs(dat))

def load_data():
    mix = np.loadtxt('mix.dat')
    return mix

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def unmixer(X):
    M, N = X.shape
    W = np.eye(N)

    anneal = [0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01, 0.01,
              0.005, 0.005, 0.002, 0.002, 0.001, 0.001]
    print('Separating tracks ...')

    for alpha in anneal:
        np.random.permutation(X)
        for i in range(M):
            x = X[i, :].reshape(-1, 1)
            WX = W.dot(x)
            grad = (1 - 2 * sigmoid(WX)).dot(x.T) + np.linalg.inv(W.T)
            W += alpha * grad

    print('Separation done.')

    return W

def unmix(X, W):
    S = np.dot(X, W.T)
    return S

def main():
    X = normalize(load_data())

    for i in range(X.shape[1]):
        track_filename = f"original_track_{i}.wav"
        wavfile.write(track_filename, Fs, X[:, i].astype(np.float32))
        print(f'Saved original track {i} as {track_filename}')

    W = unmixer(X)
    S = normalize(unmix(X, W))

    for i in range(S.shape[1]):
        print(S[:, i])
        track_filename = f"separated_track_{i}.wav"
        wavfile.write(track_filename, Fs, S[:, i].astype(np.float32))
        print(f'Saved separated track {i} as {track_filename}')

if __name__ == '__main__':
    main()
