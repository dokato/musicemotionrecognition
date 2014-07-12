#!/usr/bin/python
#-*- coding: utf-8 -*-

import numpy as np
import scipy.stats as st
import librosa
import matplotlib.pyplot as plt 

def tempo(signal,fs,hop_len = 64, **kwargs):
    """tempo for a music piece *signal*"""
    tempo, beats = librosa.beat.beat_track(y=signal, sr=fs, hop_length=hop_len)
    return tempo 
def beats(signal,fs,hop_len = 64, **kwargs):
    """number of beats for a music piece *signal*"""
    tempo, beat_ = librosa.beat.beat_track(y=signal, sr=fs, hop_length=hop_len)
    return len(beat_)
def rms(signal, **kwargs):
    """RMS value for music piece *signal*"""
    return np.sqrt(np.sum(signal**2))
def spectral_centroids(signal, fs, **kwargs):
    """SC for *signal*"""
    S = np.abs(np.fft.fft(signal))
    fv = np.fft.fftfreq(len(S), 1./fs)
    idx = fv >= 0
    S_plus = S[idx]
    fv_plus = fv[idx]
    return np.sum(S_plus*fv_plus)/np.sum(S_plus)
def simple_hoc(signal, **kwargs):
    """Count the number of zero crossings"""
    X = np.int_(signal >= 0)
    return np.sum(np.abs(X[1:] - X[:-1]))
def chromagram(signal,fs,n_fft=4096, hop_len=64, **kwargs):
    """Suggested by 'Prediction of multidimensional emotional ratings in music [...]'
    by Eerola T. et all"""
    y_harmonic, y_percussive = librosa.effects.hpss(signal)
    C = librosa.feature.chromagram(y=y_harmonic, sr=fs, n_fft=n_fft, hop_length=hop_len)
    return C 
def chromagram_feat(signal,fs,func = spectral_centroids,n_fft=4096, hop_len=64, **kwargs):
    "It makes *chromagram* flat by doing *func* on its rows"
    C = chromagram(signal,fs,n_fft,hop_len)
    feat_C = np.zeros(C.shape[0])
    for x in xrange(C.shape[0]):
        if hasattr(np,func.__name__):
            feat_C[x] = func(C[x])
        else:
            feat_C[x] = func(C[x],fs=fs)
    return feat_C
def irregularity(signal,fs, **kwargs):
    """Descriptor defined in 'Extracting emotions from music data' by Wieczorkowska A."""
    S = np.abs(np.fft.fft(signal))
    fv = np.fft.fftfreq(len(S), 1./fs)
    idx = fv >= 0
    S_plus = S[idx]
    fv_plus = fv[idx]
    S_k = S_plus[1:-1]
    S_left = S_plus[2:]
    S_right = S_plus[:-2]
    return np.log(20*np.sum(np.abs(np.log(S_k/(S_left*S_k*S_right)**(1./3)))))
def tunning(signal,fs, **kwargs):
    "It estimates *signal*'s tuning offset (in fractions of a bin) relative to A440=440.0Hz."
    return librosa.feature.estimate_tuning(y=signal,sr=fs)
if __name__ == '__main__':
    y, sr = librosa.load('music/GoHome.mp3',duration=30)
    print "sampling rate %f [Hz]"%sr 
    print "Music length %f [s]"%((len(y)*1./sr)/60,)
    print rms(y)
    print tunning(y,sr)
    print spectral_centroids(y,sr)
    print chromagram_feat(y,sr)
