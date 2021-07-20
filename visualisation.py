from IPython.display import Audio 
from librosa import display
from matplotlib import pyplot as plt
from pathlib import Path
from ipywidgets import interactive
from mpl_toolkits import mplot3d 
from matplotlib import cm

import glob
import librosa
import numpy as np
import os
import pandas as pd
import scipy
import seaborn as sns
import soundfile as sf

file_path = r"file.wav"
samples, sampling_rate = librosa.load(file_path,sr=None, mono=True, offset=0.0, duration=None)
params = {'n_fft': 160*16,'frame_step': 160*8,'lower_edge_hertz': 0,'upper_edge_hertz': 8000,'num_mel_bins': 160}
def logMelSpectrogram(audio, params, fe):
    stfts = librosa.stft(audio, n_fft = int(params['n_fft']), hop_length = int(params["frame_step"]), center = False).T
    power_spectrograms = np.real(stfts * np.conj(stfts))
    linear_to_mel_weight_matrix = librosa.filters.mel(sr=fe, n_fft=int(params['n_fft']) + 1,n_mels=params['num_mel_bins'], fmin=params['lower_edge_hertz'], fmax=params['upper_edge_hertz']).T
    mel_spectrograms = np.tensordot( power_spectrograms, linear_to_mel_weight_matrix, 1)
    return (np.log(mel_spectrograms + 1e-8).astype(np.float16))

def plot_logMelSpectrogram(audio, params, fe, audio_type):
    cmap = sns.color_palette("hot", 100)
    sns.set(style="white")

    fig, ax = plt.subplots(2, 2, figsize=(16,7), gridspec_kw={'width_ratios':[100,5], 'height_ratios':[4, 1] })
    #fig.suptitle(f'Extraction Audio', fontsize=26)
    sns.heatmap(np.rot90(logMelSpectrogram(audio, params, fe)), cmap='inferno', ax=ax[0,0], cbar_ax=ax[0,1])
                    
    ax[0, 0].set_title('logMelSpectrogram', fontsize=16)
    ax[0,0].set_ylabel("Frequency (Mel)")
    ax[0,0].set_yticks([])
    ax[0,0].set_xticks([])
    ax[1,1].remove()  # remove unused upper right axes
                    
    librosa.display.waveplot(y=audio, sr=fe, ax=ax[1, 0])
                    
    ax[1, 0].set_title('Audio wave', fontsize=16)
    ax[1, 0].spines['right'].set_visible(False)
    ax[1, 0].spines['top'].set_visible(False)

