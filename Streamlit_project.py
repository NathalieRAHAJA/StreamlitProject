import streamlit as st

import glob
import librosa
import numpy as np
import os
import joblib
import pandas as pd
import scipy
import seaborn as sns
import shutil
import soundfile as sf
import tensorflow
import tensorflow as tf

from IPython.display import Audio 
from librosa import display
from matplotlib import pyplot as plt
from pathlib import Path
from ipywidgets import interactive
from mpl_toolkits import mplot3d 
from matplotlib import cm
#from tqdm import tqdm

#from enum import Enum
from fastapi import FastAPI, UploadFile, File
#from pydantic import BaseModel
from tempfile import NamedTemporaryFile
from pathlib import Path

from model.result import Result
#from src.api import get_result
#from src.visualisation import plot_logMelSpectrogram
from extract.feature_extraction_classification import ExtractorClassification
from nn.classification_model import Classification
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score

extractor = ExtractorClassification(r'C:/Users/Mimnat/Documents/DAS/etc/config.yaml')

def load_model(model_file):
    loaded_model=joblib.load(open(model_file),(os.path.join(model_file),"rb"))
    return loaded_model

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

 
def main():
    st.title("Détection de son anormal dans les pièces industrielles")
    menu = ["Visualisation","Prediction"]
    choice = st.sidebar.selectbox("Menu",menu)
    
    if choice == "Visualisation":
        machines=["fan","pump","slider","ToyCar","ToyConveyor","valve"]
        machine=st.sidebar.selectbox("Choisir une machine",machines)
        if machine == "fan":
            fan_train_folder= "C:/Users/Mimnat/Documents/DATASCIENTEST/Projet ds_son_anormal/Detection-son-anormal/StreamlitDoc/data/fan/train"
            #st.write(fan_train_folder)
            #choisir un fichier
            filenames=os.listdir(fan_train_folder)
            sound_files=st.selectbox("Choisir un fichier",filenames)
            #écouter l'audio:
            sound_file=os.path.join(fan_train_folder,sound_files)
            audio_file=open(sound_file,"rb")
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/wav')
            #Affichage du logmelspectrogramme et de l'audio wave : 
            st.write("Affichage du logmelspectrogramme et de l'audio wave:")
            audio_type=['wav']
            file_path= sound_file
            samples, sampling_rate = librosa.load(file_path,sr=None, mono=True, offset=0.0, duration=None)
            figure= plot_logMelSpectrogram(samples, params, sampling_rate,audio_type)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot(figure)    
        
        elif machine == "pump":
            pump_train_folder= "C:/Users/Mimnat/Documents/DATASCIENTEST/Projet ds_son_anormal/Detection-son-anormal/StreamlitDoc/data/pump/train"
            #st.write(pump_train_folder)
            #choisir un fichier
            filenames=os.listdir(pump_train_folder)
            sound_files=st.selectbox("Choisir un fichier",filenames)
            #écouter l'audio:
            sound_file=os.path.join(pump_train_folder,sound_files)
            audio_file=open(sound_file,"rb")
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/wav')    
            #Affichage du logmelspectrogramme et de l'audio wave : 
            st.write("Affichage du logmelspectrogramme et de l'audio wave:")
            audio_type=['wav']
            file_path= sound_file
            samples, sampling_rate = librosa.load(file_path,sr=None, mono=True, offset=0.0, duration=None)
            figure= plot_logMelSpectrogram(samples, params, sampling_rate,audio_type)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot(figure)    

        elif machine == "slider":
            slider_train_folder= "C:/Users/Mimnat/Documents/DATASCIENTEST/Projet ds_son_anormal/Detection-son-anormal/StreamlitDoc/data/slider/train"
            st.write(slider_train_folder)
            #choisir un fichier
            filenames=os.listdir(slider_train_folder)
            sound_files=st.selectbox("Choisir un fichier",filenames)
            #écouter l'audio:
            sound_file=os.path.join(slider_train_folder,sound_files)
            audio_file=open(sound_file,"rb")
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/wav')    
            #Affichage du logmelspectrogramme et de l'audio wave : 
            st.write("Affichage du logmelspectrogramme et de l'audio wave:")
            audio_type=['wav']
            file_path= sound_file
            samples, sampling_rate = librosa.load(file_path,sr=None, mono=True, offset=0.0, duration=None)
            figure= plot_logMelSpectrogram(samples, params, sampling_rate,audio_type)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot(figure)   

        elif machine == "ToyCar":
            ToyCar_train_folder= "C:/Users/Mimnat/Documents/DATASCIENTEST/Projet ds_son_anormal/Detection-son-anormal/StreamlitDoc/data/ToyCar/train"
            #st.write(ToyCar_train_folder)
            #choisir un fichier
            filenames=os.listdir(ToyCar_train_folder)
            sound_files=st.selectbox("Choisir un fichier",filenames)
            #écouter l'audio:
            sound_file=os.path.join(ToyCar_train_folder,sound_files)
            audio_file=open(sound_file,"rb")
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/wav')    
            #Affichage du logmelspectrogramme et de l'audio wave : 
            st.write("Affichage du logmelspectrogramme et de l'audio wave:")
            audio_type=['wav']
            file_path= sound_file
            samples, sampling_rate = librosa.load(file_path,sr=None, mono=True, offset=0.0, duration=None)
            figure= plot_logMelSpectrogram(samples, params, sampling_rate,audio_type)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot(figure)   

        elif machine == "ToyConveyor":
            ToyConveyor_train_folder= "C:/Users/Mimnat/Documents/DATASCIENTEST/Projet ds_son_anormal/Detection-son-anormal/StreamlitDoc/data/ToyConveyor/train"
            st.write(ToyConveyor_train_folder)
            #choisir un fichier
            filenames=os.listdir(ToyConveyor_train_folder)
            sound_files=st.selectbox("Choisir un fichier",filenames)
            #écouter l'audio:
            sound_file=os.path.join(ToyConveyor_train_folder,sound_files)
            audio_file=open(sound_file,"rb")
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/wav')    
            #Affichage du logmelspectrogramme et de l'audio wave : 
            st.write("Affichage du logmelspectrogramme et de l'audio wave:")
            audio_type=['wav']
            file_path= sound_file
            samples, sampling_rate = librosa.load(file_path,sr=None, mono=True, offset=0.0, duration=None)
            figure= plot_logMelSpectrogram(samples, params, sampling_rate,audio_type)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot(figure)   

        elif machine== "valve":
            valve_train_folder= "C:/Users/Mimnat/Documents/DATASCIENTEST/Projet ds_son_anormal/Detection-son-anormal/StreamlitDoc/data/valve/train"
            #st.write(valve_train_folder)
            #choisir un fichier
            filenames=os.listdir(valve_train_folder)
            sound_files=st.selectbox("Choisir un fichier",filenames)
            #écouter l'audio:
            sound_file=os.path.join(valve_train_folder,sound_files)
            audio_file=open(sound_file,"rb")
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/wav')    
            #Affichage du logmelspectrogramme et de l'audio wave : 
            st.write("Affichage du logmelspectrogramme et de l'audio wave:")
            audio_type=['wav']
            file_path= sound_file
            samples, sampling_rate = librosa.load(file_path,sr=None, mono=True, offset=0.0, duration=None)
            figure= plot_logMelSpectrogram(samples, params, sampling_rate,audio_type)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot(figure)   

    elif choice == "Prediction":
        machines=["fan","pump","slider","ToyCar","ToyConveyor","valve"]
        machine=st.sidebar.selectbox("choisir une machine",machines)
        id_m= extractor.id_machines[machine]
        if machine:
            id_m = st.selectbox("Choisir un ID", extractor.id_machines[machine])
        if machine=="fan":
            folder_path= "C:/Users/Mimnat/Documents/DATASCIENTEST/Projet ds_son_anormal/Detection-son-anormal/StreamlitDoc/data/fan/test"
            #st.write(folder_path)
            filenames=os.listdir(folder_path)
            sound_files=st.selectbox("Choisir un fichier avec ce même ID",filenames)
            sound_file=os.path.join(folder_path,sound_files)
            audio_file=open(sound_file,"rb")
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/wav')
            #classification
            filename= audio_file
            if st.button('Predict'):
                result = Result(machine, id_m, extractor)
                suffix = Path(filename.name).suffix
                st.write(suffix)
                try:
                    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        shutil.copyfileobj(filename, tmp)
                        out = result.predict(tmp.name)
                        if out == 0:
                            st.write("Audio '{filename.name}' sounds normal for machine {machine} {id_m}")
                        else:
                            st.write("Audio '{filename.name}' sounds anormal for machine {machine} {id_m}")
                finally:
                    tmp.close()
                    os.unlink(tmp.name)

        elif machine=="pump":
            folder_path= "C:/Users/Mimnat/Documents/DATASCIENTEST/Projet ds_son_anormal/Detection-son-anormal/StreamlitDoc/data/pump/test"
            #st.write(folder_path)
            filenames=os.listdir(folder_path)
            sound_files=st.selectbox("Choisir un fichier avec ce même ID",filenames)
            sound_file=os.path.join(folder_path,sound_files)
            audio_file=open(sound_file,"rb")
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/wav')
            #classification
            filename= audio_file
            if st.button('Predict'):
                result = Result(machine, id_m, extractor)
                suffix = Path(filename.name).suffix
                st.write(suffix)
                try:
                    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        shutil.copyfileobj(filename, tmp)
                        out = result.predict(tmp.name)
                        if out == 0:
                            st.write("Audio '{filename.name}' sounds normal for machine {machine} {id_m}")
                        else:
                            st.write("Audio '{filename.name}' sounds anormal for machine {machine} {id_m}")
                finally:
                    tmp.close()
                    os.unlink(tmp.name)
        if machine=="slider":
            folder_path= "C:/Users/Mimnat/Documents/DATASCIENTEST/Projet ds_son_anormal/Detection-son-anormal/StreamlitDoc/data/slider/test"
            #st.write(folder_path)
            filenames=os.listdir(folder_path)
            sound_files=st.selectbox("Choisir un fichier avec ce même ID",filenames)
            sound_file=os.path.join(folder_path,sound_files)
            audio_file=open(sound_file,"rb")
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/wav')
            #classification
            filename= audio_file
            if st.button('Predict'):
                result = Result(machine, id_m, extractor)
                suffix = Path(filename.name).suffix
                st.write(suffix)
                try:
                    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        shutil.copyfileobj(filename, tmp)
                        out = result.predict(tmp.name)
                        if out == 0:
                            st.write("Audio '{filename.name}' sounds normal for machine {machine} {id_m}")
                        else:
                            st.write("Audio '{filename.name}' sounds anormal for machine {machine} {id_m}")
                finally:
                    tmp.close()
                    os.unlink(tmp.name)
        elif machine=="ToyCar":
            folder_path= "C:/Users/Mimnat/Documents/DATASCIENTEST/Projet ds_son_anormal/Detection-son-anormal/StreamlitDoc/data/ToyCar/test"
            #st.write(folder_path)
            filenames=os.listdir(folder_path)
            sound_files=st.selectbox("Choisir un fichier avec ce même ID",filenames)
            sound_file=os.path.join(folder_path,sound_files)
            audio_file=open(sound_file,"rb")
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/wav')
            #classification
            filename= audio_file
            if st.button('Predict'):
                result = Result(machine, id_m, extractor)
                suffix = Path(filename.name).suffix
                st.write(suffix)
                try:
                    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        shutil.copyfileobj(filename, tmp)
                        out = result.predict(tmp.name)
                        if out == 0:
                            st.write("Audio '{filename.name}' sounds normal for machine {machine} {id_m}")
                        else:
                            st.write("Audio '{filename.name}' sounds anormal for machine {machine} {id_m}")
                finally:
                    tmp.close()
                    os.unlink(tmp.name)    
        elif machine=="ToyConveyor":
            folder_path= "C:/Users/Mimnat/Documents/DATASCIENTEST/Projet ds_son_anormal/Detection-son-anormal/StreamlitDoc/data/ToyConveyor/test"
            #st.write(folder_path)
            filenames=os.listdir(folder_path)
            sound_files=st.selectbox("Choisir un fichier avec ce même ID",filenames)
            sound_file=os.path.join(folder_path,sound_files)
            audio_file=open(sound_file,"rb")
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/wav')
            #classification
            filename= audio_file
            if st.button('Predict'):
                result = Result(machine, id_m, extractor)
                suffix = Path(filename.name).suffix
                st.write(suffix)
                try:
                    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        shutil.copyfileobj(filename, tmp)
                        out = result.predict(tmp.name)
                        if out == 0:
                            st.write("Audio '{filename.name}' sounds normal for machine {machine} {id_m}")
                        else:
                            st.write("Audio '{filename.name}' sounds anormal for machine {machine} {id_m}")
                finally:
                    tmp.close()
                    os.unlink(tmp.name)   
        elif machine=="valve":
            folder_path= "C:/Users/Mimnat/Documents/DATASCIENTEST/Projet ds_son_anormal/Detection-son-anormal/StreamlitDoc/data/valve/test"
            #st.write(folder_path)
            filenames=os.listdir(folder_path)
            sound_files=st.selectbox("Choisir un fichier avec ce même ID",filenames)
            sound_file=os.path.join(folder_path,sound_files)
            audio_file=open(sound_file,"rb")
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/wav')
            #classification
            filename= audio_file
            if st.button('Predict'):
                result = Result(machine, id_m, extractor)
                suffix = Path(filename.name).suffix
                st.write(suffix)
                try:
                    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        shutil.copyfileobj(filename, tmp)
                        out = result.predict(tmp.name)
                        if out == 0:
                            st.write("Audio '{filename.name}' sounds normal for machine {machine} {id_m}")
                        else:
                            st.write("Audio '{filename.name}' sounds anormal for machine {machine} {id_m}")
                finally:
                    tmp.close()
                    os.unlink(tmp.name)



if __name__ == '__main__':
        main()