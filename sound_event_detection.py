import numpy as np
import pyaudio
from matplotlib import pyplot as plt
import pandas as pd
import sounddevice as sd

from keras_yamnet.yamnet import YAMNet, class_names
from keras_yamnet.preprocessing import preprocess_input
from keras.models import Model

import tensorflow as tf

from plot import Plotter

import soundfile as sf
import sounddevice as sd
import threading
import os
import pickle
import time
from settings import YAMNET_CLASSES, N_WINDOWS, PLT_CLASSES, class_labels, FORMAT, CHANNELS, RATE, CHUNK, RECORD_SECONDS, MIC, wav_detection, WINDOW_SIZE, STRIDE, N_CLASSES

def process_and_cache(audio_path, audio_wave, sample_rate, model, force=False):
    cache_dir = 'cache'
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, os.path.basename(audio_path) + '.pkl')

    if os.path.exists(cache_file) and not force:
        print(f"Loading cached result for {audio_path}")
        with open(cache_file, 'rb') as f:
            variables = pickle.load(f)
    else:
        print(f'Processing and caching: {audio_path}')
        data = preprocess_input(audio_wave, sample_rate)

        windows = []
        predictions = []
        for start in range(0, data.shape[0] - WINDOW_SIZE + 1, STRIDE):
            window = data[start:start+WINDOW_SIZE, :] # shape: (96, 64)
            prediction = model.predict(np.expand_dims(window,0))[0]
            
            windows.append(window)
            predictions.append(prediction) 

        variables = {
            "predictions": predictions,
            "windows": windows
        }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(variables, f)
    
    return variables

if __name__ == "__main__":

    #################### BASE-MODEL #####################
    
    base_model = YAMNet(weights='keras_yamnet/yamnet.h5')
    # Freeze model
    base_model.trainable = False

    #################### DATA ####################

    wav_path = "data\car-passing-city-364146.wav"
    waveform, sr = sf.read(wav_path)

    #################### STREAM ####################
    audio = pyaudio.PyAudio()

    # start Recording
    stream = audio.open(format=FORMAT,
                        input_device_index=MIC,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
    print("Recording...")

    if PLT_CLASSES is not None:
        plt_classes_lab = YAMNET_CLASSES[PLT_CLASSES]
        n_classes = len(PLT_CLASSES)
    else:
        PLT_CLASSES = [k for k in range(len(YAMNET_CLASSES))]
        plt_classes_lab = YAMNET_CLASSES if class_labels else None
        n_classes = len(YAMNET_CLASSES)

    if n_classes != N_CLASSES:
        raise ValueError('The length of plt_classes does not correlate with the settings of N_CLASSES')

    monitor = Plotter(n_classes=n_classes, win_size=WINDOW_SIZE, n_wins=N_WINDOWS, FIG_SIZE=(12,6), msd_labels=plt_classes_lab)



    if wav_detection:
        
        # Get results and visualization data
        variables = process_and_cache(wav_path, waveform, sr, base_model)
        predictions = variables['predictions']
        windows = variables['windows']

        n_window = len(windows)
        dur_per_window = 10 / n_window #TODO:  find actual duration of the audio file

        # Update visualization
        # sd.play(waveform, sr)
        for i in range(0,len(windows)):
            curr_window = windows[i]
            curr_prediction = predictions[i]
            monitor(curr_window.transpose(), np.expand_dims(curr_prediction[PLT_CLASSES], -1))
            time.sleep(dur_per_window)
    else:

        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            # Waveform
            data = preprocess_input(np.fromstring(
                stream.read(CHUNK), dtype=np.float32), RATE)
            prediction = base_model.predict(np.expand_dims(data,0))[0]

            monitor(data.transpose(), np.expand_dims(prediction[PLT_CLASSES],-1))

    print("Finished recording")
    

    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    print("Press Enter to close the plot...")
    input() 

    plt.close('all')