import numpy as np
import pyaudio
from matplotlib import pyplot as plt
import pandas as pd
import sounddevice as sd

from keras_yamnet import params
from keras_yamnet.yamnet import YAMNet, class_names
from keras_yamnet.preprocessing import preprocess_input

from plot import Plotter

import soundfile as sf
import sounddevice as sd
import threading
import os
import pickle
import time
        

def process_and_cache(wav_path, model, force=False):
    cache_dir = 'cache'
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, os.path.basename(wav_path) + '.pkl')

    if os.path.exists(cache_file) and not force:
        print(f"Loading cached result for {wav_path}")
        with open(cache_file, 'rb') as f:
            variables = pickle.load(f)
    else:
        print(f'Processing and caching: {wav_path}')
        data = preprocess_input(waveform, sr)

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

    ################### SETTINGS ###################
    plt_classes = [294,300,279,494] # Vehicle, Motot vehicle (road), Wind noise (microphone), Silence 
    class_labels=True
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = params.SAMPLE_RATE
    WIN_SIZE_SEC = 0.975
    CHUNK = int(WIN_SIZE_SEC * RATE)
    RECORD_SECONDS = 500

    print(sd.query_devices())
    MIC = None
    WAV_DETECTION = True

    WINDOW_SIZE = 96
    STRIDE = 96

    #################### MODEL #####################
    
    base_model = YAMNet(weights='keras_yamnet/yamnet.h5')
    # Freeze model
    base_model.trainable = False
    yamnet_classes = class_names('keras_yamnet/yamnet_class_map.csv')

    #################### DATA ####################

    wav_path = "data\car-passing-city-364146.wav"
    waveform, sr = sf.read(wav_path)
    duration = len(waveform) / sr

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

    if plt_classes is not None:
        plt_classes_lab = yamnet_classes[plt_classes]
        n_classes = len(plt_classes)
    else:
        plt_classes = [k for k in range(len(yamnet_classes))]
        plt_classes_lab = yamnet_classes if class_labels else None
        n_classes = len(yamnet_classes)

    monitor = Plotter(n_classes=n_classes, FIG_SIZE=(12,6), msd_labels=plt_classes_lab)

    if WAV_DETECTION:
        
        # Get results and visualization data
        variables = process_and_cache(wav_path, base_model)
        predictions = variables['predictions']
        windows = variables['windows']

        n_window = len(windows)
        dur_per_window = duration / n_window

        # Update visualization
        sd.play(waveform, sr)
        for i in range(0,len(windows)):
            curr_window = windows[i]
            curr_prediction = predictions[i]
            monitor(curr_window.transpose(), np.expand_dims(curr_prediction[plt_classes], -1))
            time.sleep(dur_per_window)
    else:

        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            # Waveform
            data = preprocess_input(np.fromstring(
                stream.read(CHUNK), dtype=np.float32), RATE)
            prediction = base_model.predict(np.expand_dims(data,0))[0]

            monitor(data.transpose(), np.expand_dims(prediction[plt_classes],-1))

    print("Finished recording")
    

    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    print("Press Enter to close the plot...")
    input() 

    plt.close('all')