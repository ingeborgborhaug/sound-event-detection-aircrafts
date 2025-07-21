import numpy as np
import pyaudio
from matplotlib import pyplot as plt
import pandas as pd
import sounddevice as sd

from keras_yamnet.yamnet import YAMNet, class_names
from keras_yamnet.preprocessing import preprocess_input
import tensorflow as tf
from plot import Plotter

import soundfile as sf
import sounddevice as sd
import os
import pickle
import time
import settings 
from keras.models import Model


def process_and_cache(audio_path, audio_wave, sample_rate, model_extention, model_base, force=False):
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
        for start in range(0, data.shape[0] - settings.WINDOW_SIZE + 1, settings.STRIDE):
            window = data[start:start+settings.WINDOW_SIZE, :] # shape: (96, 64)
            embedding = model_base.predict(np.expand_dims(window,0))
            prediction = model_extention(embedding)[0]
            
            windows.append(window)
            predictions.append(prediction) 

        variables = {
            "predictions": predictions,
            "windows": windows
        }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(variables, f)
    
    return variables

def get_newest_timestamp_folder(parent_dir):
    subfolders = [f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))]
    if not subfolders:
        return None
    newest = max(subfolders)
    return os.path.join(parent_dir, newest)

if __name__ == "__main__":

    #################### BASE-MODEL #####################
    
    yamnet_model = YAMNet(weights='keras_yamnet/yamnet.h5')
    base_model = Model(
    inputs=yamnet_model.input,
    outputs=yamnet_model.get_layer('global_average_pooling2d').output
)
    modified_model = tf.saved_model.load(f'{get_newest_timestamp_folder("history")}\modified_model')
    print(f'\nUsing model: {get_newest_timestamp_folder("history")}\modified_model \n')

    #################### DATA ####################

    wav_path = "data\car-passing-city-364146.wav"
    waveform, sr = sf.read(wav_path)

    #################### STREAM ####################
    """ audio = pyaudio.PyAudio()

    # start Recording
    stream = audio.open(format=settings.FORMAT,
                        input_device_index=settings.MIC,
                        channels=settings.CHANNELS,
                        rate=settings.RATE,
                        input=True,
                        frames_per_buffer=settings.CHUNK)
    print("Recording...") """

    """ if settings.PLT_CLASSES is not None:
        plt_classes_lab = settings.YAMNET_CLASSES[settings.PLT_CLASSES]
        n_classes = len(settings.PLT_CLASSES)
    else:
        PLT_CLASSES = [k for k in range(len(settings.YAMNET_CLASSES))]
        plt_classes_lab = settings.YAMNET_CLASSES if settings.class_labels else None
        n_classes = len(settings.YAMNET_CLASSES)

    if n_classes != settings.N_CLASSES:
        raise ValueError('The length of plt_classes does not correlate with the settings of N_CLASSES') """

    monitor = Plotter(n_classes=settings.N_CLASSES, 
                      win_size=settings.WINDOW_SIZE, 
                      n_wins=settings.N_WINDOWS, 
                      FIG_SIZE=(12,6), 
                      msd_labels=settings.CLASS_NAMES
    )


    if settings.wav_detection:
        
        # Get results and visualization data
        variables = process_and_cache(wav_path, waveform, sr, modified_model, base_model)
        predictions = variables['predictions']
        windows = variables['windows']

        n_window = len(windows)
        dur_per_window = 10 / n_window #TODO:  find actual duration of the audio file

        # Update visualization
        # sd.play(waveform, sr)
        for i in range(0, len(windows)):
            curr_window = windows[i]
            curr_prediction = predictions[i]
            monitor(curr_window.transpose(), np.expand_dims(curr_prediction, -1))
            time.sleep(dur_per_window)

    """ else:

        for i in range(0, int(settings.RATE / settings.CHUNK * settings.RECORD_SECONDS)):
            # Waveform
            data = preprocess_input(np.fromstring(
                stream.read(settings.CHUNK), dtype=np.float32), settings.RATE)
            prediction = model.predict(np.expand_dims(data,0))[0]

            monitor(data.transpose(), np.expand_dims(prediction[PLT_CLASSES],-1)) """

    print("Finished recording")
    

    """ # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate() """

    print("Press Enter to close the plot...")
    input() 

    plt.close('all')