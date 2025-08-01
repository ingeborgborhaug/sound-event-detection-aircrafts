import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

import numpy as np
import pyaudio
from matplotlib import pyplot as plt
import pandas as pd
import sounddevice as sd

from keras_yamnet.yamnet import YAMNet, class_names
from keras_yamnet.preprocessing import preprocess_input
from keras_yamnet.postprocessing import postprocess_output
import tensorflow as tf
import tensorflow as tf
from demonstration.interactive.plot import Plotter

import soundfile as sf
import sounddevice as sd
import pickle
import settings 
from keras.models import Model
from keras_yamnet import params

  
def process_and_cache(audio_path, audio_wave, sample_rate, model_extention, model_base, force=settings.FORCE_RELOAD_SED):
    """ cache_dir = 'cache_SED'
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, os.path.basename(audio_path) + '0-30sek' +'.pkl') """

    cache_file = os.path.splitext(audio_path)[0] + '_demo' + '.pkl'

    if os.path.exists(cache_file) and not force:
        print(f"Loading cached result for {audio_path}")
        with open(cache_file, 'rb') as f:
            variables = pickle.load(f)
    else:
        print(f'Processing and caching: {audio_path}')

        data_patches, spectrogram = preprocess_input(audio_wave, sample_rate)
        spectrogram = spectrogram[:data_patches.shape[0] * params.PATCH_HOP_FRAMES, :]

        embedding = model_base.predict(data_patches)
        prediction = model_extention(embedding)

        variables = {
            "prediction": prediction,
            "spectrogram": spectrogram
        }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(variables, f)
    
    print(f"Cached result saved to {cache_file}")
    
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

    wav_path_car = "data\data_testing\car-passing-city-364146.wav"
    wav_path_talking = "data\data_testing\people-talking-from-distant-271396.wav"
    wav_path_skjetten = "data/data_collected/22072025/A1-0002_skjetten_OPT_C_008_0001_Tr1.wav"
    wav_path_messa = "data/data_collected/01082025/A1-0001_OPT_C_003_Tr1_25m.wav"
    wav_path_car_from_train = settings.dcase_folder + 'unbalanced_train_segments_testing_set_audio_formatted_and_segmented_downloads/Y--zbPxnl27o_20.000_30.000.wav'
    wav_path = wav_path_messa
    gt_path = "data/data_collected/01082025/ground_truth/0001_processed.csv"
    info = sf.info(wav_path)
    sr = info.samplerate
    start_time = 130
    end_time = 180
    start_frame = int(start_time * sr)
    stop_frame = int(end_time * sr)
    waveform, _ = sf.read(wav_path, start= start_frame, stop=stop_frame)
    # waveform = waveform / np.max(np.abs(waveform))  # Normalize waveform
    

    #################### STREAM ####################
        
    # Get results and visualization data
    variables = process_and_cache(wav_path, waveform, sr, modified_model, base_model)
    prediction = variables['prediction']
    # prediction = postprocess_output(prediction)
    spectrogram = variables['spectrogram']

    monitor = Plotter(n_classes=settings.N_CLASSES, 
                    win_size=settings.WINDOW_SIZE, 
                    n_wins=len(prediction), 
                    spec= spectrogram,
                    pred= prediction,
                    FIG_SIZE=(12,6), 
                    msd_labels=settings.CLASS_NAMES,
                    waveform= waveform,
                    sr= sr,
                    start= start_time,
                    end= end_time
                    #gt_path= gt_path
    )

    """ n_window = len(windows)
    dur_per_window = 10 / n_window #TODO:  find actual duration of the audio file

    # Update visualization
    # sd.play(waveform, sr)
    for i in range(0, len(windows)):
        curr_window = windows[i]
        curr_prediction = predictions[i]
        monitor(curr_window.transpose(), np.expand_dims(curr_prediction, -1))
        time.sleep(dur_per_window) """

    """ else:

    for i in range(0, int(settings.RATE / settings.CHUNK * settings.RECORD_SECONDS)):
    # Waveform
    data = preprocess_input(np.fromstring(
        stream.read(settings.CHUNK), dtype=np.float32), settings.RATE)
    prediction = model.predict(np.expand_dims(data,0))[0]

    monitor(data.transpose(), np.expand_dims(prediction[PLT_CLASSES],-1)) """

    print("Finished recording")

    """ print("Press Enter to close the plot...")
    input()  """

    plt.close('all')