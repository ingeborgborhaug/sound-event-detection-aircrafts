
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from matplotlib import pyplot as plt

from keras_yamnet.preprocessing import preprocess_input
import tensorflow as tf
from demonstration.interactive.plot import Plotter

import soundfile as sf
import pickle
import settings 
from pathlib import Path
import numpy as np
import pandas as pd
from keras_yamnet import params
import functions 

  
def process_and_cache(audio_path, audio_wave, sample_rate, model, force=False):

    cache_file = os.path.splitext(audio_path)[0] + '_demo' + '.pkl'

    if os.path.exists(cache_file) and not force:
        print(f"Loading cached result for {audio_path}")
        with open(cache_file, 'rb') as f:
            variables = pickle.load(f)
    else:
        print(f'Processing and caching: {audio_path}')

        data_patches, spectrogram = preprocess_input(audio_wave, sample_rate)

        prediction = model.predict(data_patches)
        #prediction = prediction.detach().cpu().numpy()  # <-- Add this line


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
    
    baseline_model = tf.keras.models.load_model('history/baseline-aerosonicdb/best_model.keras')

    #################### DATA ####################

    # Set input
    wav_folder = Path("D:\\dataset_master\\280126")
    wav_file = wav_folder / "loc_2_280126.wav"
    ground_truth_path = Path("D:\\dataset_master\\280126\\loc_2_280126_AUTOSAVE_sphere.csv")


    info = sf.info(wav_file)
    sr = info.samplerate
    start_time = 0
    end_time = 36
    start_frame = int(start_time * sr)
    stop_frame = int(end_time * sr)
    waveform, sr = sf.read(wav_file, start=start_frame, stop=stop_frame, dtype='int16')
    # waveform = waveform / np.max(np.abs(waveform))  # Normalize waveform
    

    #################### STREAM ####################
        
    # Get results and visualization data
    variables = process_and_cache(wav_file, waveform, sr, baseline_model, force=True)
    prediction = variables['prediction']
    #prediction = postprocess_output(prediction)
    spectrogram = variables['spectrogram']

    _, y_test, _ = functions.get_data_from_dict({ground_truth_path : [wav_folder]}, force_reload=False)
    n_wins = len(prediction)
    y_test = y_test[:n_wins]

    print(f'First 20 elements of y_test: {y_test[:20]}')

    monitor = Plotter(n_classes=settings.N_CLASSES, 
                    n_wins=n_wins, 
                    spec= spectrogram,
                    pred= prediction,
                    gt=y_test,
                    FIG_SIZE=(12,6), 
                    msd_labels=None,
                    waveform= waveform,
                    sr= sr
    )


    plt.close('all')