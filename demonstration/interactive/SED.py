import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from matplotlib import pyplot as plt

from keras_yamnet.yamnet import YAMNet
from keras_yamnet.preprocessing import preprocess_input
import tensorflow as tf
from demonstration.interactive.plot import Plotter

import soundfile as sf
import pickle
import settings 
from keras.models import Model
from keras_yamnet import params
from pathlib import Path

  
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
    # Choose model 
    modified_model =  tf.saved_model.load(f'{get_newest_timestamp_folder("history")}/modified_model')
    print(f'\nUsing model: {get_newest_timestamp_folder("history")}/modified_model \n')

    #################### DATA ####################

    # Set input
    wav_path = 'dataset/test/aircraft-248663.wav'

    info = sf.info(wav_path)
    sr = info.samplerate
    start_time = 0
    end_time = 36
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
                    n_wins=len(prediction), 
                    spec= spectrogram,
                    pred= prediction,
                    FIG_SIZE=(12,6), 
                    msd_labels=settings.CLASS_NAMES,
                    waveform= waveform,
                    sr= sr
    )


    plt.close('all')