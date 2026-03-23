
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from matplotlib import pyplot as plt

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
from dataset import gt_conversion_functions as cf



def get_newest_timestamp_folder(parent_dir):
    subfolders = [f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))]
    if not subfolders:
        return None
    newest = max(subfolders)
    return os.path.join(parent_dir, newest)


if __name__ == "__main__":

    #################### BASE-MODEL #####################
    
    baseline_model = tf.keras.models.load_model('history/baseline-aerosonicdb/best_model.keras')

    #################### CONFIG #####################
    
    datasetname = 'AerosonicDB' # 'Skatval' or 'AerosonicDB'
    
    # Set start and end time for visualization (in seconds)
    start_time = 0
    end_time = 60

    #################### DATA ####################

    # Set input to collected data
    if datasetname == 'Skatval':
        wav_folder = [Path("D:\\dataset_master\\280126")]
        wav_file = wav_folder / "loc_2_280126.wav"
        ground_truth_path = Path("D:\\dataset_master\\280126\\loc_2_280126_AUTOSAVE_sphere.csv")

    # Set input to AerosonicDB data
    if datasetname == 'AerosonicDB':
        wav_folder = [Path("D:\\AeroSonicDB-YPAD0523\\data\\raw\\audio\\1"), Path("D:\\AeroSonicDB-YPAD0523\\data\\raw\\audio\\0")]
        wav_file = wav_folder[0] / "7C68D4_2023-05-03_10-11-12_0_1.wav"
        ground_truth_path = Path("C:\\Users\\imborhau\\Documents\\sound-event-detection-aircrafts\\dataset\\AeroSonicDB\\gt_test.csv")


    #################### STREAM ####################
        
    # Get results and visualization data
    

    _, y_test, _ = functions.get_data_from_dict({ground_truth_path : wav_folder}, force_reload=False)
    y_test = y_test[cf.sec_to_start_index(start_time):cf.sec_to_start_index(end_time)]
    pdf_output_path = wav_folder[0] / f"{wav_file.stem}_interactive_plot.pdf"

    monitor = Plotter(n_classes=settings.N_CLASSES, 
                    starttime= start_time,
                    model=baseline_model,
                    endtime= end_time,
                    gt=y_test,
                    FIG_SIZE=(12,6), 
                    msd_labels=None,
                    wavfile=wav_file,
                    save_pdf_path=pdf_output_path,
    )


    plt.close('all')