
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from matplotlib import pyplot as plt

import zipfile
import tempfile
import tensorflow as tf
from demonstration.interactive.plot import Plotter
from src.models.yamnet_finetune import build_yamnet_classifier

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
    
    model_path = r'D:\final_runs\aerosonic_to_norwegian_ex3\20260526-171718Z\radius_dual_pos_2km_neg_10km_wnone\aero_only_to_norwegian\fold_4_test_4\training\best_model.keras'
    try:
        baseline_model = tf.keras.models.load_model(model_path, compile=False)
    except Exception:
        # Fallback: reconstruct model architecture from project builder and load archived HDF5 weights by name
        baseline_model = build_yamnet_classifier(freeze_backbone=False)
        try:
            with zipfile.ZipFile(model_path, 'r') as z:
                if 'model.weights.h5' in z.namelist():
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.h5')
                    tmp.write(z.read('model.weights.h5'))
                    tmp.flush(); tmp.close()
                    baseline_model.load_weights(tmp.name, by_name=True)
                else:
                    # re-raise original error if no weights found
                    raise
        except Exception:
            # If fallback fails, re-raise to show original problem
            raise

    #################### CONFIG #####################
    
    datasetname = 'Skatval' # 'Skatval' or 'AerosonicDB'
    loc = 'gardemoen' # loc_1, loc_2, loc_3 for Skatval; gardemoen for AerosonicDB
    session = "300925"
    km = 2.0

    # Set start and end time for visualization (in seconds)
    start_time = 620 #7460
    end_time = start_time + 50 #start_time + 60 # Visualize 60 seconds of audio

    #################### DATA ####################

    # Set input to collected data
    if datasetname == 'Skatval':
        wav_folder = [Path(f"dataset\\Skatval\\{session}")]
        wav_file = wav_folder[0] / f"{loc}_{session}.wav"
        ground_truth_path = Path(f"dataset\\Skatval\\{session}\\Newly_generated\\{loc}_{session}_AUTOSAVE_sphere_{km}KM.csv")

    # Set input to AerosonicDB data
    if datasetname == 'AerosonicDB':
        wav_folder = [Path("D:\\AeroSonicDB-YPAD0523\\data\\raw\\audio\\1"), Path("D:\\AeroSonicDB-YPAD0523\\data\\raw\\audio\\0")]
        wav_file = wav_folder[0] / "7C68D4_2023-05-03_10-11-12_0_1.wav"
        ground_truth_path = Path("C:\\Users\\imborhau\\Documents\\sound-event-detection-aircrafts\\dataset\\AeroSonicDB\\gt_test.csv")


    #################### STREAM ####################
        
    # Get results and visualization data
    

    _, y_test, _ = functions.get_data_from_dict({ground_truth_path : wav_folder}, force_reload=False)
    y_test = y_test[cf.sec_to_start_index(start_time):cf.sec_to_end_index(end_time)]
    pdf_output_path = f"{wav_file.stem}_interactive_plot.pdf"

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