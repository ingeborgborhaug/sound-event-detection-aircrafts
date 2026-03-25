from keras_yamnet import params
import pyaudio
from keras_yamnet.yamnet import class_names
import numpy as np
import os

# Dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
audio_folder_aero = 'C:\\Users\\imborhau\\OneDrive - NTNU\\Documents\\Prosjektoppgave\\AeroSonicDB-YPAD0523\\data\\raw'
gt_folder_aero = os.path.join(current_dir, "dataset", "AeroSonicDB")
port_folder = '/Volumes/Samsung USB'

print(f'audio_folder_aero: {audio_folder_aero}')
data_pairs_env = {gt_folder_aero + '/env_audio_gt.csv': [audio_folder_aero + '/env_audio']}
data_pairs_test = {gt_folder_aero + '/gt_test.csv' : [audio_folder_aero + '/audio/0', audio_folder_aero + '/audio/1']}
data_pairs_train = { gt_folder_aero + '/gt_train.csv': [ audio_folder_aero + '/audio/0', audio_folder_aero + '/audio/1']}

TRAIN_SIZE = 0.8
VAL_SIZE = 1 - TRAIN_SIZE

# Training and evaluation metric parameters
# GT_CONFIDENCE = 1.0 Ikke i bruk lenger
PREDICTION_THRESHOLD = 0.3 # Threshold for considering a class as present in a segment

# Pre-defined parameters
# YAMNET_CLASSES = class_names('keras_yamnet/yamnet_class_map.csv')
# PLT_CLASSES = [329]
# CLASS_NAMES = YAMNET_CLASSES[PLT_CLASSES]
CLASS_NAMES = ['Aircraft']
N_CLASSES = len(CLASS_NAMES)

# Parameters for demonstration/regular/..
""" print(sd.query_devices()) """
MIC = None
wav_detection = True
WINDOW_SIZE = 96
N_WINDOWS_SHOWING = 10
STRIDE = WINDOW_SIZE
#MAX_PATCHES_PER_AUDIO = 28
#FORMAT = pyaudio.paFloat32
#CHANNELS = 1
RATE = params.SAMPLE_RATE
WIN_SIZE_SEC = 0.975
CHUNK = int(WIN_SIZE_SEC * RATE) # Frames per window = 15600
RECORD_SECONDS = 500