from keras_yamnet import params
import pyaudio
from keras_yamnet.yamnet import class_names
import numpy as np
import os

# Dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
audio_folder = os.path.join(parent_dir, "AeroSonicDB-YPAD0523", "data", "raw")
gt_folder = os.path.join(current_dir, "dataset", "AeroSonicDB")

data_pairs_train = {gt_folder + '/gt_train.csv' : [audio_folder + '/audio/0', audio_folder + '/audio/1']}
data_pairs_test = {gt_folder + '/gt_test.csv' : [audio_folder + '/audio/0', audio_folder + '/audio/1']}
data_pair_env = {gt_folder + '/env_audio_gt.csv' : [audio_folder + '/env_audio']}

data_pair_mix_snr1 = {'dataset/synthetic_data/snr1_gt_A2-0002_OPT_G_002_0001_Tr2.csv' : ['dataset/synthetic_data/part1']}
data_pair_mix_snr2 = {'dataset/synthetic_data/snr2_gt_A2-0002_OPT_G_002_0001_Tr2.csv' : ['dataset/synthetic_data/part1']}
data_pair_mix_snr3 = {'dataset/synthetic_data/snr3_gt_A2-0002_OPT_G_002_0001_Tr2.csv' : ['dataset/synthetic_data/part1']}
data_pair_mix_snr5 = {'dataset/synthetic_data/snr5_gt_A2-0002_OPT_G_002_0001_Tr2.csv' : ['dataset/synthetic_data/part1']}
data_pair_mix_snr10 = {'dataset/synthetic_data/snr10_gt_A2-0002_OPT_G_002_0001_Tr2.csv' : ['dataset/synthetic_data/part1']}
data_pair_mix_snr15 = {'dataset/synthetic_data/snr15_gt_A2-0002_OPT_G_002_0001_Tr2.csv' : ['dataset/synthetic_data/part1']}
data_pair_mix_snr20 = {'dataset/synthetic_data/snr20_gt_A2-0002_OPT_G_002_0001_Tr2.csv' : ['dataset/synthetic_data/part1']}

TRAIN_SIZE = 0.6
VAL_SIZE = 1 - TRAIN_SIZE

# To cache or not to cache
FORCE_RELOAD_TRAIN = True
FORCE_RELOAD_SED = True

# Training and evaluation metric parameters
GT_CONFIDENCE = 1.0
PREDICTION_THRESHOLD = 0.85 # Threshold for considering a class as present in a segment

# Pre-defined parameters
# YAMNET_CLASSES = class_names('keras_yamnet/yamnet_class_map.csv')
# PLT_CLASSES = [329]
# CLASS_NAMES = YAMNET_CLASSES[PLT_CLASSES]
CLASS_NAMES = ['No aircraft', 'Aircraft']
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