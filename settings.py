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

#data_pairs_train = {gt_folder + '/gt_train.csv' : [audio_folder + '/audio/0', audio_folder + '/audio/1']}
data_pairs_test = {gt_folder + '/gt_test.csv' : [audio_folder + '/audio/0', audio_folder + '/audio/1']}
#data_pair_env = {gt_folder + '/env_audio_gt.csv' : [audio_folder + '/env_audio']}
data_pairs_train = {
    gt_folder + '/env_audio_gt.csv': [audio_folder + '/env_audio'],
    gt_folder + '/gt_train.csv': [ audio_folder + '/audio/0', audio_folder + '/audio/1']
}

# Part 1 data
dataset_folder1 = 'dataset/synthetic_data/part1/'

data_pair1_snr1 = {dataset_folder1 + 'snr_1_gt.csv' : [dataset_folder1]}
data_pair1_snr5 = {dataset_folder1 + 'snr_5_gt.csv' : [dataset_folder1]}
data_pair1_snr10 = {dataset_folder1 + 'snr_10_gt.csv' : [dataset_folder1]}
data_pair1_snr15 = {dataset_folder1 + 'snr_15_gt.csv' : [dataset_folder1]}
data_pair1_snr20 = {dataset_folder1 + 'snr_20_gt.csv' : [dataset_folder1]}

# Part 2 data
dataset_folder2 = 'dataset/synthetic_data/part2/'

data_pair2_snr1_2 = {(dataset_folder2 + 'snr_1_gt.csv', dataset_folder2 + 'snr_2_gt.csv') : [dataset_folder2]}
data_pair2_snr5_7 = {(dataset_folder2 + 'snr_5_gt.csv', dataset_folder2 + 'snr_7_gt.csv') : [dataset_folder2]}
data_pair2_snr10_12 = {(dataset_folder2 + 'snr_10_gt.csv', dataset_folder2 + 'snr_12_gt.csv') : [dataset_folder2]}
data_pair2_snr15_17 = {(dataset_folder2 + 'snr_15_gt.csv', dataset_folder2 + 'snr_17_gt.csv') : [dataset_folder2]}
data_pair2_snr20_23 = {(dataset_folder2 + 'snr_20_gt.csv', dataset_folder2 + 'snr_23_gt.csv') : [dataset_folder2]}

# data_pair2_snr2 = {dataset_folder2 + 'snr_2_gt.csv' : [dataset_folder2]}
# data_pair2_snr7 = {dataset_folder2 + 'snr_7_gt.csv' : [dataset_folder2]}
# data_pair2_snr12 = {dataset_folder2 + 'snr_12_gt.csv' : [dataset_folder2]}
# data_pair2_snr17 = {dataset_folder2 + 'snr_17_gt.csv' : [dataset_folder2]}
# data_pair2_snr23 = {dataset_folder2 + 'snr_23_gt.csv' : [dataset_folder2]}


TRAIN_SIZE = 0.8
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