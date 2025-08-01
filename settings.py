from keras_yamnet import params
import pyaudio
from keras_yamnet.yamnet import class_names
import numpy as np

# User-defined parameters
dcase_folder = 'data/data_dcase17_task4/'
dcase_gt_folder = dcase_folder + 'groundtruth_release/'

collected_data_folder = 'data/data_collected/'

""" audios_folder = [data_folder_path + 'unbalanced_train_segments_testing_set_audio_formatted_and_segmented_downloads/'] 
                       #data_folder_path + 'unbalanced_train_segments_training_set_audio_formatted_and_segmented_downloads/']


gt_paths = [gt_folder_path + 'groundtruth_strong_label_testing_set.csv'] """

data_pairs_train = {collected_data_folder + '22072025/ground_truth/gt_0001_0002.csv' : collected_data_folder + '22072025/',
              dcase_gt_folder + 'groundtruth_strong_label_testing_set.csv' : dcase_folder + 'unbalanced_train_segments_testing_set_audio_formatted_and_segmented_downloads/'
              }

data_pairs_test_10m = {collected_data_folder + '01082025/ground_truth/0001_A2-0001_OPT_E_003_Tr1_10m.csv' : collected_data_folder + '01082025/'}
data_pairs_test_25m = {collected_data_folder + '01082025/ground_truth/0001_A1-0001_OPT_C_003_Tr1_25m.csv' : collected_data_folder + '01082025/'}
data_pairs_test_50m = {collected_data_folder + '01082025/ground_truth/0001_A4-0001_OPT_H_001_Tr1_50m.csv' : collected_data_folder + '01082025/'}
data_pairs_test_75m = {collected_data_folder + '01082025/ground_truth/0001_A4-0001_OPT_H_001_Tr1_50m.csv' : collected_data_folder + '01082025/'}

FORCE_RELOAD_GT_TRAIN = True
FORCE_RELOAD_SED = True

TRAIN_SIZE = 0.6
TEST_SIZE = 0.2
VAL_SIZE = 1 - (TRAIN_SIZE + TEST_SIZE)

# Training and evaluation metric parameters
GT_CONFIDENCE = 1.0
PREDICTION_THRESHOLD = 0.5 # Threshold for considering a class as present in a segment

# Pre-defined parameters
YAMNET_CLASSES = class_names('keras_yamnet/yamnet_class_map.csv')
# PLT_CLASSES = [308,301,279,321] # (Car passing by), (Car), (Wind noise (microphone)), (Traffic noise, roadway noise) 
PLT_CLASSES = [301]
# WEIGHTS_CLASSES = [0.25, 0.25, 0.25, 0.25]
CLASS_NAMES = YAMNET_CLASSES[PLT_CLASSES]
class_labels=True
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = params.SAMPLE_RATE
WIN_SIZE_SEC = 0.975
CHUNK = int(WIN_SIZE_SEC * RATE) # Frames per window = 15600
print(f'CHUNK: {CHUNK}, RATE: {RATE}, WIN_SIZE_SEC: {WIN_SIZE_SEC}')
RECORD_SECONDS = 500

""" print(sd.query_devices()) """
MIC = None
wav_detection = True

WINDOW_SIZE = 96
N_WINDOWS_SHOWING = 10
STRIDE = WINDOW_SIZE
N_CLASSES = len(CLASS_NAMES)
MAX_PATCHES_PER_AUDIO = 28