from keras_yamnet import params
import pyaudio
from keras_yamnet.yamnet import class_names
import numpy as np

# User-defined parameters
data_folder_path = 'data_dcase17_task4/'
gt_folder_path = data_folder_path + 'groundtruth_release/'

audio_folder_paths = [data_folder_path + 'unbalanced_train_segments_testing_set_audio_formatted_and_segmented_downloads/', 
                       data_folder_path + 'unbalanced_train_segments_training_set_audio_formatted_and_segmented_downloads/']

gt_file_names = [gt_folder_path + 'groundtruth_strong_label_testing_set.csv']
                 # gt_folder_path + 'groundtruth_strong_label_evaluation_set.csv']

TRAIN_SIZE = 0.6
TEST_SIZE = 0.2
VAL_SIZE = 1 - (TRAIN_SIZE + TEST_SIZE)


# Pre-defined parameters
YAMNET_CLASSES = class_names('keras_yamnet/yamnet_class_map.csv')
PLT_CLASSES = [308,301,279,321] # (Car passing by), (Car), (Wind noise (microphone)), (Traffic noise, roadway noise) 
CLASS_NAMES = np.append(YAMNET_CLASSES[PLT_CLASSES], 'Other') 
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
N_WINDOWS = 10
STRIDE = WINDOW_SIZE
N_CLASSES = 4
MAX_PATCHES_PER_AUDIO = 28
