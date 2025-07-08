from keras_yamnet import params
import pyaudio
from keras_yamnet.yamnet import class_names

YAMNET_CLASSES = class_names('keras_yamnet/yamnet_class_map.csv')
PLT_CLASSES = [308,301,279,321] # (Car passing by), (Car), (Wind noise (microphone)), (Traffic noise, roadway noise) 
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
