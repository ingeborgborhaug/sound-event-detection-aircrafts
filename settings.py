from keras_yamnet import params
from keras_yamnet.yamnet import class_names
import numpy as np
import os
from pathlib import Path

try:
    import pyaudio  # Optional dependency for interactive microphone demos.
except Exception:  # pragma: no cover - training paths do not require pyaudio.
    pyaudio = None

# Dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
gt_folder_aero = os.path.join(current_dir, "dataset", "AeroSonicDB")

# Override with env var on any OS: SED_DATASETS_FOLDER=/path/to/datasets
_default_audio_folder_aero_windows = parent_dir + "/AeroSonicDB-YPAD0523/data/raw"
_default_audio_folder_aero_macos = "/Users/ingeborgborhaug/Skole/AeroSonicDB-YPAD0523/data/raw"
audio_folder_aero = os.getenv(
    "SED_DATASETS_FOLDER",
    _default_audio_folder_aero_windows if os.name == "nt" else _default_audio_folder_aero_macos,
)
datasets_folder = os.path.join(current_dir, "dataset")

data_pairs_env = {gt_folder_aero + '/env_audio_gt.csv': [audio_folder_aero + '/env_audio']}
data_pairs_test = {gt_folder_aero + '/gt_test.csv' : [audio_folder_aero + '/audio/0', audio_folder_aero + '/audio/1']}
data_pairs_train = { gt_folder_aero + '/gt_train.csv': [ audio_folder_aero + '/audio/0', audio_folder_aero + '/audio/1']}

def make_radius_data_pairs(session: str, loc: str, min_km: float = 1, max_km: float = 15):
    gt_folder = os.path.join(datasets_folder, session) 
    clipped_folder = os.path.join(datasets_folder, session, "Clipped")
    return {
        km: {
            os.path.join(gt_folder, f"loc_{loc}_{session}_AUTOSAVE_sphere_{km}KM.csv"): [clipped_folder]
        }
        for km in np.arange(float(min_km), float(max_km) + 1, 1)
    }

data_pairs_030326_loc_2_by_radius = make_radius_data_pairs("030326", "2", 1, 15)
data_pairs_030326_loc_3_by_radius = make_radius_data_pairs("030326", "3", 1, 15)

data_pairs_230226_loc_1_by_radius = make_radius_data_pairs("230226", "1", 1, 15)
data_pairs_230226_loc_2_by_radius = make_radius_data_pairs("230226", "2", 1, 15)
data_pairs_230226_loc_3_by_radius = make_radius_data_pairs("230226", "3", 1, 15)

data_pairs_260326_part1_loc_1_by_radius = make_radius_data_pairs("260326_part1", "1", 1, 15)
data_pairs_260326_part1_loc_2_by_radius = make_radius_data_pairs("260326_part1", "2", 1, 15)
data_pairs_260326_part1_loc_3_by_radius = make_radius_data_pairs("260326_part1", "3", 1, 15)

data_pairs_260326_part2_loc_1_by_radius = make_radius_data_pairs("260326_part2", "1", 1, 15)
data_pairs_260326_part2_loc_2_by_radius = make_radius_data_pairs("260326_part2", "2", 1, 15)
data_pairs_260326_part2_loc_3_by_radius = make_radius_data_pairs("260326_part2", "3", 1, 15)

data_pairs_280126_loc_1_by_radius = make_radius_data_pairs("280126", "1", 1, 15)
data_pairs_280126_loc_2_by_radius = make_radius_data_pairs("280126", "2", 1, 15)
data_pairs_280126_loc_3_by_radius = make_radius_data_pairs("280126", "3", 1, 15)

data_pairs_300925_loc_gardemoen_by_radius = make_radius_data_pairs("300925", "gardemoen", 1, 15)

TRAIN_SIZE = 0.8
VAL_SIZE = 1 - TRAIN_SIZE
MAX_PATCHES = 20

# Training and evaluation metric parameters
# GT_CONFIDENCE = 1.0 Ikke i bruk lenger
PREDICTION_THRESHOLD = 0.3 # Threshold for considering a class as present in a segment

# Pre-defined parameters
# YAMNET_CLASSES = class_names('keras_yamnet/yamnet_class_map.csv')
# PLT_CLASSES = [329]
# CLASS_NAMES = YAMNET_CLASSES[PLT_CLASSES]
CLASS_NAMES = ['Aircraft']
N_CLASSES = len(CLASS_NAMES)
print(f'Monitoring for {N_CLASSES} classes: {CLASS_NAMES}')

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