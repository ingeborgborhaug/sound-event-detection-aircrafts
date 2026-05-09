from keras_yamnet import params
from keras_yamnet.yamnet import class_names
import numpy as np
import os
from pathlib import Path

try:
    import pyaudio  # Optional dependency for interactive microphone demos.
except Exception:  # pragma: no cover - training paths do not require pyaudio.
    pyaudio = None


current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent

gt_folder_aero = current_dir / "dataset" / "AeroSonicDB"
datasets_folder = current_dir / "dataset" / "Skatval"

# Set this in WSL/Linux:
# export SED_DATASETS_FOLDER=/mnt/c/Users/kampfly/Documents/Ingeborg/Prosjektoppgave/AeroSonicDB-YPAD0523/data/raw
_default_audio_folder_aero = parent_dir / "AeroSonicDB-YPAD0523" / "data" / "raw"

audio_folder_aero = Path(
    os.getenv("SED_DATASETS_FOLDER", str(_default_audio_folder_aero))
)

data_pairs_env = {
    str(gt_folder_aero / "env_audio_gt.csv"): [
        str(audio_folder_aero / "env_audio")
    ]
}

data_pairs_test = {
    str(gt_folder_aero / "gt_test.csv"): [
        str(audio_folder_aero / "audio" / "0"),
        str(audio_folder_aero / "audio" / "1"),
    ]
}

data_pairs_train = {
    str(gt_folder_aero / "gt_train.csv"): [
        str(audio_folder_aero / "audio" / "0"),
        str(audio_folder_aero / "audio" / "1"),
    ]
}

def make_radius_data_pairs(session: str, loc: str, min_km: float = 1, max_km: float = 15):
    gt_folder = os.path.join(datasets_folder, session) 
    clipped_folder = os.path.join(datasets_folder, session, "Clipped")
    return {
        km: {
            os.path.join(gt_folder, f"loc_{loc}_{session}_AUTOSAVE_sphere_{km}KM.csv"): [clipped_folder]
        }
        for km in np.arange(float(min_km), float(max_km) + 1, 1)
    }

data_pairs_030326_loc_2_by_radius = make_radius_data_pairs("030326", "2", 1, 9)
data_pairs_030326_loc_3_by_radius = make_radius_data_pairs("030326", "3", 1, 9)

data_pairs_230226_loc_1_by_radius = make_radius_data_pairs("230226", "1", 1, 9)
data_pairs_230226_loc_2_by_radius = make_radius_data_pairs("230226", "2", 1, 9)
data_pairs_230226_loc_3_by_radius = make_radius_data_pairs("230226", "3", 1, 9)

data_pairs_260326_part1_loc_1_by_radius = make_radius_data_pairs("260326_part1", "1", 1, 9)
data_pairs_260326_part1_loc_2_by_radius = make_radius_data_pairs("260326_part1", "2", 1, 9)
data_pairs_260326_part1_loc_3_by_radius = make_radius_data_pairs("260326_part1", "3", 1, 9)

data_pairs_260326_part2_loc_1_by_radius = make_radius_data_pairs("260326_part2", "1", 1, 9)
data_pairs_260326_part2_loc_2_by_radius = make_radius_data_pairs("260326_part2", "2", 1, 9)
data_pairs_260326_part2_loc_3_by_radius = make_radius_data_pairs("260326_part2", "3", 1, 9)

data_pairs_280126_loc_1_by_radius = make_radius_data_pairs("280126", "1", 1, 9)
data_pairs_280126_loc_2_by_radius = make_radius_data_pairs("280126", "2", 1, 9)
data_pairs_280126_loc_3_by_radius = make_radius_data_pairs("280126", "3", 1, 9)

data_pairs_300925_loc_gardemoen_by_radius = make_radius_data_pairs("300925", "gardemoen", 1, 9)

TRAIN_SIZE = 0.8
VAL_SIZE = 1 - TRAIN_SIZE
MAX_PATCHES = 1

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