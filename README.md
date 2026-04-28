# Sound Event Detection of aircraft

This repository contains a Python implementation of a Sound Event Detection (SED) system for detecting aircraft-related events from .wav audio files.

## Clone repo to your computer

```bash
git clone https://github.com/ingeborgborhaug/sound-event-detection-aircrafts.git
```

## Setting up the environment

Install Python 3.10.7
Execute the following commands to setup you project.

Windows: 
```bash
py -3.10 -m venv realtimevenv
realtimevenv\Scripts\activate 
```
Mac:
```bash
python3.10 -m venv realtimevenv
source realtimevenv/bin/activate
```
Ubuntu/Linux:
```bash
python3.10 -m venv .venv-wsl
source .venv-wsl/bin/activate
```

If Python 3.10 is not installed on Ubuntu:
```bash
sudo apt update
sudo apt install -y python3.10 python3.10-venv
```
NB! 
You might have to run this on mac if you have issues with building wheels:
```bash
brew install portaudio
```

On Ubuntu/Linux, if PortAudio headers are missing:
```bash
sudo apt install -y portaudio19-dev
```

If UnauthorizedAccess do this to temporarily allow scrips in your session: 

```bash
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
```

## Install requirements

```bash
pip install -r requirements.txt
```

All script commands in this README can be run with `python ...` when your environment is activated.
For Ubuntu/WSL, the current environment in this repo is `.venv-wsl`.
`realtimevenv` is not a required training environment.

## Quick start (AeroSonic CV)

Use this when you want the shortest path from preprocessing to final CV metrics.

```bash
# 1) Precompute patches + train manifest
python scripts/01_preprocess.py \
  --pair-filter data_pairs_train \
  --out-dir data/processed/aerosonic_train \
  --manifest data/processed/aerosonic_train_manifest.csv \
  --force

# 2) Generate fold splits from manifest
python scripts/02_generate_splits.py \
  --manifest data/processed/aerosonic_train_manifest.csv \
  --dataset aerosonic \
  --out-dir data/splits/aerosonic

# 3) Train all folds
python scripts/04_train.py \
  --manifest data/processed/aerosonic_train_manifest.csv \
  --splits-dir data/splits/aerosonic \
  --epochs 30 \
  --batch-size 16 \
  --max-patches 20

# 4) Evaluate (replace <run-id> with the newest folder in history/prosjektoppgave)
python scripts/05_evaluate.py \
  --cv-results history/prosjektoppgave/<run-id>/cv_results.json
```

Files produced in quick start:
- `data/processed/aerosonic_train_manifest.csv`
- `data/splits/aerosonic/*.json`
- `history/prosjektoppgave/<run-id>/cv_results.json`

## End-to-end experiments (from raw data to final metrics)

This section is the recommended workflow for thesis experiments.

### Input files you must have before running

1. AeroSonic audio and GT files:
- audio folders (used by `settings.py`):
  - `<AEROSONIC_ROOT>/env_audio`
  - `<AEROSONIC_ROOT>/audio/0`
  - `<AEROSONIC_ROOT>/audio/1`
- GT CSV files:
  - `dataset/AeroSonicDB/env_audio_gt.csv`
  - `dataset/AeroSonicDB/gt_train.csv`
  - `dataset/AeroSonicDB/gt_test.csv`

2. Optional Norwegian/Skatval GT and audio folders:

**Expected folder structure:**
```
dataset/Skatval/
├── loc_1_20260128_124313.csv          (ground truth file with date in name)
├── loc_1_20260128_124313_audio/       (matching audio folder)
│   ├── audio_file_1.wav
│   ├── audio_file_2.wav
│   └── ...
├── loc_2_20260128_124313.csv
├── loc_2_20260128_124313_audio/
└── ...
```

**Expected CSV format (same columns required):**
```
filename    start_time  end_time    class
audio_file_1.wav    0.5    2.3    Aircraft
audio_file_1.wav    5.1    7.8    Aircraft
audio_file_2.wav    1.2    3.4    Aircraft
```

**How to add to `settings.py`:**

Add a `data_pairs_*` dictionary with your Skatval paths. The script automatically extracts `session` (e.g., `20260128`) and `location` (e.g., `loc_1`) from the filename patterns:

```python
# At the end of settings.py, add:
data_pairs_skatval = {
    'dataset/Skatval/loc_1_20260128.csv': ['dataset/Skatval/loc_1_20260128_audio'],
    'dataset/Skatval/loc_1_20260201.csv': ['dataset/Skatval/loc_1_20260201_audio'],
    'dataset/Skatval/loc_2_20260128.csv': ['dataset/Skatval/loc_2_20260128_audio'],
    # Add more sessions/locations as needed
}
```

**How to run preprocessing:**

```powershell
realtimevenv\Scripts\python.exe scripts/01_preprocess.py `
  --pair-filter skatval `
  --out-dir data/processed/skatval `
  --manifest data/processed/skatval_manifest.csv `
  --force
```

The session and location are automatically extracted from your CSV filenames using regex:
- Session: looks for date patterns like `20260128` or `280126` in the GT filename
- Location: looks for patterns like `loc_1`, `location_2`, `loc_a`, etc. in the GT filename

Example filename parsing:
- `loc_1_20260128_124313.csv` → session=`20260128`, location=`loc_1`
- `session_1_location_a.csv` → session=`session_1`, location=`location_a`

3. Configure paths in `settings.py`:
- `audio_folder_aero` is resolved from environment variable `SED_DATASETS_FOLDER` (if set), otherwise from the OS-specific fallback in `settings.py`
- `data_pairs_train`, `data_pairs_test`, `data_pairs_env` define which GT CSV and audio folders are read

### Step 1: Compute cached patches and manifest (AeroSonic)

Run preprocessing for AeroSonic train set:

```powershell
realtimevenv\Scripts\python.exe scripts/01_preprocess.py `
  --pair-filter data_pairs_train `
  --out-dir data/processed/aerosonic_train `
  --manifest data/processed/aerosonic_train_manifest.csv `
  --force
```

Optional preprocessing for AeroSonic test/env sets:

```powershell
realtimevenv\Scripts\python.exe scripts/01_preprocess.py `
  --pair-filter data_pairs_test `
  --out-dir data/processed/aerosonic_test `
  --manifest data/processed/aerosonic_test_manifest.csv `
  --force
```

```powershell
realtimevenv\Scripts\python.exe scripts/01_preprocess.py `
  --pair-filter data_pairs_env `
  --out-dir data/processed/aerosonic_env `
  --manifest data/processed/aerosonic_env_manifest.csv `
  --force
```

To run all three at one: 

```powershell
realtimevenv\Scripts\python.exe scripts/01_preprocess.py --pair-filter data_pairs_train --out-dir data/processed/aerosonic_train --manifest data/processed/aerosonic_train_manifest.csv --force; realtimevenv\Scripts\python.exe scripts/01_preprocess.py --pair-filter data_pairs_test --out-dir data/processed/aerosonic_test --manifest data/processed/aerosonic_test_manifest.csv --force; realtimevenv\Scripts\python.exe scripts/01_preprocess.py --pair-filter data_pairs_env --out-dir data/processed/aerosonic_env --manifest data/processed/aerosonic_env_manifest.csv --force
```

Files computed in this step:
- many cached patch files: `data/processed/.../*.npy`
- manifest CSV with one row per segment: `data/processed/aerosonic_train_manifest.csv`

### Step 2: Compute split files

For AeroSonic fold-based CV:

```powershell
realtimevenv\Scripts\python.exe scripts/02_generate_splits.py `
  --manifest data/processed/aerosonic_train_manifest.csv `
  --dataset aerosonic `
  --out-dir data/splits/aerosonic
```

Files computed in this step:
- fold JSON files: `data/splits/aerosonic/*.json`

### Step 3: Train folds

```powershell
realtimevenv\Scripts\python.exe scripts/04_train.py `
  --manifest data/processed/aerosonic_train_manifest.csv `
  --splits-dir data/splits/aerosonic `
  --epochs 30 `
  --batch-size 16 `
  --max-patches 20
```

Files computed in this step:
- one run directory: `history/prosjektoppgave/<run-id>/`
- per-fold outputs (models + logs) inside the run directory
- aggregate fold metrics file: `history/prosjektoppgave/<run-id>/cv_results.json`

### Step 4: Aggregate final metrics

```powershell
realtimevenv\Scripts\python.exe scripts/05_evaluate.py `
  --cv-results history/prosjektoppgave/<run-id>/cv_results.json
```

Output:
- prints mean ± std for each metric across folds

### Ubuntu/Linux run commands (same workflow)

Use these command equivalents on Ubuntu/Linux terminals:

```bash
# 1) Preprocess AeroSonic train set
python scripts/01_preprocess.py \
  --pair-filter data_pairs_train \
  --out-dir data/processed/aerosonic_train \
  --manifest data/processed/aerosonic_train_manifest.csv \
  --force

# 2) Create CV splits
python scripts/02_generate_splits.py \
  --manifest data/processed/aerosonic_train_manifest.csv \
  --dataset aerosonic \
  --out-dir data/splits/aerosonic

# 3) Train
python scripts/04_train.py \
  --manifest data/processed/aerosonic_train_manifest.csv \
  --splits-dir data/splits/aerosonic \
  --epochs 30 \
  --batch-size 16 \
  --max-patches 20

# 4) Evaluate
python scripts/05_evaluate.py \
  --cv-results history/prosjektoppgave/<run-id>/cv_results.json
```

Cross-dataset scripts on Ubuntu/Linux:

```bash
# Build Norwegian manifest
python scripts/00_build_norwegian_manifest.py \
  --spec configs/norwegian_sessions.json \
  --manifest data/processed/norwegian_manifest.csv \
  --out-dir data/processed/norwegian \
  --force

# Build leakage-free experiments
python scripts/03_build_experiments.py \
  --aerosonic-manifest data/processed/aerosonic_train_manifest.csv \
  --norwegian-manifest data/processed/norwegian_manifest.csv \
  --out-dir insert-path-here/experiments \
  --experiment aero_only_to_norwegian \
  --experiment aero_aug_noise_to_norwegian \
  --experiment aero_plus_norwegian_with_aug

# Radius hyperparameter search (all experiments, 1.0 km to 15.0 km)
python scripts/06_search_radius_hyperparams.py \
  --aerosonic-manifest data/processed/aerosonic_train_manifest.csv \
  --norwegian-manifest data/processed/norwegian_manifest.csv \
  --out-dir data/radius_search/aero_only_to_norwegian \
  --experiment aero_only_to_norwegian \
  --radius-min 1.0 \
  --radius-max 15.0 \
  --radius-step 1.0

python scripts/06_search_radius_hyperparams.py \
  --aerosonic-manifest data/processed/aerosonic_train_manifest.csv \
  --norwegian-manifest data/processed/norwegian_manifest.csv \
  --out-dir data/radius_search/aero_aug_noise_to_norwegian \
  --experiment aero_aug_noise_to_norwegian \
  --radius-min 1.0 \
  --radius-max 15.0 \
  --radius-step 1.0

python scripts/06_search_radius_hyperparams.py \
  --aerosonic-manifest data/processed/aerosonic_train_manifest.csv \
  --norwegian-manifest data/processed/norwegian_manifest.csv \
  --out-dir data/radius_search/aero_plus_norwegian_with_aug \
  --experiment aero_plus_norwegian_with_aug \
  --radius-min 1.0 \
  --radius-max 15.0 \
  --radius-step 1.0
```

## Cross-dataset experiments (AeroSonic -> Norwegian/Skatval)

Use this when testing leakage-free transfer from AeroSonic to Norwegian/Skatval.

### A) Compute Norwegian manifest and cached patches

Create `configs/norwegian_sessions.json` from `configs/norwegian_sessions.example.json` and fill real paths.

Then run:

```powershell
realtimevenv\Scripts\python.exe scripts/00_build_norwegian_manifest.py `
  --spec configs/norwegian_sessions.json `
  --manifest data/processed/norwegian_manifest.csv `
  --out-dir data/processed/norwegian `
  --force
```

Files computed in this step:
- `data/processed/norwegian/**/*.npy`
- `data/processed/norwegian_manifest.csv`

### B) Compute leakage-free experiment folds

```powershell
realtimevenv\Scripts\python.exe scripts/03_build_experiments.py `
  --aerosonic-manifest data/processed/aerosonic_train_manifest.csv `
  --norwegian-manifest data/processed/norwegian_manifest.csv `
  --out-dir insert-path-here/experiments `
  --experiment aero_only_to_norwegian `
  --experiment aero_aug_noise_to_norwegian `
  --experiment aero_plus_norwegian_with_aug
```

Files computed in this step (per experiment, per fold):
- `data/experiments/<experiment>/fold_*/manifest.csv`
- `data/experiments/<experiment>/fold_*/split.json`
- `data/experiments/<experiment>/fold_*/augmented/*.npy` (for augmentation experiments)

**Reusing augmented files across experiments (recommended)**

The experiment builder now supports a `cached_augmented_dir` parameter to reuse pre-computed augmented files across different experiments without regeneration.

**How to set it up:**

1. After computing augmented files for one experiment (e.g., `aero_aug_noise_to_norwegian`), they are stored by radius:
```
E:\data\augmented_cache\radius_1km\fold_0_test_0\augmented\
E:\data\augmented_cache\radius_1km\fold_1_test_1\augmented\
E:\data\augmented_cache\radius_2km\fold_0_test_0\augmented\
...
```

2. When building experiments for the second experiment (e.g., `aero_plus_norwegian_with_aug`), point to the cached directory. For example, in your Python code:
```python
from src.datasets.experiment_builder import build_leakage_free_cv_experiments

build_leakage_free_cv_experiments(
    aerosonic_manifest="data/processed/aerosonic_train_manifest.csv",
    norwegian_manifest="data/processed/norwegian_manifest.csv",
    out_dir="data/experiments",
    experiment="aero_plus_norwegian_with_aug",
    cached_augmented_dir="E:/data/augmented_cache/radius_1km",  # Reference cached files
)
```

3. The augmented files will be reused instead of regenerated, saving computation time and disk space.

The function checks for files in `cached_augmented_dir / fold_{fold_id}_test_{test_group} / augmented/` and only generates new files if the cache misses.


### C) Train one cross-dataset experiment

```powershell
realtimevenv\Scripts\python.exe scripts/04_train.py `
  --splits-dir data/experiments/aero_aug_noise_to_norwegian `
  --epochs 30 `
  --batch-size 16 `
  --max-patches 20
```

The trainer automatically uses the fold-local `manifest.csv` located next to each `split.json`.

### D) Evaluate radius-search results

Use this after `scripts/06_search_radius_hyperparams.py` has finished.

**Overall results for the full radius search**

```powershell
.venv-wsl/bin/python scripts/05_evaluate.py `
  --results /mnt/e/data/radius_search/aero_aug_noise_to_norwegian/radius_search_summary.json
```

**Performance of one specific fold across all radii**

```powershell
.venv-wsl/bin/python scripts/05_evaluate.py `
  --results /mnt/e/data/radius_search/aero_aug_noise_to_norwegian/radius_search_summary.json `
  --fold-id 0
```

**Performance for one specific radius**

```powershell
.venv-wsl/bin/python scripts/05_evaluate.py `
  --results /mnt/e/data/radius_search/aero_aug_noise_to_norwegian/radius_5km/radius_summary.json
```

The evaluator prints mean test accuracy, precision, recall, AUC, and loss. For the fold-specific view, it also prints which radius was best for that fold.

### E) Aggregate cross-dataset metrics

```powershell
realtimevenv\Scripts\python.exe scripts/05_evaluate.py `
  --cv-results history/prosjektoppgave/<run-id>/cv_results.json
```

### Leakage rule used by the builder

For each Norwegian test fold:
- test fold is untouched
- augmentation noise is sampled only from the remaining training folds
- held-out fold is never used as augmentation noise

### E) Radius hyperparameter search for Skatval

If you want to compare radii from 1.0 to 15.0 in each cross-validation fold, first preprocess the radius-specific Skatval CSVs and rebuild `data/processed/norwegian_manifest.csv` so it contains radius-tagged rows (`radius_km` or pair names ending with `_Xkm`).

Patch caching note:
- Radius variants reuse the same cached patch files for the same wav segment path.
- Do not use `--force` when rebuilding the Norwegian manifest if you want to avoid recomputing existing patches.

Then run:

```powershell
.venv-wsl\bin\python scripts/06_search_radius_hyperparams.py `
  --aerosonic-manifest data/processed/aerosonic_train_manifest.csv `
  --norwegian-manifest data/processed/norwegian_manifest.csv `
  --out-dir data/radius_search/aero_only_to_norwegian `
  --experiment aero_only_to_norwegian `
  --radius-min 1.0 `
  --radius-max 15.0 `
  --radius-step 1.0

.venv-wsl\bin\python scripts/06_search_radius_hyperparams.py `
  --aerosonic-manifest data/processed/aerosonic_train_manifest.csv `
  --norwegian-manifest data/processed/norwegian_manifest.csv `
  --out-dir data/radius_search/aero_aug_noise_to_norwegian `
  --experiment aero_aug_noise_to_norwegian `
  --radius-min 1.0 `
  --radius-max 15.0 `
  --radius-step 1.0

.venv-wsl\bin\python scripts/06_search_radius_hyperparams.py `
  --aerosonic-manifest data/processed/aerosonic_train_manifest.csv `
  --norwegian-manifest data/processed/norwegian_manifest.csv `
  --out-dir data/radius_search/aero_plus_norwegian_with_aug `
  --experiment aero_plus_norwegian_with_aug `
  --radius-min 1.0 `
  --radius-max 15.0 `
  --radius-step 1.0
```

The script trains one model per fold and per radius, selects the best radius by mean validation AUC, and writes a summary JSON file to the output directory.

## If problems with cuda

Run this to uninstall possible CPU-only torch
```bash
python -m pip uninstall -y torch torchvision torchaudio
```

Run this to install correct torch
```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio
```


## Dataset

### Download AeroSonicDB-YPAD0523

Follow README-file in following repo to download the dataset. 
```bash
git clone https://github.com/aerosonicdb/AeroSonicDB-YPAD0523.git
```
Convert the ground truth files to the correct format by editing the "gt_dir" variable to the directory of the file "sample_meta.csv". The variable "gt_dir" is found on line 7 of sound-event-detection-aircrafts/dataset/AeroSonicDB/conversion.py. 
Run the conversion:

```bash
./realtimevenv/bin/python dataset/AeroSonicDB/conversion.py
```
To use the converted dataset in experiments, configure `data_pairs_train` and `data_pairs_test` in `settings.py` as described in the "End-to-end experiments" section.

### By use of other datasets follow these guidelines

#### GT-files
The program expects the following format of ground truth files:

```bash
filename    start_time  end_time    class
-5QrBL6MzLg_60.000_70.000.wav	0.917	2.029	Train horn
```
Youtube ID (5QrBL6MzLg) of video from 60 to 70 seconds, whereof a Train horn is present from 0.917 s to 2.029 s in the extracted 10 second clip.

#### Annotation
To annotate wav-files, label-studio was used. 
To annotate, run this in a terminal
```bash
label-studio
```
The annotations should then be downloaded as a csv-file, but need to be converted to the correct format using data/processing/gt_processing.py.

#### Wav-files
The files can not consist of spaces. If they do, check out data/processing/audio_name_processing.py and change 'audio_folder' to the folder you want to check for wav-files with spacings. 

## Legacy training note

The old `python -m train` path is kept for backwards compatibility, but thesis experiments should use:
- `scripts/01_preprocess.py`
- `scripts/02_generate_splits.py`
- `scripts/04_train.py`
- `scripts/05_evaluate.py`


## Demonstration of detection
The demonstration can be found in demonstration/. 

The folders in this directory is an interactive- and a regular-demonstration of the detection of cars in wav-files. 

The 'interactive' demonstation, is the one active and up to date. It is not guaranteed that the 'regular' demonstation is comptatibel to the current version of the program.

The most recently trained model is automatically chosen from history/. 

### Set duration of detection
'wav_path' is the path to the wav file you want to detect and can be found on line 80 in demonstration/interactive/SED.py. You can define how much of the audio you want to process by editing the variabels 'start_time' and 'end_time' on line 84 and 85. 

For testing, you can use 'aircraft-248663.wav' in 'sound-event-detection-aircrafts/dataset/test/'. 


## Run annotation of new dataset

```bash 
python -m dataset.annotate
```

### Run the demonstration
cd into sound-event-detection-aircrafts in the terminal, and run:
```bash 
python demonstration/interactive/SED.py
```
Mac: 
```bash
./realtimevenv/bin/python demonstration/interactive/SED.py
```

### Generate Skatval dataset summary + thesis figures

Use this script when your Skatval sessions are stored on an external drive.

```bash
python scripts/03_dataset_statistics.py --dataset-root "D:/Skatval" --radius-km 3.0
```

Optional output path:

```bash
python scripts/03_dataset_statistics.py --dataset-root "D:/Skatval" --radius-km 3.0 --output-dir outputs/dataset_statistics
```

The script exports:
- `outputs/dataset_statistics/tables/dataset_summary_by_session.csv`
- `outputs/dataset_statistics/tables/events_per_location.csv`
- `outputs/dataset_statistics/tables/class_distribution_by_session.csv`
- `outputs/dataset_statistics/figures/*.png`

### How the code works

The wav-file is either preprocessed into data-patches that are then fed into the model for getting the prediction, or it is loaded from cache, depending on the variable 'FORCE_RELOAD_SED' in settings.py

After the prediction and spectrogram is loaded, it is directed to the Plotter in demonstration/interactive/plot.py. 

The plot depends on the number of classes, and class names, both defined in settings. As they should be the same for the model, in which also is configred on settings. 


## Modify model

### Change classes to detect

To modify the classes to visualize in the plot, change the event's ids in the file `settings.py` at the line 44:

```python
PLT_CLASSES = [0,132,420,494] # Speech, Music, Explosion, Silence 
```

You can find the full list of 521 audio events in `keras_yamnet\yamnet_class_map.csv`. It follows the list of the first 50 audio events:

    0, Speech
    1, Child speech, kid speaking
    2, Conversation
    3, Narration, monologue
    4, Babbling
    5, Speech synthesizer
    6, Shout
    7, Bellow
    8, Whoop
    9, Yell
    10, Children shouting
    11, Screaming
    12, Whispering
    13, Laughter
    14, Baby laughter
    15, Giggle
    16, Snicker
    17, Belly laugh
    18, Chuckle, chortle
    19, Crying, sobbing
    20, Baby cry, infant cry
    21, Whimper
    22, Wail, moan
    23, Sigh
    24, Singing
    25, Choir
    26, Yodeling
    27, Chant
    28, Mantra
    29, Child singing
    30, Synthetic singing
    31, Rapping
    32, Humming
    33, Groan
    34, Grunt
    35, Whistling
    36, Breathing
    37, Wheeze
    38, Snoring
    39, Gasp
    40, Pant
    41, Snort
    42, Cough
    43, Throat clearing
    44, Sneeze
    45, Sniff
    46, Run
    47, Shuffle
    48, Walk, footsteps
    49, Chewing, mastication
    50, Biting


### Other modifications
Modifications can be made in settings.py or keras_yamnet/params.py.



# Tips and tricks 

#SELFMADE - Configurations of main settings made by Ingeborg, use at own risk. I think they are correct. Not 100%.

## If you want to check for large files before commiting, run this in terminal:

```bash
git diff --cached --name-only | ForEach-Object { 
    if (Test-Path $_) { 
        @{File = $_; SizeMB = [math]::Round((Get-Item $_).Length / 1MB, 2)} 
    } 
} | Sort-Object SizeMB -Descending | Select-Object -First 5 | Format-Table -AutoSize
```

## Take back last commit

```bash
git reset --soft HEAD~1
```

## Should be done 

- Data should be loaded from the same function in demonstration/interactive/SED.py and train.py
- settings.py and keras_yamnet/params.py should either be merged, or have a clear separated meaning.

# Notes


For annotation: 
- Audacity
- Label studio

Future improvements: 
- Per-Channel Energy Normalization (PCEN) ? (https://www.kaggle.com/code/mauriciofigueiredo/methods-for-sound-noise-reduction?utm_source=chatgpt.com)
- Low-pass/High-pass/Band-pass or Wiener filter for background noise
- Localization: range doppler map

