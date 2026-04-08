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
python3.10 -m venv aerovenv
source realtimevenv/bin/activate
```
NB! 
You might have to run this on mac if you have issues with building wheels:
```bash
brew install portaudio
```

If UnauthorizedAccess do this to temporarily allow scrips in your session: 

```bash
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
```

## Install requirements

```bash
pip install -r requirements.txt
```

## New thesis pipeline (cached YAMNet mel spectrograms)

The following scripts implement the new phased pipeline discussed in the implementation plan:

1. Precompute and cache YAMNet mel patches from all `data_pairs*` in settings:

```bash
python scripts/01_preprocess.py --force
```

2. Generate LOSO folds (Norwegian sessions):

```bash
python scripts/02_generate_splits.py --manifest data/processed/manifest.csv --dataset norwegian
```

3. Train Option-B model (cached mel patches -> YAMNet frontend -> temporal head):

```bash
python scripts/04_train.py --manifest data/processed/manifest.csv --splits-dir data/splits
```

4. Aggregate fold metrics:

```bash
python scripts/05_evaluate.py --cv-results history/prosjektoppgave/<run-id>/cv_results.json
```

## Run with only AerosonicDB

1. Preprocess AeroSonic training data

```bash
./realtimevenv/bin/python scripts/01_preprocess.py \
  --pair-filter data_pairs_train \
  --out-dir data/processed/aerosonic_train \
  --manifest data/processed/aerosonic_train_manifest.csv \
  --force
```

```bash
./realtimevenv/bin/python scripts/01_preprocess.py \
  --pair-filter data_pairs_test \
  --out-dir data/processed/aerosonic_test \
  --manifest data/processed/aerosonic_test_manifest.csv \
  --force
```

2. Generate AeroSonic folds

```bash
./realtimevenv/bin/python scripts/02_generate_splits.py \
  --manifest data/processed/aerosonic_train_manifest.csv \
  --dataset aerosonic \
  --out-dir data/splits/aerosonic
```

3. Train cross-validation folds

```bash
./realtimevenv/bin/python scripts/04_train.py \
  --manifest data/processed/aerosonic_train_manifest.csv \
  --splits-dir data/splits/aerosonic \
  --epochs 30 \
  --batch-size 16 \
  --max-patches 20
```

4. Evaluate CV results

```bash
./realtimevenv/bin/python scripts/05_evaluate.py \
  --cv-results history/prosjektoppgave/<run-id>/cv_results.json
```

## Cross-dataset experiments with Norwegian/Skatval

The Norwegian/Skatval dataset can be used as a leakage-free cross-validation set in three modes:

1. AeroSonic only, test on Norwegian/Skatval
2. AeroSonic with Norwegian/Skatval background-noise augmentation, test on Norwegian/Skatval
3. AeroSonic + Norwegian/Skatval training folds, with augmentation, test on Norwegian/Skatval

### 1) Preprocess the datasets

Generate one manifest for AeroSonic training data and one for Norwegian/Skatval data. The Norwegian/Skatval manifest should contain either a `fold` column or a `session`/day column; the builder will use whichever is available.

### 2) Build leakage-free experiment folds

If you want to avoid running preprocessing manually for each Norwegian/Skatval session, use the helper script [scripts/00_build_norwegian_manifest.py](scripts/00_build_norwegian_manifest.py) with a JSON spec file.

Example spec:

```json
[
  {
    "gt_path": "dataset/Skatval/loc_1_20260128_124313.csv",
    "audio_dirs": ["/path/to/norwegian/session_280126/loc_1"],
    "session": "280126",
    "location": "loc_1",
    "pair_name": "session_280126_loc_1"
  }
]
```

Run it like this:

```bash
./realtimevenv/bin/python scripts/00_build_norwegian_manifest.py \
  --spec configs/norwegian_sessions.json \
  --manifest data/processed/norwegian_manifest.csv \
  --out-dir data/processed/norwegian \
  --force
```

```bash
./realtimevenv/bin/python scripts/03_build_experiments.py \
  --aerosonic-manifest data/processed/aerosonic_train_manifest.csv \
  --norwegian-manifest data/processed/norwegian_manifest.csv \
  --out-dir data/experiments \
  --experiment aero_only_to_norwegian \
  --experiment aero_aug_noise_to_norwegian \
  --experiment aero_plus_norwegian_with_aug
```

This creates one folder per fold containing:
- `manifest.csv`
- `split.json`
- augmented `.npy` files when augmentation is enabled

If your Norwegian/Skatval GT files do not already contain a reliable `session` or `fold` column, build the manifest with explicit metadata overrides, for example:

```bash
./realtimevenv/bin/python scripts/01_preprocess.py \
  --gt-path dataset/Skatval/loc_1_20260128_124313.csv \
  --audio-dir /path/to/norwegian/audio/session_280126/loc_1 \
  --dataset-override norwegian \
  --session-override 280126 \
  --location-override loc_1 \
  --manifest data/processed/norwegian_manifest.csv \
  --append-manifest
```

Repeat once per Norwegian/Skatval session/day and audio location, then pass the resulting manifest to the experiment builder.

### 3) Train a chosen experiment

```bash
./realtimevenv/bin/python scripts/04_train.py \
  --splits-dir data/experiments/aero_aug_noise_to_norwegian \
  --epochs 30 \
  --batch-size 16 \
  --max-patches 20
```

The trainer automatically uses the fold-local `manifest.csv` next to each `split.json`.

### Leakage rule

For each Norwegian/Skatval test fold:
- the test fold is untouched
- augmentation noise is sampled only from the remaining training folds
- the held-out fold is never used for augmentation

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
To implement the dataset in training, set the variable 'data_pairs_train' and 'data_pairs_test' as described in the 'Training' section of this README to the path of the converted ground truth files 'sound-event-detection-aircrafts/dataset/AeroSonicDB/gt_train.csv', and the path of the downloaded audio folder 'AeroSonicDB-YPAD0523/data/audio/raw/1'. 

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

## Training

To train a new model, simply put the desired training data in pairs of 'gt_file.csv : audio_folder' in 'data_pairs_train' in settings. Do the same for 'data_pairs_test' to set the data meant for testing. 

To train the model run: 
```bash 
python -m train
```

The data is first fed into the baseline model 'base_model'. The output of the baseline model, the embeddings, is then fed into the 'modified_model', the last layers of the transfermodel. 

When the training is done, the model is saved under the current date and time under history/. You can also find the history of the loss and f1-score of the training in the same folder of the model. 


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

