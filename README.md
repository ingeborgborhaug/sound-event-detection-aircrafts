# Real-Time Sound Event Detection

This repository contains the python implementation of a Sound Event Detection system with the input of wav-files. 

<img src="./demo.png" style="max-width:600px; width:100%">

# Setting up the environment

Execute the following commands to setup you project.

```bash
py -m venv realtimevenv
py -3.10 -m venv realtimevenv
realtimevenv\Scripts\activate 
```
If UnauthorizedAccess this to temporarily allow scrips in your session: 

```bash
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
```

## Install requirements

```bash
pip install -r requirements.txt
```
or 
```bash
pip install -r requirements.txt --no-cache-dir
```

You might need this to run the code on your desired GPU. 
- CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit-archive
- cuDNN: https://developer.nvidia.com/rdp/cudnn-archive

For more detailed GPU setup, see; https://github.com/entbappy/Setup-NVIDIA-GPU-for-Deep-Learning

Dataset for training downloaded according to link: https://dcase.community/challenge2017/task-large-scale-sound-event-detection

# Running the code

## Data

### GT-files
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

### Wav-files
The files can not consist of spaces. If they do, check out data/processing/audio_name_processing.py and change 'audio_folder' to the folder you want to check for wav-files with spacings. 

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


## Training
To train a new model, simply put the desired training data in pairs of 'gt_file.csv : audio_folder' in the 'data_pairs_train' in settings. Do the same for 'data_pairs_test' to set the data meant for testing. 

# Normalization
To have the model be trained on normalized data, change line 34 in keras_yamnet/preprocessing.py marked with # Normalizaton. If this line is commented out, the model runs on data that is not normalized. 

At this point you have only to execute the demo by running the following command:

```bash 
python ./train.py
```

The data is first fed into the baseline model 'base_model'. The output of the baseline model, the embeddings, is then fed into the 'modified_model', the last layers of the transfermodel. 

When the training is done, the model is saved under the current date and time under history/. You can also find the history of the loss and f1-score of the training in the same folder of the model. 

At last the testing is done on each of the specific testing dataset, separated by distance from the road. 

## Demonstration of detection
The demonstration can be found in demonstration/. 

# Intro
The folders in this directory is an interactive- and a regular-demonstration of the detection of cars in wav-files. 

The 'interactive' demonstation, is the one active and up to date. It is not guaranteed that the 'regular' demonstation is comptatibel to the current version of the program.

# Choose model
'modified_model' is the model used fro detection. If you want the newest model, you can use tf.saved_model.load(f'{get_newest_timestamp_folder("history")}\modified_model').

# Normalization
If the chosen model is trained on normalized data, the data that is to be predicted in the demonstration should also be normalized. The opposite applies for a model trained on data that is not normalized. 
To decide whether the data is normalized or not, change line 34 in keras_yamnet/preprocessing.py. By commenting the line out, the data is not normalized. 

# Set input
'wav_path' is the path to the wav file you want to detect. You can define how much of the audio you want to process by editing the variabels 'start_time' and 'end_time'.

# Run the demonstration
```bash 
python demonstration/interactive/SED.py
```
# How the code works

The wav-file is either preprocessed into data-patches that are then fed into the model for getting the prediction, or it is loaded from cache, depending on the variable 'FORCE_RELOAD_SED' in settings.py

After the prediction and spectrogram is loaded, it is directed to the Plotter in demonstration/interactive/plot.py. 

The plot depends on the number of classes, and class names, both defined in settings. As they should be the same for the model, in which also is configred on settings. 



# Tips and tricks 

#SELFMADE - Configurations of main settings made by Ingeborg, use at own risk. I thiiiiink they are correct. Not 100%

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

