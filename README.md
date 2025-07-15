# Real-Time Sound Event Detection

This repository contains the python implementation of a Sound Event Detection system working in real time. 

<img src="./demo.png" style="max-width:600px; width:100%">

## Getting started

Execute the following commands to setup you project.

```bash
py -m venv realtimevenv
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

Install 
- CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit-archive
- cuDNN: https://developer.nvidia.com/rdp/cudnn-archive

For more detailed GPU setup, see; https://github.com/entbappy/Setup-NVIDIA-GPU-for-Deep-Learning

Dataset for training downloaded according to link: https://dcase.community/challenge2017/task-large-scale-sound-event-detection

## Running the code

At this point you have only to execute the demo by running the following command:

```bash 
python ./sound_event_detection.py
python ./train.py
```

### Change the classes to detect

To modify the classes to visualize in the plot, change the event's ids in the file `sound_event_detection.py` at the line 16:

```python
plt_classes = [0,132,420,494] # Speech, Music, Explosion, Silence 
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

# GT-files

bash```
-5QrBL6MzLg_60.000_70.000.wav	0.917	2.029	Train horn
```
Youtube ID (5QrBL6MzLg) of video from 60 to 70 seconds, whereof a Train horn is present from 0.917 s to 2.029 s in the extracted 10 second clip.


## Other 

#SELFMADE - Made by Ingeborg, use at own risk. I thiiiiink they are correct. Not 100%

## Tips and tricks

# If you want to check for large files before commiting, run this in terminal:

bash```
git diff --cached --name-only | ForEach-Object { 
    if (Test-Path $_) { 
        @{File = $_; SizeMB = [math]::Round((Get-Item $_).Length / 1MB, 2)} 
    } 
} | Sort-Object SizeMB -Descending | Select-Object -First 5 | Format-Table -AutoSize
```

# Take back last commit

bash```
git reset --soft HEAD~1
```