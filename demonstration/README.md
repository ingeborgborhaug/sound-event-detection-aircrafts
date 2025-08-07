## How to run demonstration

# Intro
The folders in this directory is an interactive- and a regular-demonstration of the detection of cars in wav-files. 

The 'interactive' demonstation, is the one active and up to date. It is not guaranteed that the 'regular' demonstation is comptatibel to the current version of the program.

# Choose model
'modified_model' is the model used fro detection. If you want the newest model, you can use tf.saved_model.load(f'{get_newest_timestamp_folder("history")}\modified_model').

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
