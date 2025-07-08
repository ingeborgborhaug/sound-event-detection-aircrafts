import numpy as np
import pyaudio
from matplotlib import pyplot as plt
import pandas as pd
import sounddevice as sd

from keras_yamnet import params
from keras_yamnet.yamnet import YAMNet, class_names
from keras_yamnet.preprocessing import preprocess_input
from keras.models import Model

import tensorflow as tf

from plot import Plotter

import soundfile as sf
import sounddevice as sd
import threading
import os
import pickle
import time
import math
from tqdm import tqdm

from settings import MAX_PATCHES_PER_AUDIO, PLT_CLASSES, WINDOW_SIZE, STRIDE, N_CLASSES, N_WINDOWS, WIN_SIZE_SEC

import torch
import h5py

print("Number of GPU: ", torch.cuda.device_count())
print("GPU Name: ", torch.cuda.get_device_name())


device = torch.device('cuda:0')
print('Using device:', device)

def save_arrays_with_progress(x, y, cache_file):
    with h5py.File(cache_file, 'w') as f:
        dset_x = f.create_dataset('x', shape=x.shape, dtype=x.dtype)
        dset_y = f.create_dataset('y', shape=y.shape, dtype=y.dtype)
        for i in tqdm(range(x.shape[0]), desc='Saving x to cache'):
            dset_x[i] = x[i]
        for i in tqdm(range(y.shape[0]), desc='Saving y to cache'):
            dset_y[i] = y[i]

def load_arrays_with_progress(cache_file):
    with h5py.File(cache_file, 'r') as f:
        x = f['x'][:]
        y = f['y'][:]
    return x, y

def process_and_cache(audio_path, gt_file_name, force=False):

    cache_dir = 'cache'
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, os.path.basename(gt_file_name) + '.npz')

    if os.path.exists(cache_file) and not force:
        print(f"Loading cached result for {gt_file_name}")
        x, y = load_arrays_with_progress(cache_file)
    else:
        print(f'Processing and caching: {gt_file_name}')
        gt_file = pd.read_csv(gt_file_name, delimiter='\t')
        x, y = load_training_data_from_gt(gt_file, audio_path)
        save_arrays_with_progress(x, y, cache_file)

    return x, y

def second_to_index(sec):
    """
    Convert seconds to index in variable output in when loading data from gt.
    """
    return round(int(sec / params.PATCH_WINDOW_SECONDS) - 1)

def class_name_to_index(class_name):
    """
    Convert class name to index based on the class names defined in the YAMNet model.
    """
    yamnet_classes = class_names('keras_yamnet/yamnet_class_map.csv')
    plt_classes_lab = yamnet_classes[PLT_CLASSES]
    if class_name in plt_classes_lab:
        return plt_classes_lab.tolist().index(class_name)
    else:
        return None

def load_training_data_from_gt(gt_file, audio_path_folder):
    """
    Load training data from ground truth file and preprocess it.

    Args:
        gt_file (pd.DataFrame): DataFrame containing ground truth data.
    Returns:
        input_array (np.ndarray): Array of preprocessed audio data patches.
        output_array (np.ndarray): Array of target outputs corresponding to the audio data.
    """
    input_patches = []
    output_targets = []
    patch_map = {}  # (audiofile, patch_idx) -> target array

    for _, row in tqdm(gt_file.iterrows(), total=len(gt_file), desc='Loading data from ground truth'):
        audiofile = 'Y' + row['filename']
        starttime = row['start_time']
        endtime = row['end_time']
        class_label = row['class']

        audio_path = os.path.join(audio_path_folder, audiofile)
        if not os.path.exists(audio_path):
            print(f"Audio file {audio_path} does not exist.")
            continue

        audio_wave, sample_rate = sf.read(audio_path)
        data = preprocess_input(audio_wave, sample_rate)
        data_patches = [data[i:i + params.PATCH_FRAMES] for i in range(0, data.shape[0] - params.PATCH_FRAMES + 1, params.PATCH_HOP_FRAMES)]
        n_patches = len(data_patches)

        patch_index_start = second_to_index(starttime)
        patch_index_end = second_to_index(endtime)
        class_idx = class_name_to_index(class_label)

        for patch_idx in range(n_patches):
            patch_key = (audiofile, patch_idx)
            if patch_key not in patch_map:
                patch_map[patch_key] = np.zeros(N_CLASSES, dtype=np.float32)
            if patch_index_start <= patch_idx < patch_index_end:
                patch_map[patch_key][class_idx] = 1.0

    # Now collect all patches and targets in order
    for (audiofile, patch_idx), target in patch_map.items():
        audio_path = os.path.join(audio_path_folder, audiofile)
        audio_wave, sample_rate = sf.read(audio_path)
        data = preprocess_input(audio_wave, sample_rate)
        patch = data[patch_idx * params.PATCH_HOP_FRAMES : patch_idx * params.PATCH_HOP_FRAMES + params.PATCH_FRAMES]
        input_patches.append(patch)
        output_targets.append(target)

    input_array = np.stack(input_patches)
    output_array = np.stack(output_targets)
    return input_array, output_array

#################### BASE-MODEL #####################
    
base_model = YAMNet(weights='keras_yamnet/yamnet.h5')
embedding_model = Model(
    inputs=base_model.input,
    outputs=base_model.get_layer('global_average_pooling2d').output
)
# Freeze model
embedding_model.trainable = False
base_model.trainable = False
# yamnet_classes = class_names('keras_yamnet/yamnet_class_map.csv')

#################### MODIFIED-MODEL ####################

modified_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1024), dtype=tf.float32,
                        name='input_embedding'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(N_CLASSES)
], name='modified_model')

# modified_model.summary()

modified_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer="adam",
                metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                        patience=3,
                                        restore_best_weights=True)

#################### TRAINING-DATA ####################

data_folder_path = 'data_dcase17_task4'
audio_folder_path = data_folder_path + '/unbalanced_train_segments_testing_set_audio_formatted_and_segmented_downloads'
gt_folder_path = data_folder_path + '/groundtruth_release'

""" gt_file_strong_test = pd.read_csv(gt_folder_path + '/groundtruth_strong_label_testing_set.csv', delimiter='\t')
gt_file_weak_test = pd.read_csv(gt_folder_path + '/groundtruth_weak_label_testing_set.csv', delimiter='\t')

gt_file_strong_eval = pd.read_csv(gt_folder_path + '/groundtruth_strong_label_evaluation_set.csv', delimiter='\t')
gt_file_weak_eval = pd.read_csv(gt_folder_path + '/groundtruth_weak_label_evaluation_set.csv', delimiter='\t')

gt_file_weak_train = pd.read_csv(gt_folder_path + '/groundtruth_weak_label_training_set.csv', delimiter='\t') """

gt_file_name = gt_folder_path + '/groundtruth_strong_label_testing_set.csv'
x, y = process_and_cache(audio_folder_path, gt_file_name, force=False)
print('Done processing and caching training data.')

#################### TRAINING ####################

""" for idx, row in gt_file_weak_train.iterrows():
    audio_file = 'Y' + row['filename']
    starttime = row['start_time']
    endtime = row['end_time']
    class_label = row['class']
    print(f"Processing {audio_file} from {starttime} to {endtime} for class {class_label}")
    audio_path = os.path.join(audio_folder_path, audio_file)
    
    if not os.path.exists(audio_path):
        print(f"Audio file {audio_path} does not exist.")
        continue
    
    audio_data, sample_rate = sf.read(audio_path)
    print(f'audio_data.shape: {audio_data.shape}, sample_rate: {sample_rate}')
    
    #if len(audio_data.shape) > 1:
    #    audio_data = audio_data[:, 0] 
    
    input_data = preprocess_input(audio_data, sample_rate)
    
    if STRIDE != WINDOW_SIZE:
        raise ValueError("STRIDE must be equal to WINDOW_SIZE for embeddings not to overlap.")

    data_windows = []
    subembeddings = []
    for start in range(0, input_data.shape[0] - WINDOW_SIZE + 1, STRIDE):
        data_window = input_data[start:start+WINDOW_SIZE, :] # shape: (96, 64)
        subembedding = embedding_model.predict(np.expand_dims(data_window,0))[0]

        data_windows.append(data_window)
        subembeddings.append(subembedding) 


    variables = {
        "subembeddings": subembeddings,
        "data_windows": data_windows
    }

    # Here you would typically save the embedding and label for training
    # For demonstration, we will just print them
    print(f"Processed {audio_file}: Embedding shape: {subembedding.shape}, Class: {class_label}") """

""" modified_model.fit(
    np.array(subembeddings),
    np.array([yamnet_classes.index(class_label)] * len(subembeddings)),
    epochs=1,
    verbose=1
) """



""" history = modified_model.fit(train_ds,
                       epochs=20,
                       validation_data=val_ds,
                       callbacks=callback) """

