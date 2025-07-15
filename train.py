import numpy as np
import pyaudio
from matplotlib import pyplot as plt
import pandas as pd

from keras_yamnet import params
from keras_yamnet.yamnet import YAMNet, class_names
from keras_yamnet.preprocessing import preprocess_input
from keras.models import Model

import tensorflow as tf

import soundfile as sf
import sounddevice as sd
import os
import time
from tqdm import tqdm

import settings

import torch
import h5py
from sklearn.model_selection import train_test_split

def save_arrays_to_cache(x, y, cache_file):
    with h5py.File(cache_file, 'w') as f:
        dset_x = f.create_dataset('x', shape=x.shape, dtype=x.dtype)
        dset_y = f.create_dataset('y', shape=y.shape, dtype=y.dtype)
        for i in tqdm(range(x.shape[0]), desc='Saving x to cache'):
            dset_x[i] = x[i]
        for i in tqdm(range(y.shape[0]), desc='Saving y to cache'):
            dset_y[i] = y[i]

def load_arrays_from_cache(cache_file):
    with h5py.File(cache_file, 'r') as f:
        x = f['x'][:]
        y = f['y'][:]
    return x, y

def load_features_and_labels_from_cache(audio_paths, gt_file_name, gt_file, force_reload_gt=False):

    cache_dir = 'cache'
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, os.path.basename(gt_file_name[:-4]) + '.npz')

    if os.path.exists(cache_file) and not force_reload_gt:
        print(f"Loading cached result from {cache_file} ...")
        x, y = load_arrays_from_cache(cache_file)
    else:
        print(f'Processing and caching in : {cache_file}')
        x, y = load_features_and_class_idx_labels(gt_file, audio_paths)
        save_arrays_to_cache(x, y, cache_file)

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
    if class_name in settings.CLASS_NAMES:
        return settings.CLASS_NAMES.tolist().index(class_name)
    else:
        return len(settings.CLASS_NAMES)-1 # Index of 'Other' class

def load_features_and_class_idx_labels(gt_file, audio_path_folders):
    """
    Load training data from ground truth file and preprocess it.

    Args:
        gt_file (pd.DataFrame): DataFrame containing ground truth data.
    Returns:
        input_array (np.ndarray): Array of preprocessed audio data patches.
        output_array (np.ndarray): Array of target outputs corresponding to the audio data. 
            Contains targets that are one-hot encoded vectors for each class.
    """
    input_patches = []
    output_targets = []
    patch_map = {}  # (audiofile, patch_idx) -> class index 
    n_files_not_found = 0

    for _, row in tqdm(gt_file.iterrows(), total=len(gt_file), desc='Loading gt'):
        audiofile = 'Y' + row['filename']
        starttime = row['start_time']
        endtime = row['end_time']
        class_name = row['class']
        audio_found = False

        for audio_path_folder in audio_path_folders:
            audio_path = os.path.join(audio_path_folder, audiofile)
            if os.path.exists(audio_path):
                audio_found = True
                break

        if not audio_found:
            n_files_not_found += 1
            print(f"Audio file {audiofile} not found in any of the specified folders.")
            continue

        audio_wave, sample_rate = sf.read(audio_path)
        data = preprocess_input(audio_wave, sample_rate)
        data_patches = [data[i:i + params.PATCH_FRAMES] for i in range(0, data.shape[0] - params.PATCH_FRAMES + 1, params.PATCH_HOP_FRAMES)]
        n_patches = len(data_patches)

        patch_index_start = second_to_index(starttime)
        patch_index_end = second_to_index(endtime)
        class_idx = class_name_to_index(class_name)

        for patch_idx in range(n_patches):
            patch_key = (audiofile, patch_idx)
            if patch_key not in patch_map:
                patch_map[patch_key] = None
            if (patch_index_start <= patch_idx < patch_index_end):
                patch_map[patch_key] = class_idx

    if n_files_not_found > 0:
        print(f"Total {n_files_not_found} audio files were not found in the specified folders. Working with {len(patch_map)} patches.")

    # Now collect all patches and targets in order
    for (audiofile, patch_idx), target in patch_map.items():

        if target is None:
            target = len(settings.CLASS_NAMES)-1

        audio_path = os.path.join(audio_path_folder, audiofile)
        audio_wave, sample_rate = sf.read(audio_path)
        data = preprocess_input(audio_wave, sample_rate)
        patch = data[patch_idx * params.PATCH_HOP_FRAMES : patch_idx * params.PATCH_HOP_FRAMES + params.PATCH_FRAMES]
        input_patches.append(patch)
        output_targets.append(target)

    input_array = np.stack(input_patches)
    output_array = np.stack(output_targets)
    return input_array, output_array

def load_features_and_one_hot_labels(gt_file, audio_path_folder): #TODO: NOT IN USE? Does not use one-hot encoding anymore.
    """
    NOT IN USE. CONVERTED TO load_features_and_class_idx_labels FOR 1D DATA.


    Load training data from ground truth file and preprocess it.

    Args:
        gt_file (pd.DataFrame): DataFrame containing ground truth data.
    Returns:
        input_array (np.ndarray): Array of preprocessed audio data patches.
        output_array (np.ndarray): Array of target outputs corresponding to the audio data. 
            Contains targets that are one-hot encoded vectors for each class.
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
                patch_map[patch_key] = np.zeros(settings.N_CLASSES, dtype=np.float32)
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

def load_wav_to_dataformat(audio_path):
    audio_wave, sample_rate = sf.read(audio_path)
    data = preprocess_input(audio_wave, sample_rate)
    total_length = len(data)
    num_segments = total_length // settings.WINDOW_SIZE
    windows = np.empty((num_segments, settings.WINDOW_SIZE, data.shape[1]), dtype=np.float32)
    window_index = 0
    for start in range(0, data.shape[0] - settings.WINDOW_SIZE + 1, settings.STRIDE):
        window = data[start:start+settings.WINDOW_SIZE, :] # shape: (96, 64)
        windows[window_index] = window
        window_index += 1

    return windows

def filter_data_by_class(gt_file, class_indices):
    """
    Filter the ground truth file to only include specified classes.
    
    Args:
        gt_file (pd.DataFrame): DataFrame containing ground truth data.
        class_indices (list): List of class indices to filter by.
    
    Returns:
        pd.DataFrame: Filtered DataFrame containing only the specified classes.
    """
    class_names = settings.YAMNET_CLASSES[class_indices]

    return gt_file[gt_file['class'].isin(class_names)].reset_index(drop=True)

def visualize_and_save_history(history, timestamp):
    """
    Visualize training history.
    
    Args:
        history (tf.keras.callbacks.History): History object containing training metrics.
    """
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    # Loss subplot
    axs[0].plot(history.history['loss'], label='Training Loss', color='#003366')  # Dark blue
    axs[0].plot(history.history['val_loss'], label='Validation Loss', color='#66b3ff')  # Light blue
    axs[0].set_title('Training and Validation Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid()

    # Accuracy subplot
    axs[1].plot(history.history['accuracy'], label='Training Accuracy', color='#006400')  # Dark green
    axs[1].plot(history.history['val_accuracy'], label='Validation Accuracy', color='#90ee90')  # Light green
    axs[1].set_title('Training and Validation Accuracy')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    axs[1].grid()

    plt.tight_layout()
    plt.show()

    os.makedirs('history', exist_ok=True)
    os.makedirs(f'history/{timestamp}', exist_ok=True)
    fig.savefig(f'history/{timestamp}/history.png')

def save_model(model, timestamp):
    os.makedirs('history', exist_ok=True)
    os.makedirs(f'history/{timestamp}', exist_ok=True)
    model.save(f'history/{timestamp}/modified_model.h5', include_optimizer=False)

#################### CHECK GPU #####################

print("Number of GPU: ", torch.cuda.device_count())
print("GPU Name: ", torch.cuda.get_device_name())

device = torch.device('cuda:0')
print('Using device:', device)

#################### TRAINING-DATA ####################

""" gt_file_name = settings.gt_folder_path + '/groundtruth_strong_label_testing_set.csv'
gt_file = pd.read_csv(gt_file_name, delimiter='\t')
get_file_filtered = filter_data_by_class(gt_file, settings.PLT_CLASSES)

X, y = load_features_and_labels_from_cache(settings.audio_folder_path, gt_file_name, get_file_filtered, force_reload_gt=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=settings.TEST_SIZE, random_state=42)
"""

X = None
y = None

for gt_file_name in settings.gt_file_names:
    gt_file = pd.read_csv(gt_file_name, delimiter='\t')
    # get_file_filtered = filter_data_by_class(gt_file, settings.PLT_CLASSES)

    X_temp, y_temp = load_features_and_labels_from_cache(settings.audio_folder_paths, gt_file_name, gt_file, force_reload_gt=True)

    print(f'X_temp shape: {X_temp.shape}, y_temp shape: {y_temp.shape}')
    if X is None and y is None:
        X = X_temp
        y = y_temp
    else:
        X = np.concatenate((X, X_temp), axis=0)
        y = np.concatenate((y, y_temp), axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=settings.TEST_SIZE, random_state=42)

#################### MODEL #####################
    
yamnet_model = YAMNet(weights='keras_yamnet/yamnet.h5')
base_model = Model(
    inputs=yamnet_model.input,
    outputs=yamnet_model.get_layer('global_average_pooling2d').output
)
yamnet_model.trainable = False
base_model.trainable = False

modified_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1024), dtype=tf.float32,
                        name='input_embedding'),
    tf.keras.layers.Dense(128, activation='relu', 
                          kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(settings.N_CLASSES)
], name='modified_model')

modified_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer="adam",
                metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                        patience=3,
                                        restore_best_weights=True)

#################### TRAINING ####################

assert base_model.trainable == False, "Base model should not be trainable."

X_embeddings_train = base_model.predict(X_train, verbose=1)
X_embeddings_test = base_model.predict(X_test, verbose=1)

time.sleep(1)  # Sleep to ensure the model is ready for training

y_pred_train = modified_model.predict(X_embeddings_train, verbose=1)

history = modified_model.fit(x= X_embeddings_train,
                             y= y_train,
                             shuffle=True,
                             #batch_size=32,
                             #class_weight={0: 1.0, 1: 1.0, 2: 0.7, 3: 0.7},
                             epochs=20,
                             validation_split=settings.VAL_SIZE/settings.TRAIN_SIZE,
                             callbacks=callback)

timestr = time.strftime("%Y%m%d-%H%M%S")
save_model(modified_model, timestr)
visualize_and_save_history(history, timestr)

loss, accuracy = modified_model.evaluate(x=X_embeddings_test, 
                                         y=y_test, 
                                         return_dict=True, 
                                         verbose=1)


#################### PREDICT ####################

#x_test = load_wav_to_dataformat(wav_file_path_test)
#embeddings_layer = base_models.predict(x_test, verbose=1)
#prediction_layer = modified_model.predict(embeddings_layer, verbose=1)



