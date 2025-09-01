import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from keras_yamnet import params
from keras_yamnet.yamnet import YAMNet
from keras_yamnet.preprocessing import preprocess_input
from keras_yamnet.postprocessing import postprocess_output
from keras.models import Model

import tensorflow as tf

import soundfile as sf
import os
import time
from tqdm import tqdm

import settings

import torch
import h5py
from sklearn.utils import shuffle
from sed_eval.sound_event import EventBasedMetrics, SegmentBasedMetrics

import random

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

def load_features_and_labels_from_cache(gt_path, audios_folder):

    cache_file = os.path.splitext(gt_path)[0] + '.npz'

    if os.path.exists(cache_file) and not settings.FORCE_RELOAD_GT_TRAIN:
        print(f"Loading cached result from {cache_file} ...")
        x, y = load_arrays_from_cache(cache_file)
    else:
        print(f'Processing and caching in : {cache_file}')
        x, y = load_features_and_array_labels(gt_path, audios_folder)
        save_arrays_to_cache(x, y, cache_file)

    return x, y

def second_to_index(sec):
    """
    Convert seconds to index in variable output in when loading data from gt.
    """
    return int(sec // params.PATCH_HOP_SECONDS)

def class_name_to_index(class_name):
    """
    Convert class name to index based on the class names defined in the YAMNet model.
    """
    if class_name in settings.CLASS_NAMES:
        return settings.CLASS_NAMES.tolist().index(class_name)
    else:
        return len(settings.CLASS_NAMES)-1 # Index of 'Other' class

def load_features_and_array_labels(gt_path, audios_folder):
    """
    Load training data from ground truth file and preprocess it.

    Args:
        gt_file (pd.DataFrame): DataFrame containing ground truth data.
    Returns:
        input_array (np.ndarray): Array of preprocessed audio data patches.
        output_array (np.ndarray): Array of target outputs corresponding to the audio data. 
            Contains targets that are one-hot encoded vectors for each class.
    """
    gt_file = pd.read_csv(gt_path, delimiter='\t')

    audiofile_to_detection = {}  
    audiofile_to_patches = {}  
    class_to_count = {key: 0 for key in settings.CLASS_NAMES}
    n_detections = 0

    for _, row in tqdm(gt_file.iterrows(), total=len(gt_file), desc='Loading gt'):
        if gt_path.startswith(settings.dcase_folder):
            audiofile = 'Y' + row['filename']
        else:
            audiofile = row['filename']

        starttime = row['start_time']
        endtime = row['end_time']
        class_name = row['class']

        patch_index_start = second_to_index(starttime)
        patch_index_end = second_to_index(endtime)
        class_idx = class_name_to_index(class_name)

        if class_name in settings.CLASS_NAMES:
            class_to_count[class_name] += 1

        n_detections += 1

        if audiofile not in audiofile_to_detection:
            audio_path = os.path.join(audios_folder, audiofile)
            audio_wave, sample_rate = sf.read(audio_path)
            data_patches, _ = preprocess_input(audio_wave, sample_rate)
            audiofile_to_patches[audiofile] = data_patches

            pred = np.zeros((len(data_patches), settings.N_CLASSES), dtype=np.float32)
            pred[patch_index_start:patch_index_end+1, class_idx] = settings.GT_CONFIDENCE
            audiofile_to_detection[audiofile] = pred

        else:
            audiofile_to_detection[audiofile][patch_index_start:patch_index_end, class_idx] = settings.GT_CONFIDENCE

    print(f'Count of detections among classes: {class_to_count}, out of a total of {n_detections} detections')
    
    input_array = []
    output_array = []

    for audiofile in audiofile_to_detection: 
        data_patches = audiofile_to_patches[audiofile]
        pred = audiofile_to_detection[audiofile]

        input_array.append(data_patches)
        output_array.append(pred)
        
    input_array = np.concatenate(input_array, axis=0)  # shape: (num_patches, PATCH_FRAMES, n_bands)
    output_array = np.concatenate(output_array, axis=0)  # shape: (num_patches, N_CLASSES)
    
    return input_array, output_array

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
    axs[0].set_ylim(0, 1.5)
    axs[0].legend()
    axs[0].grid()

    # F1-SCORE  subplot
    axs[1].plot(history.history['f1_score'], label='Training f1-score', color='#006400')  # Dark green
    axs[1].plot(history.history['val_f1_score'], label='Validation f1-score', color='#90ee90')  # Light green
    axs[1].set_title('Training and Validation f1-score')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('f1-score')
    axs[1].set_ylim(0, 1)
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
    """ input_segment = tf.keras.layers.Input(shape=(), dtype=tf.float32, name='audio')
    embedding_extraction_layer = tf.keras.models.load_model(yamnet_model_name)
    _, embeddings_output, _ = embedding_extraction_layer(input_segment)
    serving_outputs = model(embeddings_output)
    serving_model = tf.keras.Model(input_segment, serving_outputs) """
    model.save(f'history/{timestamp}/modified_model', include_optimizer=False)

def predictions_to_event_list(predictions):
    """
    Convert model predictions to a list of detected events compatible with sed_eval.
    Assumes predictions are binary (0 or 1) for each class at each segment.
    """
    output = []
    predictions = np.array(predictions)
    num_segments, num_classes = predictions.shape

    for i in range(num_segments):
        for class_idx, score in enumerate(predictions[i]):
            if score == 1:
                output.append({
                    'file': 'an_audio',
                    'event_label': settings.CLASS_NAMES[class_idx],
                    'onset': i * params.PATCH_WINDOW_SECONDS,
                    'offset': (i + 1) * params.PATCH_WINDOW_SECONDS
                })

    return output

def segment_metric(y_true, y_pred): #TODO ikke i bruk? 
    segment_metrics = SegmentBasedMetrics(event_label_list=settings.CLASS_NAMES, time_resolution=params.PATCH_WINDOW_SECONDS)

    predicted_event_list = predictions_to_event_list(y_pred)
    reference_event_list = predictions_to_event_list(y_true)

    segment_metrics.evaluate(predicted_event_list, reference_event_list)

    return segment_metrics.results()

def print_metrics(metrics, title):
    print(f"\n{title}")
    overall = metrics['overall']
    print(f"  F1: {overall['f_measure']['f_measure']:.3f} | Precision: {overall['f_measure']['precision']:.3f} | Recall: {overall['f_measure']['recall']:.3f}")
    print(f"  Error Rate: {overall['error_rate']['error_rate']:.3f}")
    if 'accuracy' in overall and overall['accuracy']:
        acc = overall['accuracy'].get('accuracy', None)
        if acc is not None:
            print(f"  Accuracy: {acc:.3f}")
    print("  Class-wise F1:")
    for cls, vals in metrics['class_wise'].items():
        f1 = vals['f_measure']['f_measure']
        print(f"    {cls}: {f1:.3f}")

def get_data_from_dict(data_dict):
    X = None
    y = None

    for gt_path, audios_folder in data_dict.items():

        X_temp, y_temp = load_features_and_labels_from_cache(gt_path, audios_folder)

        if X is None and y is None:
            X = X_temp
            y = y_temp
        else:
            X = np.concatenate((X, X_temp), axis=0)
            y = np.concatenate((y, y_temp), axis=0)

    X, y = shuffle(X, y, random_state=42)

    return X, y

#################### CHECK GPU #####################

print("Number of GPU: ", torch.cuda.device_count())
print("GPU Name: ", torch.cuda.get_device_name())

device = torch.device('cuda:0')
print('Using device:', device)
print('\n')

#################### GET DATA ####################

X_train, y_train = get_data_from_dict(settings.data_pairs_train)

X_test, y_test = get_data_from_dict(settings.data_pairs_test)

#################### COMPILE MODEL #####################

yamnet_model_name = 'keras_yamnet/yamnet.h5'
yamnet_model = YAMNet(weights=yamnet_model_name)
base_model = Model(
    inputs=yamnet_model.input,
    outputs=yamnet_model.get_layer('global_average_pooling2d').output
)
yamnet_model.trainable = False
base_model.trainable = False

modified_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1024), dtype=tf.float32,name='input_embedding'),
    tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(settings.N_CLASSES, activation='sigmoid')
], name='modified_model')

modified_model.compile(
                loss='binary_crossentropy',
                optimizer="adam",
                metrics=[tf.metrics.F1Score(threshold=settings.PREDICTION_THRESHOLD, average= 'macro')] #Frame based f1-score
)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                        patience=20,
                                        restore_best_weights=True)

#################### TRAINING ####################

assert base_model.trainable == False, "Base model should not be trainable."

X_embeddings_train = base_model.predict(X_train, verbose=1)

time.sleep(1)  # Sleep to ensure the model is ready for training

history = modified_model.fit(x= X_embeddings_train,
                             y= y_train,
                             shuffle=True,
                             batch_size=64,
                             #class_weight={0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0},
                             epochs=300,
                             validation_split=settings.VAL_SIZE,
                             callbacks=[callback]
)

timestr = time.strftime("%Y%m%d-%H%M%S")
save_model(modified_model, timestr)
visualize_and_save_history(history, timestr)

#################### TESTING ####################
def check_metrics(X_test, y_test):
    X_embeddings_test = base_model.predict(X_test, verbose=1)
    y_pred_test = modified_model.predict(X_embeddings_test, verbose=1)
    y_pred_test = postprocess_output(y_pred_test)

    event_metrics = EventBasedMetrics(event_label_list=settings.CLASS_NAMES, t_collar=params.PATCH_WINDOW_SECONDS)
    segment_metrics = SegmentBasedMetrics(event_label_list=settings.CLASS_NAMES, time_resolution=params.PATCH_WINDOW_SECONDS)

    predicted_event_list_10 = predictions_to_event_list(y_pred_test)
    reference_event_list_10 = predictions_to_event_list(y_test)

    event_metrics.evaluate(predicted_event_list_10, reference_event_list_10)
    segment_metrics.evaluate(predicted_event_list_10, reference_event_list_10)

    print_metrics(event_metrics.results(), "Event-based")
    print_metrics(segment_metrics.results(), "Segment-based")

check_metrics(X_test, y_test)


