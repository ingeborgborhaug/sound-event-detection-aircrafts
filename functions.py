import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from keras_yamnet import params
from keras_yamnet.yamnet import YAMNet
import keras_yamnet.preprocessing as kp
from keras_yamnet.postprocessing import postprocess_output

from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import PredefinedSplit

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import torch
import soundfile as sf
import platform
import time  
from tqdm import tqdm
import pickle
from pathlib import Path
import os

import h5py
from sklearn.utils import shuffle
from dataset import gt_conversion_functions as cf
import settings
import importlib
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import matplotlib as mpl

importlib.reload(settings)
importlib.reload(kp)


mpl.rcParams.update({
    "text.usetex": False,
    "font.family": 'Times New Roman',
    # "font.serif": ["Times"],
    "axes.labelsize" : 15,
    #"legend.fontsize": 50,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    #"mathtext.fontset": "cm",
    #"axes.unicode_minus": False,
    #"text.latex.preamble": r"\usepackage{amsmath}",  # <-- add this})
})
FIGSIZE = (14, 4)


def merge_skatval_datasets(x1, y1, folds1, x2, y2, folds2, x3, y3, folds3):
    X = np.concatenate([x1, x2, x3], axis=0)
    y = np.concatenate([y1, y2, y3], axis=0)
    if folds1 is not None and folds2 is not None and folds3 is not None:
        folds = np.concatenate([folds1, folds2, folds3], axis=0)
    else:
        folds = None
    return X, y, folds

    
def save_arrays_to_cache(x, y, folds, cache_file):
    with h5py.File(cache_file, 'w') as f:
        dset_x = f.create_dataset('x', shape=x.shape, dtype=x.dtype)
        dset_y = f.create_dataset('y', shape=y.shape, dtype=y.dtype)
        for i in tqdm(range(x.shape[0]), desc='Saving x to cache'):
            dset_x[i] = x[i]
        for i in tqdm(range(y.shape[0]), desc='Saving y to cache'):
            dset_y[i] = y[i]
        if folds is not None:
            dset_folds = f.create_dataset('folds', shape=folds.shape, dtype=folds.dtype)
            for i in tqdm(range(folds.shape[0]), desc='Saving folds to cache'):
                dset_folds[i] = folds[i]

def load_arrays_from_cache(cache_file):
    with h5py.File(cache_file, 'r') as f:
        x = f['x'][:]
        y = f['y'][:]
        folds = f['folds'][:] if 'folds' in f else None
    return x, y, folds

""" def load_features_and_labels_from_cache(gt_paths, audios_folders, force_reload, apply_filter):

    if apply_filter == None:
        filter_name = 'Unfiltered'
    else:
        filter_name = apply_filter

    if isinstance(gt_paths, tuple):
        name = ''
        for gt_path in gt_paths:              
            base = os.path.basename(gt_path) 
            gt_name, _ = os.path.splitext(base)  
            name += gt_name + '_'

        name = name.rstrip('_')
        cache_file = os.path.dirname(gt_path) + f'{name}_{filter_name}.npz'
    else:
        cache_file = os.path.splitext(gt_paths)[0] + f'_{filter_name}' + '.npz'

    if os.path.exists(cache_file) and not force_reload:
        print(f"Loading cached result from {cache_file} ...")
        x, y, folds = load_arrays_from_cache(cache_file)
    else:
        x, y, folds = load_features_and_array_labels(gt_paths, audios_folders, apply_filter=apply_filter)

    return x, y, folds """

def load_features_and_array_labels(gt_path, audio_folders, apply_filter):
    """
    Load training data from ground truth file and preprocess it.

    Args:
        gt_file (pd.DataFrame): DataFrame containing ground truth data.
    Returns:
        input_array (np.ndarray): Array of preprocessed audio data patches.
        output_array (np.ndarray): Array of target outputs corresponding to the audio data. 
            Contains targets that are one-hot encoded vectors for each class.
    """

    audiofile_to_detection = {}  
    audiofile_to_patches = {}  
    audiofile_to_fold = {}
    audiofile_to_ignore_indices = {}  # Track ignore patches

    gt_file = pd.read_csv(gt_path, sep='\s+')
    if len(gt_file) == 0:
        print(f"Ground truth file {gt_path} is empty. Returning empty arrays.")
        return np.array([]), np.array([]), None

    has_fold = 'fold' in gt_file.columns and gt_file['fold'].astype(str).str.isdigit().all()

    # if apply_filter == 'ica':
    #     raise ValueError('ICA filtering requires two ground truth files.')
    
    for _, row in tqdm(gt_file.iterrows(), total=len(gt_file), desc='Loading gt'):

        filename = row['filename']
        starttime = row['start_time']
        endtime = row['end_time']
        class_label = row['class']
        fold = row['fold'] if has_fold else None
        
        patch_index_start = cf.sec_to_start_index(starttime)
        patch_index_end = cf.sec_to_end_index(endtime)

        if filename not in audiofile_to_detection:

            audio_path = find_file_in_folder(audio_folders, filename)

            # Create datapatches, X
            audio_data, sample_rate = sf.read(audio_path, dtype='int16')
            
            data_patches, _ = kp.preprocess_input(audio_data, sample_rate, apply_filter=apply_filter)
            audiofile_to_patches[filename] = data_patches

            # duration_seconds = len(audio_data) / sample_rate
        
            # Create labels, y
            det = np.zeros((len(data_patches), settings.N_CLASSES), dtype=np.float32)
            audiofile_to_detection[filename] = det

            # Create fold array for this audio file
            if has_fold:
                # Keep one fold id per patch (same length as data_patches).
                folds = np.full(len(data_patches), int(fold), dtype=np.int32)
                audiofile_to_fold[filename] = folds

        if class_label != 'ignore':
            audiofile_to_detection[filename][patch_index_start:patch_index_end] = class_label
        else:
            # Track ignore indices to exclude them later
            if filename not in audiofile_to_ignore_indices:
                audiofile_to_ignore_indices[filename] = []
            audiofile_to_ignore_indices[filename].extend(range(patch_index_start, patch_index_end))
    X = []
    y = []
    folds = [] if has_fold else None

    print(f' audiofile_to_detection keys: {list(audiofile_to_detection.keys())}')

    for filename in audiofile_to_detection.keys(): 
        data_patches = audiofile_to_patches[filename]
        det = audiofile_to_detection[filename]
        
        # Create mask to exclude ignore indices
        ignore_indices = audiofile_to_ignore_indices.get(filename, [])
        keep_mask = np.ones(len(det), dtype=bool)
        if ignore_indices:
            keep_mask[ignore_indices] = False
        
        # Apply mask to exclude ignore indices
        X.append(data_patches[keep_mask])
        y.append(det[keep_mask])
        
        if has_fold:
            fold = audiofile_to_fold[filename]
            folds.append(fold[keep_mask])

    X = np.concatenate(X, axis=0)  # shape: (num_patches, PATCH_FRAMES, n_bands)
    y = np.concatenate(y, axis=0)  # shape: (num_patches, N_CLASSES)
    if has_fold:
        folds = np.concatenate(folds, axis=0)  # shape: (num_patches,)
    
    return X, y, folds

def find_file_in_folder(folders, filename):
    audio_path = None
    for audio_folder in folders:
        candidate = os.path.join(audio_folder, filename)
        if os.path.exists(candidate):
            audio_path = candidate
            break

    if audio_path is None:
        raise FileNotFoundError(f'Audio file {filename} not found in any of the specified folders : \n {folders}')
    
    return audio_path

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

def get_data_from_dict(data_dict, force_reload=False, apply_filter=None):
    X = None
    y = None
    folds = None

    if len(data_dict) != 1:
        raise ValueError("Expected exactly one entry in data_dict, but got multiple. Please ensure data_dict contains only one gt_path and corresponding audio_folders.")

    gt_path, audios_folders = next(iter(data_dict.items()))

        
    if apply_filter == None:
        filter_name = 'Unfiltered'
    else:
        filter_name = apply_filter

    if isinstance(gt_path, tuple):
        name = ''
        for gt_path in gt_path:              
            base = os.path.basename(gt_path) 
            gt_name, _ = os.path.splitext(base)  
            name += gt_name + '_'

        name = name.rstrip('_')
        cache_file = os.path.dirname(gt_path) + f'{name}_{filter_name}.npz'
    else:
        cache_file = os.path.splitext(gt_path)[0] + f'_{filter_name}' + '.npz'

    if os.path.exists(cache_file) and not force_reload:
        X, y, folds = load_arrays_from_cache(cache_file)
    else:
        X, y, folds = load_features_and_array_labels(gt_path, audios_folders, apply_filter=apply_filter)
        save_arrays_to_cache(X, y, folds, cache_file)

    # print(f'Data 2: X shape: {X.shape}, y shape: {y.shape}') #  X shape: (191, 2, 96, 64), y shape: (191, 1)
    # input shape = (96, 64)
    # N, C, F = X.shape # X_embeddings.shape : (samples, channels, input) = (1904, 2, 1024) 
    # X_flat = X.reshape(N, C * F)   # (N, C*F)

    return X, y, folds

def get_device():
    # 1) NVIDIA GPU (Windows/Linux/macOS w/ eGPU): fastest if available
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"

    # 2) Apple Silicon GPU (macOS M1/M2/M3)
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        # Optional: allow silent CPU fallback for unsupported ops on MPS
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps"), "mps"

    # 3) DirectML (Windows + AMD/Intel/NVIDIA via the 'torch-directml' package)
    #    pip install torch-directml   (import torch_directml)  -> a separate package
    try:
        if platform.system() == "Windows":
            import torch_directml  # noqa: F401
            dml = torch_directml.device()
            return dml, "directml"
    except Exception:
        pass

    # 4) CPU fallback (works everywhere)
    return torch.device("cpu"), "cpu"

import matplotlib.pyplot as plt
import numpy as np
import os


def save_pr_curves(pr_curves, title, filename):
    plt.figure(figsize=(5, 5))
    for i, (p, r) in enumerate(pr_curves, start=1):
        plt.plot(r, p, label=f"fold {i}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename, format="pdf")
    plt.show()

def plot_and_save_ap_scores(ap_scores_val, ap_scores_test, ap_scores_env, timestr):
    """
    Plot and save Average Precision scores across folds.
    
    Args:
        ap_scores_val: List of AP scores for validation set
        ap_scores_test: List of AP scores for test set
        ap_scores_env: List of AP scores for environmental set
        timestr: Timestamp string for directory name
    """
    folds = np.arange(1, len(ap_scores_test) + 1)
    fig_dir = f"history/{timestr}"
    os.makedirs(fig_dir, exist_ok=True)

    # Plot AP scores
    plt.figure(figsize=(6, 4))
    plt.plot(folds, ap_scores_val, marker="o")
    plt.plot(folds, ap_scores_test, marker="o")
    plt.plot(folds, ap_scores_env, marker="o")
    plt.xlabel("Fold")
    plt.ylabel("Average Precision (mAP macro)")
    plt.title("Average Precision across folds")
    plt.legend(["Validation", "Test", "Environmental"])
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/ap_across_folds.pdf", format="pdf")
    plt.show()

    # Save summary
    summary_path = f"{fig_dir}/map_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Mean Average Precision (mAP) summary\n")
        f.write("=" * 40 + "\n\n")

        f.write(f"Validation set:\n")
        f.write(f"  mAP = {np.mean(ap_scores_val):.4f} ± {np.std(ap_scores_val):.4f}\n\n")
        print("mAP val  :", np.mean(ap_scores_val),  "+/-", np.std(ap_scores_val))

        f.write(f"Test set:\n")
        f.write(f"  mAP = {np.mean(ap_scores_test):.4f} ± {np.std(ap_scores_test):.4f}\n\n")
        print("mAP test :", np.mean(ap_scores_test), "+/-", np.std(ap_scores_test))

        f.write(f"Environmental set:\n")
        f.write(f"  mAP = {np.mean(ap_scores_env):.4f} ± {np.std(ap_scores_env):.4f}\n")
        print("mAP env  :", np.mean(ap_scores_env),  "+/-", np.std(ap_scores_env))

def save_variable(filename, variable, timestr):
    with open(f'history/{timestr}/{filename}.pk1', "wb") as f:
        pickle.dump(variable, f)
    print(f"Saved variable to history/{timestr}/{filename}.pk1")

def save_ap_and_pr_curves(ap_scores, pr_curves, dataset_name, timestr):
    os.makedirs(f'history/{timestr}', exist_ok=True)
    save_variable(f"ap_scores_{dataset_name.lower()}", ap_scores, timestr)
    save_variable(f"pr_curves_{dataset_name.lower()}", pr_curves, timestr)
    print(f"Saving AP scores and PR curves for {dataset_name} in history/{timestr}/ ...")


def plot_losses_continuous(train_losses, val_losses, timestr, save_path=None, max_epochs=None, y_lim=100):
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    pdf_path = f'history/{timestr}/loss_plot.pdf'
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f'Saved loss plot in vector PDF: {pdf_path}')


def evaluate_dataset(
    model,
    X,
    y,
    dataset_name,
    output_dir="history/baseline-aerosonicdb",
    threshold=None,
    verbose=0,
    show_plots=True,
):
    if threshold is None:
        threshold = settings.PREDICTION_THRESHOLD

    os.makedirs(output_dir, exist_ok=True)

    y = np.asarray(y)
    preds = np.asarray(model.predict(X, verbose=verbose))

    ap = average_precision_score(y, preds, average="macro")
    precision, recall, _ = precision_recall_curve(y.ravel(), preds.ravel())
    pred_binary = (preds >= threshold).astype(int)

    pr_curve_path = os.path.join(output_dir, f"pr_curve_{dataset_name.lower()}.pdf")
    histogram_path = os.path.join(output_dir, f"histogram_{dataset_name.lower()}.pdf")

    print(
        f"{dataset_name} -> "
        f"AP: {ap:.4f} | "
        f"samples: {len(y)} | "
        f"true positives: {np.sum(y)} | "
        f"predicted positives: {np.sum(pred_binary)}"
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#2066a8")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision–Recall Curve ({dataset_name})")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(pr_curve_path, format="pdf")
    if show_plots:
        plt.show()
    plt.close(fig)

    print(f"Saved PR curve to {pr_curve_path}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        preds.ravel(),
        bins=50,
        range=(0, 1),
        color="#3e94be",
        edgecolor="black",
        alpha=0.7,
    )
    ax.set_xlabel("Model output score")
    ax.set_ylabel("Count")
    ax.set_title(f"Prediction Histogram ({dataset_name})")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig(histogram_path, format="pdf")
    if show_plots:
        plt.show()
    plt.close(fig)

    print(f"Saved histogram to {histogram_path}")

    return {
        "dataset_name": dataset_name,
        "predictions": preds,
        "ap": ap,
        "precision": precision,
        "recall": recall,
        "pred_binary": pred_binary,
        "pr_curve_path": pr_curve_path,
        "histogram_path": histogram_path,
    }

def load_variable(filename, timestr):
    with open(f'history/{timestr}/{filename}.pk1', "rb") as f:
        variable = pickle.load(f)
    return variable
