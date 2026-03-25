import numpy as np
import librosa
from matplotlib import pyplot as plt
from scipy.signal import butter, lfilter
from scipy import signal
from sklearn.decomposition import FastICA
import tensorflow as tf


from .features import mel
from . import params

def ica_separate(signal1: np.ndarray, signal2: np.ndarray):

    X = np.column_stack((signal1, signal2))

    ica = FastICA(
        n_components=len(X[0]),
        whiten="unit-variance",
        max_iter=5000,
        tol=1e-4,
        random_state=0
    )
    S_est = ica.fit_transform(X)

    # --- Step 3: Normalize recovered signals ---
    S_est = S_est / np.max(np.abs(S_est), axis=0)
    s1, s2 = S_est.T
    
    return s1, s2

def bandpass_filter(data, lowcut=40, highcut=167, order=5):
    """ The lowcut and highcut default frequencies are based on the most powerful bands of the airfraft recordings in synthetic_data_generations.ipynb """ 
    sos = butter(order, [lowcut, highcut], btype='bandpass', fs= params.SAMPLE_RATE, output='sos') 
    filtered_data = signal.sosfilt(sos, data)

    return filtered_data

def preprocess_input(waveform1: np.ndarray, sr1: int, apply_filter: str = None):
    
    assert waveform1.dtype == np.int16, 'Bad sample type: %r' % waveform1.dtype
    waveform1 = waveform1 / tf.int16.max 
    waveform1 = waveform1.astype(np.float32)

    if sr1 != params.SAMPLE_RATE:
        waveform1 = librosa.resample(waveform1, orig_sr=sr1, target_sr=params.SAMPLE_RATE, axis=0)


    # Filter the waveform if specified
    if apply_filter == 'bandpass':
        waveform1 = bandpass_filter(waveform1)

    elif apply_filter is not None:
        raise ValueError(f'Unknown apply_filter value: {apply_filter}')

    # Generate mel spectrogram and data patches
    mel_spec = mel(waveform1, params.SAMPLE_RATE)
    data_patches = [mel_spec[i:i + params.PATCH_FRAMES] for i in range(0, mel_spec.shape[0] - params.PATCH_FRAMES + 1, params.PATCH_HOP_FRAMES)]
    data_patches = np.stack(data_patches) # shape: (num_patches, PATCH_FRAMES, n_bands)

    return data_patches, mel_spec


def visualize(data1, data2, sample_rate):

    n1 = len(data1)
    frequencies1 = np.fft.rfftfreq(n1, d=1/sample_rate)
    fft_magnitude1 = np.abs(np.fft.rfft(data1))

    n2 = len(data2)
    frequencies2 = np.fft.rfftfreq(n2, d=1/sample_rate)
    fft_magnitude2 = np.abs(np.fft.rfft(data2))

    plt.figure(figsize=(12, 6))
    plt.plot(frequencies1, fft_magnitude1, label='Filtered', alpha=0.7)
    plt.plot(frequencies2, fft_magnitude2, label='Original', alpha=0.7)
    plt.title('Frequency Domain Comparison')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.xlim(0, sample_rate / 2)
    plt.legend()
    plt.tight_layout()
    plt.show()
  