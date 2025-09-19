import numpy as np
import librosa
from matplotlib import pyplot as plt

from .features import mel
from . import params
import librosa
import numpy as np

def preprocess_input(waveform: np.ndarray, sr: int):
    # waveform = waveform / np.max(np.abs(waveform))  # Normalize waveform
    
    if not waveform.shape == (waveform.shape[0],):
        if not np.all(waveform[:, 0] == 0):
            waveform = waveform[:, 0]  
            print(f'Causion! Waveform is stereo, chose first channel.')
        elif not np.all(waveform[:, 1] == 0):
            waveform = waveform[:, 1]
            print(f'Causion! Waveform is stereo, chose second channel.')
        else:
            raise ValueError("Waveform is stereo but both channels are silent. Cannot determine which channel to use.")
        
    if sr == params.SAMPLE_RATE:
        inp = waveform
    else: 
        inp = librosa.resample(waveform, orig_sr=sr, target_sr=params.SAMPLE_RATE, axis=0)

    """ # Filter with notch filter to remove 50Hz hum
    f0 = 50.0  
    b, a = signal.iirnotch(f0, Q=30, fs=sr)
    inp = signal.filtfilt(b, a, inp) """

    mel_spec = mel(inp, params.SAMPLE_RATE)
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
  