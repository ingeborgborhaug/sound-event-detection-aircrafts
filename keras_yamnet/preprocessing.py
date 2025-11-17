import numpy as np
import librosa
from matplotlib import pyplot as plt
from scipy.signal import butter, lfilter
from scipy import signal
from sklearn.decomposition import FastICA


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

def preprocess_input(waveform1: np.ndarray, sr1: int, waveform2: np.ndarray = None, sr2: int = None, apply_filter: str = None):
    
    # # Handle stereo audio and resample if needed
    # if not waveform1.shape == (waveform1.shape[0],):
    #     if not np.all(waveform1[:, 0] == 0):
    #         waveform1 = waveform1[:, 0]  
    #         print(f'Causion! Waveform is stereo, chose first channel.')
    #     elif not np.all(waveform1[:, 1] == 0):
    #         waveform1 = waveform1[:, 1]
    #         print(f'Causion! Waveform is stereo, chose second channel.')
    #     else:
    #         raise ValueError("Waveform is stereo but both channels are silent. Cannot determine which channel to use.")
        
    if sr1 == params.SAMPLE_RATE:
        waveform1 = waveform1
    else: 
        waveform1 = librosa.resample(waveform1, orig_sr=sr1, target_sr=params.SAMPLE_RATE, axis=0)

    if sr2 is not None:
        if sr2 == params.SAMPLE_RATE:
            waveform2 = waveform2
        else:
            waveform2 = librosa.resample(waveform2, orig_sr=sr2, target_sr=params.SAMPLE_RATE, axis=0)

        if len(waveform1) != len(waveform2):
            print(f'Waveform1 length: {len(waveform1)}, Waveform2 length: {len(waveform2)}')
            raise ValueError("Waveform1 and Waveform2 must have the same length for channeled data.")

    # Filter the waveform if specified
    if apply_filter == 'bandpass':
        waveform1 = bandpass_filter(waveform1)

    elif apply_filter == 'ica' and waveform2 is not None:
        # if not waveform2.shape == (waveform2.shape[0],):
        #     if not np.all(waveform2[:, 0] == 0):
        #         waveform2 = waveform2[:, 0]  
        #         print(f'Causion! Waveform2 is stereo, chose first channel.')
        #     elif not np.all(waveform2[:, 1] == 0):
        #         waveform2 = waveform2[:, 1]
        #         print(f'Causion! Waveform2 is stereo, chose second channel.')
        #     else:
        #         raise ValueError("Waveform2 is stereo but both channels are silent. Cannot determine which channel to use.")
        
        waveform1, waveform2 = ica_separate(waveform1, waveform2)

    elif apply_filter is not None:
        raise ValueError(f'Unknown apply_filter value: {apply_filter}')

    # Generate mel spectrogram and data patches
    if waveform2 is None:
        mel_spec = mel(waveform1, params.SAMPLE_RATE)
        data_patches = [mel_spec[i:i + params.PATCH_FRAMES] for i in range(0, mel_spec.shape[0] - params.PATCH_FRAMES + 1, params.PATCH_HOP_FRAMES)]
        data_patches = np.stack(data_patches) # shape: (num_patches, PATCH_FRAMES, n_bands)

        return data_patches, mel_spec
    else:
        mel_spec1 = mel(waveform1, params.SAMPLE_RATE)
        data_patches1 = [mel_spec1[i:i + params.PATCH_FRAMES] for i in range(0, mel_spec1.shape[0] - params.PATCH_FRAMES + 1, params.PATCH_HOP_FRAMES)]
        data_patches1 = np.stack(data_patches1) # shape: (num_patches, PATCH_FRAMES, n_bands)

        mel_spec2 = mel(waveform2, params.SAMPLE_RATE)
        data_patches2 = [mel_spec2[i:i + params.PATCH_FRAMES] for i in range(0, mel_spec2.shape[0] - params.PATCH_FRAMES + 1, params.PATCH_HOP_FRAMES)]
        data_patches2 = np.stack(data_patches2) # shape: (num_patches, PATCH_FRAMES, n_bands)

        return data_patches1, mel_spec1, data_patches2, mel_spec2

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
  