import numpy as np
import librosa

from .features import mel
from . import params

def preprocess_input(waveform: np.ndarray, sr: int):
    """ inp = waveform if sr == params.SAMPLE_RATE else librosa.resample(
        waveform, orig_sr=sr, target_sr=params.SAMPLE_RATE)

    return mel(waveform,params.SAMPLE_RATE) """

    if waveform.shape[1] > 1:
        waveform = waveform[:, 0]  

    if sr == params.SAMPLE_RATE:
        inp = waveform
    else: 
        print(f'waveform.shape before resampling: {waveform.shape} with sample rate {sr}')
        inp = librosa.resample(waveform, orig_sr=sr, target_sr=params.SAMPLE_RATE, axis=0)
        print(f'inp.shape after resampling: {inp.shape}')


    return mel(inp, params.SAMPLE_RATE)