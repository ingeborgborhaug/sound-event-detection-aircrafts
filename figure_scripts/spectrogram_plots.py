import keras_yamnet.preprocessing as kp
import soundfile as sf
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_mel_spectrogram(mel_spec, title="Mel spectrogram"):
    plt.figure(figsize=(10, 4))
    plt.imshow(
        mel_spec.T,
        aspect='auto',
        origin='lower',
        interpolation='nearest'
    )
    plt.title(title)
    plt.xlabel("Time frames")
    plt.ylabel("Mel bands")
    plt.colorbar(label="Amplitude")
    plt.tight_layout()
    plt.show()

skatval_dataset_folder = 'C:/Users/kampfly/Documents/Ingeborg/Masteroppgave'
audio_path = os.path.join(skatval_dataset_folder, "280126/Clipped/loc_2_280126.wav")

# First 10 seconds
audio_data1, sample_rate = sf.read(audio_path, dtype='int16', start=500*48000, stop=510*48000)
_, mel_spec1 = kp.preprocess_input(audio_data1, sample_rate)

# Second segment
audio_data2, sample_rate = sf.read(audio_path, dtype='int16', start=1335*48000, stop=1345*48000)
_, mel_spec2 = kp.preprocess_input(audio_data2, sample_rate)

# Create stacked subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

im1 = axs[0].imshow(mel_spec1.T, aspect='auto', origin='lower')
axs[0].set_title("No aircraft mel spectrogram")
axs[0].set_ylabel("Mel bands")

im2 = axs[1].imshow(mel_spec2.T, aspect='auto', origin='lower')
axs[1].set_title("Aircraft mel spectrogram")
axs[1].set_ylabel("Mel bands")
axs[1].set_xlabel("Time frames")

fig.colorbar(im1, ax=axs[0])
fig.colorbar(im2, ax=axs[1])

plt.tight_layout()
plt.show()
