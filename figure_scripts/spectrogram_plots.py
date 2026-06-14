import argparse
import keras_yamnet.preprocessing as kp
import soundfile as sf
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

laptop = "dell"

def plot_mel_spectrogram(mel_spec, fontsize=12, cmap='viridis', show=True):
    # Use Times New Roman for the figure
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    # Ensure PDF output embeds TrueType fonts
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(
        mel_spec.T,
        aspect='auto',
        origin='lower',
        interpolation='nearest',
        cmap=cmap
    )

    # No title as requested. Control font sizes for axes and ticks.
    ax.set_xlabel('Time frames', fontsize=fontsize)
    ax.set_ylabel('Mel bands', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=fontsize)
    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax


def _default_dataset_folder():
    if laptop == "dell":
        return Path(r'C:/Users/imborhau/Documents/sound-event-detection-aircrafts/dataset/Skatval')
    elif laptop == "kampfly":
        return Path(r'C:/Users/kampfly/Documents/Ingeborg/Masteroppgave')
    else:
        return Path(os.getenv('SKATVAL_DATASET_FOLDER', ''))


def main():
    parser = argparse.ArgumentParser(description='Plot a single mel spectrogram')
    parser.add_argument('--fontsize', type=int, default=12, help='Font size for labels and ticks')
    parser.add_argument('--audio', type=str, default=None, help='Path to audio file (overrides default)')
    parser.add_argument('--out', type=str, default=None, help='Output PDF file path (saves and skips showing)')
    parser.add_argument('--start_sec', type=float, default=500.0, help='Start second for the audio snippet')
    parser.add_argument('--duration_sec', type=float, default=10.0, help='Duration in seconds for the audio snippet')
    args = parser.parse_args()

    skatval_dataset_folder = _default_dataset_folder()
    if args.audio:
        audio_path = Path(args.audio)
    else:
        audio_path = skatval_dataset_folder / '280126' / 'loc_2_280126.wav'
    audio_path = audio_path.resolve()
    print(f'Using audio path: {audio_path}')
    if not audio_path.exists():
        raise FileNotFoundError(f'Audio file not found: {audio_path}')

    # Read the requested segment
    sr = 48000
    start = int(args.start_sec * sr)
    stop = int((args.start_sec + args.duration_sec) * sr)
    audio_data, sample_rate = sf.read(audio_path, dtype='int16', start=start, stop=stop)
    _, mel_spec = kp.preprocess_input(audio_data, sample_rate)

    fig, ax = plot_mel_spectrogram(mel_spec, fontsize=args.fontsize, show=(args.out is None))

    # If an output path is provided, save as vector PDF with no padding around plot
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Save as PDF (vector) with no padding
        fig.savefig(str(out_path), format='pdf', bbox_inches='tight', pad_inches=0)
        print(f'Saved spectrogram PDF to: {out_path.resolve()}')


if __name__ == '__main__':
    main()
