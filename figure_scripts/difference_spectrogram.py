import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import librosa
import librosa.display

plt.rcParams["font.family"] = "Times New Roman"

# ── paths ────────────────────────────────────────────────────────────────────
WAV_1 = "D:/dataset_master/030326/OPT F_008_0001_Tr1.WAV"
WAV_2 = "D:/dataset_master/030326/OPT F_008_0001_Tr2.WAV"

LABEL_1 = "Tr1"
LABEL_2 = "Tr2"

OUTPUT_PDF = "difference_spectrogram.pdf"

# ── STFT settings ────────────────────────────────────────────────────────────
N_FFT    = 2048
HOP      = 512
TARGET_SR = 22050   # resample both to the same rate for a fair comparison
# ─────────────────────────────────────────────────────────────────────────────


def load(path: str, sr: int):
    """Load and resample to a common sample rate."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def stft_db(y: np.ndarray) -> np.ndarray:
    """Magnitude spectrogram in dB."""
    D = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    return librosa.amplitude_to_db(D, ref=np.max)


def main() -> None:
    print("Loading audio files...")
    y1 = load(WAV_1, TARGET_SR)
    y2 = load(WAV_2, TARGET_SR)

    # Trim to the same length
    n = min(len(y1), len(y2))
    y1, y2 = y1[:n], y2[:n]

    print("Computing spectrograms...")
    S1 = stft_db(y1)
    S2 = stft_db(y2)
    S_diff = S1 - S2          # positive = louder in Tr1, negative = louder in Tr2

    times = librosa.frames_to_time(
        np.arange(S1.shape[1]), sr=TARGET_SR, hop_length=HOP
    )
    freqs = librosa.fft_frequencies(sr=TARGET_SR, n_fft=N_FFT)

    fig = plt.figure(figsize=(12, 9))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(3, 1, hspace=0.45)

    # ── Spectrogram 1 ─────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    img1 = ax1.pcolormesh(times, freqs, S1, shading="auto",
                          cmap="magma", vmin=-80, vmax=0, rasterized=True)
    ax1.set_ylabel("Frequency (Hz)", fontsize=12)
    ax1.set_title(LABEL_1, fontsize=13)
    fig.colorbar(img1, ax=ax1, format="%+2.0f dB", pad=0.01)

    # ── Spectrogram 2 ─────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    img2 = ax2.pcolormesh(times, freqs, S2, shading="auto",
                          cmap="magma", vmin=-80, vmax=0, rasterized=True)
    ax2.set_ylabel("Frequency (Hz)", fontsize=12)
    ax2.set_title(LABEL_2, fontsize=13)
    fig.colorbar(img2, ax=ax2, format="%+2.0f dB", pad=0.01)

    # ── Difference spectrogram ────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    lim = max(abs(S_diff.min()), abs(S_diff.max()))
    img3 = ax3.pcolormesh(times, freqs, S_diff, shading="auto",
                          cmap="RdBu_r", vmin=-lim, vmax=lim, rasterized=True)
    ax3.set_ylabel("Frequency (Hz)", fontsize=12)
    ax3.set_xlabel("Time (s)", fontsize=12)
    ax3.set_title(f"Difference ({LABEL_1} − {LABEL_2})", fontsize=13)
    cb3 = fig.colorbar(img3, ax=ax3, format="%+.1f dB", pad=0.01)
    cb3.set_label("dB", fontsize=11)

    for ax in (ax1, ax2, ax3):
        ax.set_facecolor("black")
        ax.spines[:].set_visible(False)

    plt.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Saved to {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
