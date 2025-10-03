import numpy as np
import soundfile as sf
from sklearn.decomposition import FastICA
from IPython.display import Audio, display

# --- Step 1: Load mixed audio files ---
files = [
    "data/mix1.wav",
    "data/mix2.wav",
    "data/mix3.wav"
]

signals = []
for f in files:
    x, sr = sf.read(f)   # waveform, sample rate
    if x.ndim > 1:       # stereo -> take first channel
        x = x[:, 0]
    signals.append(x)

# Align lengths
min_len = min(len(s) for s in signals)
signals = [s[:min_len] for s in signals]

# Stack: shape (n_samples, n_mixtures)
X = np.stack(signals, axis=1)   # (n_samples, 3)

print("Mixture matrix shape:", X.shape)

# --- Step 2: Apply FastICA ---
ica = FastICA(
    n_components=3,
    whiten="unit-variance",
    max_iter=5000,
    tol=1e-4,
    random_state=0
)
S_est = ica.fit_transform(X)

# --- Step 3: Normalize recovered signals ---
S_est = S_est / np.max(np.abs(S_est), axis=0)

# --- Step 4: Play both mixtures and recovered signals ---
print("🔊 Playing Mixtures")
for i in range(X.shape[1]):
    print(f"Mixture {i+1}")
    display(Audio(X[:, i], rate=sr))

print("\n🎶 Playing Recovered Sources (ICA)")
for i in range(S_est.shape[1]):
    print(f"Recovered Source {i+1}")
    display(Audio(S_est[:, i], rate=sr))
