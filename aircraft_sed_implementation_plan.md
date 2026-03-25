# Aircraft Sound Event Detection — Full Implementation Plan

**Master's Thesis Project — NTNU**
**Supervisor: Prof. Adil Rasheed, Dept. of Engineering Cybernetics**

---

## How to Use This Document

This document is structured as **12 phases**, each self-contained. Work through them sequentially. Each phase contains:

1. **Objective** — what this phase accomplishes
2. **Dependencies** — what must exist before starting
3. **Specification** — detailed technical requirements
4. **VS Code Prompt** — a ready-to-paste prompt for Claude Code / Cursor
5. **Validation checklist** — how to verify correctness before committing

**Workflow:** Read the spec → paste the prompt into your AI coding assistant → review the generated code against the checklist → `git commit` → move to the next phase.

---

## Project Structure (Target)

```
aircraft-sed/
├── configs/
│   ├── default.yaml              # All hyperparameters and paths
│   └── experiments/              # Per-experiment overrides
│       ├── baseline_yamnet.yaml
│       ├── finetune_yamnet.yaml
│       ├── domain_adapt.yaml
│       └── ast_comparison.yaml
├── data/
│   ├── raw/
│   │   ├── aerosonic/            # AeroSonicDB wav files + labels
│   │   └── norwegian/            # Norwegian dataset wav files + labels
│   │       ├── session1/         # Clear weather
│   │       │   ├── location_A/
│   │       │   ├── location_B/
│   │       │   └── location_C/
│   │       ├── session2/         # Windy
│   │       ├── session3/         # Rainy
│   │       ├── session4/         # Snow
│   │       └── session5/         # Wind + Rain
│   ├── processed/                # Precomputed mel spectrograms (.npy)
│   └── splits/                   # JSON split files for each LOSO fold
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── audio_loader.py       # Resampling, segmentation, I/O
│   │   ├── feature_extraction.py # Log-mel spectrogram computation
│   │   ├── noise_profiling.py    # Per-session noise floor estimation
│   │   └── augmentation.py       # Real-noise mixing, SpecAugment
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── sed_dataset.py        # PyTorch Dataset for segments
│   │   ├── split_generator.py    # LOSO fold generation
│   │   └── sampler.py            # Balanced / domain-aware sampling
│   ├── models/
│   │   ├── __init__.py
│   │   ├── yamnet_embedder.py    # Frozen YAMNet embedding extractor
│   │   ├── classifier_heads.py   # Logistic, MLP, temporal heads
│   │   ├── yamnet_finetune.py    # Fine-tunable YAMNet wrapper
│   │   ├── domain_adaptation.py  # MMD / CORAL loss modules
│   │   ├── ast_model.py          # AST backbone alternative
│   │   └── multi_location_fusion.py  # Score-level fusion module
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py            # Training loop with logging
│   │   ├── losses.py             # BCE + domain adaptation losses
│   │   └── schedulers.py         # LR schedules, warmup
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py            # F1, AUC-ROC, event-based metrics
│   │   ├── sed_eval_wrapper.py   # SED-Eval integration
│   │   ├── analysis.py           # Per-condition breakdown, DET curves
│   │   └── statistical_tests.py  # McNemar, bootstrap CIs
│   └── utils/
│       ├── __init__.py
│       ├── config.py             # YAML config loader
│       ├── logging_utils.py      # TensorBoard / W&B logging
│       └── reproducibility.py    # Seed setting, deterministic flags
├── scripts/
│   ├── 01_preprocess.py          # Run full preprocessing pipeline
│   ├── 02_generate_splits.py     # Generate LOSO split files
│   ├── 03_extract_embeddings.py  # Precompute YAMNet embeddings
│   ├── 04_train.py               # Main training entry point
│   ├── 05_evaluate.py            # Run evaluation on test folds
│   ├── 06_analyse_results.py     # Generate all figures and tables
│   └── 07_multi_location_fusion.py  # Fusion experiments
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_noise_profile_analysis.ipynb
│   ├── 03_results_visualization.ipynb
│   └── 04_statistical_analysis.ipynb
├── tests/
│   ├── test_preprocessing.py
│   ├── test_dataset.py
│   ├── test_metrics.py
│   └── test_splits.py
├── requirements.txt
├── setup.py
└── README.md
```

---

## Phase 0: Environment and Scaffolding

### Objective
Create the project skeleton, install all dependencies, set reproducibility defaults, and verify GPU access.

### Dependencies
- Python 3.10+, CUDA-capable GPU, conda or venv

### Specification

**`requirements.txt`** must include:
```
torch>=2.1.0
torchaudio>=2.1.0
tensorflow>=2.14.0        # YAMNet is TF-based
tensorflow-hub>=0.15.0
librosa>=0.10.1
soundfile>=0.12.1
numpy>=1.24.0
scipy>=1.11.0
scikit-learn>=1.3.0
pandas>=2.1.0
matplotlib>=3.8.0
seaborn>=0.13.0
sed_eval>=0.2.1
pyyaml>=6.0
tensorboard>=2.15.0
hydra-core>=1.3.0         # Optional: config management
tqdm>=4.66.0
```

**`configs/default.yaml`** structure:
```yaml
seed: 42
audio:
  target_sr: 16000
  segment_duration_s: 10.0
  segment_hop_s: 5.0
  highpass_cutoff_hz: 80
features:
  n_mels: 64
  n_fft: 400          # 25ms at 16kHz
  hop_length: 160      # 10ms at 16kHz
  fmin: 125
  fmax: 7500
  norm: per_clip       # per_clip | global | none
yamnet:
  patch_duration_s: 0.96
  patch_hop_s: 0.48
  embedding_dim: 1024
training:
  batch_size: 32
  epochs: 50
  lr: 1.0e-3
  weight_decay: 1.0e-4
  early_stopping_patience: 10
  optimizer: adam
  scheduler: cosine
augmentation:
  noise_mix_prob: 0.5
  noise_snr_range_db: [0, 20]
  spec_augment: true
  freq_mask_param: 8
  time_mask_param: 25
  num_freq_masks: 2
  num_time_masks: 2
paths:
  aerosonic_root: data/raw/aerosonic
  norwegian_root: data/raw/norwegian
  processed_dir: data/processed
  splits_dir: data/splits
  output_dir: outputs
```

### VS Code Prompt — Phase 0

```
# PHASE 0: Project Scaffolding

Create the full project directory structure for an aircraft sound event detection
system. The structure is defined below. Generate:

1. All __init__.py files (empty is fine)
2. requirements.txt with the packages listed below
3. configs/default.yaml with the configuration below
4. src/utils/config.py — a config loader using PyYAML that:
   - Loads default.yaml
   - Merges experiment-specific overrides from configs/experiments/
   - Supports CLI overrides via argparse (e.g., --training.lr 0.001)
   - Returns an OmegaConf-style dotted-access object (use a simple
     recursive namespace or dataclass)
5. src/utils/reproducibility.py — a set_seed(seed) function that:
   - Sets random.seed, np.random.seed, torch.manual_seed, torch.cuda.manual_seed_all
   - Sets torch.backends.cudnn.deterministic = True
   - Sets torch.backends.cudnn.benchmark = False
   - Sets PYTHONHASHSEED environment variable
6. A minimal setup.py with name="aircraft-sed"
7. README.md with project title, one-paragraph description, and install instructions

## Project structure:
[paste the directory tree from the implementation plan]

## configs/default.yaml content:
[paste the YAML block from the implementation plan]

## requirements.txt packages:
torch>=2.1.0, torchaudio>=2.1.0, tensorflow>=2.14.0, tensorflow-hub>=0.15.0,
librosa>=0.10.1, soundfile>=0.12.1, numpy>=1.24.0, scipy>=1.11.0,
scikit-learn>=1.3.0, pandas>=2.1.0, matplotlib>=3.8.0, seaborn>=0.13.0,
sed_eval>=0.2.1, pyyaml>=6.0, tensorboard>=2.15.0, tqdm>=4.66.0

After creating all files, run: pip install -e . && python -c "from src.utils.config import load_config; print(load_config())"
```

### Validation Checklist
- [ ] `pip install -e .` succeeds
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` returns True
- [ ] `python -c "from src.utils.config import load_config; cfg = load_config(); print(cfg.audio.target_sr)"` prints 16000
- [ ] `python -c "from src.utils.reproducibility import set_seed; set_seed(42)"` runs without error

**Git commit: `feat: project scaffolding and config system`**

---

## Phase 1: Audio Preprocessing Pipeline

### Objective
Build a robust preprocessing pipeline that loads raw audio from both datasets, resamples to 16 kHz, applies high-pass filtering, segments into fixed-length windows, and saves to disk.

### Dependencies
- Phase 0 complete
- Raw data placed in `data/raw/aerosonic/` and `data/raw/norwegian/`

### Specification

**`src/preprocessing/audio_loader.py`** must implement:

```python
class AudioLoader:
    """Load, resample, filter, and segment audio files."""

    def __init__(self, target_sr=16000, highpass_cutoff_hz=80):
        ...

    def load_and_resample(self, filepath: str) -> tuple[np.ndarray, int]:
        """
        Load any audio file, convert to mono, resample to target_sr.
        Use librosa.load with res_type='kaiser_best'.
        Returns (audio_array, sample_rate).
        """

    def apply_highpass(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply 4th-order Butterworth high-pass filter at highpass_cutoff_hz.
        Use scipy.signal.butter + scipy.signal.sosfiltfilt (zero-phase).
        """

    def segment(self, audio: np.ndarray, sr: int,
                segment_duration_s: float, hop_s: float
                ) -> list[tuple[np.ndarray, float, float]]:
        """
        Segment audio into fixed-length windows.
        Returns list of (segment_array, start_time_s, end_time_s).
        Zero-pad the final segment if shorter than segment_duration_s.
        """
```

**`src/preprocessing/feature_extraction.py`** must implement:

```python
class MelSpectrogramExtractor:
    """Compute log-mel spectrograms matching YAMNet specifications."""

    def __init__(self, sr=16000, n_mels=64, n_fft=400,
                 hop_length=160, fmin=125, fmax=7500):
        ...

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute log-mel spectrogram.
        1. librosa.feature.melspectrogram with given params
        2. Convert to dB: librosa.power_to_db(S, ref=np.max)
        3. Return shape (n_mels, time_frames)
        """

    def normalize(self, spectrogram: np.ndarray,
                  method: str = 'per_clip') -> np.ndarray:
        """
        Normalize spectrogram.
        - 'per_clip': subtract mean, divide by std (computed on this clip)
        - 'none': return as-is
        Clip std to minimum of 1e-6 to avoid division by zero.
        """

    def extract_yamnet_patches(self, spectrogram: np.ndarray,
                                patch_frames: int = 96,
                                patch_hop_frames: int = 48
                                ) -> np.ndarray:
        """
        Slice spectrogram into YAMNet-sized patches.
        Each patch: (n_mels, 96) = 0.96 seconds.
        Returns shape (n_patches, n_mels, 96).
        """
```

**`scripts/01_preprocess.py`** must:
1. Iterate over all audio files in both datasets
2. Load, resample, high-pass filter
3. Segment into 10 s windows with 5 s hop
4. Compute log-mel spectrograms
5. Save each segment's spectrogram as .npy in `data/processed/{dataset}/{session}/{location}/`
6. Save a manifest CSV: `filepath, dataset, session, location, start_s, end_s, label`
7. Print summary statistics: total segments per dataset, per session, per label

### VS Code Prompt — Phase 1

```
# PHASE 1: Audio Preprocessing Pipeline

Implement the audio preprocessing pipeline for the aircraft sound event detection
project. Read configs/default.yaml for all parameters.

## File 1: src/preprocessing/audio_loader.py

Create class AudioLoader with these methods:

- __init__(self, target_sr=16000, highpass_cutoff_hz=80)

- load_and_resample(self, filepath: str) -> tuple[np.ndarray, int]
  Use librosa.load(filepath, sr=target_sr, res_type='kaiser_best').
  Ensure mono output. Log a warning if original sr differs from target_sr.

- apply_highpass(self, audio: np.ndarray, sr: int) -> np.ndarray
  4th-order Butterworth high-pass at self.highpass_cutoff_hz.
  Use scipy.signal.butter(N=4, Wn=cutoff, btype='high', fs=sr, output='sos')
  then scipy.signal.sosfiltfilt for zero-phase filtering.

- segment(self, audio, sr, segment_duration_s, hop_s) -> list[tuple[np.ndarray, float, float]]
  Fixed-length segmentation. Zero-pad last segment.
  Return list of (segment_audio, start_time_s, end_time_s).

## File 2: src/preprocessing/feature_extraction.py

Create class MelSpectrogramExtractor with these methods:

- __init__(self, sr=16000, n_mels=64, n_fft=400, hop_length=160, fmin=125, fmax=7500)

- extract(self, audio: np.ndarray) -> np.ndarray
  Compute mel spectrogram via librosa.feature.melspectrogram, then
  librosa.power_to_db(S, ref=np.max). Return shape (n_mels, time_frames).

- normalize(self, spectrogram, method='per_clip') -> np.ndarray
  per_clip: z-score normalize (mean=0, std=1) per spectrogram.
  Clip std to min 1e-6.

- extract_yamnet_patches(self, spectrogram, patch_frames=96, patch_hop_frames=48) -> np.ndarray
  Slice into (n_patches, n_mels, 96) patches.

## File 3: scripts/01_preprocess.py

Main preprocessing script that:
1. Loads config from configs/default.yaml
2. Discovers all .wav files in data/raw/aerosonic/ and data/raw/norwegian/
   - For Norwegian data, parse the directory structure to extract session and
     location identifiers
   - For AeroSonicDB, session='aerosonic', location='default'
3. For each file: load → resample → highpass → segment → extract mel spectrogram → normalize
4. Save each spectrogram as .npy in data/processed/{dataset}/{session}/{location}/
   with filename: {original_stem}_{start_s:.1f}_{end_s:.1f}.npy
5. Build a manifest DataFrame with columns:
   [npy_path, wav_source, dataset, session, location, start_s, end_s, label, duration_s]
6. The label column should be populated from the dataset's label files.
   - For AeroSonicDB: expect a CSV or annotation file in the dataset folder.
     Add a TODO comment for the student to adapt the label-loading logic to
     their specific annotation format.
   - For Norwegian: similarly add a TODO for their annotation format.
7. Save manifest to data/processed/manifest.csv
8. Print summary: total files, total segments, segments per session, class balance

Use tqdm for progress bars. Use logging module (not print) for all messages.
Handle corrupted/unreadable files gracefully with try/except and log warnings.

## File 4: tests/test_preprocessing.py

Write pytest tests:
- test_resample: load a file, verify output sr == 16000 and audio is 1D
- test_highpass: apply to a signal with known low-freq content, verify attenuation
- test_segment: verify correct number of segments and that last segment is zero-padded
- test_mel_shape: verify spectrogram shape matches expected (n_mels, expected_frames)
- test_normalize: verify mean≈0 and std≈1 after per_clip normalization
- test_patch_extraction: verify patch shape is (n_patches, 64, 96)

For tests, generate synthetic audio with np.random.randn or sine waves rather
than requiring real data files.
```

### Validation Checklist
- [ ] `pytest tests/test_preprocessing.py` passes all tests
- [ ] Running `python scripts/01_preprocess.py` on a small subset produces .npy files and manifest.csv
- [ ] Spot-check: load a .npy file, verify shape is `(64, expected_frames)`
- [ ] manifest.csv has correct columns and no NaN in critical fields

**Git commit: `feat: audio preprocessing pipeline with resampling, filtering, segmentation, mel extraction`**

---

## Phase 2: Noise Profiling and Data Exploration

### Objective
Compute per-session average noise spectra from no-aircraft segments. Generate exploratory plots characterizing each weather condition.

### Dependencies
- Phase 1 complete, manifest.csv exists

### Specification

**`src/preprocessing/noise_profiling.py`** must implement:

```python
class NoiseProfiler:
    """Compute and store average noise profiles per session."""

    def compute_session_profile(self, manifest_df, session_id: str,
                                 extractor: MelSpectrogramExtractor
                                 ) -> dict:
        """
        For a given session:
        1. Filter manifest to label == 'no_aircraft' and session == session_id
        2. Load all corresponding spectrograms
        3. Compute: mean spectrum (average across time and segments),
           std spectrum, median spectrum, 5th/95th percentile spectra
        4. Return dict with keys: 'mean', 'std', 'median', 'p5', 'p95',
           'n_segments', 'session_id'
        """

    def compute_all_profiles(self, manifest_df, extractor) -> dict:
        """Compute profiles for all sessions. Return dict[session_id] -> profile."""

    def save_profiles(self, profiles: dict, output_dir: str):
        """Save as .npz files and a summary JSON."""
```

**`notebooks/01_data_exploration.ipynb`** should generate:
1. Waveform + spectrogram of one example aircraft event per session
2. Average noise spectrum overlay plot (all 5 sessions on one axis)
3. Class balance bar chart per session
4. Segment count table per session × location
5. Histogram of segment-level RMS energy per session

### VS Code Prompt — Phase 2

```
# PHASE 2: Noise Profiling and Data Exploration

## File 1: src/preprocessing/noise_profiling.py

Create class NoiseProfiler:

- compute_session_profile(self, manifest_df, session_id, feature_extractor):
  Filter manifest to rows where label == 'no_aircraft' and session == session_id.
  Load all .npy spectrogram files listed in the filtered manifest.
  For each spectrogram (shape n_mels × time_frames), compute the mean across
  the time axis to get a (n_mels,) vector representing average spectral energy.
  Aggregate across all segments:
    - mean_spectrum: np.mean of all per-segment mean spectra → shape (n_mels,)
    - std_spectrum: np.std across segments → shape (n_mels,)
    - median_spectrum: np.median → (n_mels,)
    - p5_spectrum / p95_spectrum: np.percentile at 5 and 95 → (n_mels,)
    - n_segments: count of segments used
  Return as dict.

- compute_all_profiles(self, manifest_df, feature_extractor) -> dict[str, dict]:
  Loop over unique sessions in manifest, call compute_session_profile for each.

- save_profiles(self, profiles, output_dir):
  For each session, save spectra as {output_dir}/{session_id}_noise_profile.npz.
  Save summary (n_segments per session, session list) as JSON.

## File 2: notebooks/01_data_exploration.ipynb

Create a Jupyter notebook that loads manifest.csv and the noise profiles, then
generates the following visualizations:

1. **Per-session example spectrograms** (5 subplots, one per session):
   For each session, pick one segment labeled 'aircraft' near the middle of the
   session. Load its .npy and plot with librosa.display.specshow. Title each
   subplot with session name and weather condition.

2. **Noise profile comparison** (single plot):
   Overlay the mean noise spectrum (n_mels,) of all 5 sessions on one plot.
   X-axis: mel band index (or approximate Hz using librosa.mel_frequencies).
   Y-axis: mean power (dB). Use distinct colors and a legend:
   Session 1=Clear, Session 2=Wind, Session 3=Rain, Session 4=Snow, Session 5=Wind+Rain.
   Add shaded region showing ±1 std for each session.

3. **Class balance** (grouped bar chart):
   For each session, show count of 'aircraft' vs 'no_aircraft' segments.

4. **Segment statistics table**:
   Pandas DataFrame displayed as a table: rows = sessions, columns = locations,
   values = segment counts. Add a 'Total' row and column.

5. **RMS energy distributions** (overlaid histograms or KDE plots):
   For each session, compute the RMS energy of each segment (from the .npy
   spectrograms: np.sqrt(np.mean(10**(S_dB/10)))) and plot the distribution.
   Separate aircraft vs no_aircraft with different line styles.

Save all figures to outputs/exploration/ as PNG at 150 DPI.
Use seaborn style 'whitegrid'. Set figure sizes to (12, 6) for single plots,
(14, 10) for multi-panel figures.
```

### Validation Checklist
- [ ] Noise profiles saved as .npz files, one per session
- [ ] Notebook runs end-to-end without errors
- [ ] Noise spectrum plot shows clear differences between sessions (wind/rain should have elevated low-frequency energy)
- [ ] Class balance is documented — note any severe imbalances

**Git commit: `feat: noise profiling and exploratory data analysis`**

---

## Phase 3: Dataset Splits (LOSO)

### Objective
Generate leave-one-session-out (LOSO) split files ensuring zero leakage. Also generate the AeroSonicDB internal train/val split.

### Dependencies
- Phase 1 complete, manifest.csv exists

### Specification

**`src/datasets/split_generator.py`** must implement:

```python
class LOSOSplitGenerator:
    """Generate leave-one-session-out splits."""

    SESSIONS = {
        'session1': 'clear',
        'session2': 'wind',
        'session3': 'rain',
        'session4': 'snow',
        'session5': 'wind_rain'
    }

    def generate_all_folds(self, manifest_df) -> list[dict]:
        """
        Generate 5 LOSO folds + 1 baseline (AeroSonicDB-only) fold.

        Each fold is a dict:
        {
            'fold_name': str,           # e.g. 'loso_session3_rain'
            'test_session': str,        # e.g. 'session3'
            'test_weather': str,        # e.g. 'rain'
            'train_indices': list[int], # manifest row indices
            'val_indices': list[int],
            'test_indices': list[int],
            'train_sessions': list[str],
            'val_strategy': str,        # description
        }

        Fold construction rules:
        - TEST: all segments from the held-out Norwegian session (all 3 locations)
        - TRAIN: all AeroSonicDB segments + Norwegian segments from the other 4 sessions
                 (locations A and B only)
        - VAL: Norwegian segments from the other 4 sessions (location C only)
                 + 15% of AeroSonicDB (held out from AeroSonicDB training set)

        Baseline fold:
        - TRAIN: 85% of AeroSonicDB
        - VAL: 15% of AeroSonicDB
        - TEST: all Norwegian sessions (full dataset)

        The AeroSonicDB train/val split must use the SAME random split across
        all folds (seed from config) to avoid information leakage via different
        validation sets.
        """

    def save_folds(self, folds: list[dict], output_dir: str):
        """Save each fold as a JSON file in output_dir."""

    def print_fold_summary(self, folds: list[dict], manifest_df):
        """Print table: fold name, #train, #val, #test, class balance per split."""
```

### VS Code Prompt — Phase 3

```
# PHASE 3: LOSO Split Generation

## File: src/datasets/split_generator.py

Implement class LOSOSplitGenerator that generates leave-one-session-out folds
from the preprocessing manifest.

### Requirements:

1. Load manifest.csv (passed as DataFrame) with columns:
   [npy_path, wav_source, dataset, session, location, start_s, end_s, label]

2. Generate 6 folds:

   FOLD 0 — "baseline_aerosonic_only":
     train = 85% of AeroSonicDB segments (stratified by label, seed=42)
     val   = 15% of AeroSonicDB segments
     test  = ALL Norwegian segments (sessions 1-5, all locations)

   FOLDS 1-5 — "loso_session{N}_{weather}":
     test  = all segments from session N (all 3 locations)
     train = all AeroSonicDB train split (same 85% as baseline)
           + all segments from the OTHER 4 Norwegian sessions, locations A and B only
     val   = 15% AeroSonicDB val split (same as baseline)
           + all segments from the OTHER 4 Norwegian sessions, location C only

3. The AeroSonicDB 85/15 split must be computed ONCE with sklearn.model_selection.
   train_test_split(stratify=labels, random_state=42) and reused across all folds.

4. Store each fold as a dict with keys:
   fold_name, test_session, test_weather, train_indices, val_indices, test_indices,
   train_sessions, n_train, n_val, n_test,
   train_aircraft_pct, val_aircraft_pct, test_aircraft_pct

5. save_folds(): serialize each fold as JSON in data/splits/{fold_name}.json
   (convert numpy int64 to Python int for JSON serialization)

6. print_fold_summary(): pretty-print a table showing fold name, split sizes,
   and class balance using pandas DataFrame.

## File: scripts/02_generate_splits.py

Script that:
1. Loads config and manifest.csv
2. Instantiates LOSOSplitGenerator
3. Calls generate_all_folds
4. Saves folds
5. Prints summary

## File: tests/test_splits.py

Write tests:
- test_no_index_overlap: for each fold, assert train ∩ val ∩ test == ∅
- test_test_session_isolation: for LOSO folds, assert NO segments from the
  test session appear in train or val
- test_location_c_in_val: for LOSO folds, assert location C segments from
  training sessions are ONLY in val, not train
- test_aerosonic_split_consistent: assert AeroSonicDB train/val indices are
  identical across all folds
- test_all_segments_covered: assert union of train+val+test covers all manifest rows
  (for baseline fold; for LOSO folds, some indices may be unused if they're
  test-session location C for val — verify no gaps)
```

### Validation Checklist
- [ ] `pytest tests/test_splits.py` passes all tests
- [ ] JSON files saved in `data/splits/`
- [ ] Summary table shows reasonable sizes and class balance
- [ ] **Critical:** No test-session segments appear in training for any LOSO fold

**Git commit: `feat: LOSO split generation with leakage-prevention tests`**

---

## Phase 4: PyTorch Dataset and DataLoaders

### Objective
Build a PyTorch Dataset that loads precomputed spectrograms from .npy files, applies augmentation, and returns batches suitable for training.

### Dependencies
- Phase 1 (spectrograms) and Phase 3 (split JSONs) complete

### Specification

**`src/preprocessing/augmentation.py`**:
```python
class AudioAugmentor:
    """Apply augmentations to log-mel spectrograms."""

    def __init__(self, noise_profiles: dict, config: dict):
        self.noise_profiles = noise_profiles  # from Phase 2
        self.config = config

    def mix_real_noise(self, spectrogram: np.ndarray,
                       session_id: str, snr_db: float) -> np.ndarray:
        """
        Mix real background noise from a randomly selected no-aircraft
        segment of the given session into the spectrogram at the given SNR.
        Noise segments come from precomputed noise profiles.
        """

    def spec_augment(self, spectrogram: np.ndarray) -> np.ndarray:
        """
        Apply SpecAugment: frequency masking + time masking.
        Use config params: freq_mask_param, time_mask_param,
        num_freq_masks, num_time_masks.
        """
```

**`src/datasets/sed_dataset.py`**:
```python
class SEDDataset(torch.utils.data.Dataset):
    """Dataset for aircraft sound event detection."""

    def __init__(self, manifest_df, indices: list[int],
                 augmentor=None, patch_mode='yamnet'):
        """
        manifest_df: full manifest DataFrame
        indices: row indices for this split
        augmentor: optional AudioAugmentor (None for val/test)
        patch_mode: 'yamnet' (extract 0.96s patches) or 'full' (return whole segment)
        """

    def __getitem__(self, idx):
        """
        Returns dict:
        {
            'spectrogram': tensor,  # shape depends on patch_mode
            'label': int,           # 0 or 1
            'session': str,
            'location': str,
            'metadata': dict        # start_s, end_s, wav_source
        }
        """
```

### VS Code Prompt — Phase 4

```
# PHASE 4: PyTorch Dataset, Augmentation, and DataLoaders

## File 1: src/preprocessing/augmentation.py

Create class AudioAugmentor with:

- __init__(self, noise_segments_dir, config):
  noise_segments_dir: path to directory containing .npy files of no-aircraft
  segments, organized by session. These serve as the noise pool.
  config: augmentation config dict from default.yaml

- mix_real_noise(self, spectrogram, allowed_sessions, snr_db=None):
  1. Randomly pick a no-aircraft .npy file from one of the allowed_sessions
  2. If snr_db is None, sample uniformly from config.noise_snr_range_db
  3. Compute scaling factor: noise_scaled = noise * 10^((S_signal_dB - S_noise_dB - snr_db) / 20)
     (operate in linear power domain, not dB)
  4. In practice for log-mel domain: convert both to linear, add, convert back to dB
  5. Handle length mismatch by tiling or cropping the noise segment
  Return augmented spectrogram, same shape as input.

- spec_augment(self, spectrogram):
  Apply SpecAugment (frequency and time masking) directly to the log-mel
  spectrogram tensor.
  For each of num_freq_masks: mask a random band of width ∈ [0, freq_mask_param]
  For each of num_time_masks: mask a random span of width ∈ [0, time_mask_param]
  Masked values set to the spectrogram's minimum value (not zero, since we're in dB).
  Return augmented spectrogram.

- __call__(self, spectrogram, allowed_sessions):
  Apply noise mixing with probability config.noise_mix_prob, then always apply
  spec_augment. Return augmented spectrogram.

## File 2: src/datasets/sed_dataset.py

Create class SEDDataset(torch.utils.data.Dataset):

- __init__(self, manifest_df, indices, augmentor=None, mode='patch'):
  Store the subset of manifest_df selected by indices.
  mode='patch': return individual YAMNet patches (0.96s)
  mode='segment': return full segment spectrogram

  If mode='patch', precompute a flat index mapping:
    For each segment, compute how many patches it contains.
    Build a list: [(segment_idx, patch_idx), ...] so __len__ returns total patches.

- __len__(): return total items (patches or segments depending on mode)

- __getitem__(self, idx):
  1. Load .npy spectrogram from manifest path
  2. If augmentor is not None, apply augmentation (pass allowed_sessions =
     sessions that appear in the training set for this fold, NOT the test session)
  3. If mode='patch': extract the specific patch from the spectrogram
  4. Convert to torch.FloatTensor, add channel dimension → (1, n_mels, time_frames)
  5. Return dict with 'spectrogram', 'label' (as torch.long), 'session', 'location',
     'metadata' (dict with start_s, end_s, source file)

## File 3: src/datasets/sampler.py

Create class BalancedDomainSampler(torch.utils.data.Sampler):
  Ensures each batch contains roughly equal aircraft / no-aircraft samples,
  and a configurable ratio of AeroSonicDB vs Norwegian samples.
  - __init__(self, dataset, batch_size, aerosonic_ratio=0.5):
    Separate indices by label and by domain. Each batch samples:
    batch_size * aerosonic_ratio from AeroSonicDB (balanced labels)
    batch_size * (1 - aerosonic_ratio) from Norwegian (balanced labels)
  - __iter__(): yield batches of indices
  - __len__(): return number of batches per epoch

## File 4: tests/test_dataset.py

Tests:
- test_dataset_length: create dataset from small manifest, verify __len__
- test_getitem_shape: verify output tensor shape is (1, 64, 96) for patch mode
- test_augmentation_changes_input: apply augmentor, verify output != input
- test_spec_augment_masking: verify some values are set to minimum
- test_balanced_sampler: verify batch composition has roughly 50% positive labels
```

### Validation Checklist
- [ ] All tests pass
- [ ] DataLoader iteration completes without errors for one epoch
- [ ] Augmented spectrograms are visually different from originals (spot-check in notebook)
- [ ] Balanced sampler produces roughly 50/50 label distribution per batch

**Git commit: `feat: PyTorch dataset with real-noise augmentation and balanced sampling`**

---

## Phase 5: YAMNet Embedding Extraction

### Objective
Build a wrapper around TensorFlow Hub's YAMNet to extract 1024-dimensional embeddings. Precompute and cache all embeddings to disk for efficient training.

### Dependencies
- Phase 1 complete (16 kHz audio segments available)

### Specification

YAMNet operates on raw waveforms at 16 kHz and outputs (N_patches, 1024) embeddings. We extract embeddings both from the TF model directly and from precomputed mel spectrograms.

### VS Code Prompt — Phase 5

```
# PHASE 5: YAMNet Embedding Extraction

## File 1: src/models/yamnet_embedder.py

Create class YAMNetEmbedder:

- __init__(self, model_url='https://tfhub.dev/google/yamnet/1'):
  Load the YAMNet model from TensorFlow Hub.
  Store it as self.model.
  This model takes raw audio waveforms (16 kHz, float32) and returns:
    scores: (N, 521) — classification scores for 521 AudioSet classes
    embeddings: (N, 1024) — per-patch embeddings
    spectrogram: (M, 64) — the log-mel spectrogram YAMNet computed internally

- extract_embeddings(self, waveform: np.ndarray) -> np.ndarray:
  Input: 1D float32 array at 16 kHz.
  Call self.model(waveform) and return the embeddings array, shape (N, 1024).
  N depends on waveform length (one embedding per 0.96s patch with 0.48s hop).

- extract_embeddings_batch(self, waveforms: list[np.ndarray]) -> list[np.ndarray]:
  Process a list of waveforms. Return list of embedding arrays.
  NOTE: YAMNet does not support batched input natively — loop over waveforms.
  Show tqdm progress bar.

## File 2: scripts/03_extract_embeddings.py

Script that precomputes YAMNet embeddings for the entire dataset.

1. Load config and manifest.csv
2. Initialize YAMNetEmbedder
3. For each row in manifest:
   a. Load the ORIGINAL wav file (not the .npy spectrogram) and extract
      the segment between start_s and end_s
   b. Ensure audio is 16 kHz float32 mono
   c. Extract embeddings → shape (N_patches, 1024)
   d. Save as .npy in data/processed/embeddings/{dataset}/{session}/{location}/
      with filename matching the spectrogram file but with _emb suffix
4. Update manifest.csv with a new column 'embedding_path'
5. Print summary: total embeddings computed, average patches per segment

This script should:
- Use tqdm for progress
- Skip files that already have embeddings (for restart-ability)
- Handle TF GPU memory: set tf.config.experimental.set_memory_growth(True)
- Log total processing time

## Key implementation notes:
- YAMNet expects raw audio at EXACTLY 16 kHz. Verify sample rate before passing.
- The TF Hub YAMNet model returns 3 outputs: scores, embeddings, spectrogram.
  We only need the embeddings tensor (index [1]).
- Each 0.96s of audio produces one 1024-dim embedding.
- A 10-second segment produces approximately 19 embeddings (with 0.48s hop).
```

### Validation Checklist
- [ ] Embedding files created in `data/processed/embeddings/`
- [ ] Spot-check: load one .npy embedding file, verify shape is `(~19, 1024)` for a 10s segment
- [ ] manifest.csv updated with `embedding_path` column
- [ ] Script completes without OOM errors

**Git commit: `feat: YAMNet embedding extraction and caching`**

---

## Phase 6: Model Architectures

### Objective
Implement the four-level model progression: (1) frozen embeddings + logistic regression, (2) frozen embeddings + temporal aggregation, (3) fine-tuned YAMNet, (4) domain-adapted YAMNet. Also implement the AST comparison model.

### Dependencies
- Phase 5 complete (embeddings available for Level 1-2)
- Phase 4 complete (dataset classes for Level 3-4)

### VS Code Prompt — Phase 6

```
# PHASE 6: Model Architectures

Implement all model architectures as PyTorch nn.Module classes. Each model must
have a consistent interface:
  - forward(self, x) -> dict with keys 'logits' (batch,) and optionally 'embeddings' (batch, dim)
  - Models operating on precomputed embeddings expect x of shape (batch, n_patches, 1024)
  - Models operating on spectrograms expect x of shape (batch, 1, n_mels, time_frames)

## File 1: src/models/classifier_heads.py

### Class 1: EmbeddingLogisticRegression
Level 1 — simplest baseline.
- __init__(self, embedding_dim=1024, aggregation='mean'):
  aggregation: how to combine multi-patch embeddings ('mean', 'max')
  self.fc = nn.Linear(embedding_dim, 1)

- forward(self, x):
  x shape: (batch, n_patches, 1024)
  1. Aggregate across patches: mean or max along dim=1 → (batch, 1024)
  2. self.fc → (batch, 1)
  3. Return {'logits': output.squeeze(-1)}

### Class 2: TemporalClassifier
Level 2 — captures temporal dynamics of flyover events.
- __init__(self, embedding_dim=1024, hidden_dim=256, n_layers=2,
           temporal_model='gru', use_attention=True, dropout=0.3):
  If temporal_model == 'gru': self.rnn = nn.GRU(embedding_dim, hidden_dim,
    n_layers, batch_first=True, dropout=dropout, bidirectional=True)
  If temporal_model == 'attention': use a small Transformer encoder
    (nn.TransformerEncoderLayer with 4 heads, 2 layers)
  If use_attention (for GRU): add self-attention pooling layer over hidden states
  self.classifier = nn.Sequential(
    nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 128),
    nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1))

- forward(self, x):
  x: (batch, n_patches, 1024)
  1. Pass through RNN → hidden states (batch, n_patches, hidden_dim*2)
  2. If use_attention: compute attention weights over patches, weighted sum
     Else: take the last hidden state
  3. Classifier → logits
  Return {'logits': logits.squeeze(-1), 'embeddings': aggregated_hidden}

### Class 3: AttentionPooling (helper module)
- __init__(self, input_dim):
  self.attention = nn.Sequential(nn.Linear(input_dim, 64), nn.Tanh(), nn.Linear(64, 1))
- forward(self, x):  # x: (batch, seq_len, dim)
  weights = F.softmax(self.attention(x).squeeze(-1), dim=1)  # (batch, seq_len)
  return (x * weights.unsqueeze(-1)).sum(dim=1)  # (batch, dim)

## File 2: src/models/yamnet_finetune.py

### Class: YAMNetFineTune
Level 3 — fine-tunable YAMNet in PyTorch.

IMPORTANT DESIGN DECISION: YAMNet is a TensorFlow model. For fine-tuning in
PyTorch, we have two options:
  Option A: Reimplement YAMNet's MobileNetV1 architecture in PyTorch and load
            converted weights. (Complex but clean.)
  Option B: Use torchaudio.models or a third-party PyTorch YAMNet port.
  Option C: Keep TF YAMNet frozen for embeddings, fine-tune only a PyTorch head.

Implement Option C as the practical choice for a master's thesis:

- __init__(self, freeze_yamnet=True, head_type='temporal', head_config={}):
  self.yamnet = YAMNetEmbedder()  # TF model, always on CPU
  self.freeze_yamnet = freeze_yamnet
  If head_type == 'logistic': self.head = EmbeddingLogisticRegression(**head_config)
  If head_type == 'temporal': self.head = TemporalClassifier(**head_config)
  self.head is a PyTorch model on GPU.

  If not freeze_yamnet, log a WARNING that YAMNet fine-tuning requires
  Option A (PyTorch reimplementation) and fall back to frozen.

- forward(self, waveforms_or_embeddings, input_type='embeddings'):
  If input_type == 'embeddings': pass directly to self.head
  If input_type == 'waveform':
    With torch.no_grad(): extract embeddings via self.yamnet
    Convert to PyTorch tensor, pass to self.head
  Return self.head output

NOTE: For true end-to-end fine-tuning (Level 3 as described in the review),
add a comment block explaining that the student should consider using
torchaudio's implementation or PANNs CNN14 as a PyTorch-native alternative
that CAN be fine-tuned end-to-end.

## File 3: src/models/domain_adaptation.py

### Class: DomainAdaptationModel
Level 4 — adds domain alignment loss.

- __init__(self, base_model, adaptation_method='coral', adaptation_weight=1.0):
  self.base_model = base_model  # any of the above models
  self.adaptation_method = adaptation_method  # 'coral' or 'mmd'
  self.adaptation_weight = adaptation_weight

- coral_loss(self, source_embeddings, target_embeddings):
  """
  CORAL: minimize || C_s - C_t ||_F^2
  where C_s, C_t are covariance matrices of source and target embeddings.
  1. Center both: x_s -= x_s.mean(0), x_t -= x_t.mean(0)
  2. Compute covariance: C = x^T x / (n-1)
  3. Return ||C_s - C_t||_F^2 / (4 * d^2) where d = embedding_dim
  """

- mmd_loss(self, source_embeddings, target_embeddings, kernel='rbf'):
  """
  Maximum Mean Discrepancy with RBF kernel.
  MMD^2 = E[k(x_s, x_s')] + E[k(x_t, x_t')] - 2*E[k(x_s, x_t)]
  Use multiple kernel bandwidths (median heuristic ± factors of 2).
  """

- forward(self, x, domain_labels=None):
  """
  x: input batch
  domain_labels: tensor of 0 (AeroSonicDB) or 1 (Norwegian) per sample
  Returns: {
    'logits': classification logits,
    'embeddings': intermediate embeddings,
    'domain_loss': scalar CORAL or MMD loss (0 if domain_labels is None)
  }
  """
  out = self.base_model(x)
  if domain_labels is not None:
    source_mask = domain_labels == 0
    target_mask = domain_labels == 1
    if source_mask.sum() > 1 and target_mask.sum() > 1:
      if self.adaptation_method == 'coral':
        d_loss = self.coral_loss(out['embeddings'][source_mask],
                                  out['embeddings'][target_mask])
      else:
        d_loss = self.mmd_loss(...)
      out['domain_loss'] = d_loss * self.adaptation_weight
    else:
      out['domain_loss'] = torch.tensor(0.0)
  return out

## File 4: src/models/multi_location_fusion.py

### Class: MultiLocationFusion
Score-level fusion across the 3 synchronized recording locations.

- __init__(self, method='mean', n_locations=3, learned_weights=False):
  method: 'mean', 'max', 'majority_vote', 'learned'
  If learned: self.weights = nn.Parameter(torch.ones(n_locations) / n_locations)

- forward(self, location_logits: dict[str, torch.Tensor]) -> torch.Tensor:
  """
  location_logits: dict mapping location_id -> logits tensor (batch,)
  Stack into (batch, n_locations), apply fusion:
  - mean: average probabilities (apply sigmoid first, then mean)
  - max: max probability across locations
  - majority_vote: threshold each at 0.5, majority wins
  - learned: sigmoid(logits) * softmax(self.weights), then sum
  Return fused logits (batch,)
  """

- fuse_predictions(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
  """
  Offline fusion for evaluation.
  predictions_df has columns: [segment_id, time_start, time_end, session,
    location, pred_prob, pred_label, true_label]
  Group by (session, time_start, time_end), apply fusion within each group.
  Return DataFrame with fused predictions.
  """

## File 5: src/models/ast_model.py

### Class: ASTClassifier
Audio Spectrogram Transformer wrapper for comparison experiments.

- __init__(self, pretrained=True, n_classes=1, freeze_backbone=False):
  Use the AST model from:
    pip install timm
    from timm import create_model
  Or implement using the MIT AST repo approach:
    1. Load a pretrained ViT (e.g., DeiT-base) from timm
    2. Modify patch embedding for audio spectrograms:
       - Input: (batch, 1, n_mels, time_frames)
       - Reshape into patches of size (16, 16)
       - Linear projection to embed_dim
    3. Add positional embeddings (learned, size depends on input length)
    4. Classification head: LayerNorm → Linear(embed_dim, 1)

  If student has limited compute, use a smaller ViT (tiny or small).

  Add a NOTE comment: "For a simpler alternative, consider using
  torchaudio.pipelines.WAV2VEC2_BASE or a pretrained PANNs CNN14 model
  from https://github.com/qiuqiangkong/audioset_tagging_cnn"

- forward(self, x):
  x: (batch, 1, n_mels, time_frames)
  Return {'logits': (batch,), 'embeddings': (batch, embed_dim)}
```

### Validation Checklist
- [ ] Each model instantiates without error
- [ ] Forward pass with dummy input produces correct output shapes
- [ ] CORAL and MMD loss return scalar tensors
- [ ] MultiLocationFusion correctly fuses 3 location predictions
- [ ] Total parameter counts are reasonable (log them)

**Git commit: `feat: all model architectures — L1 through L4 + AST + fusion`**

---

## Phase 7: Training Loop

### Objective
Build a configurable training loop supporting all model levels, domain adaptation, mixed-domain data, logging, checkpointing, and early stopping.

### Dependencies
- Phase 4 (datasets), Phase 6 (models) complete

### VS Code Prompt — Phase 7

```
# PHASE 7: Training Loop

## File 1: src/training/losses.py

### Class: SEDLoss
- __init__(self, pos_weight=None, domain_adaptation_weight=0.0, label_smoothing=0.0):
  self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
  self.domain_adaptation_weight = domain_adaptation_weight
  self.label_smoothing = label_smoothing

- forward(self, model_output: dict, targets: torch.Tensor, domain_labels=None):
  """
  model_output: dict with 'logits' and optionally 'domain_loss'
  targets: (batch,) float tensor of 0/1
  Apply label smoothing: targets = targets * (1 - eps) + 0.5 * eps

  total_loss = bce_loss
  if 'domain_loss' in model_output and domain_labels is not None:
    total_loss += self.domain_adaptation_weight * model_output['domain_loss']

  Return dict: {'total_loss': ..., 'bce_loss': ..., 'domain_loss': ...}
  """

## File 2: src/training/trainer.py

### Class: Trainer
A complete training manager.

- __init__(self, model, train_loader, val_loader, loss_fn, optimizer,
           scheduler, config, device, output_dir, fold_name):

  Store all components. Initialize:
  - TensorBoard SummaryWriter at output_dir/logs/fold_name
  - Best validation metric tracker (for early stopping)
  - Training history list

- train_epoch(self) -> dict:
  One full training epoch.
  1. model.train()
  2. Iterate over train_loader with tqdm
  3. For each batch:
     - Move data to device
     - Forward pass
     - Compute loss (pass domain_labels if available in batch)
     - Backward pass
     - Gradient clipping (max_norm=1.0)
     - Optimizer step
     - Track running loss, predictions, labels
  4. Compute epoch metrics: loss, F1, precision, recall, AUC-ROC
  5. Return metrics dict

- validate(self) -> dict:
  One full validation pass.
  1. model.eval(), torch.no_grad()
  2. Iterate val_loader, collect all predictions and labels
  3. Compute: loss, F1, precision, recall, AUC-ROC, optimal threshold
     (threshold that maximizes F1 on validation set)
  4. Return metrics dict including optimal_threshold

- train(self, n_epochs: int) -> dict:
  Full training run.
  1. For each epoch:
     a. train_epoch()
     b. validate()
     c. scheduler.step() (if ReduceLROnPlateau, pass val_loss)
     d. Log to TensorBoard: train/val loss, F1, AUC, LR
     e. If val F1 improved: save checkpoint (model state_dict, optimizer,
        epoch, metrics) to output_dir/checkpoints/fold_name_best.pt
     f. Early stopping: if val F1 hasn't improved for patience epochs, stop
  2. Load best checkpoint
  3. Return dict with best metrics, training history, total time

- save_checkpoint(self, path, epoch, metrics)
- load_checkpoint(self, path) -> dict

## File 3: scripts/04_train.py

Main training script.

Usage: python scripts/04_train.py --config configs/experiments/baseline_yamnet.yaml
                                   --fold loso_session3_rain

1. Load config (merge default + experiment-specific)
2. Set seed
3. Load manifest and fold JSON
4. Create train/val datasets and dataloaders
   - Train: with augmentation and balanced sampler
   - Val: no augmentation, sequential sampler
5. Instantiate model based on config (model.type = 'logistic' | 'temporal' |
   'finetune' | 'domain_adapt' | 'ast')
6. Compute pos_weight from training set class balance:
   pos_weight = n_negative / n_positive
7. Create loss, optimizer (AdamW), scheduler (CosineAnnealingWarmRestarts)
8. Create Trainer
9. trainer.train(config.training.epochs)
10. Save final results JSON to outputs/{experiment}/{fold_name}/results.json

Support running ALL folds sequentially:
  python scripts/04_train.py --config configs/experiments/baseline_yamnet.yaml --all-folds

This loops over all fold JSONs in data/splits/ and trains one model per fold.

## File 4: configs/experiments/baseline_yamnet.yaml

experiment_name: baseline_yamnet
model:
  type: logistic
  aggregation: mean
  input_type: embeddings
training:
  lr: 1.0e-3
  epochs: 30
  batch_size: 64
augmentation:
  enabled: false

## File 5: configs/experiments/temporal_yamnet.yaml

experiment_name: temporal_yamnet
model:
  type: temporal
  temporal_model: gru
  hidden_dim: 256
  n_layers: 2
  use_attention: true
  dropout: 0.3
  input_type: embeddings
training:
  lr: 5.0e-4
  epochs: 50
  batch_size: 32
augmentation:
  enabled: true
  noise_mix_prob: 0.5
  spec_augment: false  # not applicable to precomputed embeddings

## File 6: configs/experiments/domain_adapt.yaml

experiment_name: domain_adapted
model:
  type: domain_adapt
  base_model: temporal
  adaptation_method: coral
  adaptation_weight: 0.5
  temporal_model: gru
  hidden_dim: 256
  input_type: embeddings
training:
  lr: 5.0e-4
  epochs: 50
  batch_size: 32
augmentation:
  enabled: true

## File 7: configs/experiments/ast_comparison.yaml

experiment_name: ast_comparison
model:
  type: ast
  pretrained: true
  freeze_backbone: false
  input_type: spectrogram
training:
  lr: 1.0e-5
  epochs: 30
  batch_size: 16
augmentation:
  enabled: true
  noise_mix_prob: 0.5
  spec_augment: true
```

### Validation Checklist
- [ ] Training completes for at least 5 epochs on each model type without crashes
- [ ] TensorBoard logs show decreasing loss curves
- [ ] Checkpoints saved and loadable
- [ ] Early stopping triggers correctly (test with patience=3 on a quick run)
- [ ] `results.json` written with all metrics

**Git commit: `feat: training loop with early stopping, logging, and experiment configs`**

---

## Phase 8: Evaluation and Metrics

### Objective
Build a comprehensive evaluation suite: segment-level metrics, event-level metrics (SED-Eval), per-condition breakdowns, DET curves, and statistical significance tests.

### Dependencies
- Phase 7 complete (trained models available)

### VS Code Prompt — Phase 8

```
# PHASE 8: Evaluation and Metrics

## File 1: src/evaluation/metrics.py

### Function: compute_segment_metrics(y_true, y_pred_proba, threshold=0.5)
  Apply threshold to get y_pred.
  Return dict:
    precision, recall, f1, auc_roc, auc_pr (average precision),
    accuracy, specificity, balanced_accuracy,
    confusion_matrix (as 2x2 numpy array),
    optimal_threshold (threshold maximizing F1 via precision_recall_curve)
  Use sklearn.metrics throughout.

### Function: compute_metrics_at_optimal_threshold(y_true, y_pred_proba)
  Find optimal threshold on the data, then compute all metrics at that threshold.
  Return dict with all metrics plus the optimal_threshold value.

### Function: compute_per_condition_metrics(predictions_df)
  """
  predictions_df columns: [segment_id, session, location, y_true, y_pred_proba]
  Group by session, compute compute_segment_metrics for each.
  Return dict[session] -> metrics_dict.
  Also compute overall (all sessions pooled) metrics.
  """

## File 2: src/evaluation/sed_eval_wrapper.py

### Function: compute_event_metrics(predictions_df, collar_s=0.2, t_collar_s=0.2)
  """
  Convert segment-level predictions to event-level using sed_eval.
  1. From segment predictions, merge consecutive positive predictions into events:
     Each event has (onset_s, offset_s).
  2. Similarly, merge ground-truth labels into reference events.
  3. Use sed_eval.sound_event.SegmentBasedMetrics and
     sed_eval.sound_event.EventBasedMetrics with the specified collars.
  4. Return dict with:
     - segment_based_f1, segment_based_er (error rate)
     - event_based_f1 (with collar), event_based_precision, event_based_recall
     - deletion_rate, insertion_rate
  """

### Function: predictions_to_event_list(times, labels, min_event_duration_s=0.5)
  """Convert binary label sequence + timestamps to list of (onset, offset) events.
  Merge consecutive positive segments. Discard events shorter than min_event_duration_s."""

## File 3: src/evaluation/analysis.py

### Function: generate_det_curve(y_true, y_pred_proba, label='Model')
  """
  Generate Detection Error Tradeoff data.
  DET: x-axis = false alarm rate, y-axis = miss rate.
  Use sklearn.metrics.det_curve.
  Return (fpr, fnr) arrays and the figure.
  """

### Function: generate_per_condition_report(all_predictions: dict[str, pd.DataFrame])
  """
  all_predictions: dict[fold_name] -> predictions DataFrame
  Generate a comprehensive report:
  1. Per-condition metrics table (sessions as rows, metrics as columns)
  2. Confusion matrices per condition (as subplot figure)
  3. DET curves overlaid per condition (one plot)
  4. ROC curves overlaid per condition (one plot)
  5. Bar chart: F1 by weather condition (with confidence intervals)
  Save all figures and a summary CSV.
  """

### Function: snr_stratified_analysis(predictions_df, noise_profiles)
  """
  Estimate per-segment SNR using the session's noise profile.
  Bin segments by estimated SNR (e.g., <0 dB, 0-5, 5-10, 10-15, 15-20, >20).
  Compute metrics within each bin.
  Plot: F1 vs estimated SNR (line plot with error bars).
  """

## File 4: src/evaluation/statistical_tests.py

### Function: mcnemar_test(y_true, y_pred_a, y_pred_b)
  """
  McNemar's test comparing two classifiers.
  Build contingency table of disagreements.
  Return chi2 statistic, p-value, and whether the difference is significant at p<0.05.
  Use scipy.stats or statsmodels.
  """

### Function: bootstrap_ci(y_true, y_pred_proba, metric_fn, n_bootstraps=1000,
                            ci=0.95, seed=42)
  """
  Compute bootstrap confidence interval for a metric.
  metric_fn: callable that takes (y_true, y_pred_proba) and returns scalar.
  Resample with replacement n_bootstraps times, compute metric each time.
  Return (lower, upper, mean, std) of the bootstrap distribution.
  """

### Function: compare_conditions_significance(per_condition_predictions)
  """
  For each pair of weather conditions, run McNemar's test.
  Return a matrix of p-values.
  Apply Bonferroni correction for multiple comparisons.
  """

## File 5: scripts/05_evaluate.py

Main evaluation script.
Usage: python scripts/05_evaluate.py --experiment baseline_yamnet --all-folds

1. Load config
2. For each fold's best checkpoint:
   a. Load model
   b. Load test set from the fold JSON
   c. Run inference (model.eval, torch.no_grad) on all test segments
   d. Collect: segment_id, session, location, start_s, end_s, y_true, y_pred_proba
   e. Apply the optimal threshold found during validation
   f. Save predictions to outputs/{experiment}/{fold}/predictions.csv
3. After all folds:
   a. Concatenate all test predictions
   b. Call generate_per_condition_report()
   c. Call snr_stratified_analysis()
   d. Call compare_conditions_significance()
   e. Save full report to outputs/{experiment}/evaluation_report/

## File 6: scripts/06_analyse_results.py

Comparison script across ALL experiments.
1. Load predictions.csv from each experiment directory
2. Generate comparison tables:
   - Rows: weather conditions, Columns: experiment names, Values: F1 (95% CI)
3. Generate comparison plots:
   - Grouped bar chart: F1 by condition, grouped by experiment
   - DET curve comparison: best model per experiment overlaid
4. Run McNemar's test between key model pairs (e.g., baseline vs domain-adapted)
5. Save all outputs to outputs/comparison/
```

### Validation Checklist
- [ ] `compute_segment_metrics` produces correct results on synthetic data
- [ ] SED-eval event metrics run without errors
- [ ] DET curves and per-condition reports generate readable figures
- [ ] Bootstrap CIs are reasonable (not too wide, not degenerate)
- [ ] Cross-experiment comparison table is generated

**Git commit: `feat: comprehensive evaluation suite with SED-eval, DET curves, and statistical tests`**

---

## Phase 9: Multi-Location Fusion Experiments

### Objective
Exploit the 3-site synchronized recording setup to fuse predictions across locations.

### Dependencies
- Phase 8 complete (per-location predictions available)

### VS Code Prompt — Phase 9

```
# PHASE 9: Multi-Location Fusion

## File: scripts/07_multi_location_fusion.py

Implement multi-location fusion as a post-processing step on existing predictions.

1. Load predictions.csv from a trained experiment (all LOSO folds).

2. For each LOSO fold (test session):
   a. Filter predictions to the test session
   b. Group by (time_start, time_end) — these should align across locations
      because recordings are time-synchronized
   c. For each time window, collect the pred_prob from each location
      (should have 3 values, one per location)

3. Apply each fusion method:
   - mean_prob: average of 3 probabilities
   - max_prob: maximum of 3 probabilities
   - majority_vote: threshold each at optimal_threshold, take majority
   - product: geometric mean of probabilities (multiply and take cube root)
   - minimum: minimum probability (conservative — all locations must agree)

4. For each fusion method:
   a. Compute segment-level metrics (F1, AUC-ROC, precision, recall)
   b. Compute event-level metrics
   c. Compare with single-best-location and single-worst-location baselines

5. Generate comparison table:
   Rows: fusion methods + single-location baselines
   Columns: F1, AUC-ROC, Precision, Recall (± bootstrap CI)

6. Generate plots:
   - Bar chart comparing fusion methods
   - Scatter plot: per-event agreement level (how many locations detected it)
     vs event SNR
   - Timeline visualization for one example session: show per-location
     predictions and fused prediction vs ground truth over time

7. Analyze: for each false alarm and missed detection, report how many
   locations agreed. This reveals whether fusion is filtering out
   single-location artifacts.

8. Save all results to outputs/{experiment}/fusion/

## IMPORTANT: Time alignment
The Norwegian data has 3 locations per session, time-synchronized. When
grouping, match segments by their time_start values. If there's slight
misalignment, use a tolerance window of ±0.5 seconds.
If a time window has fewer than 3 location predictions (edge of recording),
fall back to mean of available predictions.
```

### Validation Checklist
- [ ] Fusion produces predictions for all test time windows
- [ ] At least one fusion method improves over best single-location
- [ ] Timeline visualization clearly shows the fusion effect
- [ ] Results table saved to CSV

**Git commit: `feat: multi-location fusion experiments`**

---

## Phase 10: Self-Supervised Pretraining (Optional/Advanced)

### Objective
Continue-pretrain the audio encoder on unlabeled Norwegian audio using masked spectrogram modelling. This adapts features to the target domain without using labels.

### Dependencies
- Phase 1 complete (Norwegian audio available)

### VS Code Prompt — Phase 10

```
# PHASE 10: Self-Supervised Domain Pretraining (Advanced / Optional)

This phase implements masked spectrogram modelling (MSM) for domain adaptation.
The idea: before fine-tuning on labels, pretrain the audio feature extractor on
ALL Norwegian audio (ignoring labels) so it learns the acoustic statistics of
the Norwegian environment.

## File: src/models/self_supervised.py

### Class: MaskedSpectrogramModel(nn.Module)
  Simplified SSAST-style self-supervised pretraining.

- __init__(self, encoder, mask_ratio=0.3, n_mels=64, patch_size=16):
  encoder: a CNN or transformer that takes (batch, 1, n_mels, T) spectrograms
  and outputs (batch, embed_dim, H, W) feature maps.

  self.decoder = nn.Sequential(
    nn.ConvTranspose2d(embed_dim, 128, kernel_size=patch_size, stride=patch_size),
    nn.ReLU(),
    nn.Conv2d(128, 1, kernel_size=1)
  )  # reconstruct the masked spectrogram patches

- forward(self, spectrogram):
  1. Divide spectrogram into non-overlapping patches of size (patch_size, patch_size)
  2. Randomly mask mask_ratio fraction of patches (replace with learned mask token
     or zeros)
  3. Pass masked spectrogram through encoder
  4. Decode to reconstruct the ORIGINAL (unmasked) spectrogram
  5. Compute MSE loss ONLY on the masked patches (not the visible ones)
  Return {'reconstruction_loss': loss, 'encoder_features': features}

### Training script section in scripts/04_train.py:

Add a --pretrain flag:
  python scripts/04_train.py --pretrain --config configs/experiments/self_supervised.yaml

When --pretrain:
  1. Load ALL Norwegian audio segments (all sessions, all locations, ignore labels)
  2. Create dataset returning raw spectrograms
  3. Train MaskedSpectrogramModel for N epochs
  4. Save the encoder weights to outputs/pretrained/encoder.pt
  5. These weights are then loaded by subsequent training runs as initialization

## Config: configs/experiments/self_supervised.yaml
experiment_name: self_supervised_pretrain
model:
  type: self_supervised
  encoder: cnn14  # or small_vit
  mask_ratio: 0.3
  patch_size: 16
training:
  lr: 1.0e-4
  epochs: 100
  batch_size: 64
  weight_decay: 1.0e-5
data:
  use_all_norwegian: true
  ignore_labels: true

## NOTE to student:
This is the most complex phase and is optional. A simpler alternative is to
use a pretrained BYOL-A model (https://github.com/nttcslab/byol-a) and
fine-tune it on your data. BYOL-A provides pretrained audio representations
that may already be partially adapted to diverse acoustic conditions.
```

### Validation Checklist
- [ ] Pretraining loss decreases over epochs
- [ ] Encoder weights saved and loadable
- [ ] Downstream fine-tuning with pretrained weights shows improvement over random init

**Git commit: `feat: self-supervised pretraining for domain adaptation (optional)`**

---

## Phase 11: Subband Attention (Optional Enhancement)

### Objective
Implement frequency subband analysis where the model can learn to weight different frequency regions, allowing it to suppress weather-corrupted bands.

### VS Code Prompt — Phase 11

```
# PHASE 11: Subband Attention

## File: src/models/subband_attention.py

### Class: SubbandAttentionClassifier(nn.Module)

Decompose the mel spectrogram into frequency subbands and process them with
parallel pathways, then aggregate with learned attention.

- __init__(self, n_mels=64, n_subbands=4, subband_channels=64,
           classifier_dim=256, dropout=0.3):

  Define subband boundaries for 64 mel bands:
    subbands = [(0, 16), (16, 32), (32, 48), (48, 64)]
    # Roughly: 125-500Hz, 500-1500Hz, 1500-3500Hz, 3500-7500Hz
    # Wind noise dominates band 0; rain affects bands 0-2;
    # aircraft energy concentrated in bands 0-2

  For each subband, create a small CNN pathway:
    self.subband_cnns = nn.ModuleList([
      nn.Sequential(
        nn.Conv2d(1, subband_channels, kernel_size=(3,3), padding=1),
        nn.BatchNorm2d(subband_channels),
        nn.ReLU(),
        nn.Conv2d(subband_channels, subband_channels, kernel_size=(3,3), padding=1),
        nn.BatchNorm2d(subband_channels),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1))  # → (batch, channels, 1, 1)
      ) for _ in range(n_subbands)
    ])

  Attention mechanism over subbands:
    self.subband_attention = nn.Sequential(
      nn.Linear(subband_channels * n_subbands, 128),
      nn.ReLU(),
      nn.Linear(128, n_subbands),
      nn.Softmax(dim=-1)
    )

  Classifier:
    self.classifier = nn.Sequential(
      nn.Linear(subband_channels * n_subbands, classifier_dim),
      nn.ReLU(), nn.Dropout(dropout),
      nn.Linear(classifier_dim, 1)
    )

- forward(self, x):
  x: (batch, 1, 64, time_frames)

  1. Split x into subbands along the mel axis:
     subband_inputs = [x[:, :, start:end, :] for (start, end) in self.subbands]
  2. Process each through its CNN:
     subband_features = [cnn(sb).squeeze(-1).squeeze(-1) for cnn, sb in zip(...)]
     → list of (batch, subband_channels) tensors
  3. Concatenate: all_features = torch.cat(subband_features, dim=1)
     → (batch, subband_channels * n_subbands)
  4. Compute attention weights: attn = self.subband_attention(all_features)
     → (batch, n_subbands)
  5. Weighted aggregation:
     weighted = sum(attn[:, i].unsqueeze(1) * subband_features[i] for i in range(n_subbands))
     → (batch, subband_channels)
     OR keep the concatenated features and just use attention for analysis.
  6. Classifier: logits = self.classifier(all_features)

  Return {
    'logits': logits.squeeze(-1),
    'subband_attention_weights': attn,  # (batch, n_subbands) — for analysis
    'embeddings': all_features
  }

## Analysis code (add to scripts/06_analyse_results.py):

After training the subband model, extract attention weights for all test segments.
Plot:
- Average attention weight per subband, per weather condition (grouped bar chart)
  This should reveal: wind conditions → model downweights low-frequency subband,
  rain → model shifts attention to mid-frequency bands.
- Attention weight heatmap: x-axis = time within a flyover event, y-axis = subbands,
  showing how attention shifts as the aircraft approaches and recedes.
```

### Validation Checklist
- [ ] Forward pass produces correct shapes
- [ ] Attention weights sum to 1 across subbands
- [ ] Attention weight analysis shows interpretable patterns across weather conditions

**Git commit: `feat: subband attention model for weather-robust detection`**

---

## Phase 12: Final Report Generation

### Objective
Generate all thesis figures, tables, and LaTeX-ready outputs from the experimental results.

### VS Code Prompt — Phase 12

```
# PHASE 12: Final Thesis Report Generation

## File: scripts/generate_thesis_figures.py

A single script that reads all experiment results and generates publication-ready
figures and tables for the thesis.

1. LOAD all results:
   - For each experiment in outputs/: load predictions.csv, results.json
   - Load noise profiles from Phase 2
   - Load fusion results from Phase 9

2. GENERATE the following figures (save as both PNG@300dpi and PDF):

   Fig 1: Dataset overview
   - Spectrogram examples from each weather condition (5-panel figure)
   - Include one aircraft event and one no-aircraft segment per condition

   Fig 2: Noise profiles
   - Overlay plot of mean noise spectrum per session (from Phase 2)
   - Annotate frequency bands affected by wind vs rain

   Fig 3: Model comparison — F1 by weather condition
   - Grouped bar chart: x-axis = weather conditions, bars = model variants
   - Include 95% bootstrap CIs as error bars
   - Models: baseline (L1), temporal (L2), domain-adapted (L4), AST

   Fig 4: DET curves
   - One subplot per weather condition, all models overlaid
   - OR one plot with all conditions for the best model

   Fig 5: Confusion matrices
   - 5×1 grid: one confusion matrix per weather condition for the best model
   - Use seaborn heatmap with annotated counts

   Fig 6: SNR-stratified performance
   - Line plot: F1 vs estimated SNR, one line per weather condition
   - Shaded bootstrap CI region

   Fig 7: Multi-location fusion comparison
   - Bar chart comparing fusion methods vs single-location baselines

   Fig 8: Subband attention analysis (if Phase 11 completed)
   - Grouped bar chart: attention weights per subband per weather condition

   Fig 9: Training curves
   - Loss and F1 over epochs for the best model (train and val)

   Fig 10: Performance degradation summary
   - Single plot showing F1 drop from clear (Session 1) to each other condition
   - For each model variant, show the degradation as a bar or connected line

3. GENERATE tables (save as CSV and LaTeX):

   Table 1: Dataset statistics
   - Sessions, locations, total hours, #aircraft events, #segments, class balance

   Table 2: Main results
   - Rows: model variants, Columns: weather conditions + Overall
   - Values: F1 (95% CI)

   Table 3: Statistical significance
   - Pairwise McNemar's test p-values (Bonferroni corrected)

   Table 4: Fusion results
   - Rows: fusion methods, Columns: metrics

4. LaTeX table formatting:
   For each table, also generate a .tex file with a properly formatted
   LaTeX tabular environment that can be \input{} directly into the thesis.
   Use booktabs style (\toprule, \midrule, \bottomrule).
   Bold the best result in each column.

5. Save everything to outputs/thesis_figures/ with descriptive filenames.

## Style requirements:
- Use matplotlib style: plt.style.use('seaborn-v0_8-paper')
- Font size: 11pt for labels, 13pt for titles
- Color palette: use a colorblind-friendly palette (e.g., seaborn 'colorblind')
- Figure width: single-column = 3.5 inches, double-column = 7 inches
  (standard for IEEE/Elsevier LaTeX templates)
- All text in figures should use LaTeX rendering: plt.rcParams['text.usetex'] = True
  (if LaTeX is installed; fall back to mathtext otherwise)
```

### Validation Checklist
- [ ] All 10 figures generated without errors
- [ ] Figures are visually clean and publication-ready
- [ ] LaTeX tables compile without errors
- [ ] All numbers in tables match the predictions.csv source data

**Git commit: `feat: thesis figure and table generation pipeline`**

---

## Execution Summary

| Phase | Description | Est. Time | Key Deliverable |
|---|---|---|---|
| 0 | Scaffolding | 1 day | Project structure, config, environment |
| 1 | Preprocessing | 2 days | Resampled audio, mel spectrograms, manifest |
| 2 | Noise profiling | 1 day | Per-session noise spectra, exploration notebook |
| 3 | LOSO splits | 1 day | 6 fold JSON files with leakage tests |
| 4 | Dataset + augmentation | 2 days | PyTorch datasets, real-noise augmentation |
| 5 | YAMNet embeddings | 1 day | Cached 1024-dim embeddings for all segments |
| 6 | Model architectures | 3 days | L1–L4, AST, fusion modules |
| 7 | Training loop | 2 days | Trainer, configs, checkpointing |
| 8 | Evaluation suite | 2 days | Metrics, SED-eval, DET, bootstrap CIs |
| 9 | Multi-location fusion | 1 day | Fusion comparison and analysis |
| 10 | Self-supervised (optional) | 2 days | Pretrained encoder |
| 11 | Subband attention (optional) | 1 day | Interpretable subband model |
| 12 | Thesis figures | 1 day | All publication-ready figures and tables |
| **Total** | | **~18–20 days** | |

---

## Appendix: Git Workflow

```bash
# Start each phase on a feature branch
git checkout -b phase-N-description

# After completing the phase and passing all checks
git add -A
git commit -m "feat: [description from phase]"

# Merge to main
git checkout main
git merge phase-N-description
git tag vN.0 -m "Phase N complete"
```
