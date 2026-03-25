# Critical Review: Aircraft Sound Event Detection Under Varying Meteorological Conditions

**Reviewer notes for thesis supervision — Prof. Adil Rasheed, NTNU**

---

## 1. Observations from the Sample Audio

The uploaded sample (`06A1BD_2022-12-12_13-38-31_0_1.wav`) is a 53-second mono recording at 22,050 Hz with the following characteristics:

| Property | Value |
|---|---|
| Duration | 53.0 s |
| Sample rate | 22,050 Hz (Nyquist: 11,025 Hz) |
| Bit depth | 16-bit PCM |
| Dynamic range | \[−0.42, +0.46\] |
| RMS energy | 0.081 (≈ −22 dB FS) |
| Peak-to-RMS ratio | ~15 dB |
| Spectral centroid (mean) | 779 Hz |
| Spectral bandwidth (mean) | 1,380 Hz |

The waveform shows a clear aircraft flyover event with a gradual amplitude build-up from ~10 s, peaking around 25–35 s (RMS ≈ −13 dB), and a slower decay—consistent with a propeller aircraft or turboprop approach/departure. The spectral energy is concentrated below 2 kHz, with occasional transient high-frequency spikes (likely bioacoustic or environmental). The log-mel spectrogram reveals broadband low-frequency energy during the flyover and a relatively low background noise floor (−35 to −40 dB RMS) in the quiet segments.

**Key concern:** The 22,050 Hz sample rate is fine for aircraft detection (most aircraft acoustic energy is below 5 kHz), but it deviates from YAMNet's native expectation of 16 kHz input. The student must explicitly resample to 16 kHz or document the implication of feeding 22 kHz content through YAMNet's preprocessing. This mismatch alone can create silent performance degradation.

---

## 2. Critical Assessment of the Proposed Approach

### 2.1 YAMNet as the Base Model — Strengths and Weaknesses

**Strengths.** YAMNet provides a strong pretrained audio embedding backbone (MobileNetV1 on AudioSet), and using transfer learning from a general-purpose audio classifier is a sensible baseline for a binary detection task. The log-mel spectrogram front-end is standard and interpretable.

**Weaknesses the student should address:**

1. **YAMNet was trained on AudioSet at 16 kHz with specific mel parameters** (64 bands, 25 ms window, 10 ms hop, 125–7500 Hz). The student must replicate these *exactly* or quantify the impact of deviation. Using 22 kHz input without resampling shifts the effective mel bin frequencies.

2. **YAMNet's 0.96 s patch-based inference** (96 frames × 10 ms) may not be well-suited to the slow temporal evolution of aircraft flyover events, which unfold over 20–60 seconds. A single 0.96 s patch captures a snapshot, not the characteristic temporal envelope (gradual onset, sustained, gradual offset). The student should consider **temporal aggregation strategies** — e.g., attention pooling or an LSTM/GRU layer over a sequence of patch-level embeddings — rather than treating each patch independently.

3. **Binary classification is fragile in the presence of weather noise.** Rain, wind, and other meteorological sounds may produce broadband spectral energy that overlaps heavily with aircraft signatures below 2 kHz. The model needs to learn to disentangle these, not just detect "energy present." The student should examine whether the confusion matrix errors under weather conditions are dominated by false positives (weather → aircraft) or false negatives (aircraft masked by weather).

4. **MobileNetV1 backbone is relatively shallow.** More modern audio backbones — PANNs (CNN14), BEATs, Audio Spectrogram Transformer (AST) — have shown substantially better performance on AudioSet and downstream tasks. At minimum, the student should benchmark against one alternative backbone to justify the YAMNet choice, or acknowledge it as a limitation.

### 2.2 Critique of Idea 1: Data Augmentation with Synthetic Weather Noise

This idea has merit but is insufficient on its own, and must be implemented carefully:

**Problems:**

- **Synthetic rain/wind is not real rain/wind.** Adding white/pink noise or downloaded rain sound effects does not capture the actual spectral-temporal characteristics of the Norwegian recording environment. Real wind noise has strong low-frequency components that interact with the microphone (turbulence-induced pressure fluctuations), which synthetic additive noise does not replicate. Real rain produces impact noise on surfaces near the microphone that has site-specific spectral colouring.
- **The domain gap is not only about weather.** It also includes microphone hardware differences, acoustic propagation differences (terrain, vegetation, temperature gradients affecting sound refraction), aircraft fleet differences (Australian vs. Norwegian traffic), and recording distance/geometry differences. Augmentation addresses only one dimension of mismatch.
- **Risk of augmentation mismatch.** If the synthetic noise is unrealistic, the model learns to reject *synthetic* noise but not *real* weather, giving an illusion of robustness that collapses on the Norwegian test set.

**Improvement:** If pursuing augmentation, the student should extract *actual* background noise segments from the Norwegian dataset's "no-aircraft" periods and use those for mixing. This is far more representative than synthetic noise. However, this bleeds Norwegian acoustic characteristics into training, which must be handled carefully (see Section 3 on splitting).

### 2.3 Critique of Idea 2: Mixed-Domain Training

This is the stronger of the two ideas, but the splitting strategy is critical:

**The fundamental question is: what is the unit of independence?**

Within a single recording session at one location, consecutive audio segments are highly correlated — they share the same microphone, background noise profile, weather state, and potentially the same aircraft events. A random train/test split within a session would cause severe data leakage, because the model would learn location-specific and session-specific patterns rather than genuinely generalising.

**Recommendation — session-level leave-one-out is the only defensible strategy.** See Section 3 below.

---

## 3. Recommended Experimental Design

### 3.1 Primary Evaluation Protocol: Session-Level Cross-Validation

The Norwegian dataset's structure (5 sessions × 3 locations × ~2 hours) is its greatest asset. Each session represents a distinct, self-contained meteorological condition. The student should adopt a **leave-one-session-out** (LOSO) evaluation:

| Experiment | Train | Validate | Test | Purpose |
|---|---|---|---|---|
| Baseline | AeroSonicDB only | AeroSonicDB (held-out) | All 5 Norwegian sessions | Quantify raw domain gap |
| LOSO-1 | AeroSonicDB + Sessions 2,3,4,5 | Subset of training sessions | Session 1 (clear) | Test on clear weather |
| LOSO-2 | AeroSonicDB + Sessions 1,3,4,5 | Subset of training sessions | Session 2 (windy) | Test on wind |
| LOSO-3 | AeroSonicDB + Sessions 1,2,4,5 | Subset of training sessions | Session 3 (rainy) | Test on rain |
| LOSO-4 | AeroSonicDB + Sessions 1,2,3,5 | Subset of training sessions | Session 4 (snow) | Test on snow |
| LOSO-5 | AeroSonicDB + Sessions 1,2,3,4 | Subset of training sessions | Session 5 (wind+rain) | Test on combined |

**Why this is defensible:** The test session has no temporal, spatial, or meteorological overlap with training. Each fold produces a weather-specific performance estimate. The five results together answer the thesis question directly.

**Validation split:** Within the training sessions, hold out one location per session (e.g., Location C from each session) as validation, or use a random 80/20 split *within* the training sessions. The key constraint is that the *test session* is never touched during any training or hyperparameter decision.

**Location-level analysis:** Because each session has three synchronized locations at ~3 km separation, the student can additionally report per-location variance *within* a session, which quantifies spatial robustness at the acoustic propagation scale.

### 3.2 Handling the Validation Set

For mixed-domain training (AeroSonicDB + Norwegian subsets), the validation set should draw from **both** domains, weighted toward the Norwegian data, since that is where the model must ultimately perform. A practical split:

- AeroSonicDB validation: 10–15% of AeroSonicDB, random split (within-domain data is less correlated)
- Norwegian validation: one full location from each training session

This ensures that early stopping and hyperparameter selection reflect the target domain.

---

## 4. Five Additional Methodological Ideas

### Idea 3: Feature-Level Domain Adaptation (MMD or CORAL Loss)

Add an explicit **domain alignment loss** to the training objective. During training, minimize the maximum mean discrepancy (MMD) or the correlation alignment (CORAL) distance between the feature distributions of AeroSonicDB and Norwegian embeddings, in addition to the standard binary cross-entropy. This forces the learned representation to be domain-invariant before the classification head.

**Why this is better than naïve mixed training:** Mixed training lets the model implicitly handle domain shift, but without an explicit alignment signal, the classifier may simply learn to distinguish domains and develop separate decision boundaries for each, which does not generalize to unseen Norwegian conditions.

### Idea 4: Teacher–Student Knowledge Distillation

Train a "teacher" model on AeroSonicDB (where labels are abundant and clean), then use it to generate soft labels (probability scores) for unlabeled or weakly labeled Norwegian data. Train a "student" model on the combined hard-label and soft-label data. The soft labels provide a regularization signal that bridges the domain gap.

**Practical variant:** Use the teacher to pseudo-label the quiet (no-aircraft) periods of the Norwegian sessions with confidence scores, and include high-confidence pseudo-labels in student training.

### Idea 5: Frequency-Domain Attention / Subband Analysis

Rather than feeding the full 64-band mel spectrogram to a monolithic CNN, decompose the representation into **frequency subbands** (e.g., 0–500 Hz, 500–2000 Hz, 2000–8000 Hz) and let the model attend to each subband with learned weights. Weather noise predominantly affects specific bands: wind noise is concentrated below 500 Hz, rain noise is more broadband but with energy peaks around 1–5 kHz. A subband-attention mechanism can learn to suppress the weather-corrupted bands while preserving the aircraft-informative ones.

**Implementation:** After the mel spectrogram extraction, split the 64 mel bands into 3–4 groups, process each through parallel convolutional paths, and aggregate via a learned attention vector.

### Idea 6: Multi-Location Fusion (Late or Score-Level)

Since each session has three synchronized recording locations separated by ~3 km, the student can exploit **spatial redundancy**. An aircraft will produce a correlated signal across all three locations (with time delays), while weather noise is either correlated (rain, wind) or semi-correlated. At inference time, fuse the per-location predictions (e.g., by majority voting, average probability, or a learned fusion layer). This provides a direct mechanism to reject location-specific false alarms.

**This is arguably the thesis's most distinctive contribution opportunity** — it exploits the unique data collection setup in a way that no generic method does.

### Idea 7: Self-Supervised Pretraining on Unlabeled Norwegian Audio

Before fine-tuning, train or continue-pretrain the audio encoder using a **self-supervised objective** (e.g., masked spectrogram modelling, contrastive learning) on the full ~30 hours of Norwegian audio (all sessions, all locations, ignoring labels). This adapts the feature extractor to the Norwegian acoustic domain without requiring labels, and avoids any leakage because the model never sees labels during this phase. Then fine-tune the classifier head on the labeled data.

**Framework:** Use SSAST (Self-Supervised Audio Spectrogram Transformer) or a contrastive approach like BYOL-A on the Norwegian recordings, then fine-tune a linear probe or small MLP on top.

---

## 5. Preprocessing Recommendations

### 5.1 Resampling
Resample all audio to **16 kHz** to match YAMNet's native expectations. Document the resampling method (e.g., `librosa.resample` with `res_type='kaiser_best'`). This is non-negotiable if using YAMNet.

### 5.2 Normalization
Apply **per-clip z-score normalization** to the log-mel spectrogram (subtract mean, divide by standard deviation computed per clip). This mitigates absolute level differences between datasets and recording setups. Do *not* use global statistics computed across the full dataset, as this leaks information about dataset composition.

### 5.3 Segment Length and Overlap
Given the slow temporal evolution of aircraft events (20–60 s), the student should experiment with segment lengths longer than YAMNet's default 0.96 s. Reasonable options: segment the audio into 5 s or 10 s windows with 50% overlap, extract YAMNet embeddings for each 0.96 s sub-patch within the window, and aggregate (mean, max, or attention pooling) to produce a single embedding per window. The label for the window should be "aircraft" if any portion overlaps with a ground-truth aircraft event.

### 5.4 Background Noise Profiling
For each session, compute an **average noise spectrum** from the labeled no-aircraft segments. This serves two purposes: (a) it provides a quantitative description of the weather-specific noise floor for the thesis analysis, and (b) it can be used for spectral subtraction as a preprocessing step to reduce stationary noise before feature extraction.

### 5.5 Handling Microphone Wind Noise
Apply a **high-pass filter at 50–100 Hz** before mel spectrogram extraction. Turbulent wind pressure on the microphone capsule produces large low-frequency artifacts that are not informative for aircraft detection but can dominate the energy and confuse the model. Alternatively, remove the lowest 2–3 mel bands after extraction.

---

## 6. Model Architecture Recommendations

### 6.1 Recommended Architecture Progression

The thesis should present a progression of increasing complexity, with each step justified by empirical improvement:

**Level 1 — Frozen YAMNet + Logistic Regression.** Extract YAMNet embeddings (1024-dimensional) for each patch, train a logistic regression classifier on top. This is the simplest baseline and sets a floor.

**Level 2 — Frozen YAMNet + Temporal Aggregation.** Feed sequences of patch-level embeddings (e.g., covering 10 s of audio) into a GRU or temporal attention layer, then classify. This captures the temporal structure of flyover events.

**Level 3 — Fine-tuned YAMNet.** Unfreeze the top N layers of YAMNet and fine-tune end-to-end on the combined AeroSonicDB + Norwegian training data. Use a low learning rate (1e-5 to 1e-4) for the pretrained layers and a higher rate for the classifier head.

**Level 4 — Fine-tuned YAMNet + Domain Adaptation Loss.** Add MMD or CORAL loss as per Idea 3.

At each level, report the same metrics on the same LOSO folds. This progression tells a coherent story and isolates the contribution of each component.

### 6.2 Alternative Backbone: AST or BEATs

As a secondary experiment (not the main thesis contribution), swap YAMNet for the **Audio Spectrogram Transformer (AST)** and compare. AST accepts the same mel-spectrogram input but uses a Vision Transformer architecture and has shown state-of-the-art results on AudioSet-derived tasks. If the student has compute budget, this comparison adds significant value to the thesis by disentangling "the method works" from "the backbone works."

---

## 7. Evaluation Metrics

### 7.1 Primary Metrics
For binary aircraft detection, the student should report:

- **Segment-level metrics:** Precision, Recall, F1-score, and AUC-ROC at the chosen segment granularity (e.g., 1 s or 5 s segments).
- **Event-level metrics:** Use the **SED-Eval** framework (segment-based and event-based F1). Event-based evaluation counts a detection as correct if the predicted event overlaps with a ground-truth event beyond a threshold (e.g., 200 ms onset tolerance). This is more meaningful than segment-level metrics for operational use.
- **Per-condition breakdown:** Report all metrics *separately* for each session/weather condition. The interesting result is not a single number but the **performance degradation curve** across conditions.

### 7.2 Avoid Accuracy

Do **not** use classification accuracy as a primary metric. Aircraft events are sparse (most of each 2-hour recording is "no aircraft"), so a trivial classifier predicting "no aircraft" always achieves high accuracy. F1-score and AUC-ROC are far more informative for imbalanced detection tasks.

### 7.3 Recommended Additional Analyses

- **Confusion matrix per weather condition** — Reveal whether errors are primarily false positives (weather misclassified as aircraft) or false negatives (aircraft masked by weather).
- **Detection Error Tradeoff (DET) curve** — Plot miss rate vs. false alarm rate across operating thresholds, per condition. This is standard in acoustic event detection literature and directly shows the cost of weather degradation.
- **Statistical significance testing** — Use McNemar's test or bootstrap confidence intervals to determine whether performance differences between weather conditions (or between models) are statistically significant, not just artefacts of small sample sizes.
- **SNR-stratified analysis** — Estimate per-segment SNR (aircraft energy relative to the noise floor of that session) and report detection performance as a function of SNR. This decouples weather type from signal strength and provides a more mechanistic understanding.

---

## 8. Risks and Pitfalls

### 8.1 Data Leakage Vectors
- **Temporal leakage within sessions:** If segments from the same aircraft flyover appear in both training and test, the model memorises the event. Ensure that entire events (not just segments) are kept together in the same split.
- **Location leakage across sessions:** If the same physical location appears in both training and test sessions, the model may learn location-specific background noise. This is partially mitigated by LOSO, but the student should check whether the three locations are the same across sessions. If they are, this is a confound that should be discussed.
- **Label leakage via augmentation:** If Norwegian background noise is used for augmentation (as recommended in Section 2.2), segments used for noise extraction must come only from the training sessions, never the test session.

### 8.2 Confounds That Could Invalidate Conclusions
- **Aircraft traffic differences across sessions.** If Session 2 (windy) happens to have fewer aircraft events, or different aircraft types, than Session 1 (clear), the performance difference may reflect traffic variation rather than weather impact. The student must report the number and type distribution of aircraft events per session and discuss this confound.
- **Time-of-day and seasonal effects.** Background noise varies with time of day (bird activity, human activity). If sessions were recorded at different times of day or in different seasons, this is a confound.
- **Microphone placement and housing.** If microphones were sheltered from weather differently across sessions, the acoustic impact of weather on the recordings is partly a hardware artefact, not a pure environmental effect. This should be documented.

### 8.3 Limitations to Discuss Explicitly
- **Small-sample weather conditions.** Each weather condition is represented by a single session (~6 hours × 3 locations). This means the thesis characterises *one instance* of rain, not rain in general. The student should acknowledge that generalization to other rain intensities, durations, or locations is not established.
- **Binary detection only.** The thesis does not address aircraft type classification, distance estimation, or direction of arrival — all of which are operationally relevant and also weather-dependent.
- **Single geographic site for Norwegian data.** Results may not generalise to other Norwegian locations with different terrain, vegetation, or traffic patterns.
- **Label quality.** How were the aircraft / no-aircraft labels generated? If by manual annotation, inter-annotator agreement should be reported. If by ADS-B correlation, the matching tolerance should be documented and its impact on label noise analysed.

---

## 9. Summary of Recommendations

| Aspect | Recommendation |
|---|---|
| **Splitting** | Leave-one-session-out (LOSO), never random within sessions |
| **Preprocessing** | Resample to 16 kHz; per-clip z-norm; high-pass at 50–100 Hz |
| **Augmentation** | Use real Norwegian background noise from training sessions, not synthetic |
| **Architecture** | Progressive: frozen embeddings → temporal aggregation → fine-tuning → domain adaptation |
| **Domain adaptation** | MMD/CORAL loss or self-supervised pretraining on unlabeled Norwegian audio |
| **Unique contribution** | Multi-location score fusion exploiting the 3-site synchronized setup |
| **Metrics** | F1, AUC-ROC, event-based F1 (SED-Eval), all per weather condition |
| **Backbone comparison** | Benchmark YAMNet vs. AST or PANNs CNN14 |
| **Statistical rigour** | Bootstrap CIs or McNemar's test; SNR-stratified analysis |
| **Key pitfall** | Ensure aircraft event counts are comparable across sessions |
