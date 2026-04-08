from __future__ import annotations

import numpy as np

from keras_yamnet import preprocessing as kp


class YAMNetFeatureExtractor:
    """Wrapper around local keras_yamnet preprocessing pipeline."""

    def __init__(self, apply_filter: str | None = None):
        self.apply_filter = apply_filter

    def extract(self, audio_int16: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (patches, mel_spec) from keras_yamnet.preprocessing.preprocess_input."""
        if audio_int16.dtype != np.int16:
            audio_int16 = np.asarray(audio_int16, dtype=np.int16)

        patches, mel_spec = kp.preprocess_input(audio_int16, int(sr), apply_filter=self.apply_filter)
        return patches.astype(np.float32), mel_spec.astype(np.float32)
