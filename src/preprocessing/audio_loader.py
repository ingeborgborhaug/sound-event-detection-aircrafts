from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf


class AudioLoader:
    """Load, convert to mono, and segment audio clips."""

    def load_mono_int16(self, filepath: str | Path) -> tuple[np.ndarray, int]:
        audio, sr = sf.read(str(filepath), dtype="int16", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
            audio = np.clip(audio, np.iinfo(np.int16).min, np.iinfo(np.int16).max).astype(np.int16)
        return audio.astype(np.int16), int(sr)

    @staticmethod
    def segment(
        audio: np.ndarray,
        sr: int,
        segment_duration_s: float,
        hop_s: float,
    ) -> Iterator[tuple[np.ndarray, float, float]]:
        """Yield fixed-size segments as int16 arrays with zero-padding for last chunk."""
        segment_len = int(round(segment_duration_s * sr))
        hop_len = int(round(hop_s * sr))
        if segment_len <= 0 or hop_len <= 0:
            raise ValueError("segment_duration_s and hop_s must be positive")

        n = len(audio)
        if n == 0:
            return

        starts = list(range(0, max(1, n - segment_len + 1), hop_len))
        if starts[-1] + segment_len < n:
            starts.append(max(0, n - segment_len))

        for start in starts:
            end = start + segment_len
            segment = audio[start:end]
            if len(segment) < segment_len:
                segment = np.pad(segment, (0, segment_len - len(segment)), mode="constant")
            yield segment.astype(np.int16), start / sr, min(end, n) / sr
