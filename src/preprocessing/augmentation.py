import glob
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


class AudioAugmentor:
    """Apply real-noise mixing and SpecAugment to log-mel spectrograms."""

    def __init__(self, noise_segments_dir: str, config: dict):
        self.noise_segments_dir = str(noise_segments_dir)
        self.config = dict(config)

        aug_cfg = self.config.get("augmentation", self.config)
        self.noise_mix_prob = float(aug_cfg.get("noise_mix_prob", 0.5))
        self.noise_snr_range_db = aug_cfg.get("noise_snr_range_db", [0, 20])
        self.spec_augment_enabled = bool(aug_cfg.get("spec_augment", True))
        self.freq_mask_param = int(aug_cfg.get("freq_mask_param", 8))
        self.time_mask_param = int(aug_cfg.get("time_mask_param", 25))
        self.num_freq_masks = int(aug_cfg.get("num_freq_masks", 2))
        self.num_time_masks = int(aug_cfg.get("num_time_masks", 2))

        self._eps = 1e-10
        self.noise_pool_by_session = self._discover_noise_segments()

    def _discover_noise_segments(self) -> Dict[str, List[str]]:
        """Discover noise .npy files under noise_segments_dir/<session>/*.npy."""
        pool: Dict[str, List[str]] = {}
        root = Path(self.noise_segments_dir)
        if not root.exists():
            return pool

        for session_dir in root.iterdir():
            if not session_dir.is_dir():
                continue
            files = sorted(glob.glob(str(session_dir / "*.npy")))
            if files:
                pool[session_dir.name] = files
        return pool

    @staticmethod
    def _ensure_2d(spec: np.ndarray) -> np.ndarray:
        spec = np.asarray(spec, dtype=np.float32)
        if spec.ndim != 2:
            raise ValueError(f"Expected 2D spectrogram, got shape {spec.shape}")
        return spec

    def _sample_noise_path(self, allowed_sessions: Sequence[str]) -> Optional[str]:
        candidates: List[str] = []
        for session in allowed_sessions:
            candidates.extend(self.noise_pool_by_session.get(str(session), []))

        if not candidates:
            for files in self.noise_pool_by_session.values():
                candidates.extend(files)

        if not candidates:
            return None

        idx = np.random.randint(0, len(candidates))
        return candidates[idx]

    def _match_shape(self, noise: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        """Match noise to target (n_mels, time) via transpose/interpolation and crop/tile."""
        n_mels, n_time = target_shape
        noise = self._ensure_2d(noise)

        if noise.shape[0] != n_mels:
            if noise.shape[1] == n_mels:
                noise = noise.T
            else:
                x_old = np.linspace(0.0, 1.0, noise.shape[0])
                x_new = np.linspace(0.0, 1.0, n_mels)
                resized = np.zeros((n_mels, noise.shape[1]), dtype=np.float32)
                for t in range(noise.shape[1]):
                    resized[:, t] = np.interp(x_new, x_old, noise[:, t])
                noise = resized

        if noise.shape[1] < n_time:
            reps = int(np.ceil(n_time / max(1, noise.shape[1])))
            noise = np.tile(noise, (1, reps))

        noise = noise[:, :n_time]
        return noise.astype(np.float32, copy=False)

    def mix_real_noise(
        self,
        spectrogram: np.ndarray,
        allowed_sessions: Sequence[str],
        snr_db: Optional[float] = None,
    ) -> np.ndarray:
        """Mix sampled session noise with spectrogram in linear-power domain."""
        signal_db = self._ensure_2d(spectrogram)
        noise_path = self._sample_noise_path(allowed_sessions)
        if noise_path is None:
            return signal_db

        noise_db = np.load(noise_path)
        noise_db = self._match_shape(noise_db, signal_db.shape)

        if snr_db is None:
            lo, hi = float(self.noise_snr_range_db[0]), float(self.noise_snr_range_db[1])
            snr_db = float(np.random.uniform(lo, hi))

        signal_power = np.power(10.0, signal_db / 10.0)
        noise_power = np.power(10.0, noise_db / 10.0)

        signal_rms = np.sqrt(np.mean(signal_power) + self._eps)
        noise_rms = np.sqrt(np.mean(noise_power) + self._eps)

        scale = signal_rms / (noise_rms * (10.0 ** (snr_db / 20.0)) + self._eps)
        mixed_power = signal_power + scale * noise_power
        mixed_db = 10.0 * np.log10(np.maximum(mixed_power, self._eps))
        return mixed_db.astype(np.float32, copy=False)

    def spec_augment(self, spectrogram: np.ndarray) -> np.ndarray:
        """Apply frequency and time masking to a log-mel spectrogram."""
        spec = self._ensure_2d(spectrogram).copy()
        if not self.spec_augment_enabled:
            return spec

        n_mels, n_time = spec.shape
        fill_value = float(np.min(spec))

        for _ in range(self.num_freq_masks):
            width = np.random.randint(0, self.freq_mask_param + 1)
            if width <= 0 or width >= n_mels:
                continue
            start = np.random.randint(0, n_mels - width + 1)
            spec[start:start + width, :] = fill_value

        for _ in range(self.num_time_masks):
            width = np.random.randint(0, self.time_mask_param + 1)
            if width <= 0 or width >= n_time:
                continue
            start = np.random.randint(0, n_time - width + 1)
            spec[:, start:start + width] = fill_value

        return spec.astype(np.float32, copy=False)

    def __call__(self, spectrogram: np.ndarray, allowed_sessions: Sequence[str]) -> np.ndarray:
        """Apply noise mix with probability and then SpecAugment."""
        out = self._ensure_2d(spectrogram)
        if np.random.rand() < self.noise_mix_prob:
            out = self.mix_real_noise(out, allowed_sessions=allowed_sessions)
        out = self.spec_augment(out)
        return out
