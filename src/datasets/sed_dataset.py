from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SEDDataset(Dataset):
    """PyTorch dataset for aircraft SED segments or patches."""

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        indices: List[int],
        augmentor=None,
        mode: str = "patch",
        patch_frames: int = 96,
        patch_hop_frames: int = 48,
        allowed_sessions_for_noise: Optional[Sequence[str]] = None,
    ):
        if mode not in {"patch", "segment", "embedding"}:
            raise ValueError("mode must be 'patch', 'segment', or 'embedding'")

        self.manifest_df = manifest_df
        self.indices = list(map(int, indices))
        self.rows = manifest_df.loc[self.indices].reset_index(drop=True).copy()
        self.augmentor = augmentor
        self.mode = mode
        self.patch_frames = int(patch_frames)
        self.patch_hop_frames = int(patch_hop_frames)
        self.allowed_sessions_for_noise = list(allowed_sessions_for_noise or [])

        required_cols = {"label", "session", "location"}
        if self.mode == "embedding":
            required_cols.add("embedding_path")
        else:
            required_cols.add("npy_path")
        missing = [c for c in required_cols if c not in self.rows.columns]
        if missing:
            raise ValueError(f"Manifest subset missing required columns: {missing}")

        self.patch_index_map: List[Tuple[int, int]] = []
        if self.mode == "patch":
            self._build_patch_index_map()

    def _load_embedding(self, row: pd.Series) -> np.ndarray:
        path = str(row["embedding_path"])
        emb = np.load(path)
        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim == 1:
            emb = emb[np.newaxis, :]
        if emb.ndim != 2:
            raise ValueError(f"Expected embedding shape (n_patches, dim), got {emb.shape}")
        return emb

    def _load_spectrogram(self, row: pd.Series) -> np.ndarray:
        path = str(row["npy_path"])
        ext = Path(path).suffix.lower()

        if ext == ".npy":
            spec = np.load(path)
        else:
            seg_idx = int(row.get("segment_idx", 0))
            with h5py.File(path, "r") as f:
                x = f["x"]
                if seg_idx < 0 or seg_idx >= x.shape[0]:
                    raise IndexError(f"segment_idx {seg_idx} out of range for {path}")
                spec = x[seg_idx]

        return np.asarray(spec, dtype=np.float32)

    @staticmethod
    def _to_mel_time(spec: np.ndarray) -> np.ndarray:
        """Normalize common shapes to (n_mels, time_frames)."""
        spec = np.asarray(spec, dtype=np.float32)

        if spec.ndim == 2:
            return spec if spec.shape[0] <= spec.shape[1] else spec.T

        if spec.ndim == 3:
            if spec.shape[0] <= 4:
                # channel-first: (C, n_mels, T) or (C, T, n_mels)
                merged = np.mean(spec, axis=0)
                return merged if merged.shape[0] <= merged.shape[1] else merged.T
            # fallback: average axis 0
            merged = np.mean(spec, axis=0)
            return merged if merged.shape[0] <= merged.shape[1] else merged.T

        raise ValueError(f"Unsupported spectrogram shape {spec.shape}")

    def _extract_patches_from_segment(self, spec_mel_time: np.ndarray) -> np.ndarray:
        n_mels, n_frames = spec_mel_time.shape

        if n_frames < self.patch_frames:
            pad = self.patch_frames - n_frames
            padded = np.pad(spec_mel_time, ((0, 0), (0, pad)), mode="constant")
            return padded[np.newaxis, :, :]

        starts = range(0, n_frames - self.patch_frames + 1, self.patch_hop_frames)
        patches = [spec_mel_time[:, s:s + self.patch_frames] for s in starts]
        return np.stack(patches, axis=0).astype(np.float32)

    def _build_patch_index_map(self) -> None:
        for row_idx in range(len(self.rows)):
            row = self.rows.iloc[row_idx]
            spec = self._load_spectrogram(row)

            if spec.ndim == 3 and spec.shape[0] > 1:
                n_patches = int(spec.shape[0])
            else:
                spec_mt = self._to_mel_time(spec)
                patches = self._extract_patches_from_segment(spec_mt)
                n_patches = int(patches.shape[0])

            for patch_idx in range(n_patches):
                self.patch_index_map.append((row_idx, patch_idx))

    def __len__(self) -> int:
        return len(self.patch_index_map) if self.mode == "patch" else len(self.rows)

    def _item_from_row(self, row_idx: int, patch_idx: Optional[int]) -> Dict[str, Any]:
        row = self.rows.iloc[row_idx]
        label = int(row["label"])
        session = str(row["session"])
        location = str(row["location"])

        if self.mode == "embedding":
            emb = self._load_embedding(row)
            x = torch.tensor(emb, dtype=torch.float32)
            y = torch.tensor(label, dtype=torch.long)
            metadata = {
                "start_s": row.get("start_s", None),
                "end_s": row.get("end_s", None),
                "wav_source": row.get("wav_source", None),
                "embedding_path": row.get("embedding_path", None),
            }
            return {
                "spectrogram": x,
                "label": y,
                "session": session,
                "location": location,
                "metadata": metadata,
            }

        spec = self._load_spectrogram(row)

        if self.mode == "patch":
            if spec.ndim == 3 and spec.shape[0] > 1:
                patch = np.asarray(spec[int(patch_idx)], dtype=np.float32)
                spec_mt = self._to_mel_time(patch)
            else:
                spec_mt = self._to_mel_time(spec)
                patches = self._extract_patches_from_segment(spec_mt)
                spec_mt = patches[int(patch_idx)]
        else:
            spec_mt = self._to_mel_time(spec)

        if self.augmentor is not None:
            spec_mt = self.augmentor(spec_mt, allowed_sessions=self.allowed_sessions_for_noise)

        x = torch.tensor(spec_mt, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(label, dtype=torch.long)

        metadata = {
            "start_s": row.get("start_s", None),
            "end_s": row.get("end_s", None),
            "wav_source": row.get("wav_source", None),
            "npy_path": row.get("npy_path", None),
            "segment_idx": row.get("segment_idx", None),
        }

        return {
            "spectrogram": x,
            "label": y,
            "session": session,
            "location": location,
            "metadata": metadata,
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.mode == "patch":
            row_idx, patch_idx = self.patch_index_map[int(idx)]
            return self._item_from_row(row_idx=row_idx, patch_idx=patch_idx)
        return self._item_from_row(row_idx=int(idx), patch_idx=None)
