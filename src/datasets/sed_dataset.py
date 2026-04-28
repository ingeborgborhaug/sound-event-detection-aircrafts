from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import settings


class CachedSpectrogramDataset(Dataset):
    """Loads cached YAMNet patch tensors (.npy) using manifest/split information."""

    def __init__(
        self,
        manifest_path: str | Path,
        split_json: str | Path | None = None,
        split_name: str | None = None,
        max_patches: int | None = None,
    ):
        self.df = pd.read_csv(manifest_path)
        if max_patches is None:
            max_patches = int(settings.MAX_PATCHES)
        self.max_patches = int(max_patches)

        if split_json and split_name:
            split = json.loads(Path(split_json).read_text(encoding="utf-8"))
            key = f"{split_name}_paths"
            keep = set(split.get(key, []))
            self.df = self.df[self.df["npy_path"].isin(keep)].copy()

        if self.df.empty:
            raise ValueError("Dataset is empty after filtering")

        self.df = self.df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def _pad_or_trim(self, patches: np.ndarray) -> tuple[np.ndarray, int]:
        t = patches.shape[0]
        if t >= self.max_patches:
            return patches[: self.max_patches], self.max_patches

        out = np.zeros((self.max_patches, patches.shape[1], patches.shape[2]), dtype=np.float32)
        out[:t] = patches
        return out, t

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        patches = np.load(row["npy_path"]).astype(np.float32)
        patches, valid_len = self._pad_or_trim(patches)

        x = torch.from_numpy(patches)
        y = torch.tensor(float(row["label"]), dtype=torch.float32)
        session = str(row.get("session", "session_unknown"))
        return {
            "x": x,
            "y": y,
            "valid_len": torch.tensor(valid_len, dtype=torch.long),
            "session": session,
            "path": row["npy_path"],
        }


def pad_collate(batch: list[dict]):
    x = torch.stack([b["x"] for b in batch], dim=0)
    y = torch.stack([b["y"] for b in batch], dim=0)
    valid_len = torch.stack([b["valid_len"] for b in batch], dim=0)
    session = [b["session"] for b in batch]
    path = [b["path"] for b in batch]
    return {"x": x, "y": y, "valid_len": valid_len, "session": session, "path": path}
