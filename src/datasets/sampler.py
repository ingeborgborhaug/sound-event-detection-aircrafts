from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler


def make_balanced_sampler(df: pd.DataFrame, label_col: str = "label") -> WeightedRandomSampler:
    labels = df[label_col].astype(int).values
    class_counts = {c: max(1, int((labels == c).sum())) for c in [0, 1]}
    weights = [1.0 / class_counts[int(y)] for y in labels]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
