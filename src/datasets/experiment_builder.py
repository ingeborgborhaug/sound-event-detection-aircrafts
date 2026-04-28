from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.preprocessing.augmentation import AudioSegmentRef, mix_segment_refs


@dataclass(frozen=True)
class ExperimentFoldSpec:
    fold_id: int
    test_fold: int
    val_fold: int
    train_folds: list[int]
    experiment: str


def _load_manifest(manifest_path: str | Path, dataset: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    if dataset is not None and "dataset" in df.columns:
        df = df[df["dataset"] == dataset].copy()
    return df.reset_index(drop=True)


def _ensure_fold_int(df: pd.DataFrame, fold_col: str = "fold") -> pd.DataFrame:
    if fold_col in df.columns:
        df = df.copy()
        df[fold_col] = pd.to_numeric(df[fold_col], errors="coerce")
        df = df.dropna(subset=[fold_col]).copy()
        df[fold_col] = df[fold_col].astype(int)
    return df.reset_index(drop=True)


def _detect_group_column(df: pd.DataFrame) -> str:
    if "fold" in df.columns and df["fold"].notna().any():
        return "fold"
    if "session" in df.columns and df["session"].notna().any():
        return "session"
    raise ValueError("Could not find a grouping column; expected 'fold' or 'session'")


def _write_rows_manifest(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _rows_from_df(df: pd.DataFrame) -> list[dict]:
    return df.to_dict(orient="records")


def _mix_and_cache_augmentations(
    source_df: pd.DataFrame,
    noise_df: pd.DataFrame,
    out_dir: Path,
    experiment: str,
    fold_id: int,
    snr_range_db: tuple[float, float],
    augment_only_positive: bool,
    augments_per_source: int,
    seed: int,
    cached_augmented_dir: Path | None = None,
) -> pd.DataFrame:
    if noise_df.empty:
        raise ValueError("Noise pool is empty after leakage filtering")

    source_df = source_df.copy()
    if augment_only_positive and "label" in source_df.columns:
        source_df = source_df[source_df["label"].astype(int) == 1].copy()

    rows: list[dict] = []
    augmented_dir = out_dir / "augmented"
    augmented_dir.mkdir(parents=True, exist_ok=True)

    # Check if we should use cached augmented files

    print(f"Checking for cached augmented files in {cached_augmented_dir} for fold {fold_id}...")
    if cached_augmented_dir is not None and cached_augmented_dir.exists():
        npy_files = sorted(cached_augmented_dir.glob(f"fold_{fold_id}_*.npy"))
        if npy_files:
            for npy_path in npy_files:
                src_idx = len(rows) // augments_per_source
                if src_idx >= len(source_df):
                    src_idx = len(source_df) - 1
                
                row = source_df.iloc[src_idx].to_dict()
                row.update(
                    {
                        "npy_path": str(npy_path),
                        "dataset": f"{row.get('dataset', 'aerosonic')}_augmented",
                        "augmented_from": row.get("npy_path", ""),
                        "fold_id": fold_id,
                        "is_augmented": True,
                    }
                )
                rows.append(row)
            return pd.DataFrame(rows)

    rng = np.random.default_rng(seed)

    for src_idx, src_row in tqdm(
        source_df.iterrows(),
        total=len(source_df),
        desc=f"Augment fold {fold_id}",
        leave=False,
        unit="src",
    ):
        src_ref = AudioSegmentRef(
            audio_path=str(src_row["audio_path"]),
            start_s=float(src_row["start_s"]),
            end_s=float(src_row["end_s"]),
            dataset=str(src_row.get("dataset", "aerosonic")),
            fold=int(src_row["fold"]) if "fold" in src_row and pd.notna(src_row["fold"]) else None,
            label=int(src_row["label"]) if "label" in src_row and pd.notna(src_row["label"]) else None,
        )

        for aug_idx in range(augments_per_source):
            bg_row = noise_df.iloc[int(rng.integers(0, len(noise_df)))]
            bg_ref = AudioSegmentRef(
                audio_path=str(bg_row["audio_path"]),
                start_s=float(bg_row["start_s"]),
                end_s=float(bg_row["end_s"]),
                dataset=str(bg_row.get("dataset", "norwegian")),
                fold=int(bg_row["fold"]) if "fold" in bg_row and pd.notna(bg_row["fold"]) else None,
                label=int(bg_row["label"]) if "label" in bg_row and pd.notna(bg_row["label"]) else None,
            )

            snr_db = float(rng.uniform(*snr_range_db))
            out_path = augmented_dir / f"fold_{fold_id}_{src_idx:05d}_{aug_idx:02d}.npy"
            if not out_path.exists():
                patches = mix_segment_refs(src_ref, bg_ref, snr_db=snr_db, rng=rng)
                np.save(out_path, patches.astype(np.float32))

            row = src_row.to_dict()
            row.update(
                {
                    "npy_path": str(out_path),
                    "dataset": f"{row.get('dataset', 'aerosonic')}_augmented",
                    "augmented_from": row.get("npy_path", ""),
                    "noise_audio_path": bg_row["audio_path"],
                    "noise_start_s": float(bg_row["start_s"]),
                    "noise_end_s": float(bg_row["end_s"]),
                    "snr_db": snr_db,
                    "fold_id": fold_id,
                    "is_augmented": True,
                }
            )
            rows.append(row)

    return pd.DataFrame(rows)


def build_leakage_free_cv_experiments(
    aerosonic_manifest: str | Path,
    norwegian_manifest: str | Path,
    out_dir: str | Path,
    experiment: str,
    augments_per_source: int = 1,
    snr_range_db: tuple[float, float] = (0.0, 20.0),
    augment_only_positive: bool = True,
    seed: int = 42,
    cached_augmented_dir: str | Path | None = None,
) -> list[Path]:
    """Create fold manifests/splits for AeroSonic-to-Norwegian experiments.

    experiment values:
      - aero_only_to_norwegian
      - aero_aug_noise_to_norwegian
      - aero_plus_norwegian_with_aug

    cached_augmented_dir: optional path to a cached augmented files directory.
      If provided, will reuse augmented files instead of generating new ones.
      Expected structure: cached_augmented_dir/fold_{fold_id}_test_{test_group}/augmented/*.npy
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    cached_aug_dir: Path | None = None
    if cached_augmented_dir is not None:
        cached_aug_dir = Path(cached_augmented_dir)

    aero_df = _ensure_fold_int(_load_manifest(aerosonic_manifest, dataset="aerosonic"), fold_col="fold")
    nor_df = _ensure_fold_int(_load_manifest(norwegian_manifest, dataset="norwegian"), fold_col="fold")

    if aero_df.empty:
        raise ValueError("AeroSonic manifest is empty")
    if nor_df.empty:
        raise ValueError("Norwegian manifest is empty")

    if "label" not in nor_df.columns:
        raise ValueError("Norwegian manifest must contain label column")

    group_col = _detect_group_column(nor_df)
    groups = sorted(nor_df[group_col].dropna().astype(str).unique().tolist())
    if len(groups) < 2:
        raise ValueError("Need at least two Norwegian folds")

    split_files: list[Path] = []
    for fold_id, test_group in enumerate(groups):
        val_group = groups[(fold_id + 1) % len(groups)]
        train_groups = [g for g in groups if g not in {test_group, val_group}]
        if not train_groups:
            raise ValueError("Not enough folds left for training after excluding test/val")

        train_nor_df = nor_df[nor_df[group_col].astype(str).isin(train_groups)].copy()
        val_nor_df = nor_df[nor_df[group_col].astype(str) == str(val_group)].copy()
        test_nor_df = nor_df[nor_df[group_col].astype(str) == str(test_group)].copy()

        noise_pool = nor_df[nor_df[group_col].astype(str).isin(train_groups)].copy()
        noise_pool = noise_pool[noise_pool["label"].astype(int) == 0].copy()

        fold_dir = out_dir / experiment / f"fold_{fold_id}_test_{test_group}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_parts: list[pd.DataFrame] = [aero_df.copy()]
        if experiment == "aero_aug_noise_to_norwegian" or experiment == "aero_plus_norwegian_with_aug":
            # For cached augmented directory, point to fold-specific cache
            fold_cache_dir: Path | None = None
            if cached_aug_dir is not None:
                fold_cache_dir = cached_aug_dir / f"fold_{fold_id}_test_{test_group}" / "augmented"
            
            aug_df = _mix_and_cache_augmentations(
                source_df=aero_df,
                noise_df=noise_pool,
                out_dir=fold_dir,
                experiment=experiment,
                fold_id=fold_id,
                snr_range_db=snr_range_db,
                augment_only_positive=augment_only_positive,
                augments_per_source=augments_per_source,
                seed=seed + fold_id,
                cached_augmented_dir=fold_cache_dir,
            )
            if not aug_df.empty:
                train_parts.append(aug_df)

        if experiment == "aero_plus_norwegian_with_aug":
            train_parts.append(train_nor_df.copy())

        train_df = pd.concat(train_parts, ignore_index=True)
        train_df = train_df.reset_index(drop=True)
        val_df = val_nor_df.reset_index(drop=True)
        test_df = test_nor_df.reset_index(drop=True)

        combined_rows = pd.concat(
            [
                train_df.assign(split="train"),
                val_df.assign(split="val"),
                test_df.assign(split="test"),
            ],
            ignore_index=True,
        )

        manifest_path = fold_dir / "manifest.csv"
        _write_rows_manifest(_rows_from_df(combined_rows), manifest_path)

        split = {
            "fold_id": fold_id,
            "experiment": experiment,
            "group_column": group_col,
            "test_group": test_group,
            "val_group": val_group,
            "train_groups": train_groups,
            "noise_groups": train_groups,
            "test_fold": test_group,
            "val_fold": val_group,
            "train_folds": train_groups,
            "noise_folds": train_groups,
            "train_paths": train_df["npy_path"].astype(str).tolist(),
            "val_paths": val_df["npy_path"].astype(str).tolist(),
            "test_paths": test_df["npy_path"].astype(str).tolist(),
        }

        split_path = fold_dir / "split.json"
        split_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
        split_files.append(split_path)

    return split_files
