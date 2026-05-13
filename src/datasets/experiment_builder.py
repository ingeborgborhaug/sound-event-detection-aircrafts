from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.preprocessing.augmentation import AudioSegmentRef, mix_segment_refs
from keras_yamnet.params import PATCH_WINDOW_SECONDS


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


def _split_items_from_df(df: pd.DataFrame) -> list[dict]:
    items: list[dict] = []
    for _, row in df.iterrows():
        patch_index_value = row.get("patch_index", None)
        patch_index = None if pd.isna(patch_index_value) else int(patch_index_value)
        label_value = row.get("label", None)
        label = None if pd.isna(label_value) else int(label_value)

        items.append(
            {
                "npz_path": str(row["npz_path"]),
                "y_path": str(row["y_path"]) if row.get("y_path", None) is not None else None,
                "patch_index": patch_index,
                "label": label,
            }
        )
    return items


def _find_existing_split_files(out_dir: Path, experiment: str) -> list[Path] | None:
    """Return existing split.json files for an experiment if they already exist."""
    experiment_dir = out_dir / experiment
    if not experiment_dir.exists():
        return None

    split_files = sorted(experiment_dir.glob("fold_*/split.json"))
    return split_files if split_files else None


def _expand_npz_manifest_to_patches(manifest_df: pd.DataFrame, dataset: str, cache_dir: Path | None = None) -> pd.DataFrame:
    """Expand session-level manifest (one row per .npz file) to patch-level (one row per patch with label).
    
    Loads each .npz file referenced in the manifest and creates one row per patch,
    preserving the per-patch label value needed for filtering source/noise pools.
    
    If cache_dir is provided, will check for a cached expanded manifest (CSV) and reuse it if available.
    Otherwise computes the expansion and caches it for future runs.
    """
    # Determine cache path if caching is enabled
    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{dataset}_manifest_expanded.csv"
        
        # Check if cache exists and can be read
        if cache_path.exists():
            try:
                print(f"Loading cached expanded {dataset} manifest from {cache_path}")
                return pd.read_csv(cache_path, low_memory=False)
            except Exception as e:
                print(f"Warning: Failed to read cached manifest {cache_path}: {e}. Recomputing...")
    
    rows: list[dict] = []
    
    for _, row in tqdm(manifest_df.iterrows(), total=len(manifest_df), desc="Expanding NPZ manifest", leave=False):
        npz_path = Path(row["npz_path"])

        try:
            data = np.load(npz_path)
            if dataset == 'aero':
                y = data["y"].astype(int)  # per-patch labels
                start_s = data["start_s"].astype(np.float32)
                fold = int(row["fold"])
                audio_path = row["audio_path"]
            
            if dataset == 'norwegian':
                audio_path = row["audio_path"]
                start_s = data["start_s"].astype(np.float32)
                y_path = Path(row["y_path"])
                y = np.load(y_path).astype(int)  # per-patch labels
                fold = int(row["fold"])
            
            # Create one row per patch
            for patch_idx in range(len(y)):
                new_row = row.to_dict()
                new_row.update({
                    "audio_path": audio_path,
                    "X_path": str(npz_path),
                    "y_path": str(y_path) if dataset == 'norwegian' else None,
                    "start_s": float(start_s[patch_idx]),
                    "label": int(y[patch_idx]),
                    "fold": fold,
                    "patch_index": patch_idx,
                })
                rows.append(new_row)
        except Exception as e:
            print(f"Warning: Failed to load {npz_path}: {e}")
            continue
    
    result_df = pd.DataFrame(rows).reset_index(drop=True)
    
    # Save to cache if enabled
    if cache_path is not None:
        try:
            result_df.to_csv(cache_path, index=False)
            print(f"Cached expanded {dataset} manifest to {cache_path}")
        except Exception as e:
            print(f"Warning: Failed to cache expanded manifest to {cache_path}: {e}")
    
    return result_df


def _mix_and_cache_augmentations(
    source_df: pd.DataFrame,
    noise_df: pd.DataFrame,
    out_dir: Path,
    experiment: str,
    fold_id: int,
    snr_range_db: tuple[float, float],
    augment_source_percent: float,
    augments_per_source: int,
    seed: int,
    cached_augmented_dir: Path | None = None,
) -> pd.DataFrame:
    if noise_df.empty:
        raise ValueError("Noise pool is empty after leakage filtering")

    source_df = source_df.copy()
    rows: list[dict] = []
    augmented_dir = out_dir / "augmented"
    augmented_dir.mkdir(parents=True, exist_ok=True)

    # Check if we should use cached augmented files
    print(f"Checking for cached augmented files in {cached_augmented_dir} for fold {fold_id}...")
    if cached_augmented_dir is not None and cached_augmented_dir.exists():
        npz_files = sorted(cached_augmented_dir.glob(f"fold_{fold_id}_*.npz"))
        if npz_files:
            for npz_path in npz_files:
                src_idx = len(rows) // augments_per_source
                if src_idx >= len(source_df):
                    src_idx = len(source_df) - 1

                row = source_df.iloc[src_idx].to_dict()
                row.update(
                    {
                        "npz_path": str(npz_path),
                        "dataset": f"{row.get('dataset', 'aerosonic')}_augmented",
                        "augmented_from": row.get("npz_path", ""),
                        "fold_id": fold_id,
                        "is_augmented": True,
                    }
                )
                rows.append(row)
            return pd.DataFrame(rows)

    rng = np.random.default_rng(seed)

    # Percentage-based source selection: select unique sources without replacement
    num_total_sources = len(source_df)
    num_to_augment = max(1, int(np.ceil(num_total_sources * augment_source_percent / 100.0)))
    selected_source_indices = rng.choice(num_total_sources, size=min(num_to_augment, num_total_sources), replace=False)
    selected_source_indices = sorted(selected_source_indices)

    # Target selection: ensure unique targets (no reuse)
    num_targets_available = len(noise_df)
    num_targets_needed = len(selected_source_indices)
    
    if num_targets_needed > num_targets_available:
        print(f"WARNING: Not enough unique targets ({num_targets_available}) for selected sources ({num_targets_needed}). "
              f"Auto-reducing to {num_targets_available} augmentations.")
        selected_source_indices = selected_source_indices[:num_targets_available]
        num_targets_needed = num_targets_available

    # Create unique source-to-target assignment
    selected_target_indices = rng.choice(num_targets_available, size=num_targets_needed, replace=False)
    source_target_pairs = list(zip(selected_source_indices, selected_target_indices))

    # Track statistics
    count_label_0_augmented = 0
    count_label_1_augmented = 0

    for pair_idx, (src_idx, tgt_idx) in enumerate(tqdm(source_target_pairs, desc=f"Augment fold {fold_id}", leave=False, unit="aug")):
        src_row = source_df.iloc[src_idx]
        bg_row = noise_df.iloc[tgt_idx]

        # Strict duration validation before mixing
        src_duration = float(src_row["end_s"]) - float(src_row["start_s"])
        if abs(src_duration - PATCH_WINDOW_SECONDS) > 1e-4:  # Allow tiny floating-point tolerance
            error_msg = (f"ERROR: Source segment duration {src_duration:.6f}s != {PATCH_WINDOW_SECONDS:.6f}s\n"
                        f"  npz_path: {src_row.get('npz_path', 'N/A')}\n"
                        f"  patch_index: {src_row.get('patch_index', 'N/A')}\n"
                        f"  audio_path: {src_row.get('audio_path', 'N/A')}\n"
                        f"  start_s: {src_row['start_s']}\n"
                        f"  end_s: {src_row['end_s']}")
            print(error_msg)
            raise ValueError(error_msg)

        # Track label statistics
        src_label = int(src_row["label"]) if "label" in src_row and pd.notna(src_row["label"]) else 1
        if src_label == 0:
            count_label_0_augmented += 1
        else:
            count_label_1_augmented += 1

        src_ref = AudioSegmentRef(
            audio_path=str(src_row["audio_path"]),
            start_s=float(src_row["start_s"]),
            end_s=float(src_row["end_s"]),
            dataset=str(src_row.get("dataset", "aerosonic")),
            fold=int(src_row["fold"]) if "fold" in src_row and pd.notna(src_row["fold"]) else None,
            label=src_label,
        )

        bg_ref = AudioSegmentRef(
            audio_path=str(bg_row["audio_path"]),
            start_s=float(bg_row["start_s"]),
            end_s=float(bg_row["end_s"]),
            dataset=str(bg_row.get("dataset", "norwegian")),
            fold=int(bg_row["fold"]) if "fold" in bg_row and pd.notna(bg_row["fold"]) else None,
            label=int(bg_row["label"]) if "label" in bg_row and pd.notna(bg_row["label"]) else 0,
        )

        snr_db = float(rng.uniform(*snr_range_db))
        out_path = augmented_dir / f"fold_{fold_id}_{src_idx:05d}_{pair_idx:02d}.npz"
        
        if not out_path.exists():
            patches = mix_segment_refs(src_ref, bg_ref, snr_db=snr_db, rng=rng)

            X = patches.astype(np.float32)
            y = np.full((X.shape[0],), src_label, dtype=np.int32)
            fold_array = np.full((X.shape[0],), int(src_row.get("fold", -1) if pd.notna(src_row.get("fold", pd.NA)) else -1), dtype=np.int32)
            start_s = np.zeros((X.shape[0],), dtype=np.float32)
            end_s = start_s + float(PATCH_WINDOW_SECONDS)
            np.savez_compressed(
                out_path,
                X=X,
                y=y,
                fold=fold_array,
                start_s=start_s,
                end_s=end_s,
                audio_path=np.array(str(src_row.get("audio_path", ""))),
                gt_path=np.array(str(src_row.get("gt_path", ""))),
            )

        row = src_row.to_dict()
        row.update(
            {
                "npz_path": str(out_path),
                "dataset": f"{row.get('dataset', 'aerosonic')}_augmented",
                "augmented_from": row.get("npz_path", ""),
                "noise_audio_path": bg_row["audio_path"],
                "noise_start_s": float(bg_row["start_s"]),
                "noise_end_s": float(bg_row["end_s"]),
                "snr_db": snr_db,
                "fold_id": fold_id,
                "is_augmented": True,
                "patch_index": None,
            }
        )
        rows.append(row)

    # Report statistics
    total_augmented = count_label_0_augmented + count_label_1_augmented
    if total_augmented > 0:
        pct_label_0 = 100.0 * count_label_0_augmented / total_augmented
        pct_label_1 = 100.0 * count_label_1_augmented / total_augmented
        print(f"Fold {fold_id}: Augmented {count_label_1_augmented} label-1 rows ({pct_label_1:.1f}%) and "
              f"{count_label_0_augmented} label-0 rows ({pct_label_0:.1f}%)")

    return pd.DataFrame(rows)

def build_leakage_free_cv_experiments(
    aerosonic_manifest: str | Path,
    norwegian_manifest: str | Path,
    out_dir: str | Path,
    experiment: str,
    augments_per_source: int = 1,
    snr_range_db: tuple[float, float] = (0.0, 20.0),
    augment_source_percent: float = 100.0,
    seed: int = 42,
    cached_augmented_dir: str | Path | None = None, 
    split_cache_root: str | Path | None = None
) -> list[Path]:
    """Create fold manifests/splits for AeroSonic-to-Norwegian experiments.

        experiment values:
            - aero_only_to_norwegian
            - aero_aug_noise_to_norwegian
            - aero_plus_norwegian_with_aug
            - norwegian_only

        cached_augmented_dir: optional path to a cached augmented files directory.
            If provided, will reuse augmented files instead of generating new ones.
            Expected structure: cached_augmented_dir/fold_{fold_id}_test_{test_group}/augmented/*.npz
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_splits = _find_existing_split_files(out_dir, experiment)
    if existing_splits:
        print(
            f"Found existing split files for experiment '{experiment}' in {out_dir}: "
            f"reusing {len(existing_splits)} fold(s)"
        )
        return existing_splits
    
    cached_aug_dir: Path | None = None
    if cached_augmented_dir is not None:
        cached_aug_dir = Path(cached_augmented_dir)

    # Load Norwegian manifest (always required)
    nor_df = _ensure_fold_int(_load_manifest(norwegian_manifest, dataset="norwegian"), fold_col="fold")
    if nor_df.empty:
        raise ValueError("Norwegian manifest is empty")

    # Load AeroSonic only when experiment requires it
    aero_df = pd.DataFrame()
    aero_aug_df = pd.DataFrame()
    if experiment != "norwegian_only":
        aero_df = _ensure_fold_int(_load_manifest(aerosonic_manifest, dataset="aerosonic"), fold_col="fold")
        if aero_df.empty:
            raise ValueError("AeroSonic manifest is empty")

    # If manifests are session-level (have npz_path), expand to patch-level for augmentation / label filtering.
    # Expand AeroSonic to patch-level only when used
    if experiment != "norwegian_only":
        aero_aug_df = aero_df
        if "npz_path" in aero_df.columns and not {"start_s", "end_s", "label"}.issubset(aero_df.columns):
            print("Expanding AeroSonic manifest from session-level to patch-level for augmentation...")
            aero_aug_df = _expand_npz_manifest_to_patches(aero_df, 'aero', cache_dir=split_cache_root)
            if aero_aug_df.empty:
                raise ValueError("AeroSonic manifest is empty after expanding NPZ files")

    # If Norwegian manifest is session-level (has npz_path), expand to patch-level
    if "npz_path" in nor_df.columns and "label" not in nor_df.columns:
        print("Expanding Norwegian manifest from session-level to patch-level...")
        nor_df = _expand_npz_manifest_to_patches(nor_df, 'norwegian', cache_dir=split_cache_root)
        if nor_df.empty:
            raise ValueError("Norwegian manifest is empty after expanding NPZ files")

    if "label" not in nor_df.columns:
        raise ValueError("Norwegian manifest must contain label column")

    group_col = _detect_group_column(nor_df)
    groups = sorted(nor_df[group_col].dropna().astype(str).unique().tolist()) # folds (0,1,2,3,4) or sessions (1,2,3,4,5)
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

        # Build training parts depending on experiment
        if experiment == "norwegian_only":
            train_parts: list[pd.DataFrame] = [train_nor_df.copy()]
        else:
            train_parts: list[pd.DataFrame] = [aero_df.copy()]
            if experiment == "aero_aug_noise_to_norwegian" or experiment == "aero_plus_norwegian_with_aug":
                # For cached augmented directory, point to fold-specific cache
                fold_cache_dir: Path | None = None
                if cached_aug_dir is not None:
                    fold_cache_dir = cached_aug_dir / f"fold_{fold_id}_test_{test_group}" / "augmented"

                aug_df = _mix_and_cache_augmentations(
                    source_df=aero_aug_df,
                    noise_df=noise_pool,
                    out_dir=fold_dir,
                    experiment=experiment,
                    fold_id=fold_id,
                    snr_range_db=snr_range_db,
                    augment_source_percent=augment_source_percent,
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
            "train_items": _split_items_from_df(train_df),
            "val_items": _split_items_from_df(val_df),
            "test_items": _split_items_from_df(test_df),
        }

        split_path = fold_dir / "split.json"
        split_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
        split_files.append(split_path)

    return split_files
