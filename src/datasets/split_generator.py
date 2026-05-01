from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _make_split_dict(fold_id: int, dataset: str, train_vals, val_val, test_val, df: pd.DataFrame, key_col: str) -> dict:
    train_df = df[df[key_col].isin(train_vals)]
    val_df = df[df[key_col] == val_val]
    test_df = df[df[key_col] == test_val]

    return {
        "fold_id": fold_id,
        "dataset": dataset,
        "train_key_values": list(train_vals),
        "val_key_value": val_val,
        "test_key_value": test_val,
        "path_type": "npz",
        "train_paths": train_df["npz_path"].tolist(),
        "val_paths": val_df["npz_path"].tolist(),
        "test_paths": test_df["npz_path"].tolist(),
    }


def generate_loso_splits(
    manifest_path: str | Path,
    out_dir: str | Path,
    dataset: str = "norwegian",
) -> list[Path]:
    """Generate LOSO split JSON files from manifest sessions."""
    manifest_path = Path(manifest_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    if "dataset" in df.columns:
        df = df[df["dataset"] == dataset].copy()

    df["session"] = df["session"].astype(str)

    if df.empty:
        raise ValueError(f"No rows found for dataset={dataset!r} in {manifest_path}")

    sessions = sorted(df["session"].dropna().astype(str).unique())
    if len(sessions) < 2:
        raise ValueError("Need at least two sessions for LOSO splitting")

    split_files: list[Path] = []
    for fold_id, test_session in enumerate(sessions):
        remaining = [s for s in sessions if s != test_session]
        val_session = remaining[fold_id % len(remaining)]
        train_sessions = [s for s in remaining if s != val_session]

        train_df = df[df["session"].isin(train_sessions)]
        val_df = df[df["session"] == val_session]
        test_df = df[df["session"] == test_session]

        split = {
            "fold_id": fold_id,
            "dataset": dataset,
            "train_sessions": train_sessions,
            "val_session": val_session,
            "test_session": test_session,
            "path_type": "npz",
            "train_paths": train_df["npz_path"].tolist(),
            "val_paths": val_df["npz_path"].tolist(),
            "test_paths": test_df["npz_path"].tolist(),
        }

        split_path = out_dir / f"loso_fold_{fold_id}_{test_session}.json"
        split_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
        split_files.append(split_path)

    return split_files


def generate_fold_splits(
    manifest_path: str | Path,
    out_dir: str | Path,
    dataset: str = "aerosonic",
    fold_column: str = "fold",
) -> list[Path]:
    """Generate fold-based train/val/test JSON files from a manifest."""
    manifest_path = Path(manifest_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    if "dataset" in df.columns:
        df = df[df["dataset"] == dataset].copy()

    if fold_column not in df.columns:
        raise ValueError(f"Manifest {manifest_path} does not contain column {fold_column!r}")

    df = df.copy()
    df[fold_column] = pd.to_numeric(df[fold_column], errors="coerce")
    df = df.dropna(subset=[fold_column]).copy()
    df[fold_column] = df[fold_column].astype(int)

    folds = sorted(df[fold_column].unique().tolist())
    if len(folds) < 2:
        raise ValueError(f"Need at least two folds in column {fold_column!r}")

    split_files: list[Path] = []
    for i, test_fold in enumerate(folds):
        remaining = [f for f in folds if f != test_fold]
        val_fold = remaining[i % len(remaining)]
        train_folds = [f for f in remaining if f != val_fold]

        split = _make_split_dict(i, dataset, train_folds, val_fold, test_fold, df, fold_column)
        split_path = out_dir / f"{dataset}_fold_{i}_test_{test_fold}.json"
        split_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
        split_files.append(split_path)

    return split_files

def generate_train_manifest_val_fold_test_manifest_splits(
    train_manifest_path: str | Path,
    test_manifest_path: str | Path,
    out_dir: str | Path,
    train_dataset: str | None = None,
    test_dataset: str | None = None,
    fold_column: str = "fold",
) -> list[Path]:
    """Use one manifest as fixed test set, and split train manifest by folds into train/val."""
    train_manifest_path = Path(train_manifest_path)
    test_manifest_path = Path(test_manifest_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_manifest_path)
    test_df = pd.read_csv(test_manifest_path)

    for name, df in [("train", train_df), ("test", test_df)]:
        if "npz_path" not in df.columns:
            raise ValueError(f"{name} manifest must contain 'npz_path'")

    if fold_column not in train_df.columns:
        raise ValueError(f"Train manifest must contain fold column {fold_column!r}")

    train_df = train_df.copy()
    train_df[fold_column] = pd.to_numeric(train_df[fold_column], errors="coerce")
    train_df = train_df.dropna(subset=[fold_column]).copy()
    train_df[fold_column] = train_df[fold_column].astype(int)

    # Ignore foldless rows in train manifest
    train_df = train_df[train_df[fold_column] >= 0].copy()

    folds = sorted(train_df[fold_column].unique().tolist())
    if len(folds) < 2:
        raise ValueError("Need at least two folds in the train manifest")

    split_files: list[Path] = []

    for i, val_fold in enumerate(folds, start=1):
        fold_train_df = train_df[train_df[fold_column] != val_fold]
        fold_val_df = train_df[train_df[fold_column] == val_fold]

        split = {
            "fold_id": i,
            "split_type": "train_manifest_val_fold_test_manifest",
            "path_type": "npz",
            "train_manifest": str(train_manifest_path),
            "test_manifest": str(test_manifest_path),
            "train_dataset": train_dataset,
            "test_dataset": test_dataset,
            "val_fold": int(val_fold),
            "train_folds": [int(f) for f in folds if f != val_fold],
            "train_paths": fold_train_df["npz_path"].tolist(),
            "val_paths": fold_val_df["npz_path"].tolist(),
            "test_paths": test_df["npz_path"].tolist(),
        }

        split_path = out_dir / f"fold_{i}_val_{val_fold}_external_test.json"
        split_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
        split_files.append(split_path)

    return split_files
