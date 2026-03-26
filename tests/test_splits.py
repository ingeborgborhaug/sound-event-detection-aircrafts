import sys
from pathlib import Path

import pandas as pd

# Allow running this file directly: `python tests/test_splits.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.split_generator import LOSOSplitGenerator


def _build_manifest() -> pd.DataFrame:
    rows = []
    idx = 0

    # Norwegian: 3 sessions, locations A/B/C, balanced labels.
    for session in ["session1", "session2", "session3"]:
        for location in ["A", "B", "C"]:
            for label in [0, 1, 0, 1]:
                rows.append(
                    {
                        "row_id": idx,
                        "npy_path": f"/tmp/{session}_{location}_{idx}.npy",
                        "wav_source": f"{session}_{location}.wav",
                        "dataset": "norwegian",
                        "session": session,
                        "location": location,
                        "start_s": 0.0,
                        "end_s": 10.0,
                        "label": label,
                    }
                )
                idx += 1

    # AeroSonic: enough rows for stratified split.
    for k in range(60):
        rows.append(
            {
                "row_id": idx,
                "npy_path": f"/tmp/aerosonic_{k}.npy",
                "wav_source": "aerosonic.wav",
                "dataset": "aerosonic",
                "session": "aerosonic",
                "location": "all",
                "start_s": 0.0,
                "end_s": 10.0,
                "label": 1 if (k % 4 == 0) else 0,
            }
        )
        idx += 1

    return pd.DataFrame(rows)


def _folds():
    manifest = _build_manifest()
    generator = LOSOSplitGenerator(seed=42)
    folds = generator.generate_all_folds(manifest)
    return manifest, folds


def test_no_index_overlap():
    _, folds = _folds()
    for fold in folds:
        train = set(fold["train_indices"])
        val = set(fold["val_indices"])
        test = set(fold["test_indices"])
        assert train.isdisjoint(val)
        assert train.isdisjoint(test)
        assert val.isdisjoint(test)


def test_test_session_isolation():
    manifest, folds = _folds()

    for fold in folds:
        if fold["fold_name"].startswith("baseline"):
            continue

        heldout = fold["test_session"]
        train_val_idx = fold["train_indices"] + fold["val_indices"]
        if not train_val_idx:
            continue

        tv_sessions = set(manifest.loc[train_val_idx, "session"].astype(str))
        assert heldout not in tv_sessions


def test_location_c_in_val():
    manifest, folds = _folds()

    for fold in folds:
        if fold["fold_name"].startswith("baseline"):
            continue

        heldout = fold["test_session"]
        train_idx = fold["train_indices"]
        val_idx = fold["val_indices"]

        train_nor_other = manifest.loc[
            train_idx,
            ["dataset", "session", "location"],
        ]
        train_nor_other = train_nor_other[
            (train_nor_other["dataset"] == "norwegian")
            & (train_nor_other["session"] != heldout)
        ]
        assert not (train_nor_other["location"].astype(str).str.upper() == "C").any()

        val_nor_other = manifest.loc[
            val_idx,
            ["dataset", "session", "location"],
        ]
        val_nor_other = val_nor_other[
            (val_nor_other["dataset"] == "norwegian")
            & (val_nor_other["session"] != heldout)
        ]
        assert (val_nor_other["location"].astype(str).str.upper() == "C").any()


def test_aerosonic_split_consistent():
    manifest, folds = _folds()

    baseline = [f for f in folds if f["fold_name"].startswith("baseline")][0]
    base_train = set(
        manifest.loc[baseline["train_indices"]]
        .query("dataset == 'aerosonic'")
        .index
        .tolist()
    )
    base_val = set(
        manifest.loc[baseline["val_indices"]]
        .query("dataset == 'aerosonic'")
        .index
        .tolist()
    )

    for fold in folds:
        fold_train = set(
            manifest.loc[fold["train_indices"]]
            .query("dataset == 'aerosonic'")
            .index
            .tolist()
        )
        fold_val = set(
            manifest.loc[fold["val_indices"]]
            .query("dataset == 'aerosonic'")
            .index
            .tolist()
        )
        assert fold_train == base_train
        assert fold_val == base_val


def test_all_segments_covered():
    manifest, folds = _folds()
    all_idx = set(range(len(manifest)))

    baseline = [f for f in folds if f["fold_name"].startswith("baseline")][0]
    union_baseline = set(baseline["train_indices"]) | set(baseline["val_indices"]) | set(
        baseline["test_indices"]
    )
    assert union_baseline == all_idx

    for fold in folds:
        if fold["fold_name"].startswith("baseline"):
            continue
        union_fold = set(fold["train_indices"]) | set(fold["val_indices"]) | set(
            fold["test_indices"]
        )
        assert union_fold == all_idx
