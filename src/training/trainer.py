from __future__ import annotations

import json
import gc
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score


import settings
from keras_yamnet.params import PATCH_BANDS, PATCH_FRAMES
from src.models.yamnet_finetune import build_yamnet_classifier
from src.training.losses import bce_loss
from src.training.schedulers import cosine_decay
from settings import PREDICTION_THRESHOLD


def _normalize_path(path_value: str) -> str:
    """Normalize manifest/split path strings across Windows/WSL/Linux."""
    path_str = str(path_value).strip().replace("\\", "/")

    # Convert Windows absolute paths (e.g. C:/...) to WSL mount paths when needed.
    if os.name != "nt" and re.match(r"^[A-Za-z]:/", path_str):
        drive = path_str[0].lower()
        path_str = f"/mnt/{drive}{path_str[2:]}"

    return path_str


def _materialize_path(path_value: str, base_dir: Path) -> str:
    """Return an absolute path string ready for np.load."""
    normalized = _normalize_path(path_value)
    candidate = Path(normalized)
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return str(candidate)


def _load_split_index(
    split_json: str | Path,
    split_name: str,
) -> tuple[list[str], np.ndarray]:
    repo_root = Path(__file__).resolve().parents[2]
    split = json.loads(Path(split_json).read_text(encoding="utf-8"))

    items_key = f"{split_name}_items"
    if items_key not in split:
        raise ValueError(f"Split file {split_json} must contain '{items_key}'")
    
    items = split[items_key]

    y_paths = {item["y_path"] for item in items}

    if len(y_paths) != 1:
        raise ValueError(
            f"Expected all y_path values to be identical for split file {split_json}, found {len(y_paths)} different values:\n"
            + "\n".join(sorted(y_paths))
        )
    else:
        arr = np.load(list(y_paths)[0]).astype(np.float32)


    """ paths_key = f"{split_name}_paths"
    labels_key = f"{split_name}_labels"

    if paths_key not in split or labels_key not in split:
        raise ValueError(f"Split file {split_json} must contain '{paths_key}' and '{labels_key}'")

    paths = [_materialize_path(path, repo_root) for path in split[paths_key]]
    labels = np.array(split[labels_key], dtype=np.float32) """

    return y_paths, arr


def _load_split_records(
    split_json: str | Path,
    split_name: str,
) -> tuple[list[dict], np.ndarray]:
    repo_root = Path(__file__).resolve().parents[2]
    split = json.loads(Path(split_json).read_text(encoding="utf-8"))

    items_key = f"{split_name}_items"
    if items_key in split:
        records: list[dict] = []
        for item in split[items_key]:
            patch_index_value = item.get("patch_index", None)
            patch_index = None if patch_index_value is None else int(patch_index_value)
            label_value = item.get("label", None)
            y_path = _materialize_path(item["y_path"], repo_root) if item.get("y_path") is not None else None
            records.append(
                {
                    "npz_path": _materialize_path(item["npz_path"], repo_root),
                    "y_path": y_path,
                    "patch_index": patch_index,
                    "label": label_value
                }
            )
        return records
    
    else:
        raise ValueError(f"Split file {split_json} must contain '{items_key}'")

    #paths, labels = _load_split_index(split_json, split_name)
    #records = [{"npz_path": path, "patch_index": None} for path in paths]
    #return records, labels


def _resolve_shuffle_buffer() -> int:
    raw = os.getenv("SED_SHUFFLE_BUFFER", "2000")
    try:
        value = int(raw)
    except ValueError:
        print(f"[train_fold] invalid SED_SHUFFLE_BUFFER={raw!r}, falling back to 2000", flush=True)
        return 2000

    if value < 1:
        print(f"[train_fold] non-positive SED_SHUFFLE_BUFFER={value}, falling back to 2000", flush=True)
        return 2000
    return value


def _label_distribution_stats(labels: np.ndarray, threshold: float) -> tuple[int, int, int, float]:
    total = int(len(labels))
    pos = int((labels >= threshold).sum()) if total else 0
    neg = total - pos
    pos_rate = (pos / total) if total else 0.0
    return total, pos, neg, pos_rate


def _build_dataset(
    npz_paths: list[str],
    batch_size: int,
    shuffle: bool,
    y_paths: list[str] | None = None,
) -> tf.data.Dataset:
    """Build dataset from X and y paths. Supports both old (combined) and new (split) formats."""
    if not npz_paths:
        raise ValueError("No npz paths were provided to _build_dataset().")

    # Determine if we're using split format (X_path + y_path) or legacy format (X_path only)
    use_split_format = y_paths is not None and len(y_paths) == len(npz_paths)

    def _generator():
        for idx, x_path in enumerate(npz_paths):
            x_npz_path = Path(x_path)
            if not x_npz_path.exists():
                raise FileNotFoundError(f"Missing X file: {x_npz_path}")

            if use_split_format:
                # New split format: load X from npz_path, y from y_paths[idx]
                y_npy_path = Path(y_paths[idx])
                if not y_npy_path.exists():
                    raise FileNotFoundError(f"Missing y file: {y_npy_path}")

                with np.load(x_npz_path, allow_pickle=True) as data:
                    X = data["X"].astype(np.float32)
                y = np.load(y_npy_path).astype(np.float32)
            else:
                # Legacy format: load both X and y from npz_path
                with np.load(x_npz_path, allow_pickle=True) as data:
                    X = data["X"].astype(np.float32)
                    y = data["y"].astype(np.float32)

            if y.ndim == 1:
                y = y[:, None]
            if X.shape[0] != len(y):
                y_file_info = str(Path(y_paths[idx]) if use_split_format else x_npz_path)
                raise ValueError(f"X/y length mismatch in {x_npz_path} / {y_file_info}: {X.shape[0]} vs {len(y)}")

            for x_item, y_item in zip(X, y):
                yield x_item, y_item

    output_signature = (
        tf.TensorSpec(shape=(PATCH_FRAMES, PATCH_BANDS), dtype=tf.float32),
        tf.TensorSpec(shape=(1,), dtype=tf.float32),
    )

    ds = tf.data.Dataset.from_generator(_generator, output_signature=output_signature)
    if shuffle:
        ds = ds.shuffle(buffer_size=_resolve_shuffle_buffer(), reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def _build_dataset_from_records(
    records: list[dict],
    batch_size: int,
    shuffle: bool,
) -> tuple[tf.data.Dataset, np.ndarray]:
    if not records:
        raise ValueError("No split records were provided to _build_dataset_from_records().")

    # New split behavior:
    # - patch_index is metadata only (do not subset by it)
    # - use y_path when provided; otherwise fallback to npz y
    unique_sources: dict[tuple[str, str | None], None] = {}
    for record in records:
        npz_path = str(record["npz_path"])
        y_path_value = record.get("y_path")
        y_path = None if y_path_value is None else str(y_path_value)
        unique_sources[(npz_path, y_path)] = None

    grouped_items = list(unique_sources.keys())

    all_y = []

    for npz_path_str, y_path_str in grouped_items:
        npz_path = Path(npz_path_str)

        if y_path_str is None:
            with np.load(npz_path, allow_pickle=True) as data:
                y = data["y"].astype(np.float32)
        else:
            y = np.load(y_path_str, allow_pickle=True).astype(np.float32)

        if y.ndim == 1:
            y = y[:, None]

        all_y.append(y)

    def _generator():
        for npz_path_str, y_path_str in grouped_items:
            npz_path = Path(npz_path_str)

            with np.load(npz_path, allow_pickle=True) as data:
                X = data["X"].astype(np.float32)

                if y_path_str is None:
                    y = data["y"].astype(np.float32)
                else:
                    y = np.load(y_path_str, allow_pickle=True).astype(np.float32)

            if y.ndim == 1:
                y = y[:, None]

            if X.shape[0] != len(y):
                y_file_info = str(y_path_str) if y_path_str is not None else str(npz_path)
                raise ValueError(f"X/y length mismatch in {npz_path} / {y_file_info}: {X.shape[0]} vs {len(y)}")

            for x_item, y_item in zip(X, y):
                yield x_item, y_item

    output_signature = (
        tf.TensorSpec(shape=(PATCH_FRAMES, PATCH_BANDS), dtype=tf.float32),
        tf.TensorSpec(shape=(1,), dtype=tf.float32),
    )

    ds = tf.data.Dataset.from_generator(_generator, output_signature=output_signature)
    if shuffle:
        ds = ds.shuffle(buffer_size=_resolve_shuffle_buffer(), reshuffle_each_iteration=True)

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE), np.concatenate(all_y, axis=0)

def _split_labels_from_npz_paths(npz_paths: list[str], y_paths: list[str] | None = None) -> np.ndarray:
    """Load labels from npz/npy files. Supports both old (combined) and new (split) formats."""
    labels: list[np.ndarray] = []
    use_split_format = y_paths is not None and len(y_paths) == len(npz_paths)
    
    for idx, path in enumerate(npz_paths):
        if use_split_format:
            # New split format: load y from y_paths[idx]
            y = np.load(y_paths[idx], allow_pickle=True).astype(np.float32)
        else:
            # Legacy format: load y from npz_path
            with np.load(path, allow_pickle=True) as data:
                y = data["y"].astype(np.float32)
        if y.ndim == 1:
            y = y[:, None]
        labels.append(y)

    if not labels:
        return np.empty((0, 1), dtype=np.float32)
    return np.concatenate(labels, axis=0)


def _log_step(message: str, started_at: float) -> float:
    return time.perf_counter()

def find_best_threshold(model, val_ds, y_val):
    """Find the best threshold for classification based on F1-score."""
    y_prob = model.predict(val_ds, verbose=0).flatten()
    y_true = np.asarray(y_val).reshape(-1)
    thresholds = np.arange(0.05, 1.0, 0.01)
    f1_scores = [f1_score(y_true, y_prob >= t) for t in thresholds]
    best_threshold = thresholds[np.argmax(f1_scores)]
    best_f1 = max(f1_scores)
    print(f"Best threshold: {best_threshold:.4f} with F1-score: {best_f1:.4f}")
    return best_threshold

def _predict_dataset(model: tf.keras.Model, dataset: tf.data.Dataset) -> np.ndarray:
    return model.predict(dataset, verbose=0).reshape(-1)


def _dataset_labels(dataset: tf.data.Dataset) -> np.ndarray:
    labels: list[np.ndarray] = []
    for _, y_batch in dataset:
        labels.append(np.asarray(y_batch).reshape(-1))
    if not labels:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(labels, axis=0)


class ValPredictionStats(tf.keras.callbacks.Callback):
    def __init__(self, val_dataset):
        super().__init__()
        self.val_dataset = val_dataset

    def on_epoch_end(self, epoch, logs=None):
        p = self.model.predict(self.val_dataset, verbose=0).ravel()
        print(
            f"\nval_pred_stats epoch={epoch+1}: "
            f"min={p.min():.6f}, max={p.max():.6f}, "
            f"mean={p.mean():.6f}, std={p.std():.6f}"
        )


def train_fold(
    split_json: str | Path,
    output_dir: str | Path,
    threshold: float,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-3,
    freeze_backbone: bool = True,
) -> dict:
    
    print(f'Using threshold: {threshold} \nFreeze backbone: {freeze_backbone}')
    started_at = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    #split = json.loads(Path(split_json).read_text(encoding="utf-8"))
    #repo_root = Path(__file__).resolve().parents[2]
    train_records = _load_split_records(split_json, "train")
    val_records = _load_split_records(split_json, "val")
    test_records = _load_split_records(split_json, "test")

    train_ds, train_y = _build_dataset_from_records(train_records, batch_size=batch_size, shuffle=True)
    val_ds, val_y = _build_dataset_from_records(val_records, batch_size=batch_size, shuffle=False)
    test_ds, test_y = _build_dataset_from_records(test_records, batch_size=batch_size, shuffle=False)

    train_total, train_pos, train_neg, train_pos_rate = _label_distribution_stats(train_y, threshold)
    val_total, val_pos, val_neg, val_pos_rate = _label_distribution_stats(val_y, threshold)
    test_total, test_pos, test_neg, test_pos_rate = _label_distribution_stats(test_y, threshold)

    print(
        "[train_fold] dataset_summary "
        #f"train={len(train_paths)} val={len(val_paths)} test={len(test_paths)} "
        f"train_pos={train_pos}/{train_total} ({train_pos_rate:.4f}) "
        f"val_pos={val_pos}/{val_total} ({val_pos_rate:.4f}) "
        f"test_pos={test_pos}/{test_total} ({test_pos_rate:.4f}) "
        f"batch_size={batch_size} "
    )

    model = None
    optimizer = None
    history = None
    try:
        started_at = _log_step("building model", started_at)
        model = build_yamnet_classifier(freeze_backbone=freeze_backbone)

        started_at = _log_step("creating optimizer", started_at)
        optimizer = tf.keras.optimizers.Adam(learning_rate=cosine_decay(lr, epochs))

        started_at = _log_step("compiling model", started_at)
        model.compile(
            optimizer=optimizer,
            loss=bce_loss(),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(threshold=threshold, name="acc"),
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
                tf.keras.metrics.Precision(thresholds=threshold, name="precision"),
                tf.keras.metrics.Recall(thresholds=threshold, name="recall"),
            ],
        )

        started_at = _log_step("starting model.fit", started_at)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=8, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(output_dir / "best_model.keras"),
                monitor="val_auc",
                mode="max",
                save_best_only=True,
            ),
        ]

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )

        best_threshold = find_best_threshold(model, val_ds, val_y)
        print(f"Best threshold found on validation set: {best_threshold:.4f}")

        test_probs = _predict_dataset(model, test_ds)
        test_preds = (test_probs >= best_threshold).astype(int)

        pred_df = pd.DataFrame({
            "y_true": test_y.flatten().astype(int),
            "y_prob": test_probs,
            "y_pred": test_preds,
        })

        pred_df.to_csv(output_dir / "test_predictions.csv", index=False)

        started_at = _log_step("computing metrics from predictions", started_at)

        y_true = test_y.flatten().astype(int)
        acc = accuracy_score(y_true, test_preds)
        auc = roc_auc_score(y_true, test_probs)
        precision = precision_score(y_true, test_preds, zero_division=0)
        recall = recall_score(y_true, test_preds, zero_division=0)
        bce = tf.keras.losses.binary_crossentropy(y_true, test_probs).numpy().mean()

        metrics = {
            "loss": float(bce),
            "acc": float(acc),
            "auc": float(auc),
            "precision": float(precision),
            "recall": float(recall),
        }
        started_at = _log_step("metrics computed", started_at)

        result = {
            "split_json": str(split_json),
            "best_threshold (used for metrics in test)": best_threshold,
            "metrics": metrics,
            "history": {k: [float(vv) for vv in v] for k, v in history.history.items()},
            "label_distribution": {
                "train": {"total": train_total, "pos": train_pos, "neg": train_neg, "pos_rate": train_pos_rate},
                "val": {"total": val_total, "pos": val_pos, "neg": val_neg, "pos_rate": val_pos_rate},
                "test": {"total": test_total, "pos": test_pos, "neg": test_neg, "pos_rate": test_pos_rate},
            },
            "warnings": [],
        }

        (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    finally:
        del train_ds, val_ds, test_ds, train_records, val_records, test_records, train_y, val_y, test_y
        model = None
        optimizer = None
        history = None
        tf.keras.backend.clear_session()
        gc.collect()