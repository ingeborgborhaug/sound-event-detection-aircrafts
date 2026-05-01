from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score


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

    paths_key = f"{split_name}_paths"
    labels_key = f"{split_name}_labels"

    if paths_key not in split or labels_key not in split:
        raise ValueError(f"Split file {split_json} must contain '{paths_key}' and '{labels_key}'")

    paths = [_materialize_path(path, repo_root) for path in split[paths_key]]
    labels = np.array(split[labels_key], dtype=np.float32)

    return paths, labels


def _prepare_patches(path: str, max_patches: int) -> np.ndarray:
    patches = np.load(path).astype(np.float32)
    t = patches.shape[0]
    if t >= max_patches:
        return patches[:max_patches]

    pad = np.zeros((max_patches - t, patches.shape[1], patches.shape[2]), dtype=np.float32)
    return np.concatenate([patches, pad], axis=0)


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
    paths: list[str],
    labels: np.ndarray,
    max_patches: int,
    batch_size: int,
    shuffle: bool,
) -> tf.data.Dataset:
    if len(paths) != len(labels):
        raise ValueError("paths/labels length mismatch")

    def _load_from_path(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        def _np_load(path_bytes: bytes) -> np.ndarray:
            return _prepare_patches(path_bytes.decode("utf-8"), max_patches)

        patches = tf.numpy_function(_np_load, [path], tf.float32)
        patches.set_shape((max_patches, PATCH_FRAMES, PATCH_BANDS))
        return patches, label

    ds = tf.data.Dataset.from_tensor_slices((np.asarray(paths, dtype=np.str_), labels.astype(np.float32)))
    if shuffle:
        shuffle_cap = _resolve_shuffle_buffer()
        ds = ds.shuffle(buffer_size=min(len(paths), shuffle_cap), reshuffle_each_iteration=True)

    ds = ds.map(_load_from_path, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _log_step(message: str, started_at: float) -> float:
    return time.perf_counter()

def find_best_threshold(model, X_val, y_val):
    """Find the best threshold for classification based on F1-score."""
    y_prob = model.predict(X_val, verbose=0).flatten()
    thresholds = np.arange(0.05, 1.0, 0.01)
    f1_scores = [f1_score(y_val, y_prob >= t) for t in thresholds]
    best_threshold = thresholds[np.argmax(f1_scores)]
    best_f1 = max(f1_scores)
    print(f"Best threshold: {best_threshold:.4f} with F1-score: {best_f1:.4f}")
    return best_threshold

class ValPredictionStats(tf.keras.callbacks.Callback):
    def __init__(self, X_val, y_val, batch_size):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        p = self.model.predict(self.X_val, batch_size=self.batch_size, verbose=0).ravel()
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
    max_patches: int | None = None,
    lr: float = 1e-3,
    freeze_backbone: bool = True,
) -> dict:
    
    print(f'Using threshold: {threshold} \n Freeze backbone: {freeze_backbone}')
    if max_patches is None:
        max_patches = int(settings.MAX_PATCHES)

    started_at = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split = json.loads(Path(split_json).read_text(encoding="utf-8"))

    train_paths = split["train_paths"]
    val_paths = split["val_paths"]
    test_paths = split.get("test_paths", [])

    X_train, y_train = load_npz_paths(train_paths)
    X_val, y_val = load_npz_paths(val_paths)
    X_test, y_test = load_npz_paths(test_paths)

    train_labels = y_train
    val_labels = y_val
    test_labels = y_test
    
    train_total, train_pos, train_neg, train_pos_rate = _label_distribution_stats(train_labels, threshold)
    val_total, val_pos, val_neg, val_pos_rate = _label_distribution_stats(val_labels, threshold)
    test_total, test_pos, test_neg, test_pos_rate = _label_distribution_stats(test_labels, threshold)

    print(
        "[train_fold] dataset_summary "
        f"train={len(train_paths)} val={len(val_paths)} test={len(test_paths)} "
        f"train_pos={train_pos}/{train_total} ({train_pos_rate:.4f}) "
        f"val_pos={val_pos}/{val_total} ({val_pos_rate:.4f}) "
        f"test_pos={test_pos}/{test_total} ({test_pos_rate:.4f}) "
        f"batch_size={batch_size} max_patches={max_patches} "
    )

    started_at = _log_step("building model", started_at)
    model = build_yamnet_classifier(freeze_backbone=freeze_backbone)
    print("Trainable variables:")
    for v in model.trainable_variables:
        print(v.name, v.shape)
    print("Number of trainable variables:", len(model.trainable_variables))
    started_at = _log_step("creating optimizer", started_at)
    optimizer = tf.keras.optimizers.Adam(learning_rate=cosine_decay(lr, epochs))

    started_at = _log_step("compiling model", started_at)
    model.compile(
        optimizer=optimizer,
        loss=bce_loss(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(threshold=threshold,name="acc"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(thresholds=threshold, name="precision"),
            tf.keras.metrics.Recall(thresholds=threshold, name="recall"),
        ],
    )

    started_at = _log_step("starting model.fit", started_at)

    callbacks = [
        ValPredictionStats(X_val, y_val, batch_size),
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_model.keras"),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=True,
        verbose=1,
    )

    best_threshold = find_best_threshold(model, X_val, y_val)
    print(f"Best threshold found on validation set: {best_threshold:.4f}")

    test_probs = model.predict(X_test, verbose=0).reshape(-1)
    test_preds = (test_probs >= best_threshold).astype(int)

    pred_df = pd.DataFrame({
        "y_true": y_test.flatten().astype(int),
        "y_prob": test_probs,
        "y_pred": test_preds,
    })

    pred_df.to_csv(output_dir / "test_predictions.csv", index=False)

    started_at = _log_step("model.fit finished, starting evaluation", started_at)

    eval_result = model.evaluate(X_test, y_test, batch_size=batch_size,verbose=0)
    started_at = _log_step("evaluation finished", started_at)
    metric_names = model.metrics_names
    metrics = {name: float(value) for name, value in zip(metric_names, eval_result)}

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



def load_npz_paths(paths):
    X_parts = []
    y_parts = []

    for path in paths:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Missing preprocessed file: {path}")

        data = np.load(path, allow_pickle=True)

        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.float32)

        if y.ndim == 1:
            y = y[:, None]

        X_parts.append(X)
        y_parts.append(y)

    if not X_parts:
        raise ValueError("No input files were provided to load_npz_paths().")

    X_all = np.concatenate(X_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)

    return X_all, y_all