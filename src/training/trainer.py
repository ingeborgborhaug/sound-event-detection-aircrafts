from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

import settings
from keras_yamnet.params import PATCH_BANDS, PATCH_FRAMES
from src.models.yamnet_finetune import build_yamnet_temporal_classifier
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
    manifest_path: str | Path,
    split_json: str | Path,
    split_name: str,
) -> tuple[list[str], np.ndarray]:
    manifest_path = Path(manifest_path)
    repo_root = Path(__file__).resolve().parents[2]

    split_started_at = time.perf_counter()
    print(f"[load_split_index +0.00s] reading manifest: {manifest_path}", flush=True)
    df = pd.read_csv(manifest_path, dtype={"session": "string"}, low_memory=False)
    print(f"[load_split_index +{time.perf_counter() - split_started_at:8.2f}s] manifest loaded ({len(df)} rows)", flush=True)

    print(f"[load_split_index +{time.perf_counter() - split_started_at:8.2f}s] reading split json: {split_json}", flush=True)
    split = json.loads(Path(split_json).read_text(encoding="utf-8"))
    print(f"[load_split_index +{time.perf_counter() - split_started_at:8.2f}s] split json loaded", flush=True)

    print(f"[load_split_index +{time.perf_counter() - split_started_at:8.2f}s] normalizing kept paths", flush=True)
    keep_paths = {_normalize_path(path) for path in split[f"{split_name}_paths"]}
    print(f"[load_split_index +{time.perf_counter() - split_started_at:8.2f}s] normalizing manifest paths", flush=True)
    df["_npy_path_norm"] = df["npy_path"].astype(str).map(_normalize_path)
    print(f"[load_split_index +{time.perf_counter() - split_started_at:8.2f}s] filtering rows", flush=True)
    sdf = df[df["_npy_path_norm"].isin(keep_paths)].copy()
    if sdf.empty:
        raise ValueError(f"No rows for split={split_name} in {split_json}")

    print(f"[load_split_index +{time.perf_counter() - split_started_at:8.2f}s] materializing {len(sdf)} paths", flush=True)
    paths = [_materialize_path(path, repo_root) for path in sdf["_npy_path_norm"].tolist()]
    labels = sdf["label"].astype(float).to_numpy(dtype=np.float32)
    print(f"[load_split_index +{time.perf_counter() - split_started_at:8.2f}s] done ({len(paths)} paths)", flush=True)
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


def _label_distribution_stats(labels: np.ndarray) -> tuple[int, int, int, float]:
    total = int(len(labels))
    pos = int((labels >= PREDICTION_THRESHOLD).sum()) if total else 0
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


def train_fold(
    manifest_path: str | Path,
    split_json: str | Path,
    output_dir: str | Path,
    epochs: int = 30,
    batch_size: int = 16,
    max_patches: int | None = None,
    lr: float = 1e-3,
    freeze_backbone: bool = True,
) -> dict:
    if max_patches is None:
        max_patches = int(settings.MAX_PATCHES)

    started_at = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train_fold +0.00s] starting fold from {split_json}", flush=True)

    started_at = _log_step("loading train split index", started_at)
    train_paths, train_labels = _load_split_index(manifest_path, split_json, "train")
    started_at = _log_step(f"loaded train split: {len(train_paths)} rows", started_at)

    started_at = _log_step("loading val split index", started_at)
    val_paths, val_labels = _load_split_index(manifest_path, split_json, "val")
    started_at = _log_step(f"loaded val split: {len(val_paths)} rows", started_at)

    started_at = _log_step("loading test split index", started_at)
    test_paths, test_labels = _load_split_index(manifest_path, split_json, "test")
    started_at = _log_step(f"loaded test split: {len(test_paths)} rows", started_at)

    train_total, train_pos, train_neg, train_pos_rate = _label_distribution_stats(train_labels)
    val_total, val_pos, val_neg, val_pos_rate = _label_distribution_stats(val_labels)
    test_total, test_pos, test_neg, test_pos_rate = _label_distribution_stats(test_labels)

    shuffle_cap = _resolve_shuffle_buffer()
    effective_shuffle = min(len(train_paths), shuffle_cap)
    print(
        "[train_fold] dataset_summary "
        f"train={len(train_paths)} val={len(val_paths)} test={len(test_paths)} "
        f"train_pos={train_pos}/{train_total} ({train_pos_rate:.4f}) "
        f"val_pos={val_pos}/{val_total} ({val_pos_rate:.4f}) "
        f"test_pos={test_pos}/{test_total} ({test_pos_rate:.4f}) "
        f"batch_size={batch_size} max_patches={max_patches} "
        f"shuffle_buffer={effective_shuffle} (SED_SHUFFLE_BUFFER={shuffle_cap})",
        flush=True,
    )

    fold_warnings: list[str] = []
    min_pos_rate_raw = os.getenv("SED_MIN_POS_RATE", "0.01")
    try:
        min_pos_rate = float(min_pos_rate_raw)
    except ValueError:
        print(f"[train_fold] invalid SED_MIN_POS_RATE={min_pos_rate_raw!r}, falling back to 0.01", flush=True)
        min_pos_rate = 0.01

    if min_pos_rate < 0.0:
        min_pos_rate = 0.0
    if min_pos_rate > 1.0:
        min_pos_rate = 1.0

    for split_name, total, pos, neg, pos_rate in [
        ("train", train_total, train_pos, train_neg, train_pos_rate),
        ("val", val_total, val_pos, val_neg, val_pos_rate),
        ("test", test_total, test_pos, test_neg, test_pos_rate),
    ]:
        if total == 0:
            fold_warnings.append(f"{split_name} split is empty")
        elif pos == 0:
            fold_warnings.append(f"{split_name} split has zero positives ({pos}/{total})")
        elif neg == 0:
            fold_warnings.append(f"{split_name} split has zero negatives ({neg}/{total})")
        elif pos_rate < min_pos_rate:
            fold_warnings.append(
                f"{split_name} split positive rate {pos_rate:.4f} is below threshold {min_pos_rate:.4f}"
            )

    for warning in fold_warnings:
        print(f"[train_fold][warning] {warning}", flush=True)

    strict_guard = os.getenv("SED_STRICT_FOLD_GUARD", "0").strip().lower() in {"1", "true", "yes", "on"}
    if strict_guard and fold_warnings:
        raise ValueError(
            "Fold guard failed due to problematic split distribution. "
            "Disable strict mode with SED_STRICT_FOLD_GUARD=0 if you only want warnings. "
            f"Warnings: {fold_warnings}"
        )

    started_at = _log_step("building train dataset", started_at)
    train_ds = _build_dataset(train_paths, train_labels, max_patches, batch_size, shuffle=True)
    started_at = _log_step("building val dataset", started_at)
    val_ds = _build_dataset(val_paths, val_labels, max_patches, batch_size, shuffle=False)
    started_at = _log_step("building test dataset", started_at)
    test_ds = _build_dataset(test_paths, test_labels, max_patches, batch_size, shuffle=False)

    started_at = _log_step("building model", started_at)
    model = build_yamnet_temporal_classifier(max_patches=max_patches, freeze_backbone=freeze_backbone)
    started_at = _log_step("creating optimizer", started_at)
    optimizer = tf.keras.optimizers.Adam(learning_rate=cosine_decay(lr, epochs))

    started_at = _log_step("compiling model", started_at)
    model.compile(
        optimizer=optimizer,
        loss=bce_loss(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(threshold=PREDICTION_THRESHOLD,name="acc"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
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

    callbacks.append(
        tf.keras.callbacks.LambdaCallback(
            on_train_begin=lambda logs: print("[train_fold] train_begin", flush=True),
            on_epoch_begin=lambda epoch, logs: print(f"[train_fold] epoch_begin {epoch}", flush=True),
            on_train_batch_begin=lambda batch, logs: print(f"[train_fold] batch_begin {batch}", flush=True) if batch == 0 else None,
        )
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    started_at = _log_step("model.fit finished, starting evaluation", started_at)

    eval_result = model.evaluate(test_ds, verbose=0)
    started_at = _log_step("evaluation finished", started_at)
    metric_names = model.metrics_names
    metrics = {name: float(value) for name, value in zip(metric_names, eval_result)}

    result = {
        "split_json": str(split_json),
        "metrics": metrics,
        "history": {k: [float(vv) for vv in v] for k, v in history.history.items()},
        "label_distribution": {
            "train": {"total": train_total, "pos": train_pos, "neg": train_neg, "pos_rate": train_pos_rate},
            "val": {"total": val_total, "pos": val_pos, "neg": val_neg, "pos_rate": val_pos_rate},
            "test": {"total": test_total, "pos": test_pos, "neg": test_neg, "pos_rate": test_pos_rate},
        },
        "warnings": fold_warnings,
    }

    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
