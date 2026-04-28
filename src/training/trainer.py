from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import tensorflow as tf

from keras_yamnet.params import PATCH_BANDS, PATCH_FRAMES
from src.models.yamnet_finetune import build_yamnet_temporal_classifier
from src.training.losses import bce_loss
from src.training.schedulers import cosine_decay


def _load_split_index(
    manifest_path: str | Path,
    split_json: str | Path,
    split_name: str,
) -> tuple[list[str], np.ndarray]:
    df = pd.read_csv(manifest_path, dtype={"session": "string"})
    split = json.loads(Path(split_json).read_text(encoding="utf-8"))
    keep_paths = set(split[f"{split_name}_paths"])
    sdf = df[df["npy_path"].isin(keep_paths)].copy()
    if sdf.empty:
        raise ValueError(f"No rows for split={split_name} in {split_json}")

    paths = sdf["npy_path"].astype(str).tolist()
    labels = sdf["label"].astype(float).to_numpy(dtype=np.float32)
    return paths, labels


def _prepare_patches(path: str, max_patches: int) -> np.ndarray:
    patches = np.load(path).astype(np.float32)
    t = patches.shape[0]
    if t >= max_patches:
        return patches[:max_patches]

    pad = np.zeros((max_patches - t, patches.shape[1], patches.shape[2]), dtype=np.float32)
    return np.concatenate([patches, pad], axis=0)


def _build_dataset(
    paths: list[str],
    labels: np.ndarray,
    max_patches: int,
    batch_size: int,
    shuffle: bool,
) -> tf.data.Dataset:
    if len(paths) != len(labels):
        raise ValueError("paths/labels length mismatch")

    def _generator() -> Iterator[tuple[np.ndarray, np.float32]]:
        for path, label in zip(paths, labels):
            yield _prepare_patches(path, max_patches), np.float32(label)

    output_signature = (
        tf.TensorSpec(shape=(max_patches, PATCH_FRAMES, PATCH_BANDS), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
    )

    ds = tf.data.Dataset.from_generator(_generator, output_signature=output_signature)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(paths), 10_000), reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def train_fold(
    manifest_path: str | Path,
    split_json: str | Path,
    output_dir: str | Path,
    epochs: int = 30,
    batch_size: int = 16,
    max_patches: int = 20,
    lr: float = 1e-3,
    freeze_backbone: bool = True,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_paths, train_labels = _load_split_index(manifest_path, split_json, "train")
    val_paths, val_labels = _load_split_index(manifest_path, split_json, "val")
    test_paths, test_labels = _load_split_index(manifest_path, split_json, "test")

    train_ds = _build_dataset(train_paths, train_labels, max_patches, batch_size, shuffle=True)
    val_ds = _build_dataset(val_paths, val_labels, max_patches, batch_size, shuffle=False)
    test_ds = _build_dataset(test_paths, test_labels, max_patches, batch_size, shuffle=False)

    model = build_yamnet_temporal_classifier(max_patches=max_patches, freeze_backbone=freeze_backbone)
    optimizer = tf.keras.optimizers.Adam(learning_rate=cosine_decay(lr, epochs))

    model.compile(
        optimizer=optimizer,
        loss=bce_loss(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="acc"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

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

    eval_result = model.evaluate(test_ds, verbose=0)
    metric_names = model.metrics_names
    metrics = {name: float(value) for name, value in zip(metric_names, eval_result)}

    result = {
        "split_json": str(split_json),
        "metrics": metrics,
        "history": {k: [float(vv) for vv in v] for k, v in history.history.items()},
    }

    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
