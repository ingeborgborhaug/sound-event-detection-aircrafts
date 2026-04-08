from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.models.yamnet_finetune import build_yamnet_temporal_classifier
from src.training.losses import bce_loss
from src.training.schedulers import cosine_decay


def _load_split_arrays(
    manifest_path: str | Path,
    split_json: str | Path,
    split_name: str,
    max_patches: int,
) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(manifest_path)
    split = json.loads(Path(split_json).read_text(encoding="utf-8"))
    keep_paths = set(split[f"{split_name}_paths"])
    sdf = df[df["npy_path"].isin(keep_paths)].copy()
    if sdf.empty:
        raise ValueError(f"No rows for split={split_name} in {split_json}")

    x = []
    y = []
    for _, row in sdf.iterrows():
        patches = np.load(row["npy_path"]).astype(np.float32)
        t = patches.shape[0]
        if t >= max_patches:
            patches = patches[:max_patches]
        else:
            pad = np.zeros((max_patches - t, patches.shape[1], patches.shape[2]), dtype=np.float32)
            patches = np.concatenate([patches, pad], axis=0)
        x.append(patches)
        y.append(float(row["label"]))

    return np.stack(x), np.array(y, dtype=np.float32)


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

    x_train, y_train = _load_split_arrays(manifest_path, split_json, "train", max_patches)
    x_val, y_val = _load_split_arrays(manifest_path, split_json, "val", max_patches)
    x_test, y_test = _load_split_arrays(manifest_path, split_json, "test", max_patches)

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
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    eval_result = model.evaluate(x_test, y_test, verbose=0)
    metric_names = model.metrics_names
    metrics = {name: float(value) for name, value in zip(metric_names, eval_result)}

    result = {
        "split_json": str(split_json),
        "metrics": metrics,
        "history": {k: [float(vv) for vv in v] for k, v in history.history.items()},
    }

    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
