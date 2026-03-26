import argparse
import json
from pathlib import Path
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.datasets.sed_dataset import SEDDataset
from src.datasets.sampler import BalancedDomainSampler
from src.models import ASTClassifier, DomainAdaptationModel, EmbeddingLogisticRegression, SpectrogramPatchClassifier, TemporalClassifier
from src.preprocessing.augmentation import AudioAugmentor
from src.training.losses import SEDLoss
from src.training.schedulers import build_scheduler
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.reproducibility import set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Phase 7 training entrypoint")
    p.add_argument("--config", type=str, required=False, default=None)
    p.add_argument("--manifest", type=str, default=None)
    p.add_argument("--fold", type=str, default=None)
    p.add_argument("--all-folds", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_model(cfg):
    model_type = getattr(cfg.model, "type", "temporal")
    input_type = str(getattr(cfg.model, "input_type", "spectrogram")).lower()

    if model_type == "logistic":
        if input_type == "spectrogram":
            return SpectrogramPatchClassifier(
                hidden_dim=int(getattr(cfg.model, "hidden_dim", 128)),
                dropout=float(getattr(cfg.model, "dropout", 0.3)),
            )
        return EmbeddingLogisticRegression(
            embedding_dim=int(getattr(cfg.yamnet, "embedding_dim", 1024)),
            aggregation=str(getattr(cfg.model, "aggregation", "mean")),
        )

    if model_type == "temporal":
        if input_type == "spectrogram":
            return SpectrogramPatchClassifier(
                hidden_dim=int(getattr(cfg.model, "hidden_dim", 128)),
                dropout=float(getattr(cfg.model, "dropout", 0.3)),
            )
        return TemporalClassifier(
            embedding_dim=int(getattr(cfg.yamnet, "embedding_dim", 1024)),
            hidden_dim=int(getattr(cfg.model, "hidden_dim", 256)),
            n_layers=int(getattr(cfg.model, "n_layers", 2)),
            temporal_model=str(getattr(cfg.model, "temporal_model", "gru")),
            use_attention=bool(getattr(cfg.model, "use_attention", True)),
            dropout=float(getattr(cfg.model, "dropout", 0.3)),
        )

    if model_type == "ast":
        return ASTClassifier(
            pretrained=bool(getattr(cfg.model, "pretrained", True)),
            freeze_backbone=bool(getattr(cfg.model, "freeze_backbone", False)),
        )

    if model_type == "domain_adapt":
        if input_type == "spectrogram":
            base = SpectrogramPatchClassifier(
                hidden_dim=int(getattr(cfg.model, "hidden_dim", 128)),
                dropout=float(getattr(cfg.model, "dropout", 0.3)),
            )
        else:
            base = TemporalClassifier(
                embedding_dim=int(getattr(cfg.yamnet, "embedding_dim", 1024)),
                hidden_dim=int(getattr(cfg.model, "hidden_dim", 256)),
                n_layers=int(getattr(cfg.model, "n_layers", 2)),
                temporal_model=str(getattr(cfg.model, "temporal_model", "gru")),
                use_attention=bool(getattr(cfg.model, "use_attention", True)),
                dropout=float(getattr(cfg.model, "dropout", 0.3)),
            )
        return DomainAdaptationModel(
            base,
            adaptation_method=str(getattr(cfg.model, "adaptation_method", "coral")),
            adaptation_weight=float(getattr(cfg.model, "adaptation_weight", 0.5)),
        )

    raise ValueError(f"Unknown model type: {model_type}")


def _to_pos_weight(labels: pd.Series) -> torch.Tensor:
    y = labels.astype(int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        return torch.tensor(1.0)
    return torch.tensor(float(n_neg / max(n_pos, 1)))


def _load_fold(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _collate_with_padding(batch):
    x_list = [b["spectrogram"] for b in batch]
    y = torch.stack([b["label"] for b in batch], dim=0)

    if x_list[0].ndim == 2:
        # Embedding mode: (n_patches, emb_dim), pad by n_patches.
        max_len = max(x.shape[0] for x in x_list)
        emb_dim = x_list[0].shape[1]
        x = torch.zeros((len(x_list), max_len, emb_dim), dtype=torch.float32)
        for i, item in enumerate(x_list):
            x[i, : item.shape[0], :] = item
    else:
        x = torch.stack(x_list, dim=0)

    return {
        "spectrogram": x,
        "label": y,
        "session": [b["session"] for b in batch],
        "location": [b["location"] for b in batch],
        "metadata": [b["metadata"] for b in batch],
    }


def _get_fold_files(cfg, specific_fold: str = None):
    splits_dir = Path(cfg.paths.splits_dir)
    if specific_fold:
        fold_file = splits_dir / f"{specific_fold}.json"
        if not fold_file.exists():
            raise FileNotFoundError(f"Fold JSON not found: {fold_file}")
        return [fold_file]
    return sorted(splits_dir.glob("*.json"))


@torch.no_grad()
def _predict_loader(model, loader, device) -> pd.DataFrame:
    model.eval()
    rows = []

    for batch in loader:
        x = batch["spectrogram"].to(device)
        out = model(x)
        probs = torch.sigmoid(out["logits"]).detach().cpu().numpy()
        y = batch["label"].detach().cpu().numpy()

        for i in range(len(probs)):
            md = batch["metadata"][i] if isinstance(batch["metadata"], list) else {}
            rows.append(
                {
                    "session": str(batch["session"][i]),
                    "location": str(batch["location"][i]),
                    "y_true": int(y[i]),
                    "y_pred_proba": float(probs[i]),
                    "start_s": md.get("start_s", None),
                    "end_s": md.get("end_s", None),
                    "npy_path": md.get("npy_path", None),
                    "segment_idx": md.get("segment_idx", None),
                }
            )

    return pd.DataFrame(rows)


def _aggregate_patch_predictions(df: pd.DataFrame) -> pd.DataFrame:
    # For patch mode, average probabilities back to segment-level rows.
    keys = ["session", "location", "npy_path", "segment_idx", "start_s", "end_s"]
    usable_keys = [k for k in keys if k in df.columns]

    if not {"npy_path", "segment_idx"}.issubset(df.columns):
        return df

    grouped = (
        df.groupby(usable_keys, dropna=False)
        .agg(y_true=("y_true", "first"), y_pred_proba=("y_pred_proba", "mean"))
        .reset_index()
    )
    return grouped


def run_one_fold(cfg, manifest_df, fold, device):
    train_idx = fold["train_indices"]
    val_idx = fold["val_indices"]
    test_idx = fold["test_indices"]
    fold_name = fold["fold_name"]

    augmentation_enabled = bool(getattr(cfg.augmentation, "enabled", True))
    augmentor = None
    if augmentation_enabled:
        augmentor = AudioAugmentor(
            str(cfg.paths.noise_segments_dir),
            config={"augmentation": vars(cfg.augmentation)},
        )

    input_type = str(getattr(cfg.model, "input_type", "spectrogram"))
    if input_type == "embeddings":
        ds_mode = "embedding"
    else:
        ds_mode = "patch"

    train_dataset = SEDDataset(
        manifest_df,
        train_idx,
        augmentor=augmentor if ds_mode != "embedding" else None,
        mode=ds_mode,
        patch_frames=int(getattr(cfg.model, "patch_frames", 96)),
        patch_hop_frames=int(getattr(cfg.model, "patch_hop_frames", 48)),
        allowed_sessions_for_noise=fold.get("train_sessions", []),
    )
    val_dataset = SEDDataset(
        manifest_df,
        val_idx,
        augmentor=None,
        mode=ds_mode,
        patch_frames=int(getattr(cfg.model, "patch_frames", 96)),
        patch_hop_frames=int(getattr(cfg.model, "patch_hop_frames", 48)),
    )
    test_dataset = SEDDataset(
        manifest_df,
        test_idx,
        augmentor=None,
        mode=ds_mode,
        patch_frames=int(getattr(cfg.model, "patch_frames", 96)),
        patch_hop_frames=int(getattr(cfg.model, "patch_hop_frames", 48)),
    )

    batch_size = int(cfg.training.batch_size)
    if ds_mode == "patch":
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(cfg.training.num_workers),
            collate_fn=_collate_with_padding,
        )
    else:
        sampler = BalancedDomainSampler(train_dataset, batch_size=batch_size, aerosonic_ratio=0.5)
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            num_workers=int(cfg.training.num_workers),
            collate_fn=_collate_with_padding,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
        collate_fn=_collate_with_padding,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
        collate_fn=_collate_with_padding,
    )

    model = build_model(cfg).to(device)

    pos_weight = _to_pos_weight(manifest_df.loc[train_idx, "label"]).to(device)
    loss_fn = SEDLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.lr),
        weight_decay=float(getattr(cfg.training, "weight_decay", 1.0e-4)),
    )
    scheduler = build_scheduler(optimizer, str(getattr(cfg.training, "scheduler", "cosine")), int(cfg.training.epochs))

    out_root = Path(cfg.paths.output_dir) / str(getattr(cfg, "experiment_name", "experiment"))
    out_root.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        config=cfg,
        device=device,
        output_dir=str(out_root),
        fold_name=fold_name,
    )

    results = trainer.train(int(cfg.training.epochs))

    fold_dir = out_root / fold_name
    fold_dir.mkdir(parents=True, exist_ok=True)
    with (fold_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Export test predictions expected by Phase 8 evaluation script.
    pred_df = _predict_loader(model, test_loader, device=device)
    if ds_mode == "patch":
        pred_df = _aggregate_patch_predictions(pred_df)
    pred_df.to_csv(fold_dir / "predictions.csv", index=False)

    # Optional: keep validation predictions for debugging.
    val_pred_df = _predict_loader(model, val_loader, device=device)
    if ds_mode == "patch":
        val_pred_df = _aggregate_patch_predictions(val_pred_df)
    val_pred_df.to_csv(fold_dir / "val_predictions.csv", index=False)

    if len(pred_df):
        y_true = pred_df["y_true"].astype(int).values
        y_prob = pred_df["y_pred_proba"].astype(float).values
        y_pred = (y_prob >= 0.5).astype(int)
        test_f1 = float((2 * ((y_pred == 1) & (y_true == 1)).sum()) / max((y_pred == 1).sum() + (y_true == 1).sum(), 1))
    else:
        test_f1 = 0.0
    results["test_f1@0.5"] = test_f1
    with (fold_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    args = parse_args()
    cfg = load_config(experiment_path=args.config)
    set_seed(int(cfg.seed))

    manifest_path = Path(args.manifest or cfg.paths.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest_df = pd.read_csv(manifest_path)

    fold_files = _get_fold_files(cfg, specific_fold=args.fold)
    if not fold_files:
        raise RuntimeError("No fold JSON files found in splits directory.")

    if not args.all_folds and args.fold is None:
        # Default to first fold to make single-run convenient.
        fold_files = [fold_files[0]]

    device = torch.device(args.device)

    all_results = {}
    for fold_file in fold_files:
        fold = _load_fold(fold_file)
        print(f"\nTraining fold: {fold['fold_name']}")
        all_results[fold["fold_name"]] = run_one_fold(cfg, manifest_df, fold, device)

    out_root = Path(cfg.paths.output_dir) / str(getattr(cfg, "experiment_name", "experiment"))
    with (out_root / "all_results.json").open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
