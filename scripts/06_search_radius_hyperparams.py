from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.experiment_builder import build_leakage_free_cv_experiments
from src.training.trainer import train_fold


def _radius_candidates(radius_min: float, radius_max: float, radius_step: float) -> list[float]:
    if radius_step <= 0:
        raise ValueError("radius_step must be positive")
    if radius_max < radius_min:
        raise ValueError("radius_max must be greater than or equal to radius_min")

    radii = np.arange(radius_min, radius_max + (radius_step * 0.5), radius_step, dtype=float)
    return [round(float(radius), 6) for radius in radii]


def _radius_mask(df: pd.DataFrame, radius_km: float) -> pd.Series:
    if "radius_km" in df.columns:
        values = pd.to_numeric(df["radius_km"], errors="coerce")
        return np.isclose(values.to_numpy(dtype=float), radius_km, atol=1e-6, rtol=0.0)

    if "pair_name" in df.columns:
        radius_tag = f"_{radius_km:g}km"
        return df["pair_name"].astype(str).str.contains(radius_tag, case=False, regex=False).to_numpy()

    raise ValueError("Manifest must contain either a radius_km column or pair_name column")


def _load_radius_manifest(manifest_path: str | Path, radius_km: float) -> pd.DataFrame:
    df = pd.read_csv(
        manifest_path,
        dtype={
            "session": "string",
            "location": "string",
            "pair_name": "string",
        },
        low_memory=False,
    )
    mask = _radius_mask(df, radius_km)
    radius_df = df[mask].copy()
    if radius_df.empty:
        raise ValueError(f"No rows found for radius={radius_km:g} km in {manifest_path}")
    return radius_df.reset_index(drop=True)


def _best_history_metric(history: dict[str, list[float]], metric_name: str) -> float:
    values = history.get(metric_name, [])
    if not values:
        return float("nan")
    return float(np.nanmax(np.asarray(values, dtype=float)))


def _mean(values: list[float]) -> float:
    finite_values = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not finite_values:
        return float("nan")
    return float(np.mean(finite_values))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a radius hyperparameter search for each cross-validation fold."
    )
    parser.add_argument("--aerosonic-manifest", required=True, help="AeroSonic manifest CSV")
    parser.add_argument("--norwegian-manifest", required=True, help="Norwegian/Skatval manifest CSV containing radius rows")
    parser.add_argument("--out-dir", default="data/radius_search", help="Output directory for search artifacts")
    parser.add_argument(
        "--experiment",
        default="aero_only_to_norwegian",
        choices=["aero_only_to_norwegian", "aero_aug_noise_to_norwegian", "aero_plus_norwegian_with_aug"],
        help="Cross-dataset experiment variant to evaluate for each radius",
    )
    parser.add_argument("--radius-min", type=float, default=1.0)
    parser.add_argument("--radius-max", type=float, default=8.0)
    parser.add_argument("--radius-step", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--augments-per-source", type=int, default=1)
    parser.add_argument("--snr-min", type=float, default=0.0)
    parser.add_argument("--snr-max", type=float, default=20.0)
    parser.add_argument("--augment-all-labels", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cached-augmented-dir",
        type=str,
        default=None,
        help="Base path to cached augmented files (e.g., E:/data/augmented_cache). Script will look for radius-specific subdirectories like radius_1km, radius_2km, etc. If provided, reuses augmented files instead of regenerating for each radius.",
    )
    args = parser.parse_args()

    if args.max_patches is None:
        import settings

        args.max_patches = int(settings.MAX_PATCHES)

    radii = _radius_candidates(args.radius_min, args.radius_max, args.radius_step)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    search_results: list[dict[str, Any]] = []

    for radius_index, radius_km in enumerate(tqdm(radii, desc="Radii", unit="radius"), start=1):
        print(f"Starting radius {radius_index}/{len(radii)}: {radius_km:g} km")
        radius_dir = out_dir / f"radius_{radius_km:g}km"
        radius_dir.mkdir(parents=True, exist_ok=True)

        radius_manifest_df = _load_radius_manifest(args.norwegian_manifest, radius_km)
        radius_manifest_path = radius_dir / "norwegian_manifest.csv"
        radius_manifest_df.to_csv(radius_manifest_path, index=False)

        # Construct radius-specific cache path if base cache dir is provided
        radius_cache_dir = None
        if args.cached_augmented_dir:
            radius_cache_dir = Path(args.cached_augmented_dir) / f"radius_{radius_km:g}km"

        split_files = build_leakage_free_cv_experiments(
            aerosonic_manifest=args.aerosonic_manifest,
            norwegian_manifest=radius_manifest_path,
            out_dir=radius_dir,
            experiment=args.experiment,
            augments_per_source=args.augments_per_source,
            snr_range_db=(args.snr_min, args.snr_max),
            augment_only_positive=not args.augment_all_labels,
            seed=args.seed,
            cached_augmented_dir=radius_cache_dir,
        )

        radius_fold_results: list[dict[str, Any]] = []
        for split_file in tqdm(split_files, desc=f"Radius {radius_km:g} folds", leave=False, unit="fold"):
            split_data = json.loads(Path(split_file).read_text(encoding="utf-8"))
            fold_dir = Path(split_file).parent
            train_output_dir = fold_dir / "training"
            result = train_fold(
                manifest_path=fold_dir / "manifest.csv",
                split_json=split_file,
                output_dir=train_output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                max_patches=args.max_patches,
                lr=args.lr,
                freeze_backbone=not args.unfreeze_backbone,
            )

            val_auc = _best_history_metric(result.get("history", {}), "val_auc")
            val_loss = _best_history_metric(result.get("history", {}), "val_loss")
            test_metrics = result.get("metrics", {})
            radius_fold_results.append(
                {
                    "radius_km": radius_km,
                    "fold_id": split_data.get("fold_id"),
                    "split_json": str(split_file),
                    "manifest_path": str(fold_dir / "manifest.csv"),
                    "train_output_dir": str(train_output_dir),
                    "best_val_auc": val_auc,
                    "best_val_loss": val_loss,
                    "test_metrics": test_metrics,
                    "history": result.get("history", {}),
                }
            )

        best_fold_val_aucs = [entry["best_val_auc"] for entry in radius_fold_results]
        mean_val_auc = _mean(best_fold_val_aucs)
        mean_test_auc = _mean([float(entry["test_metrics"].get("auc", float("nan"))) for entry in radius_fold_results])
        mean_test_loss = _mean([float(entry["test_metrics"].get("loss", float("nan"))) for entry in radius_fold_results])

        radius_summary = {
            "radius_km": radius_km,
            "mean_best_val_auc": mean_val_auc,
            "mean_test_auc": mean_test_auc,
            "mean_test_loss": mean_test_loss,
            "fold_results": radius_fold_results,
        }
        (radius_dir / "radius_summary.json").write_text(json.dumps(radius_summary, indent=2), encoding="utf-8")
        search_results.append(radius_summary)

        print(
            f"Radius {radius_km:g} km: mean best val AUC={mean_val_auc:.4f}, mean test AUC={mean_test_auc:.4f}"
        )

    best_radius = max(search_results, key=lambda item: item["mean_best_val_auc"])
    selected_fold_results = []
    for fold_id in sorted({entry["fold_id"] for item in search_results for entry in item["fold_results"]}):
        candidates = [
            entry
            for item in search_results
            for entry in item["fold_results"]
            if entry["fold_id"] == fold_id
        ]
        if not candidates:
            continue
        best_fold = max(candidates, key=lambda entry: entry["best_val_auc"])
        selected_fold_results.append(best_fold)

    selected_mean_test_auc = _mean([float(entry["test_metrics"].get("auc", float("nan"))) for entry in selected_fold_results])
    selected_mean_test_loss = _mean([float(entry["test_metrics"].get("loss", float("nan"))) for entry in selected_fold_results])

    summary = {
        "experiment": args.experiment,
        "radius_candidates": radii,
        "best_radius_by_mean_val_auc": best_radius["radius_km"],
        "best_radius_mean_val_auc": best_radius["mean_best_val_auc"],
        "selected_mean_test_auc": selected_mean_test_auc,
        "selected_mean_test_loss": selected_mean_test_loss,
        "radius_results": search_results,
        "selected_fold_results": selected_fold_results,
    }
    (out_dir / "radius_search_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Best radius by mean validation AUC: {best_radius['radius_km']:g} km")
    print(f"Search summary written to {out_dir / 'radius_search_summary.json'}")


if __name__ == "__main__":
    main()
