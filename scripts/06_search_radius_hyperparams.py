from __future__ import annotations

import argparse
import gc
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
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


def _extract_partition_labels(split_data: dict[str, Any], split_name: str) -> np.ndarray:
    """Extract labels from split data using the new items format.
    
    The new split format (from build_leakage_free_cv_experiments) uses:
    - {split_name}_items: list of dicts with 'npz_path', 'y_path', 'patch_index', 'label'
    
    Falls back to old format if available: {split_name}_labels
    """
    items_key = f"{split_name}_items"
    labels_key = f"{split_name}_labels"

    if items_key in split_data:
        labels = [item.get("label", 0) for item in split_data[items_key]]
        return np.asarray(labels, dtype=np.float32)

    if labels_key in split_data:
        return np.asarray(split_data[labels_key], dtype=np.float32)

    raise ValueError(f"Split data is missing '{items_key}' and '{labels_key}'")


def _exact_positive_stats(labels: np.ndarray) -> dict[str, float | int]:
    flat = np.asarray(labels, dtype=np.float32).reshape(-1)
    total = int(flat.size)
    # Labels are expected to be 0/1. Use exact ==1 counting to verify patch labels.
    pos = int(np.sum(flat == 1.0))
    neg = total - pos
    pos_rate = (pos / total) if total else float("nan")
    return {
        "total": total,
        "pos": pos,
        "neg": neg,
        "pos_rate": float(pos_rate),
        "pos_rate_percent": float(pos_rate * 100.0) if total else float("nan"),
    }


def _split_label_overview(split_data: dict[str, Any]) -> dict[str, dict[str, float | int]]:
    overview: dict[str, dict[str, float | int]] = {}
    for split_name in ("train", "val", "test"):
        labels = _extract_partition_labels(split_data, split_name)
        overview[split_name] = _exact_positive_stats(labels)
    return overview


def _find_existing_split_files(radius_dir: Path, experiment: str) -> list[Path] | None:
    """Find existing split.json files for a given radius and experiment.
    
    Returns list of split paths if experiment folder exists, None otherwise.
    """
    experiment_dir = radius_dir / experiment
    if not experiment_dir.exists():
        return None
    
    # Look for all fold_* subdirectories with split.json files
    split_files = sorted(experiment_dir.glob("fold_*/split.json"))
    return split_files if split_files else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_radius_summary(
    radius_km: float,
    radius_fold_results: list[dict[str, Any]],
    skipped_folds: list[dict[str, Any]],
    completed: bool,
) -> dict[str, Any]:
    best_fold_val_aucs = [entry["best_val_auc"] for entry in radius_fold_results]
    mean_val_auc = _mean(best_fold_val_aucs)
    mean_test_auc = _mean([float(entry["test_metrics"].get("auc", float("nan"))) for entry in radius_fold_results])
    mean_test_loss = _mean([float(entry["test_metrics"].get("loss", float("nan"))) for entry in radius_fold_results])
    
    # Extract backbone freezing status (should be consistent across all folds for a radius)
    freeze_backbone = None
    if radius_fold_results:
        freeze_backbone = radius_fold_results[0].get("freeze_backbone", None)
    
    result_dict = {
        "radius_km": radius_km,
        "completed": completed,
        "mean_best_val_auc": mean_val_auc,
        "mean_test_auc": mean_test_auc,
        "mean_test_loss": mean_test_loss,
        "num_trained_folds": len(radius_fold_results),
        "num_skipped_folds": len(skipped_folds),
        "freeze_backbone": freeze_backbone,
        "skipped_folds": skipped_folds,
        "fold_results": radius_fold_results,
    }
    return result_dict


def _build_grid_summary(
    pos_radius_km: float,
    aircraft_weight: float | None,
    radius_fold_results: list[dict[str, Any]],
    skipped_folds: list[dict[str, Any]],
    completed: bool,
) -> dict[str, Any]:
    best_fold_val_aucs = [entry["best_val_auc"] for entry in radius_fold_results]
    mean_val_auc = _mean(best_fold_val_aucs)
    mean_test_auc = _mean([float(entry["test_metrics"].get("auc", float("nan"))) for entry in radius_fold_results])
    mean_test_loss = _mean([float(entry["test_metrics"].get("loss", float("nan"))) for entry in radius_fold_results])

    freeze_backbone = None
    if radius_fold_results:
        freeze_backbone = radius_fold_results[0].get("freeze_backbone", None)

    return {
        "pos_radius_km": pos_radius_km,
        "aircraft_weight": aircraft_weight,
        "completed": completed,
        "mean_best_val_auc": mean_val_auc,
        "mean_test_auc": mean_test_auc,
        "mean_test_loss": mean_test_loss,
        "num_trained_folds": len(radius_fold_results),
        "num_skipped_folds": len(skipped_folds),
        "freeze_backbone": freeze_backbone,
        "skipped_folds": skipped_folds,
        "fold_results": radius_fold_results,
    }


def _parse_optional_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    normalized = str(raw).strip().lower()
    if normalized in {"", "none", "null"}:
        return None
    return float(raw)


def _run_single_fold_mode(args: argparse.Namespace) -> None:
    if not args.single_split_json or not args.single_output_dir or not args.single_result_json:
        raise ValueError("--run-single-fold requires --single-split-json, --single-output-dir, and --single-result-json")

    freeze_backbone = str(args.single_freeze_backbone).strip().lower() == "true"
    aircraft_weight = _parse_optional_float(args.single_aircraft_weight)

    result = train_fold(
        aircraft_weight=aircraft_weight,
        split_json=args.single_split_json,
        output_dir=args.single_output_dir,
        threshold=float(args.single_threshold),
        epochs=int(args.single_epochs),
        batch_size=int(args.single_batch_size),
        lr=float(args.single_lr),
        freeze_backbone=freeze_backbone,
    )
    Path(args.single_result_json).write_text(json.dumps({"result": result}, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a radius hyperparameter search for each cross-validation fold."
    )
    parser.add_argument("--aerosonic-manifest", default="/mnt/e/data/processed/aerosonic_train_manifest.csv", help="AeroSonic manifest CSV")
    parser.add_argument("--norwegian-manifest", default="/mnt/e/data/processed/norwegian_manifest.csv" , help="Norwegian/Skatval manifest CSV containing radius rows")
    parser.add_argument("--out-dir", default="/mnt/e/data/radius_search_plus", help="Output directory for search artifacts")
    parser.add_argument(
        "--experiment",
        default="aero_only_to_norwegian",
        choices=["aero_only_to_norwegian", "aero_aug_noise_to_norwegian", "aero_plus_norwegian_with_aug", "norwegian_only", "norwegian_dual_radius_posneg"],
        help="Cross-dataset experiment variant to evaluate for each radius",
    )
    parser.add_argument("--radius-min", type=float, default=1.0)
    parser.add_argument("--radius-max", type=float, default=9.0)
    parser.add_argument("--radius-step", type=float, default=1.0)
    parser.add_argument("--pos-radius-km", type=float, default=None,
                        help="(Required for norwegian_dual_radius_posneg) Radius in km for positive (label=1) patches")
    parser.add_argument("--neg-radius-km", type=float, default=None,
                        help="(Required for norwegian_dual_radius_posneg) Radius in km for negative (label=0) patches")
    parser.add_argument("--cap-neg-to-pos-train", action="store_true",
                        help="(For norwegian_dual_radius_posneg) Cap training negatives to positive count; val/test unchanged")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--aircraft-weight", type=float, default=None)
    parser.add_argument(
        "--pos-radius-kms",
        type=float,
        nargs="+",
        default=None,
        help="Optional grid of positive radii for norwegian_dual_radius_posneg. If set, overrides --pos-radius-km.",
    )
    parser.add_argument(
        "--aircraft-weights",
        type=float,
        nargs="+",
        default=None,
        help="Optional grid of aircraft weights. If set, overrides --aircraft-weight.",
    )
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--augments-per-source", type=int, default=1)
    parser.add_argument(
        "--augment-source-percent",
        type=float,
        default=50.0,
        help="Percentage of source rows to augment per fold.",
    )
    parser.add_argument("--snr-min", type=float, default=0.0)
    parser.add_argument("--snr-max", type=float, default=20.0)
    #parser.add_argument("--augment-all-labels", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cached-augmented-dir",
        type=str,
        default=None,
        help="Base path to cached augmented files (e.g., E:/data/augmented_cache). Script will look for radius-specific subdirectories like radius_1km, radius_2km, etc. If provided, reuses augmented files instead of regenerating for each radius.",
    )
    parser.add_argument(
        "--split-cache-dir",
        type=str,
        default=None,
        help="Stable directory for cached manifests/splits. Defaults to <out-dir>/_cache.",
    )
    parser.add_argument(
        "--one-fold-per-process",
        action="store_true",
        help="Run each fold in a fresh Python process to reduce memory accumulation.",
    )

    # Internal mode used by --one-fold-per-process. Hidden from normal CLI help.
    parser.add_argument("--run-single-fold", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--single-split-json", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-output-dir", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-threshold", type=float, default=0.3, help=argparse.SUPPRESS)
    parser.add_argument("--single-epochs", type=int, default=30, help=argparse.SUPPRESS)
    parser.add_argument("--single-batch-size", type=int, default=16, help=argparse.SUPPRESS)
    parser.add_argument("--single-lr", type=float, default=1e-3, help=argparse.SUPPRESS)
    parser.add_argument("--single-freeze-backbone", type=str, default="true", help=argparse.SUPPRESS)
    parser.add_argument("--single-aircraft-weight", type=str, default="none", help=argparse.SUPPRESS)
    parser.add_argument("--single-result-json", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.run_single_fold:
        _run_single_fold_mode(args)
        return

    pos_radius_values = args.pos_radius_kms if args.pos_radius_kms is not None else ([args.pos_radius_km] if args.pos_radius_km is not None else None)
    if args.experiment == "norwegian_dual_radius_posneg" and pos_radius_values is None:
        raise ValueError("Experiment 'norwegian_dual_radius_posneg' requires either --pos-radius-km or --pos-radius-kms")
    if pos_radius_values is None:
        pos_radius_values = [None]

    aircraft_weight_values = args.aircraft_weights if args.aircraft_weights is not None else ([args.aircraft_weight] if args.aircraft_weight is not None else [None])

    radii = _radius_candidates(args.radius_min, args.radius_max, args.radius_step)
    out_dir = Path(args.out_dir)
    split_cache_root = Path(args.split_cache_dir) if args.split_cache_dir else (out_dir / "_cache")
    split_cache_root.mkdir(parents=True, exist_ok=True)

    # Handle dual-radius mode specially: override radii to the selected positive-radius grid.
    if args.experiment == "norwegian_dual_radius_posneg":
        if args.neg_radius_km is None:
            raise ValueError("Experiment 'norwegian_dual_radius_posneg' requires --neg-radius-km")
        radii = [float(radius) for radius in pos_radius_values]

    # Create a timestamped run folder so repeated runs do not overwrite previous outputs
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%SZ")
    run_out_dir = out_dir / timestamp
    run_out_dir.mkdir(parents=True, exist_ok=True)

    search_results: list[dict[str, Any]] = []
    completed_radii: list[dict[str, Any]] = []
    current_radius_state: dict[str, Any] | None = None

    try:
        for pos_radius_index, pos_radius_km in enumerate(tqdm(radii, desc="Positive radii", unit="radius"), start=1):
            for weight_index, aircraft_weight in enumerate(tqdm(aircraft_weight_values, desc=f"Weights for {pos_radius_km:g} km", leave=False, unit="weight"), start=1):
                print(
                    f"Starting combo {pos_radius_index}/{len(radii)} radius={pos_radius_km:g} km "
                    f"weight={aircraft_weight if aircraft_weight is not None else 'none'} "
                    f"({weight_index}/{len(aircraft_weight_values)})"
                )

                if args.experiment == "norwegian_dual_radius_posneg":
                    combo_run_label = f"radius_dual_pos_{pos_radius_km:g}km_neg_{args.neg_radius_km:g}km_w{aircraft_weight if aircraft_weight is not None else 'none'}"
                    combo_cache_label = combo_run_label
                    combo_run_dir = out_dir / combo_run_label
                    combo_cache_dir = split_cache_root / combo_cache_label
                    radius_manifest_df = pd.read_csv(
                        args.norwegian_manifest,
                        dtype={"session": "string", "location": "string", "pair_name": "string"},
                        low_memory=False,
                    )
                else:
                    combo_run_label = f"radius_{pos_radius_km:g}km_w{aircraft_weight if aircraft_weight is not None else 'none'}"
                    combo_cache_label = combo_run_label
                    combo_run_dir = run_out_dir / combo_run_label
                    combo_cache_dir = split_cache_root / combo_cache_label
                    radius_manifest_df = _load_radius_manifest(args.norwegian_manifest, pos_radius_km)

                combo_run_dir.mkdir(parents=True, exist_ok=True)
                combo_cache_dir.mkdir(parents=True, exist_ok=True)

                radius_manifest_path = combo_cache_dir / "norwegian_manifest.csv"
                print(f"Saving radius-specific manifest to {radius_manifest_path}")
                radius_manifest_df.to_csv(radius_manifest_path, index=False)

                radius_augmented_cache_dir = Path(args.cached_augmented_dir) if args.cached_augmented_dir else None

                existing_splits = _find_existing_split_files(combo_cache_dir, args.experiment)
                if existing_splits:
                    print(f"Found existing splits for radius {pos_radius_km:g} km, experiment '{args.experiment}': reusing {len(existing_splits)} fold(s)")
                    split_files = existing_splits
                else:
                    print(f"Generating new splits for radius {pos_radius_km:g} km, experiment '{args.experiment}'...")
                    split_files = build_leakage_free_cv_experiments(
                        aerosonic_manifest=args.aerosonic_manifest,
                        norwegian_manifest=radius_manifest_path,
                        out_dir=combo_cache_dir,
                        experiment=args.experiment,
                        augments_per_source=args.augments_per_source,
                        snr_range_db=(args.snr_min, args.snr_max),
                        augment_source_percent=args.augment_source_percent,
                        seed=args.seed,
                        cached_augmented_dir=radius_augmented_cache_dir,
                        pos_radius_km=pos_radius_km,
                        neg_radius_km=args.neg_radius_km,
                        cap_neg_to_pos_train=args.cap_neg_to_pos_train,
                        split_cache_root="/mnt/e/data/processed/expanded_manifests"
                    )

                radius_fold_results = []
                skipped_folds = []
                radius_label_overview_rows = []
                current_radius_state = {
                    "pos_radius_km": pos_radius_km,
                    "aircraft_weight": aircraft_weight,
                    "combo_run_dir": str(combo_run_dir),
                    "combo_cache_dir": str(combo_cache_dir),
                    "fold_results": radius_fold_results,
                    "skipped_folds": skipped_folds,
                    "label_overview_rows": radius_label_overview_rows,
                }

                for split_file in tqdm(split_files, desc=f"Radius {pos_radius_km:g} folds", leave=False, unit="fold"):
                    try:
                        print(f"DEBUG: Processing {split_file}")
                        split_data = json.loads(Path(split_file).read_text(encoding="utf-8"))
                        fold_dir = Path(split_file).parent
                        fold_run_dir = combo_run_dir / args.experiment / fold_dir.name
                        fold_run_dir.mkdir(parents=True, exist_ok=True)
                        train_output_dir = fold_run_dir / "training"

                        label_overview = _split_label_overview(split_data)
                        fold_label_overview_path = fold_run_dir / "label_overview_pretrain.json"
                        fold_label_overview_path.write_text(json.dumps(label_overview, indent=2), encoding="utf-8")

                        for split_name in ("train", "val", "test"):
                            stats = label_overview[split_name]
                            radius_label_overview_rows.append(
                                {
                                    "radius_km": pos_radius_km,
                                    "aircraft_weight": aircraft_weight,
                                    "fold_id": split_data.get("fold_id"),
                                    "split": split_name,
                                    "total": stats["total"],
                                    "pos": stats["pos"],
                                    "neg": stats["neg"],
                                    "pos_rate": stats["pos_rate"],
                                    "pos_rate_percent": stats["pos_rate_percent"],
                                    "split_json": str(split_file),
                                    "label_overview_pretrain_json": str(fold_label_overview_path),
                                }
                            )

                        val_pos = int(label_overview["val"]["pos"])
                        test_pos = int(label_overview["test"]["pos"])
                        skip_reasons: list[str] = []
                        if val_pos == 0:
                            skip_reasons.append("validation split has 0 positive labels")
                        if test_pos == 0:
                            skip_reasons.append("test split has 0 positive labels")

                        if skip_reasons:
                            skip_info = {
                                "radius_km": pos_radius_km,
                                "aircraft_weight": aircraft_weight,
                                "fold_id": split_data.get("fold_id"),
                                "split_json": str(split_file),
                                "status": "skipped",
                                "reason": "; ".join(skip_reasons),
                                "label_overview_pretrain": label_overview,
                            }
                            skip_path = fold_run_dir / "skip_reason.json"
                            _write_json(skip_path, skip_info)
                            skipped_folds.append({**skip_info, "skip_reason_path": str(skip_path)})

                            print(
                                "[skip_fold] "
                                f"fold={split_data.get('fold_id')} "
                                f"reason={skip_info['reason']} "
                                f"saved={skip_path}"
                            )
                            _write_json(combo_run_dir / "radius_progress.json", _build_grid_summary(pos_radius_km, aircraft_weight, radius_fold_results, skipped_folds, completed=False))
                            _write_json(run_out_dir / "search_progress.json", {"completed_radii": completed_radii, "active_radius": current_radius_state})
                            continue

                        if args.one_fold_per_process:
                            fold_result_json = fold_run_dir / "single_fold_result.json"
                            fold_cmd = [
                                sys.executable,
                                str(Path(__file__).resolve()),
                                "--run-single-fold",
                                "--single-split-json",
                                str(split_file),
                                "--single-output-dir",
                                str(train_output_dir),
                                "--single-threshold",
                                str(args.threshold),
                                "--single-epochs",
                                str(args.epochs),
                                "--single-batch-size",
                                str(args.batch_size),
                                "--single-lr",
                                str(args.lr),
                                "--single-freeze-backbone",
                                str(not args.unfreeze_backbone).lower(),
                                "--single-aircraft-weight",
                                "none" if aircraft_weight is None else str(aircraft_weight),
                                "--single-result-json",
                                str(fold_result_json),
                            ]
                            subprocess.run(fold_cmd, check=True)
                            fold_payload = json.loads(fold_result_json.read_text(encoding="utf-8"))
                            result = fold_payload["result"]
                        else:
                            result = train_fold(
                                aircraft_weight=aircraft_weight,
                                split_json=split_file,
                                output_dir=train_output_dir,
                                threshold=args.threshold,
                                epochs=args.epochs,
                                batch_size=args.batch_size,
                                lr=args.lr,
                                freeze_backbone=not args.unfreeze_backbone,
                            )

                        val_auc = _best_history_metric(result.get("history", {}), "val_auc")
                        val_loss = _best_history_metric(result.get("history", {}), "val_loss")
                        test_metrics = result.get("metrics", {})
                        radius_fold_results.append(
                            {
                                "radius_km": pos_radius_km,
                                "aircraft_weight": aircraft_weight,
                                "freeze_backbone": not args.unfreeze_backbone,
                                "fold_id": split_data.get("fold_id"),
                                "split_json": str(split_file),
                                "manifest_path": str(fold_dir / "manifest.csv"),
                                "train_output_dir": str(train_output_dir),
                                "best_val_auc": val_auc,
                                "best_val_loss": val_loss,
                                "test_metrics": test_metrics,
                                "history": result.get("history", {}),
                                "pretrain_label_overview": label_overview,
                            }
                        )

                    except Exception as e:
                        print(f"ERROR in train_fold: {type(e).__name__}: {e}")
                        import traceback
                        traceback.print_exc()
                        _write_json(combo_run_dir / "radius_progress.json", _build_grid_summary(pos_radius_km, aircraft_weight, radius_fold_results, skipped_folds, completed=False))
                        _write_json(run_out_dir / "search_progress.json", {"completed_radii": completed_radii, "active_radius": current_radius_state})
                        continue

                    _write_json(combo_run_dir / "radius_progress.json", _build_grid_summary(pos_radius_km, aircraft_weight, radius_fold_results, skipped_folds, completed=False))
                    _write_json(run_out_dir / "search_progress.json", {"completed_radii": completed_radii, "active_radius": current_radius_state})

                if radius_label_overview_rows:
                    pd.DataFrame(radius_label_overview_rows).to_csv(
                        combo_run_dir / "label_overview_pretrain.csv",
                        index=False,
                    )

                if skipped_folds:
                    pd.DataFrame(skipped_folds).to_csv(
                        combo_run_dir / "skipped_folds.csv",
                        index=False,
                    )

                combo_summary = _build_grid_summary(pos_radius_km, aircraft_weight, radius_fold_results, skipped_folds, completed=True)
                _write_json(combo_run_dir / "radius_summary.json", combo_summary)
                _write_json(combo_run_dir / "radius_progress.json", combo_summary)
                search_results.append(combo_summary)
                completed_radii.append(combo_summary)

                print(f"\n=== Clearing session and collecting garbage after radius {pos_radius_km:g} km weight={aircraft_weight if aircraft_weight is not None else 'none'} ===")
                tf.keras.backend.clear_session()
                gc.collect()
                print("Session cleared and garbage collected.\n")

                print(
                    f"Radius {pos_radius_km:g} km weight={aircraft_weight if aircraft_weight is not None else 'none'}: "
                    f"mean best val AUC={combo_summary['mean_best_val_auc']:.4f}, mean test AUC={combo_summary['mean_test_auc']:.4f}"
                )

    except KeyboardInterrupt:
        print("Interrupted by user; writing partial progress files before exit.")
        if current_radius_state is not None:
            radius_km = float(current_radius_state["pos_radius_km"])
            radius_run_dir = Path(current_radius_state["combo_run_dir"])
            partial_summary = _build_radius_summary(
                radius_km,
                current_radius_state["fold_results"],
                current_radius_state["skipped_folds"],
                completed=False,
            )
            _write_json(radius_run_dir / "radius_progress.json", partial_summary)
            print(f"Partial progress for radius {radius_km:g} km written to {radius_run_dir / 'radius_progress.json'}")
        _write_json(run_out_dir / "search_progress.json", {"completed_radii": completed_radii, "active_radius": current_radius_state, "interrupted": True})
        print(f'Search progress written to {run_out_dir / "search_progress.json"}')
        raise
    finally:
        _write_json(run_out_dir / "search_progress.json", {"completed_radii": completed_radii, "active_radius": current_radius_state, "interrupted": False})

    if not search_results:
        print("No completed radii were available to summarize.")
        return

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
        "freeze_backbone": not args.unfreeze_backbone,
        "radius_candidates": radii,
        "split_cache_root": str(split_cache_root),
        "best_radius_by_mean_val_auc": best_radius["radius_km"],
        "best_aircraft_weight_by_mean_val_auc": best_radius.get("aircraft_weight"),
        "best_radius_mean_val_auc": best_radius["mean_best_val_auc"],
        "best_radius_freeze_backbone": best_radius.get("freeze_backbone"),
        "selected_mean_test_auc": selected_mean_test_auc,
        "selected_mean_test_loss": selected_mean_test_loss,
        "radius_results": search_results,
        "selected_fold_results": selected_fold_results,
    }
    (run_out_dir / "radius_search_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Best radius by mean validation AUC: {best_radius['radius_km']:g} km")
    print(f"Search summary written to {run_out_dir / 'radius_search_summary.json'}")


if __name__ == "__main__":
    import time, datetime
    from pathlib import Path as _Path
    _start = time.perf_counter()
    try:
        main()
    finally:
        _elapsed = time.perf_counter() - _start
        print(f"{_Path(__file__).name} elapsed: {_elapsed:.2f}s")
        _log_path = _Path(__file__).with_suffix('.runtime.log')
        with open(_log_path, 'a', encoding='utf-8') as _f:
            _f.write(f"{datetime.datetime.utcnow().isoformat()}Z {_elapsed:.6f}s\n")
