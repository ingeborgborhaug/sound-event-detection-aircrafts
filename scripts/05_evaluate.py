from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.array(values, dtype=float)
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def _print_metric_summary(results: list[dict[str, Any]], title: str) -> None:
    keys = sorted({k for r in results for k in r.get("metrics", {}).keys()})
    print(title)
    for key in keys:
        vals = [r["metrics"][key] for r in results if key in r.get("metrics", {})]
        mean, std = _mean_std(vals)
        print(f"{key}: {mean:.4f} ± {std:.4f}")


def _mean_metric_from_fold_results(fold_results: list[dict[str, Any]], metric_name: str) -> tuple[float, float]:
    vals = []
    for entry in fold_results:
        test_metrics = entry.get("test_metrics", {})
        if metric_name in test_metrics:
            vals.append(float(test_metrics[metric_name]))
    if not vals:
        return float("nan"), float("nan")
    return _mean_std(vals)


def _print_radius_summary(summary: dict[str, Any]) -> None:
    radius = summary.get("radius_km")
    print("Radius search summary")
    if radius is not None:
        print(f"radius_km: {radius:g}")
    if "mean_best_val_auc" in summary:
        print(f"mean_best_val_auc: {summary['mean_best_val_auc']:.4f}")
    if "mean_test_auc" in summary:
        print(f"mean_test_auc: {summary['mean_test_auc']:.4f}")
    if "mean_test_loss" in summary:
        print(f"mean_test_loss: {summary['mean_test_loss']:.4f}")

    fold_results = summary.get("fold_results", [])
    if fold_results:
        print("Fold metrics")
        for entry in fold_results:
            fold_id = entry.get("fold_id", "?")
            test_metrics = entry.get("test_metrics", {})
            best_val_auc = entry.get("best_val_auc", float("nan"))
            test_auc = test_metrics.get("auc", float("nan"))
            test_loss = test_metrics.get("loss", float("nan"))
            test_acc = test_metrics.get("acc", float("nan"))
            test_precision = test_metrics.get("precision", float("nan"))
            test_recall = test_metrics.get("recall", float("nan"))
            print(
                f"fold {fold_id}: best_val_auc={best_val_auc:.4f}, test_auc={test_auc:.4f}, "
                f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}, "
                f"test_precision={test_precision:.4f}, test_recall={test_recall:.4f}"
            )


def _print_search_summary(summary: dict[str, Any]) -> None:
    print("Radius hyperparameter search summary")
    print(f"experiment: {summary.get('experiment', 'unknown')}")
    if "best_radius_by_mean_val_auc" in summary:
        print(f"best_radius_by_mean_val_auc: {summary['best_radius_by_mean_val_auc']:g}")
    if "best_radius_mean_val_auc" in summary:
        print(f"best_radius_mean_val_auc: {summary['best_radius_mean_val_auc']:.4f}")
    if "selected_mean_test_auc" in summary:
        print(f"selected_mean_test_auc: {summary['selected_mean_test_auc']:.4f}")
    if "selected_mean_test_loss" in summary:
        print(f"selected_mean_test_loss: {summary['selected_mean_test_loss']:.4f}")

    selected_fold_results = summary.get("selected_fold_results", [])
    if selected_fold_results:
        selected_mean_test_acc, _ = _mean_metric_from_fold_results(selected_fold_results, "acc")
        selected_mean_test_precision, _ = _mean_metric_from_fold_results(selected_fold_results, "precision")
        selected_mean_test_recall, _ = _mean_metric_from_fold_results(selected_fold_results, "recall")
        print(f"selected_mean_test_acc: {selected_mean_test_acc:.4f}")
        print(f"selected_mean_test_precision: {selected_mean_test_precision:.4f}")
        print(f"selected_mean_test_recall: {selected_mean_test_recall:.4f}")

    radius_results = summary.get("radius_results", [])
    if radius_results:
        print("Per-radius metrics")
        for entry in radius_results:
            radius_km = entry.get("radius_km", float("nan"))
            mean_best_val_auc = entry.get("mean_best_val_auc", float("nan"))
            mean_test_auc = entry.get("mean_test_auc", float("nan"))
            mean_test_loss = entry.get("mean_test_loss", float("nan"))
            fold_results = entry.get("fold_results", [])
            mean_test_acc, _ = _mean_metric_from_fold_results(fold_results, "acc")
            mean_test_precision, _ = _mean_metric_from_fold_results(fold_results, "precision")
            mean_test_recall, _ = _mean_metric_from_fold_results(fold_results, "recall")
            print(
                f"radius {radius_km:g} km: mean_best_val_auc={mean_best_val_auc:.4f}, "
                f"mean_test_auc={mean_test_auc:.4f}, mean_test_loss={mean_test_loss:.4f}, "
                f"mean_test_acc={mean_test_acc:.4f}, mean_test_precision={mean_test_precision:.4f}, "
                f"mean_test_recall={mean_test_recall:.4f}"
            )


def _print_fold_across_radii(summary: dict[str, Any], fold_id: int) -> None:
    radius_results = summary.get("radius_results", [])
    if not radius_results:
        print(f"No radius results available for fold {fold_id}")
        return

    print(f"Fold {fold_id} across radii")
    found = False
    fold_candidates: list[dict[str, Any]] = []
    for radius_entry in radius_results:
        radius_km = radius_entry.get("radius_km", float("nan"))
        fold_results = radius_entry.get("fold_results", [])
        matching_fold = next((entry for entry in fold_results if entry.get("fold_id") == fold_id), None)
        if matching_fold is None:
            continue

        found = True
        fold_candidates.append({"radius_km": radius_km, **matching_fold})
        test_metrics = matching_fold.get("test_metrics", {})
        best_val_auc = float(matching_fold.get("best_val_auc", float("nan")))
        best_val_loss = float(matching_fold.get("best_val_loss", float("nan")))
        test_auc = float(test_metrics.get("auc", float("nan")))
        test_loss = float(test_metrics.get("loss", float("nan")))
        test_acc = float(test_metrics.get("acc", float("nan")))
        test_precision = float(test_metrics.get("precision", float("nan")))
        test_recall = float(test_metrics.get("recall", float("nan")))

        print(
            f"radius {radius_km:g} km: best_val_auc={best_val_auc:.4f}, best_val_loss={best_val_loss:.4f}, "
            f"test_auc={test_auc:.4f}, test_loss={test_loss:.4f}, test_acc={test_acc:.4f}, "
            f"test_precision={test_precision:.4f}, test_recall={test_recall:.4f}"
        )

    if not found:
        print(f"Fold {fold_id} was not found in any radius result")
        return

    best_by_val_auc = max(fold_candidates, key=lambda entry: float(entry.get("best_val_auc", float("nan"))))
    best_radius_km = float(best_by_val_auc.get("radius_km", float("nan")))
    best_val_auc = float(best_by_val_auc.get("best_val_auc", float("nan")))
    best_val_loss = float(best_by_val_auc.get("best_val_loss", float("nan")))
    best_test_metrics = best_by_val_auc.get("test_metrics", {})
    print(
        f"Best radius for fold {fold_id}: {best_radius_km:g} km "
        f"(best_val_auc={best_val_auc:.4f}, best_val_loss={best_val_loss:.4f}, "
        f"test_auc={float(best_test_metrics.get('auc', float('nan'))):.4f}, "
        f"test_loss={float(best_test_metrics.get('loss', float('nan'))):.4f}, "
        f"test_acc={float(best_test_metrics.get('acc', float('nan'))):.4f}, "
        f"test_precision={float(best_test_metrics.get('precision', float('nan'))):.4f}, "
        f"test_recall={float(best_test_metrics.get('recall', float('nan'))):.4f})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate CV or radius-search metrics")
    parser.add_argument(
        "--results",
        help="Path to cv_results.json, radius_summary.json, or radius_search_summary.json",
    )
    parser.add_argument(
        "--cv-results",
        dest="results",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--fold-id",
        type=int,
        default=None,
        help="When reading a radius_search_summary.json, print one fold's results across all radii",
    )
    args = parser.parse_args()

    if not args.results:
        raise ValueError("Provide --results or --cv-results")

    results = json.loads(Path(args.results).read_text(encoding="utf-8"))

    if isinstance(results, list):
        _print_metric_summary(results, "Cross-validation summary")
        return

    if not isinstance(results, dict):
        raise ValueError("Expected results to be a JSON list or object")

    if "radius_results" in results:
        if args.fold_id is not None:
            _print_fold_across_radii(results, args.fold_id)
            return
        _print_search_summary(results)
        return

    if "fold_results" in results:
        _print_radius_summary(results)
        return

    raise ValueError("Unrecognized results format")


if __name__ == "__main__":
    main()
