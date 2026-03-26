import argparse
import json
from pathlib import Path

import pandas as pd

from src.evaluation.analysis import generate_per_condition_report, snr_stratified_analysis


def parse_args():
    p = argparse.ArgumentParser(description="Phase 8 evaluation entrypoint")
    p.add_argument("--experiment", type=str, required=True)
    p.add_argument("--all-folds", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    exp_dir = Path("outputs") / args.experiment
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment output folder not found: {exp_dir}")

    fold_csvs = sorted(exp_dir.glob("*/predictions.csv"))
    if not fold_csvs:
        raise RuntimeError("No predictions.csv files found under experiment folder.")

    all_predictions = {}
    for csv in fold_csvs:
        fold_name = csv.parent.name
        all_predictions[fold_name] = pd.read_csv(csv)

    report_dir = exp_dir / "evaluation_report"
    out = generate_per_condition_report(all_predictions, output_dir=str(report_dir))

    full_df = pd.concat(all_predictions.values(), ignore_index=True)
    snr_stratified_analysis(full_df, noise_profiles={}, output_dir=str(report_dir))

    with (report_dir / "report_index.json").open("w", encoding="utf-8") as f:
        json.dump({"report": out}, f, indent=2)

    print(f"Saved evaluation report to: {report_dir}")


if __name__ == "__main__":
    main()
