from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.experiment_builder import build_leakage_free_cv_experiments


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-free AeroSonic/Norwegian CV experiments")
    parser.add_argument("--aerosonic-manifest", required=True)
    parser.add_argument("--norwegian-manifest", required=True)
    parser.add_argument("--out-dir", default="data/experiments")
    parser.add_argument(
        "--experiment",
        action="append",
        choices=["aero_only_to_norwegian", "aero_aug_noise_to_norwegian", "aero_plus_norwegian_with_aug"],
        help="Experiment(s) to build. Repeat to build several.",
    )
    parser.add_argument("--augments-per-source", type=int, default=1)
    parser.add_argument("--snr-min", type=float, default=0.0)
    parser.add_argument("--snr-max", type=float, default=20.0)
    parser.add_argument("--augment-all-labels", action="store_true", help="Override and augment all source rows")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    experiments = args.experiment or ["aero_only_to_norwegian", "aero_aug_noise_to_norwegian", "aero_plus_norwegian_with_aug"]
    augment_only_positive = not args.augment_all_labels

    for exp in experiments:
        split_files = build_leakage_free_cv_experiments(
            aerosonic_manifest=args.aerosonic_manifest,
            norwegian_manifest=args.norwegian_manifest,
            out_dir=args.out_dir,
            experiment=exp,
            augments_per_source=args.augments_per_source,
            snr_range_db=(args.snr_min, args.snr_max),
            augment_only_positive=augment_only_positive,
            seed=args.seed,
        )
        print(f"Built {len(split_files)} folds for {exp} in {args.out_dir}")


if __name__ == "__main__":
    main()
