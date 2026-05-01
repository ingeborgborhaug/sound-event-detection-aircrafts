from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.split_generator import (
    generate_fold_splits,
    generate_loso_splits,
    generate_train_manifest_val_fold_test_manifest_splits,
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LOSO split files from preprocessing manifest")
    parser.add_argument("--manifest", default="data/processed/manifest.csv")
    parser.add_argument("--out-dir", default="data/splits")
    parser.add_argument("--dataset", default="norwegian", choices=["norwegian", "aerosonic"])
    parser.add_argument("--mode", default="default", choices=["default", "external-test"])
    parser.add_argument("--train-manifest", default=None)
    parser.add_argument("--test-manifest", default=None)
    parser.add_argument("--train-dataset", default=None)
    parser.add_argument("--test-dataset", default=None)
    args = parser.parse_args()

    if args.mode == "external-test":
        if args.train_manifest is None or args.test_manifest is None:
            raise ValueError("--train-manifest and --test-manifest are required for external-test mode")

        paths = generate_train_manifest_val_fold_test_manifest_splits(
            train_manifest_path=args.train_manifest,
            test_manifest_path=args.test_manifest,
            out_dir=args.out_dir,
            train_dataset=args.train_dataset,
            test_dataset=args.test_dataset,
            fold_column="fold",
        )

    else:
        if args.dataset == "aerosonic":
            paths = generate_fold_splits(args.manifest, args.out_dir, dataset=args.dataset, fold_column="fold")
        else:
            paths = generate_loso_splits(args.manifest, args.out_dir, dataset=args.dataset)

    print(f"Generated {len(paths)} split files in {args.out_dir}")


if __name__ == "__main__":
    main()
