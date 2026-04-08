from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.split_generator import generate_fold_splits, generate_loso_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LOSO split files from preprocessing manifest")
    parser.add_argument("--manifest", default="data/processed/manifest.csv")
    parser.add_argument("--out-dir", default="data/splits")
    parser.add_argument("--dataset", default="norwegian", choices=["norwegian", "aerosonic"])
    args = parser.parse_args()

    if args.dataset == "aerosonic":
        paths = generate_fold_splits(args.manifest, args.out_dir, dataset=args.dataset, fold_column="fold")
    else:
        paths = generate_loso_splits(args.manifest, args.out_dir, dataset=args.dataset)

    print(f"Generated {len(paths)} split files in {args.out_dir}")


if __name__ == "__main__":
    main()
