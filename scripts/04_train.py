from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import settings
from src.training.trainer import train_fold


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Option-B model on LOSO splits")
    #parser.add_argument("--manifest", default="data/processed/manifest.csv")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--output-root", default="/mnt/e/history/sound-event-detection-aircrafts/runs")
    parser.add_argument("--threshold", type=float, default=settings.PREDICTION_THRESHOLD)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-patches", type=int, default=int(settings.MAX_PATCHES))
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--unfreeze-backbone", action="store_true")
    args = parser.parse_args()

    split_files = sorted(Path(args.splits_dir).rglob("split.json"))
    if not split_files:
        split_files = sorted(Path(args.splits_dir).glob("*.json"))
    if not split_files:
        raise RuntimeError(f"No split JSON files in {args.splits_dir}")

    run_dir = Path(args.output_root) / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for split_file in reversed(split_files):
        fold_name = split_file.stem
        fold_dir = run_dir / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)


        print(f'Using threshold: {args.threshold} and splitfile {split_file}')
        result = train_fold(
            split_json=split_file,
            output_dir=fold_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            #max_patches=args.max_patches,
            lr=args.lr,
            freeze_backbone=not args.unfreeze_backbone,
            threshold=args.threshold,
        )
        all_results.append(result)

    (run_dir / "cv_results.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"Finished cross-validation. Results in {run_dir}")


if __name__ == "__main__":
    main()
