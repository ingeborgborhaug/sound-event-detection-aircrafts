import argparse
import glob
import os
import re
import sys
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.split_generator import LOSOSplitGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Phase 3 LOSO split files")
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "manifest.csv"),
        help="Path to manifest CSV. If missing, script builds a manifest from per-session CSV caches.",
    )
    parser.add_argument(
        "--splits-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "splits"),
        help="Output folder for split JSON files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for AeroSonic train/val split.",
    )
    parser.add_argument(
        "--norwegian-root",
        type=str,
        default=r"C:\Users\kampfly\Documents\Ingeborg\Prosjektoppgave\sound-event-detection-aircrafts\dataset\Skatval",
        help="Root folder containing per-session Norwegian CSV files.",
    )
    parser.add_argument(
        "--sessions",
        type=str,
        default="280126,230226",
        help="Comma-separated Norwegian session IDs to include when building fallback manifest.",
    )
    parser.add_argument(
        "--norwegian-csv-glob",
        type=str,
        default="loc_*_*AUTOSAVE*.csv",
        help="Glob pattern for per-location CSV files in each session folder.",
    )
    parser.add_argument(
        "--aerosonic-cache-glob",
        type=str,
        default=str(PROJECT_ROOT / "dataset" / "AeroSonicDB" / "*_Unfiltered.npz"),
        help="Glob for AeroSonic cache files (.npz HDF5) used in fallback manifest mode.",
    )
    parser.add_argument(
        "--save-generated-manifest",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "manifest_phase3_generated.csv"),
        help="Where to save fallback generated manifest CSV.",
    )
    return parser.parse_args()


def _read_labels_from_cache(cache_path: str) -> np.ndarray:
    with h5py.File(cache_path, "r") as f:
        y = np.asarray(f["y"][:]).reshape(-1)
    return y.astype(int)


def _rows_from_cache(
    cache_path: str,
    dataset: str,
    session: str,
    location: str,
    wav_source: str,
) -> List[Dict[str, object]]:
    y = _read_labels_from_cache(cache_path)
    rows: List[Dict[str, object]] = []
    for i, label in enumerate(y.tolist()):
        rows.append(
            {
                "npy_path": cache_path,
                "segment_idx": int(i),
                "wav_source": wav_source,
                "dataset": dataset,
                "session": session,
                "location": location,
                "start_s": None,
                "end_s": None,
                "label": int(label),
            }
        )
    return rows


def _cache_path_from_csv(csv_path: str, apply_filter: str = "Unfiltered") -> str:
    return str(Path(csv_path).with_suffix("")) + f"_{apply_filter}.npz"


def _infer_loc_and_session(csv_path: str) -> tuple[str, str]:
    name = Path(csv_path).name
    m = re.search(r"loc_(\d+)_(\d{6})", name)
    if not m:
        raise ValueError(
            f"Cannot parse location/session from filename: {name}. "
            "Expected pattern loc_<num>_<session>..."
        )
    loc_num, session = m.group(1), m.group(2)
    return f"loc_{loc_num}", session


def build_manifest_from_csv_caches(
    norwegian_root: str,
    sessions: List[str],
    norwegian_csv_glob: str,
    aerosonic_cache_glob: str,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for session in sessions:
        csv_paths = sorted(glob.glob(str(Path(norwegian_root) / session / norwegian_csv_glob)))
        for csv_path in csv_paths:
            location, session_id = _infer_loc_and_session(csv_path)
            cache_path = _cache_path_from_csv(csv_path)
            if not os.path.exists(cache_path):
                print(f"Skipping missing cache: {cache_path}")
                continue

            rows.extend(
                _rows_from_cache(
                    cache_path=cache_path,
                    dataset="norwegian",
                    session=session_id,
                    location=location,
                    wav_source=os.path.basename(csv_path),
                )
            )

    # Add AeroSonic caches if available (baseline and fixed val split need these).
    for aero_cache in sorted(glob.glob(aerosonic_cache_glob)):
        rows.extend(
            _rows_from_cache(
                cache_path=aero_cache,
                dataset="aerosonic",
                session="aerosonic",
                location="all",
                wav_source=os.path.basename(aero_cache),
            )
        )

    if not rows:
        raise RuntimeError("No rows could be created for fallback manifest.")

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    manifest_path = Path(args.manifest)
    if manifest_path.exists():
        manifest_df = pd.read_csv(manifest_path)
        print(f"Loaded manifest: {manifest_path} ({len(manifest_df)} rows)")
    else:
        sessions = [s.strip() for s in args.sessions.split(",") if s.strip()]
        manifest_df = build_manifest_from_csv_caches(
            norwegian_root=args.norwegian_root,
            sessions=sessions,
            norwegian_csv_glob=args.norwegian_csv_glob,
            aerosonic_cache_glob=args.aerosonic_cache_glob,
        )

        out_manifest = Path(args.save_generated_manifest)
        out_manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest_df.to_csv(out_manifest, index=False)
        print(f"Generated fallback manifest: {out_manifest} ({len(manifest_df)} rows)")

    generator = LOSOSplitGenerator(seed=args.seed)
    folds = generator.generate_all_folds(manifest_df)
    generator.save_folds(folds, args.splits_dir)

    print(f"Saved {len(folds)} fold files to: {args.splits_dir}")
    generator.print_fold_summary(folds, manifest_df)


if __name__ == "__main__":
    main()
