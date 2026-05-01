from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import settings
from keras_yamnet import params as yamnet_params
from src.preprocessing import AudioLoader, YAMNetFeatureExtractor

LOGGER = logging.getLogger("preprocess")


@dataclass
class PairSpec:
    gt_path: Path
    audio_dirs: list[Path]
    pair_name: str


def _flatten_data_pairs_from_settings() -> list[PairSpec]:
    pairs: list[PairSpec] = []

    for name in dir(settings):
        if not name.startswith("data_pairs"):
            continue
        value = getattr(settings, name)
        if not isinstance(value, dict):
            continue

        if value and all(isinstance(k, (int, float, np.floating)) for k in value.keys()):
            for radius_value, nested in value.items():
                if isinstance(nested, dict):
                    for gt, dirs in nested.items():
                        pairs.append(
                            PairSpec(
                                gt_path=Path(gt),
                                audio_dirs=[Path(d) for d in dirs],
                                pair_name=f"{name}_{radius_value:g}km",
                            )
                        )
        else:
            for gt, dirs in value.items():
                pairs.append(
                    PairSpec(
                        gt_path=Path(gt),
                        audio_dirs=[Path(d) for d in dirs],
                        pair_name=name,
                    )
                )

    dedup: dict[tuple[str, tuple[str, ...]], PairSpec] = {}
    for p in pairs:
        key = (str(p.gt_path), tuple(sorted(str(x) for x in p.audio_dirs)))
        dedup[key] = p
    return list(dedup.values())


def _manual_pair_specs(args: argparse.Namespace) -> list[PairSpec]:
    if not args.gt_path:
        return []
    if not args.audio_dir:
        raise ValueError("--audio-dir is required when --gt-path is provided")
    return [
        PairSpec(
            gt_path=Path(args.gt_path),
            audio_dirs=[Path(d) for d in args.audio_dir],
            pair_name=args.pair_name,
        )
    ]


def _load_gt_dataframe(gt_path: Path) -> pd.DataFrame:
    if not gt_path.exists():
        raise FileNotFoundError(f"GT file not found: {gt_path}")

    try:
        gt = pd.read_csv(gt_path, sep=None, engine="python")
    except Exception:
        gt = pd.read_csv(gt_path, sep=r"\s+", engine="python")

    required = {"filename", "start_time", "end_time", "class"}
    if not required.issubset(set(gt.columns)):
        raise ValueError(f"{gt_path} missing required columns {required}")

    gt = gt.copy()
    gt["start_time"] = pd.to_numeric(gt["start_time"], errors="coerce")
    gt["end_time"] = pd.to_numeric(gt["end_time"], errors="coerce")
    gt = gt.dropna(subset=["filename", "start_time", "end_time", "class"])
    gt["class"] = gt["class"].astype(str)
    if "fold" in gt.columns:
        gt["fold"] = pd.to_numeric(gt["fold"], errors="coerce").astype("Int64")
    else:
        gt["fold"] = pd.Series([pd.NA] * len(gt), dtype="Int64")
    return gt


def _build_audio_index(audio_dirs: list[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for root in audio_dirs:
        if not root.exists():
            continue
        for wav in root.rglob("*.wav"):
            index.setdefault(wav.name, wav)
    return index


def _infer_dataset(pair_name: str, gt_path: Path) -> str:
    token = f"{pair_name} {gt_path}".lower()
    if "aero" in token:
        return "aerosonic"
    return "norwegian"


def _extract_session_and_location(dataset: str, pair_name: str, gt_path: Path, filename: str) -> tuple[str, str]:
    if dataset == "aerosonic":
        return gt_path.stem, "aerosonic_default"

    token = f"{pair_name} {gt_path.stem} {filename}".lower()

    session_match = re.search(r"(20\d{6}|\d{6}|session[_-]?\d+)", token)
    location_match = re.search(r"loc(?:ation)?[_-]?(\d+|[abc])", token)

    session = session_match.group(1) if session_match else "session_unknown"
    location = f"loc_{location_match.group(1)}" if location_match else "loc_unknown"
    return session, location


def _extract_radius_km(pair_name: str) -> float | None:
    match = re.search(r"_(\d+(?:\.\d+)?)km$", pair_name.lower())
    return float(match.group(1)) if match else None


def _label_to_state(raw: str) -> int | None:
    value = str(raw).strip().lower()
    if value in {"", "nan", "none", "ignore"}:
        return None
    if value in {"1", "true", "aircraft", "plane", "positive", "event"}:
        return 1
    if value in {"0", "false", "no_aircraft", "background", "noise", "negative", "noevent"}:
        return 0
    try:
        return 1 if float(value) > 0 else 0
    except Exception:
        return 0


def _segment_label(segment_start: float, segment_end: float, events: pd.DataFrame) -> int:
    if events.empty:
        return 0
    overlap = events[(events["start_time"] < segment_end) & (events["end_time"] > segment_start)]
    if overlap.empty:
        return 0

    states = overlap["class"].map(_label_to_state).tolist()
    if any(s == 1 for s in states):
        return 1
    return 0


def _segment_fold(segment_start: float, segment_end: float, events: pd.DataFrame) -> int | None:
    if "fold" not in events.columns:
        return None
    overlap = events[(events["start_time"] < segment_end) & (events["end_time"] > segment_start)]
    if overlap.empty:
        return None
    folds = overlap["fold"].dropna().astype(int).unique().tolist()
    return int(folds[0]) if folds else None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Precompute YAMNet mel spectrogram patches and cache them.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"), help="Output root for cached npy files")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.csv"), help="Manifest CSV path")
    parser.add_argument("--apply-filter", type=str, default=None, help="Optional filter name for keras_yamnet preprocessing")
    parser.add_argument("--force", action="store_true", help="Recompute existing cache files")
    parser.add_argument("--pair-filter", type=str, default=None, help="Only process pairs whose name contains this text")
    parser.add_argument("--max-audios", type=int, default=None, help="Optional limit on number of audio files to process")
    parser.add_argument("--gt-path", type=str, default=None, help="Manual GT CSV path (overrides settings data_pairs)")
    parser.add_argument("--audio-dir", action="append", default=None, help="Manual audio directory, repeatable")
    parser.add_argument("--pair-name", type=str, default="manual_pair", help="Pair name when using --gt-path")
    parser.add_argument("--dataset-override", type=str, default=None, help="Force dataset name in manifest rows")
    parser.add_argument("--session-override", type=str, default=None, help="Force session/day value in manifest rows")
    parser.add_argument("--location-override", type=str, default=None, help="Force location value in manifest rows")
    parser.add_argument("--fold-override", type=int, default=None, help="Force fold value in manifest rows")
    parser.add_argument("--append-manifest", action="store_true", help="Append to manifest if it already exists")
    args = parser.parse_args()

    loader = AudioLoader()
    extractor = YAMNetFeatureExtractor(apply_filter=args.apply_filter)

    rows: list[dict[str, Any]] = []
    pairs = _manual_pair_specs(args) or _flatten_data_pairs_from_settings()

    if not pairs:
        raise RuntimeError("No data_pairs* dictionaries found in settings.py")

    if args.pair_filter:
        pairs = [p for p in pairs if args.pair_filter.lower() in p.pair_name.lower()]
    if not pairs:
        raise RuntimeError("No pair specs matched --pair-filter")

    processed_audio_count = 0
    skipped_missing_audio = 0
    skipped_missing_gt = 0

    for pair in pairs:
        gt = _load_gt_dataframe(pair.gt_path)
        if gt.empty:
            raise ValueError(f"GT file is empty after parsing required rows: {pair.gt_path}")

        audio_index = _build_audio_index(pair.audio_dirs)
        if not audio_index:
            raise FileNotFoundError(f"No audio files found for pair={pair.pair_name} under {pair.audio_dirs}")
        grouped = gt.groupby("filename", sort=False)
        radius_km = _extract_radius_km(pair.pair_name)

        for filename, events in tqdm(grouped, desc=f"{pair.pair_name}", leave=False):
            if args.max_audios is not None and processed_audio_count >= args.max_audios:
                break

            audio_path = audio_index.get(filename)
            if audio_path is None:
                raise FileNotFoundError(
                    f"Audio file referenced in GT not found: {filename} (pair={pair.pair_name}, gt={pair.gt_path})"
                )

            audio, sr = loader.load_mono_int16(audio_path)
            dataset = args.dataset_override or _infer_dataset(pair.pair_name, pair.gt_path)
            session, location = _extract_session_and_location(dataset, pair.pair_name, pair.gt_path, filename)
            if args.session_override is not None:
                session = args.session_override
            if args.location_override is not None:
                location = args.location_override

                        # Extract all YAMNet patches from the full wav, same principle as Pipeline 1
            patches, _ = extractor.extract(audio, sr)

            # One label per patch, initially background
            labels = np.zeros(len(patches), dtype=np.int32)

            # One fold per patch
            if args.fold_override is not None:
                fold_value = int(args.fold_override)
            else:
                fold_values = events["fold"].dropna().astype(int).unique().tolist()
                fold_value = int(fold_values[0]) if fold_values else -1

            # Track patches to remove
            keep_mask = np.ones(len(patches), dtype=bool)

            for _, row in events.iterrows():
                start_s = float(row["start_time"])
                end_s = float(row["end_time"])
                class_label = str(row["class"]).strip().lower()

                patch_start = int(np.floor(start_s / yamnet_params.PATCH_HOP_SECONDS))
                patch_end = int(np.ceil(end_s / yamnet_params.PATCH_HOP_SECONDS))

                patch_start = max(0, patch_start)
                patch_end = min(len(patches), patch_end)

                if class_label == "ignore":
                    keep_mask[patch_start:patch_end] = False
                else:
                    state = _label_to_state(class_label)
                    if state == 1:
                        labels[patch_start:patch_end] = 1
                    elif state == 0:
                        labels[patch_start:patch_end] = 0

            out_dir = args.out_dir / dataset / session / location
            out_dir.mkdir(parents=True, exist_ok=True)

            kept_indices = np.where(keep_mask)[0]

            X = patches[kept_indices].astype(np.float32)
            y = labels[kept_indices].astype(np.int32)

            start_times = kept_indices * yamnet_params.PATCH_HOP_SECONDS
            end_times = start_times + yamnet_params.PATCH_WINDOW_SECONDS
            fold_array = np.full(len(kept_indices), fold_value, dtype=np.int32)

            out_dir = args.out_dir / dataset / session / location
            out_dir.mkdir(parents=True, exist_ok=True)

            fold_tag = f"fold{fold_value}" if fold_value >= 0 else "nofold"
            out_name = f"{Path(filename).stem}_{fold_tag}.npz"
            out_path = out_dir / out_name

            if args.force or not out_path.exists():
                np.savez_compressed(
                    out_path,
                    X=X,
                    y=y,
                    fold=fold_array,
                    start_s=start_times.astype(np.float32),
                    end_s=end_times.astype(np.float32),
                    filename=np.array(filename),
                    audio_path=np.array(str(audio_path)),
                    gt_path=np.array(str(pair.gt_path)),
                )

            rows.append(
                {
                    "npz_path": str(out_path),
                    "audio_path": str(audio_path),
                    "gt_path": str(pair.gt_path),
                    "dataset": dataset,
                    "session": session,
                    "location": location,
                    "radius_km": radius_km,
                    "filename": filename,
                    "fold": fold_value,
                    "num_patches": int(len(X)),
                    "num_positive": int(y.sum()),
                    "num_negative": int(len(y) - y.sum()),
                    "shape": str(tuple(X.shape)),
                    "pair_name": pair.pair_name,
                }
            )

            processed_audio_count += 1

        if args.max_audios is not None and processed_audio_count >= args.max_audios:
            break

    manifest_df = pd.DataFrame(rows)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    if args.append_manifest and args.manifest.exists():
        existing = pd.read_csv(args.manifest)
        manifest_df = pd.concat([existing, manifest_df], ignore_index=True)
        dedup_cols = [
            col
            for col in ["npz_path", "pair_name", "radius_km", "filename", "fold"]
            if col in manifest_df.columns
        ]
        if dedup_cols:
            manifest_df = manifest_df.drop_duplicates(subset=dedup_cols, keep="last")
    manifest_df.to_csv(args.manifest, index=False)

    LOGGER.info("Saved manifest with %d rows to %s", len(manifest_df), args.manifest)
    LOGGER.info("Processed audio files: %d", processed_audio_count)
    LOGGER.info("Skipped files missing in audio dirs: %d", skipped_missing_audio)
    LOGGER.info("Skipped empty/missing GT pairs: %d", skipped_missing_gt)


if __name__ == "__main__":
    main()
