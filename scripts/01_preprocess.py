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


def _process_audio_file(
    filename: str,
    audio_path: Path,
    events: pd.DataFrame,
    pair: PairSpec,
    out_dir: Path,
    radius_km: float | None,
    args: argparse.Namespace,
    loader: Any,
    extractor: Any,
) -> dict[str, Any] | None:
    """Process a single audio file and return manifest row or None if skipped."""
    
    # Determine fold value
    if args.fold_override is not None:
        fold_value = int(args.fold_override)
    else:
        fold_values = events["fold"].dropna().astype(int).unique().tolist()
        fold_value = int(fold_values[0]) if fold_values else -1

    fold_tag = f"fold{fold_value}" if fold_value >= 0 else "nofold"
    x_out_name = f"{Path(filename).stem}_{fold_tag}.npz"
    y_out_name = f"{Path(filename).stem}_{fold_tag}_y_radius{radius_km}km.npy"
    x_out_path = out_dir / x_out_name
    y_out_path = out_dir / y_out_name
    out_path = x_out_path

    x_exists = x_out_path.exists()
    y_exists = y_out_path.exists()

    # Pre-compute dataset, session, location for use in all code paths
    dataset = args.dataset_override or _infer_dataset(pair.pair_name, pair.gt_path)
    inferred_session, inferred_location = _extract_session_and_location(dataset, pair.pair_name, pair.gt_path, filename)
    session = args.session_override or inferred_session
    location = args.location_override or inferred_location

    # Try loading from cache
    if x_exists and y_exists and not args.force:
        try:
            with np.load(x_out_path) as data:
                X = data["X"]    
            y = np.load(y_out_path)
            if X.shape[0] != y.shape[0]:
                raise ValueError(f"Shape mismatch between X and y in cache for {filename}")
            print(f'Percentage of positive patches for {filename}: {y.mean() * 100:.2f}%')
            return {
                "npz_path": str(out_path),
                "y_path": str(y_out_path),
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
        except Exception as e:
            print(f"Error loading cache for {filename}: {e}. Recomputing.")
            x_exists = False
            y_exists = False

    elif x_exists and not y_exists and not args.force:
        try:
            with np.load(x_out_path) as data:
                X = data["X"]
                start_times = data["start_s"]
                end_times = data["end_s"]

            y = np.zeros(len(X), dtype=np.int32)

            for _, row in events.iterrows():
                start_s = float(row["start_time"])
                end_s = float(row["end_time"])
                class_label = str(row["class"]).strip().lower()

                overlaps = (start_times < end_s) & (end_times > start_s)

                if class_label == "ignore":
                    continue
                state = _label_to_state(class_label)
                if state == 1:
                    y[overlaps] = 1
                elif state == 0:
                    y[overlaps] = 0

            print(f'Percentage of positive patches for {filename}: {y.mean() * 100:.2f}%')
            np.save(y_out_path, y)
            return {
                "npz_path": str(out_path),
                "y_path": str(y_out_path),
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
        except Exception as e:
            print(f"Error loading X cache for {filename}: {e}. Recomputing both X and y.")
            x_exists = False
            y_exists = False

    # Compute from scratch
    print('Cache not found for', out_path, 'processing audio and GT to create it.')
    audio, sr = loader.load_mono_int16(audio_path)
    patches, _ = extractor.extract(audio, sr)

    labels = np.zeros(len(patches), dtype=np.int32)
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

    kept_indices = np.where(keep_mask)[0]

    X = patches[kept_indices].astype(np.float32)
    y = labels[kept_indices].astype(np.int32)

    np.save(y_out_path, y)

    start_times = kept_indices * yamnet_params.PATCH_HOP_SECONDS
    end_times = start_times + yamnet_params.PATCH_WINDOW_SECONDS
    fold_array = np.full(len(kept_indices), fold_value, dtype=np.int32)

    if args.force or not out_path.exists():
        np.savez_compressed(
            out_path,
            X=X,
            fold=fold_array,
            start_s=start_times.astype(np.float32),
            end_s=end_times.astype(np.float32),
            filename=np.array(filename),
            audio_path=np.array(str(audio_path)),
            gt_path=np.array(str(pair.gt_path)),
        )

    print(f'Percentage of positive patches for {filename}: {y.mean() * 100:.2f}%')
    return {
        "npz_path": str(out_path),
        "y_path": str(y_out_path),
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


def _process_single_entry(args: argparse.Namespace, loader: Any, extractor: Any) -> None:
    """Legacy mode: process a single pair/entry (backward compatible)."""
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
            
            dataset = args.dataset_override or _infer_dataset(pair.pair_name, pair.gt_path)
            session, location = _extract_session_and_location(dataset, pair.pair_name, pair.gt_path, filename)
            if args.session_override is not None:
                session = args.session_override
            if args.location_override is not None:
                location = args.location_override
            
            out_dir = args.out_dir / dataset / session / location
            out_dir.mkdir(parents=True, exist_ok=True)

            row = _process_audio_file(filename, audio_path, events, pair, out_dir, radius_km, args, loader, extractor)
            if row:
                rows.append(row)
            processed_audio_count += 1

        if args.max_audios is not None and processed_audio_count >= args.max_audios:
            break

    manifest_df = pd.DataFrame(rows)
    if args.append_manifest and args.manifest.exists():
        existing_df = pd.read_csv(args.manifest)
        manifest_df = pd.concat([existing_df, manifest_df], ignore_index=True)
    
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(args.manifest, index=False)
    LOGGER.info(f"Saved manifest with {len(manifest_df)} rows to {args.manifest}")
    LOGGER.info(f"Processed audio files: {processed_audio_count}")
    LOGGER.info(f"Skipped files missing in audio dirs: {skipped_missing_audio}")
    LOGGER.info(f"Skipped empty/missing GT pairs: {skipped_missing_gt}")


def _process_batch_entries(args: argparse.Namespace, loader: Any, extractor: Any) -> None:
    """Batch mode: process all entries from spec file in a single Python process."""
    import json
    
    spec_path = args.spec_file
    if not spec_path.exists():
        raise FileNotFoundError(f"Spec file not found: {spec_path}")
    
    data = json.loads(spec_path.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict) and "entries" in data:
        entries = data["entries"]
    elif isinstance(data, list):
        entries = data
    else:
        raise ValueError("Spec file must be a JSON list or dict with 'entries' key")
    
    if not entries:
        raise ValueError("Spec file contains no entries")

    rows: list[dict[str, Any]] = []
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    
    if args.manifest.exists():
        args.manifest.unlink()

    for idx, entry in enumerate(entries, start=1):
        gt_path = entry.get("gt_path") or entry.get("gt")
        if not gt_path:
            raise ValueError(f"Missing gt_path in entry: {entry}")
        if str(gt_path).startswith("/path/to/"):
            raise ValueError(
                f"Placeholder gt_path detected: {gt_path}. "
                "Update configs/norwegian_sessions.json with real GT files."
            )
        gt_path = Path(gt_path)
        if not gt_path.exists():
            raise FileNotFoundError(f"GT file not found: {gt_path}")

        audio_dirs_raw = entry.get("audio_dirs") or entry.get("audio_dir") or entry.get("audio_folders")
        if audio_dirs_raw is None:
            raise ValueError(f"Missing audio_dirs for entry: {entry}")
        audio_dirs = [Path(audio_dirs_raw)] if isinstance(audio_dirs_raw, str) else [Path(d) for d in audio_dirs_raw]
        
        for d in audio_dirs:
            if not d.exists():
                raise FileNotFoundError(f"Audio directory not found: {d}")

        dataset = entry.get("dataset_override", "norwegian")
        session_override = entry.get("session_override") or entry.get("session")
        location_override = entry.get("location_override") or entry.get("location")
        fold_override = entry.get("fold_override")
        pair_name = entry.get("pair_name") or f"norwegian_session_{idx}"

        # Create a minimal PairSpec for this entry
        pair = PairSpec(
            gt_path=gt_path,
            audio_dirs=audio_dirs,
            pair_name=pair_name,
        )

        gt = _load_gt_dataframe(pair.gt_path)
        if gt.empty:
            raise ValueError(f"GT file is empty after parsing required rows: {pair.gt_path}")

        audio_index = _build_audio_index(pair.audio_dirs)
        if not audio_index:
            raise FileNotFoundError(f"No audio files found for pair={pair.pair_name} under {pair.audio_dirs}")
        
        grouped = gt.groupby("filename", sort=False)
        radius_km = _extract_radius_km(pair.pair_name)

        print(f"[{idx}/{len(entries)}] preprocessing {pair_name}")
        for filename, events in tqdm(grouped, desc=f"{pair_name}", leave=False):
            audio_path = audio_index.get(filename)
            if audio_path is None:
                raise FileNotFoundError(
                    f"Audio file referenced in GT not found: {filename} (pair={pair.pair_name}, gt={pair.gt_path})"
                )
            
            inferred_dataset = dataset or _infer_dataset(pair.pair_name, pair.gt_path)
            inferred_session, inferred_location = _extract_session_and_location(inferred_dataset, pair.pair_name, pair.gt_path, filename)
            
            final_session = session_override or inferred_session
            final_location = location_override or inferred_location
            final_fold = fold_override
            
            out_dir = args.out_dir / inferred_dataset / final_session / final_location
            out_dir.mkdir(parents=True, exist_ok=True)

            # Temporarily set args for _process_audio_file compatibility
            args.dataset_override = dataset
            args.session_override = session_override
            args.location_override = location_override
            args.fold_override = fold_override

            row = _process_audio_file(filename, audio_path, events, pair, out_dir, radius_km, args, loader, extractor)
            if row:
                rows.append(row)

    manifest_df = pd.DataFrame(rows)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(args.manifest, index=False)
    LOGGER.info(f"Saved manifest with {len(manifest_df)} rows to {args.manifest}")
    LOGGER.info(f"Processed {len(entries)} entries")


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
    parser.add_argument("--spec-file", type=Path, default=None, help="JSON spec file with entries (batch mode)")
    args = parser.parse_args()

    # Load YAMNet ONCE at the start of the entire process
    loader = AudioLoader()
    extractor = YAMNetFeatureExtractor(apply_filter=args.apply_filter)
    
    # Check if batch mode (--spec-file) or legacy mode
    if args.spec_file:
        _process_batch_entries(args, loader, extractor)
    else:
        _process_single_entry(args, loader, extractor)


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
