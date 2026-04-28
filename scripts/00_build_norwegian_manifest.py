from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUT_DIR = Path(os.environ.get("SED_NORWEGIAN_OUT_DIR", r"E:\sed_cache\norwegian"))


def _load_spec(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        example = path.with_name("norwegian_sessions.example.json")
        hint = f" Create it from {example} and update audio_dirs/session values." if example.exists() else ""
        raise FileNotFoundError(f"Spec file not found: {path}.{hint}")
    # Accept both utf-8 and utf-8 with BOM (common when edited on Windows).
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict) and "entries" in data:
        data = data["entries"]
    if not isinstance(data, list):
        raise ValueError("Session spec must be a JSON list or a dict with an 'entries' list")
    return data


def _normalize_audio_dirs(entry: dict[str, Any]) -> list[str]:
    dirs = entry.get("audio_dirs") or entry.get("audio_dir") or entry.get("audio_folders")
    if dirs is None:
        raise ValueError(f"Missing audio_dirs for entry: {entry}")
    normalized = [dirs] if isinstance(dirs, str) else [str(d) for d in dirs]
    for d in normalized:
        if d.startswith("/path/to/"):
            raise ValueError(
                f"Placeholder audio path detected: {d}. "
                "Update configs/norwegian_sessions.json with real audio folders."
            )
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Norwegian/Skatval manifest from session specs")
    parser.add_argument("--spec", required=True, type=Path, help="JSON file with preprocessing entries")
    parser.add_argument("--manifest", required=True, type=Path, help="Combined output manifest CSV")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=(
            "Cache directory for npy files "
            "(default: SED_NORWEGIAN_OUT_DIR env var, else E:\\sed_cache\\norwegian)"
        ),
    )
    parser.add_argument("--apply-filter", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    entries = _load_spec(args.spec)
    if not entries:
        raise ValueError("Spec file contains no entries")

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
        if not Path(gt_path).exists():
            raise FileNotFoundError(f"GT file not found: {gt_path}")

        audio_dirs = _normalize_audio_dirs(entry)
        for d in audio_dirs:
            if not Path(d).exists():
                raise FileNotFoundError(f"Audio directory not found: {d}")
        dataset = entry.get("dataset_override", "norwegian")
        session = entry.get("session_override") or entry.get("session")
        location = entry.get("location_override") or entry.get("location")
        fold = entry.get("fold_override")
        pair_name = entry.get("pair_name") or f"norwegian_session_{idx}"

        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "01_preprocess.py"),
            "--gt-path", str(gt_path),
            "--pair-name", str(pair_name),
            "--manifest", str(args.manifest),
            "--out-dir", str(args.out_dir),
            "--dataset-override", str(dataset),
            "--append-manifest",
        ]

        for d in audio_dirs:
            cmd.extend(["--audio-dir", d])
        if args.apply_filter is not None:
            cmd.extend(["--apply-filter", args.apply_filter])
        if args.force:
            cmd.append("--force")
        if session is not None:
            cmd.extend(["--session-override", str(session)])
        if location is not None:
            cmd.extend(["--location-override", str(location)])
        if fold is not None:
            cmd.extend(["--fold-override", str(fold)])

        print(f"[{idx}/{len(entries)}] preprocessing {pair_name}")
        subprocess.run(cmd, check=True)

    print(f"Combined manifest written to {args.manifest}")


if __name__ == "__main__":
    main()
