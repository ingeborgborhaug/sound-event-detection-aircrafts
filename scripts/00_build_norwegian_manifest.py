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

    # Call 01_preprocess.py ONCE in batch mode with the spec file
    # This loads YAMNet once and processes all entries in a single Python process
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "01_preprocess.py"),
        "--spec-file", str(args.spec),
        "--manifest", str(args.manifest),
        "--out-dir", str(args.out_dir),
    ]

    if args.apply_filter is not None:
        cmd.extend(["--apply-filter", args.apply_filter])
    if args.force:
        cmd.append("--force")

    print(f"Processing {len(entries)} entries in batch mode (YAMNet loaded once)")
    subprocess.run(cmd, check=True)

    print(f"Combined manifest written to {args.manifest}")


if __name__ == "__main__":
    main()
