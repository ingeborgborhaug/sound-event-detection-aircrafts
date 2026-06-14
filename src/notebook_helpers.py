import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd


def _resolve_mnt_path(p: str) -> Path:
    p = str(p)
    if Path(p).exists():
        return Path(p)
    m = re.match(r"^/mnt/([a-zA-Z])/(.*)$", p)
    if m:
        drive = m.group(1).upper()
        rest = m.group(2)
        candidate = Path(f"{drive}:/" + rest)
        if candidate.exists():
            return candidate
    # Fallback: try replacing leading /mnt/ with C:/
    if p.startswith("/mnt/"):
        parts = p.split("/", 3)
        if len(parts) >= 3:
            drive = parts[2].upper()
            rest = p.split("/", 3)[3] if len(parts) == 4 else ""
            candidate = Path(f"{drive}:/" + rest)
            if candidate.exists():
                return candidate
    return Path(p)


def load_split_json(path_like: str) -> Optional[Dict[str, Any]]:
    """Attempt to load a split.json file, resolving common /mnt/<drive>/ -> <DRIVE>:/ mappings.

    Returns parsed JSON dict or None if file not found / could not be read.
    """
    try_paths = [path_like]
    try_paths.append(str(_resolve_mnt_path(path_like)))

    for p in try_paths:
        try:
            pth = Path(p)
            if pth.exists():
                with pth.open("r", encoding="utf-8") as fh:
                    return json.load(fh)
        except Exception:
            continue
    return None


def format_session_radius_df(session_radius: pd.DataFrame, radius_label_fmt: str = "{r} km") -> pd.DataFrame:
    """Return a display-friendly DataFrame where the index is converted to a column
    named 'Session' so the top-left header cell shows 'Session' on the same row as radii.

    session_radius: pivot table with index=session and columns radii (numeric or str).
    """
    df = session_radius.reset_index()
    # Normalize index column name to Session
    if df.columns[0] != "Session":
        df = df.rename(columns={df.columns[0]: "Session"})

    # Rename radius columns, if numeric, to '{r} km'
    new_cols = []
    for c in df.columns[1:]:
        try:
            r = int(c)
            new_cols.append(radius_label_fmt.format(r=r))
        except Exception:
            new_cols.append(str(c))
    df.columns = ["Session"] + new_cols
    return df


def get_validation_overview_from_fold_summaries(fold_summaries: List[Dict[str, Any]],
                                                fold_to_session: Dict[int, int]) -> pd.DataFrame:
    """Build a compact DataFrame mapping test-session -> validation-session(s).

    fold_summaries: list of dicts, each ideally containing a 'split_json' entry (path or URL)
    fold_to_session: mapping of fold id -> session id used in this project
    """
    rows = []
    for fs in fold_summaries:
        split_path = fs.get("split_json") or fs.get("result_json") or fs.get("split")
        fold_dir = fs.get("fold_dir") or fs.get("fold") or ""
        # Try to parse fold id from fold_dir like 'fold_2_test_2'
        fold_id = None
        m = re.search(r"fold_(\d+)_test_(\d+)", fold_dir)
        if m:
            try:
                fold_id = int(m.group(1))
            except Exception:
                fold_id = None
        # fallback: attempt to get numeric 'fold' key
        if fold_id is None and "fold" in fs:
            try:
                fold_id = int(fs.get("fold"))
            except Exception:
                fold_id = None

        test_session = fold_to_session.get(fold_id, None) if fold_id is not None else None

        val_sessions = None
        if split_path:
            payload = load_split_json(split_path)
            if isinstance(payload, dict):
                # Heuristics to find validation sessions
                if "val_sessions" in payload:
                    val_sessions = payload["val_sessions"]
                elif "validation_sessions" in payload:
                    val_sessions = payload["validation_sessions"]
                elif "splits" in payload and isinstance(payload["splits"], dict):
                    # look for entries labeled 'val'
                    vs = [k for k, v in payload["splits"].items() if v == "val"]
                    if vs:
                        val_sessions = vs
                else:
                    # Look for any mapping where values contain 'val' or 'validation'
                    found = []
                    for k, v in payload.items():
                        if isinstance(v, (list, dict)):
                            continue
                        if isinstance(v, str) and ("val" in v.lower() or "validation" in v.lower()):
                            found.append(k)
                    if found:
                        val_sessions = found

        rows.append({
            "Test Session": test_session if test_session is not None else "unknown",
            "Validation Sessions": ", ".join(map(str, val_sessions)) if val_sessions else "(not available)"
        })

    return pd.DataFrame(rows).sort_values("Test Session")
