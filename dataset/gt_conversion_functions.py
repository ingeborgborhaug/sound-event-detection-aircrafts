from keras_yamnet import params
import pandas as pd
from pathlib import Path
import settings
import numpy as np

# def second_to_index(sec):
#     """
#     Convert seconds to index in variable output in when loading data from gt.
#     """
#     return int(sec // params.PATCH_HOP_SECONDS)

def sec_to_start_index(sec):
    return int(np.floor(sec / params.PATCH_HOP_SECONDS))

def sec_to_end_index(sec):
    return int(np.floor((sec - params.PATCH_WINDOW_SECONDS) / params.PATCH_HOP_SECONDS))

def class_name_to_index(class_name):
    """
    Convert class name to index based on the class names defined in the YAMNet model.
    """
    if class_name in settings.CLASS_NAMES:
        return settings.CLASS_NAMES.tolist().index(class_name)
    else:
        raise ValueError(f"Class name '{class_name}' not found in CLASS_NAMES.")
    
def _detect_sep(sample_line: str) -> str:
    """Guess separator from the header line."""
    if "," in sample_line and "\t" not in sample_line:
        return ","
    if "\t" in sample_line and "," not in sample_line:
        return "\t"
    # fallback: any whitespace
    return r"\s+"

def duplicate_gt_for_filenames(
    in_path: str,
    out_path: str,
    new_filenames: list[str],
    include_original: bool = True,
    drop_dupes: bool = True,
    out_sep: str | None = None,
):
    """
    Read a GT file with columns: filename, start_time, end_time, class
    and create/update a file where the same detections are duplicated for each
    filename in `new_filenames`.

    If `out_path` already exists, its contents are loaded and *extended* with
    the new duplicated rows instead of being overwritten.
    """
    in_path = Path(in_path)
    out_path = Path(out_path)

    # --- Load template from input GT file ---
    first_line_in = in_path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    in_sep = _detect_sep(first_line_in)

    df = pd.read_csv(in_path, sep=in_sep, engine="python")

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    required = ["filename", "start_time", "end_time", "class"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {list(df.columns)}")

    # Build duplicated rows (keep time/class, swap filename)
    template = df.drop(columns=["filename"])
    replicated = pd.concat(
        [template.assign(filename=Path(fn).name) for fn in new_filenames],
        ignore_index=True
    )

    # --- If out_path exists, update it instead of recreating from scratch ---
    if out_path.exists():
        first_line_out = out_path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
        existing_sep = _detect_sep(first_line_out)

        existing = pd.read_csv(out_path, sep=existing_sep, engine="python")
        existing.columns = [c.strip().lower() for c in existing.columns]

        # Ensure column order/availability
        for col in required:
            if col not in existing.columns:
                raise ValueError(f"Existing out file is missing required column '{col}'")

        existing = existing[required]

        # Append new replicated rows to the existing file
        out_df = pd.concat([existing, replicated], ignore_index=True)

        # Default: keep the existing separator unless explicitly overridden
        if out_sep is None:
            out_sep = existing_sep
    else:
        # Original behavior: create new file from input GT
        out_df = pd.concat([df, replicated], ignore_index=True) if include_original else replicated

        # Choose output separator
        if out_sep is None:
            out_sep = "," if in_sep == "," else "\t"  # write tsv for tab/whitespace inputs

    # Order columns + sort for readability
    out_df = out_df[["filename", "start_time", "end_time", "class"]]
    out_df = out_df.sort_values(["filename", "start_time", "end_time"], kind="mergesort")

    if drop_dupes:
        out_df = out_df.drop_duplicates(ignore_index=True)

    out_df.to_csv(out_path, sep=out_sep, index=False)
    print(f"Wrote duplicated/updated GT to: {out_path}")
    return out_df