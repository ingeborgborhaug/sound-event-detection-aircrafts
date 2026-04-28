from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from keras_yamnet import params


SESSION_NAMES = [
    "300925",
    "280126",
    "230226",
    "030326",
    "260326_part1",
    "260326_part2",
]
LOCATION_ORDER = ["loc_1", "loc_2", "loc_3", "gardemoen"]
SESSION_LABEL_MAP = {
    "280126": "Session 1",
    "230226": "Session 2",
    "030326": "Session 3",
    "260326": "Session 4",
    "300925": "Session 5",
}
SESSION_LABEL_ORDER = {
    "Session 1": 1,
    "Session 2": 2,
    "Session 3": 3,
    "Session 4": 4,
    "Session 5": 5,
}
SESSION_DATE_PATTERN = re.compile(r"^(?P<date>\d{6})(?:_part(?P<part>\d+))?$")

GT_NAME_PATTERN = re.compile(
    r"^(?P<location>loc_[123]|gardemoen)_(?P<session>.+?)_AUTOSAVE_sphere_(?P<radius>[0-9]+(?:\.[0-9]+)?)KM$"
)


@dataclass
class SessionData:
    session: str
    wav_files: list[Path]
    wav_total_seconds: float
    gt_files: list[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Skatval dataset summary tables and thesis figures."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root folder that contains session folders (e.g., D:/Skatval...).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dataset_statistics"),
        help="Directory where tables and figures are saved.",
    )
    parser.add_argument(
        "--radius-km",
        type=float,
        default=3.0,
        help="Only GT files for this radius are used (default: 3.0).",
    )
    parser.add_argument(
        "--sample-hop-seconds",
        type=float,
        default=params.PATCH_HOP_SECONDS,
        help="Temporal sampling resolution used for positive/negative sample counts.",
    )
    parser.add_argument(
        "--session-names",
        nargs="+",
        default=SESSION_NAMES,
        help="Session folder names to include.",
    )
    parser.add_argument(
        "--event-label",
        type=str,
        default="Aircraft",
        help="Label used in figure titles, for example Aircraft.",
    )
    return parser.parse_args()


def read_gt_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    required = ["filename", "start_time", "end_time", "class"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    out = df.copy()
    out["start_time"] = pd.to_numeric(out["start_time"], errors="coerce")
    out["end_time"] = pd.to_numeric(out["end_time"], errors="coerce")
    out = out.dropna(subset=["start_time", "end_time"])
    out = out[out["end_time"] >= out["start_time"]].copy()
    out["duration_s"] = out["end_time"] - out["start_time"]
    return out


def parse_gt_name(path: Path) -> tuple[str, str, float]:
    match = GT_NAME_PATTERN.match(path.stem)
    if not match:
        raise ValueError(f"GT filename does not match expected format: {path.name}")
    location = match.group("location")
    session = match.group("session")
    radius = float(match.group("radius"))
    return location, session, radius


def session_sort_key(name: str) -> tuple[pd.Timestamp, int, str]:
    match = SESSION_DATE_PATTERN.match(name)
    if not match:
        return (pd.Timestamp.max, 999, name)

    date_text = match.group("date")
    part_text = match.group("part")
    dt = pd.to_datetime(date_text, format="%d%m%y", errors="coerce")
    if pd.isna(dt):
        return (pd.Timestamp.max, 999, name)

    part = int(part_text) if part_text is not None else 0
    return (dt, part, name)


def sort_sessions(names: Iterable[str]) -> list[str]:
    return sorted(names, key=session_sort_key)


def sort_locations(names: Iterable[str]) -> list[str]:
    known = {name: idx for idx, name in enumerate(LOCATION_ORDER)}
    return sorted(names, key=lambda name: (known.get(name, len(known)), name))


def format_session_label(session: str) -> str:
    """Convert raw session name to display label used in plots."""
    base = session.split("_part", maxsplit=1)[0]
    return SESSION_LABEL_MAP.get(base, session)


def session_label_sort_key(label: str) -> tuple[int, str]:
    """Sort Session labels numerically (Session 1, Session 2, ...)."""
    return (SESSION_LABEL_ORDER.get(label, 999), label)


def session_display_order(sessions: Iterable[str]) -> list[str]:
    """Build ordered, deduplicated display labels from session names."""
    labels = {format_session_label(session) for session in sessions}
    return sorted(labels, key=session_label_sort_key)


def format_location_label(location: str) -> str:
    """Convert location names for display (gardemoen → location 4)."""
    if location == "gardemoen":
        return "location 4"
    return location


def is_positive_label(value: object) -> bool:
    text = str(value).strip().lower()
    if text in {"", "none", "nan"}:
        return False
    if text == "ignore":
        return False
    try:
        return float(text) > 0
    except ValueError:
        return text not in {"0", "false", "negative", "no_aircraft"}


def sum_wav_seconds(wav_files: Iterable[Path]) -> float:
    total = 0.0
    for wav in wav_files:
        info = sf.info(str(wav))
        if info.samplerate and info.frames:
            total += float(info.frames) / float(info.samplerate)
    return total


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals_sorted = sorted(intervals, key=lambda pair: (pair[0], pair[1]))
    merged = [intervals_sorted[0]]
    for start, end in intervals_sorted[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def discover_session_data(
    dataset_root: Path,
    session_names: list[str],
    radius_km: float,
) -> list[SessionData]:
    results: list[SessionData] = []
    for session in sort_sessions(session_names):
        session_dir = dataset_root / session
        if not session_dir.exists():
            print(f"[WARN] Session folder not found, skipping: {session_dir}")
            continue

        wav_files = sorted(session_dir.glob("*.wav"))
        wav_total_seconds = sum_wav_seconds(wav_files)

        gt_dir = session_dir / "Newly_generated"
        candidate_gt: list[Path] = []
        if gt_dir.exists():
            candidate_gt = sorted(gt_dir.glob("*_AUTOSAVE_sphere_*KM.csv"))
        else:
            candidate_gt = sorted(session_dir.glob("*_AUTOSAVE_sphere_*KM.csv"))

        filtered_gt: list[Path] = []
        for gt_path in candidate_gt:
            try:
                _, gt_session, radius = parse_gt_name(gt_path)
            except ValueError:
                continue
            if gt_session != session:
                continue
            if math.isclose(radius, radius_km, abs_tol=1e-6):
                filtered_gt.append(gt_path)

        if not filtered_gt:
            print(
                f"[WARN] No GT files found for session={session} and radius={radius_km:.1f}KM"
            )

        results.append(
            SessionData(
                session=session,
                wav_files=wav_files,
                wav_total_seconds=wav_total_seconds,
                gt_files=sorted(filtered_gt),
            )
        )
    return results


def build_events_dataframe(session_data: list[SessionData]) -> pd.DataFrame:
    rows: list[dict] = []
    for data in session_data:
        for gt_path in data.gt_files:
            location, session, radius = parse_gt_name(gt_path)
            gt_df = read_gt_csv(gt_path)
            gt_df["is_positive"] = gt_df["class"].apply(is_positive_label)
            gt_df = gt_df[gt_df["is_positive"]].copy()
            gt_df = gt_df[gt_df["duration_s"] > 0].copy()

            if gt_df.empty:
                continue

            for _, row in gt_df.iterrows():
                filename = str(row["filename"]).strip()
                if not filename or filename.lower() == "nan":
                    filename = f"{location}_{session}.wav"
                rows.append(
                    {
                        "session": session,
                        "location": location,
                        "radius_km": radius,
                        "filename": filename,
                        "start_time": float(row["start_time"]),
                        "end_time": float(row["end_time"]),
                        "duration_s": float(row["duration_s"]),
                        "class": str(row["class"]),
                    }
                )
    if not rows:
        return pd.DataFrame(
            columns=[
                "session",
                "location",
                "radius_km",
                "filename",
                "start_time",
                "end_time",
                "duration_s",
                "class",
            ]
        )
    return pd.DataFrame(rows)


def compute_positive_seconds(events_df: pd.DataFrame) -> dict[str, float]:
    positive_by_session: dict[str, float] = {}
    if events_df.empty:
        return positive_by_session

    grouped = events_df.groupby(["session", "filename"], dropna=False)
    for (session, _filename), group in grouped:
        intervals = list(zip(group["start_time"].tolist(), group["end_time"].tolist()))
        merged = merge_intervals(intervals)
        positive = sum(end - start for start, end in merged)
        positive_by_session[session] = positive_by_session.get(session, 0.0) + positive
    return positive_by_session


def make_event_balance_table(
    session_data: list[SessionData],
    events_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create table showing positive/negative event time and percentages per session."""
    positive_seconds = compute_positive_seconds(events_df)

    rows: list[dict] = []
    for data in session_data:
        wav_seconds = float(data.wav_total_seconds)
        pos_seconds = float(positive_seconds.get(data.session, 0.0))
        neg_seconds = max(wav_seconds - pos_seconds, 0.0)

        pos_percent = (pos_seconds / wav_seconds * 100.0) if wav_seconds > 0 else 0.0
        neg_percent = (neg_seconds / wav_seconds * 100.0) if wav_seconds > 0 else 0.0

        rows.append(
            {
                "session": data.session,
                "positive_duration_s": round(pos_seconds, 2),
                "negative_duration_s": round(neg_seconds, 2),
                "positive_percent": round(pos_percent, 2),
                "negative_percent": round(neg_percent, 2),
            }
        )

    balance_df = pd.DataFrame(rows)
    if not balance_df.empty:
        balance_df = balance_df.sort_values(by="session", key=lambda col: col.map(session_sort_key)).reset_index(drop=True)
    return balance_df


def make_summary_table(
    session_data: list[SessionData],
    events_df: pd.DataFrame,
    sample_hop_seconds: float,
) -> pd.DataFrame:
    positive_seconds = compute_positive_seconds(events_df)

    rows: list[dict] = []
    for data in session_data:
        session_events = events_df[events_df["session"] == data.session]
        event_count = int(len(session_events))
        wav_seconds = float(data.wav_total_seconds)
        pos_seconds = float(positive_seconds.get(data.session, 0.0))
        neg_seconds = max(wav_seconds - pos_seconds, 0.0)

        pos_samples = int(round(pos_seconds / sample_hop_seconds)) if sample_hop_seconds > 0 else 0
        neg_samples = int(round(neg_seconds / sample_hop_seconds)) if sample_hop_seconds > 0 else 0

        ratio = float("inf") if pos_samples == 0 else float(neg_samples / pos_samples)

        if event_count > 0:
            durations = session_events["duration_s"]
            mean_dur = float(durations.mean())
            median_dur = float(durations.median())
            min_dur = float(durations.min())
            max_dur = float(durations.max())
        else:
            mean_dur = 0.0
            median_dur = 0.0
            min_dur = 0.0
            max_dur = 0.0

        rows.append(
            {
                "session": data.session,
                "num_wav_files": len(data.wav_files),
                "recorded_hours": wav_seconds / 3600.0,
                "total_labeled_events": event_count,
                "positive_samples": pos_samples,
                "negative_samples": neg_samples,
                "neg_to_pos_ratio": ratio,
                "mean_event_duration_s": mean_dur,
                "median_event_duration_s": median_dur,
                "min_event_duration_s": min_dur,
                "max_event_duration_s": max_dur,
            }
        )

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(by="session", key=lambda col: col.map(session_sort_key)).reset_index(drop=True)
    return summary_df


def configure_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 20,
            "axes.titlesize": 20,
            "axes.labelsize": 20,
            "legend.fontsize": 20,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
        }
    )


def save_bar_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    ylabel: str,
    out_path: Path,
    order: list[str] | None = None,
) -> None:
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df, x=x, y=y, color="steelblue", order=order)
    ax.set_title(title)
    ax.set_xlabel("Session")
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def complete_grid(df: pd.DataFrame, session_col: str, value_col: str, value_name: str, sessions: list[str], categories: list[str], category_col: str) -> pd.DataFrame:
    if df.empty:
        grid = pd.MultiIndex.from_product([sessions, categories], names=[session_col, category_col]).to_frame(index=False)
        grid[value_name] = 0
        return grid

    grid = pd.MultiIndex.from_product([sessions, categories], names=[session_col, category_col]).to_frame(index=False)
    merged = grid.merge(df[[session_col, category_col, value_col]], on=[session_col, category_col], how="left")
    merged[value_col] = merged[value_col].fillna(0)
    return merged.rename(columns={value_col: value_name})


def create_figures(summary_df: pd.DataFrame, events_df: pd.DataFrame, output_dir: Path, event_label: str, radius_km: float) -> None:
    radius_folder = f"radius_{str(radius_km).replace('.', '_')}KM"
    figures_dir = output_dir / "figures" / radius_folder
    figures_dir.mkdir(parents=True, exist_ok=True)

    for existing_pdf in figures_dir.glob("*.pdf"):
        existing_pdf.unlink(missing_ok=True)
    configure_plot_style()

    session_order = sort_sessions(SESSION_NAMES)
    session_order_display = session_display_order(session_order)
    location_order = LOCATION_ORDER

    if not summary_df.empty:
        summary_display_df = summary_df.copy()
        summary_display_df["session_display"] = summary_display_df["session"].apply(format_session_label)
        summary_display_df = (
            summary_display_df.groupby("session_display", as_index=False)[
                ["recorded_hours", "total_labeled_events", "positive_samples", "negative_samples"]
            ]
            .sum()
        )

        save_bar_plot(
            summary_display_df,
            x="session_display",
            y="recorded_hours",
            title="Total Recorded Hours per Session",
            ylabel="Hours",
            out_path=figures_dir / "01_recorded_hours_per_session.pdf",
            order=session_order_display,
        )

        save_bar_plot(
            summary_display_df,
            x="session_display",
            y="total_labeled_events",
            title="Total Labeled Events per Session",
            ylabel="Events",
            out_path=figures_dir / "02_total_events_per_session.pdf",
            order=session_order_display,
        )

        stacked_df = summary_display_df[["session_display", "positive_samples", "negative_samples"]].set_index("session_display")
        stacked_df = stacked_df.reindex(session_order_display).fillna(0)
        fig, ax = plt.subplots(figsize=(12, 6))
        stacked_df.plot(kind="bar", stacked=True, ax=ax, color=["#1f77b4", "#ff7f0e"])
        ax.set_title("Positive vs Negative Samples per Session")
        ax.set_xlabel("Session")
        ax.set_ylabel("Sample Count")
        ax.legend(title="Sample Type")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(figures_dir / "03_positive_negative_samples_per_session.pdf", dpi=300)
        plt.close()

    if not events_df.empty:
        fig, ax = plt.subplots(figsize=(12, 6))

        sns.histplot(events_df["duration_s"], bins=40, kde=False, color="slateblue", ax=ax)
        plt.title(f"Histogram of {event_label} Event Durations")
        plt.xlabel("Duration (s)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(figures_dir / "04_event_duration_histogram.pdf", dpi=300)
        plt.close()

        per_location_session = (
            events_df.assign(session_display=events_df["session"].apply(format_session_label))
            .groupby(["session_display", "location"], as_index=False)
            .size()
            .rename(columns={"size": "events"})
        )
        per_location_session = complete_grid(
            per_location_session,
            session_col="session_display",
            value_col="events",
            value_name="events",
            sessions=session_order_display,
            categories=location_order,
            category_col="location",
        )
        per_location_session["session_display"] = pd.Categorical(per_location_session["session_display"], categories=session_order_display, ordered=True)
        per_location_session["location"] = pd.Categorical(per_location_session["location"], categories=location_order, ordered=True)
        per_location_session = per_location_session.sort_values(["session_display", "location"])
        per_location_session["location_display"] = per_location_session["location"].apply(format_location_label)
        plt.figure(figsize=(12, 6))
        sns.barplot(data=per_location_session, x="session_display", y="events", hue="location_display", order=session_order_display, hue_order=[format_location_label(loc) for loc in location_order])
        plt.title(f"Total {event_label} Events per Location and Session")
        plt.xlabel("Session")
        plt.ylabel("Events")
        plt.xticks(rotation=20, ha="right")
        plt.legend(title="Location")
        plt.tight_layout()
        plt.savefig(figures_dir / "05_events_per_location_and_session.pdf", dpi=300)
        plt.close()

        plt.figure(figsize=(10, 6))
        boxplot_df = events_df.copy()
        boxplot_df["location_display"] = boxplot_df["location"].apply(format_location_label)
        sns.boxplot(data=boxplot_df, x="location_display", y="duration_s", order=[format_location_label(loc) for loc in location_order])
        plt.title(f"{event_label} Event Duration per Location")
        plt.xlabel("Location")
        plt.ylabel("Duration (s)")
        plt.tight_layout()
        plt.savefig(figures_dir / "06_aircraft_event_duration_per_location.pdf", dpi=300)
        plt.close()

        per_location = pd.DataFrame({"location": location_order}).merge(
            events_df.groupby("location", as_index=False).size().rename(columns={"size": "events"}),
            on="location",
            how="left",
        ).fillna(0)
        per_location["location_display"] = per_location["location"].apply(format_location_label)
        plt.figure(figsize=(8, 6))
        sns.barplot(data=per_location, x="location_display", y="events", color="teal", order=[format_location_label(loc) for loc in location_order])
        plt.title(f"Distribution of {event_label} Events Across Locations")
        plt.xlabel("Location")
        plt.ylabel("Events")
        plt.tight_layout()
        plt.savefig(figures_dir / "07_distribution_across_locations.pdf", dpi=300)
        plt.close()

        class_session = (
            events_df.assign(session_display=events_df["session"].apply(format_session_label))
            .groupby(["class", "session_display"], as_index=False)
            .size()
            .rename(columns={"size": "events"})
        )
        class_session_pivot = class_session.pivot(index="class", columns="session_display", values="events").fillna(0)
        class_session_pivot = class_session_pivot.reindex(columns=session_order_display, fill_value=0)
        plt.figure(figsize=(12, 6))
        sns.heatmap(class_session_pivot, annot=True, fmt=".0f", cmap="Blues")
        plt.title("Class Distribution Across Sessions")
        plt.xlabel("Session")
        plt.ylabel("Class")
        plt.tight_layout()
        plt.savefig(figures_dir / "08_class_distribution_across_sessions_heatmap.pdf", dpi=300)
        plt.close()

        class_location = (
            events_df.groupby(["class", "location"], as_index=False)
            .size()
            .rename(columns={"size": "events"})
        )
        class_location_pivot = class_location.pivot(index="class", columns="location", values="events").fillna(0)
        class_location_pivot = class_location_pivot.reindex(columns=location_order, fill_value=0)
        class_location_pivot.columns = [format_location_label(col) for col in class_location_pivot.columns]
        plt.figure(figsize=(8, 6))
        sns.heatmap(class_location_pivot, annot=True, fmt=".0f", cmap="Purples")
        plt.title("Class/Location Heatmap")
        plt.xlabel("Location")
        plt.ylabel("Class")
        plt.tight_layout()
        plt.savefig(figures_dir / "09_class_location_heatmap.pdf", dpi=300)
        plt.close()


def write_additional_tables(events_df: pd.DataFrame, output_dir: Path) -> None:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    if events_df.empty:
        return

    session_order_display = session_display_order(SESSION_NAMES)

    location_breakdown = events_df.assign(session_display=events_df["session"].apply(format_session_label)).groupby(["session_display", "location"], as_index=False).agg(
        total_events=("duration_s", "size"),
        mean_event_duration_s=("duration_s", "mean"),
    )
    location_breakdown = location_breakdown.merge(
        pd.MultiIndex.from_product([session_order_display, LOCATION_ORDER], names=["session_display", "location"]).to_frame(index=False),
        on=["session_display", "location"],
        how="right",
    ).fillna({"total_events": 0, "mean_event_duration_s": 0.0})
    location_breakdown = location_breakdown.rename(columns={"session_display": "session"})
    location_breakdown = location_breakdown[["session", "location", "total_events", "mean_event_duration_s"]]
    location_breakdown["session"] = pd.Categorical(location_breakdown["session"], categories=session_order_display, ordered=True)
    location_breakdown = location_breakdown.sort_values(["session", "location"]).reset_index(drop=True)
    location_breakdown.to_csv(tables_dir / "events_per_location.csv", index=False)

    observed_classes = sorted(events_df["class"].astype(str).unique().tolist())
    class_breakdown = events_df.assign(session_display=events_df["session"].apply(format_session_label)).groupby(["session_display", "class"], as_index=False).agg(
        total_events=("duration_s", "size"),
        mean_event_duration_s=("duration_s", "mean"),
    )
    class_breakdown = class_breakdown.merge(
        pd.MultiIndex.from_product([session_order_display, observed_classes], names=["session_display", "class"]).to_frame(index=False),
        on=["session_display", "class"],
        how="right",
    ).fillna({"total_events": 0, "mean_event_duration_s": 0.0})
    class_breakdown = class_breakdown.rename(columns={"session_display": "session"})
    class_breakdown = class_breakdown[["session", "class", "total_events", "mean_event_duration_s"]]
    class_breakdown["session"] = pd.Categorical(class_breakdown["session"], categories=session_order_display, ordered=True)
    class_breakdown = class_breakdown.sort_values(["session", "class"]).reset_index(drop=True)
    class_breakdown.to_csv(tables_dir / "class_distribution_by_session.csv", index=False)


def ensure_complete_session_tables(summary_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df

    summary_df = summary_df.copy()
    summary_df["session"] = pd.Categorical(summary_df["session"], categories=SESSION_NAMES, ordered=True)
    summary_df = summary_df.sort_values("session").reset_index(drop=True)
    return summary_df


def write_overview_json(
    summary_df: pd.DataFrame,
    output_dir: Path,
    radius_km: float,
    sample_hop_seconds: float,
) -> None:
    metrics = {
        "radius_km": radius_km,
        "sample_hop_seconds": sample_hop_seconds,
        "total_recorded_hours": float(summary_df["recorded_hours"].sum()) if not summary_df.empty else 0.0,
        "total_labeled_events": int(summary_df["total_labeled_events"].sum()) if not summary_df.empty else 0,
        "total_positive_samples": int(summary_df["positive_samples"].sum()) if not summary_df.empty else 0,
        "total_negative_samples": int(summary_df["negative_samples"].sum()) if not summary_df.empty else 0,
    }
    path = output_dir / "dataset_overview.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def main() -> None:
    args = parse_args()

    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset root: {dataset_root}")
    print(f"Output dir:   {output_dir}")
    print(f"Radius:       {args.radius_km:.1f} KM")

    session_data = discover_session_data(
        dataset_root=dataset_root,
        session_names=args.session_names,
        radius_km=args.radius_km,
    )

    events_df = build_events_dataframe(session_data)
    summary_df = make_summary_table(
        session_data=session_data,
        events_df=events_df,
        sample_hop_seconds=args.sample_hop_seconds,
    )

    session_order_display = session_display_order(args.session_names)

    summary_output_df = summary_df.copy()
    if not summary_output_df.empty:
        summary_output_df["session"] = summary_output_df["session"].apply(format_session_label)
        summary_output_df = (
            summary_output_df.groupby("session", as_index=False)
            .agg(
                num_wav_files=("num_wav_files", "sum"),
                recorded_hours=("recorded_hours", "sum"),
                total_labeled_events=("total_labeled_events", "sum"),
                positive_samples=("positive_samples", "sum"),
                negative_samples=("negative_samples", "sum"),
                mean_event_duration_s=("mean_event_duration_s", "mean"),
                median_event_duration_s=("median_event_duration_s", "median"),
                min_event_duration_s=("min_event_duration_s", "min"),
                max_event_duration_s=("max_event_duration_s", "max"),
            )
        )
        summary_output_df["neg_to_pos_ratio"] = summary_output_df.apply(
            lambda row: float("inf") if row["positive_samples"] == 0 else float(row["negative_samples"] / row["positive_samples"]),
            axis=1,
        )
        summary_output_df = summary_output_df[
            [
                "session",
                "num_wav_files",
                "recorded_hours",
                "total_labeled_events",
                "positive_samples",
                "negative_samples",
                "neg_to_pos_ratio",
                "mean_event_duration_s",
                "median_event_duration_s",
                "min_event_duration_s",
                "max_event_duration_s",
            ]
        ]
        summary_output_df["session"] = pd.Categorical(summary_output_df["session"], categories=session_order_display, ordered=True)
        summary_output_df = summary_output_df.sort_values("session").reset_index(drop=True)

    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary_path = tables_dir / "dataset_summary_by_session.csv"
    summary_output_df.to_csv(summary_path, index=False)

    if not events_df.empty:
        events_output_df = events_df.copy()
        events_output_df["session"] = events_output_df["session"].apply(format_session_label)
        events_output_df["session"] = pd.Categorical(events_output_df["session"], categories=session_order_display, ordered=True)
        events_output_df.sort_values(["session", "location", "filename", "start_time"]).to_csv(
            tables_dir / "all_positive_events.csv", index=False
        )

    write_additional_tables(events_df=events_df, output_dir=output_dir)
    
    # Create and save event balance table
    balance_df = make_event_balance_table(session_data=session_data, events_df=events_df)
    balance_output_df = balance_df.copy()
    if not balance_output_df.empty:
        balance_output_df["session"] = balance_output_df["session"].apply(format_session_label)
        balance_output_df = (
            balance_output_df.groupby("session", as_index=False)
            .agg(
                positive_duration_s=("positive_duration_s", "sum"),
                negative_duration_s=("negative_duration_s", "sum"),
            )
        )
        total_duration = balance_output_df["positive_duration_s"] + balance_output_df["negative_duration_s"]
        balance_output_df["positive_percent"] = np.where(total_duration > 0, (balance_output_df["positive_duration_s"] / total_duration) * 100.0, 0.0)
        balance_output_df["negative_percent"] = np.where(total_duration > 0, (balance_output_df["negative_duration_s"] / total_duration) * 100.0, 0.0)
        balance_output_df["session"] = pd.Categorical(balance_output_df["session"], categories=session_order_display, ordered=True)
        balance_output_df = balance_output_df.sort_values("session").reset_index(drop=True)

    balance_table_path = tables_dir / "event_balance_by_session.csv"
    balance_output_df.to_csv(balance_table_path, index=False)
    
    print(f"\n{'='*80}")
    print("Event Balance by Session (Positive vs Negative Duration & Percentage):")
    print(f"{'='*80}")
    print(balance_output_df.to_string(index=False))
    print(f"{'='*80}\n")
    
    create_figures(summary_df=summary_df, events_df=events_df, output_dir=output_dir, event_label=args.event_label, radius_km=args.radius_km)
    write_overview_json(
        summary_df=summary_df,
        output_dir=output_dir,
        radius_km=args.radius_km,
        sample_hop_seconds=args.sample_hop_seconds,
    )

    print(f"Saved summary table: {summary_path}")
    print(f"Saved event balance table: {balance_table_path}")
    print(f"Saved figures in:    {output_dir / 'figures'}")


if __name__ == "__main__":
    main()
