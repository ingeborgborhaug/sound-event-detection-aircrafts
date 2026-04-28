from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf


SESSION_DATE_PATTERN = re.compile(r"^(?P<date>\d{6})(?:_part(?P<part>\d+))?$")
GT_NAME_PATTERN = re.compile(
    r"^(?P<location>loc_[123])_(?P<session>.+?)_AUTOSAVE_sphere_(?P<radius>[0-9]+(?:\.[0-9]+)?)KM$"
)

SESSION_TITLES = {
    "280126": "Clear skies (Session 1)",
    "230226": "Snow-covered ground (Session 2)",
    "030326": "Windy (Session 3)",
    "260326_part1": "Windy (Session 4)",
    "260326_part2": "Rainy (Session 4)",
    "300925": "Clear skies (Session 5)",
}

EXCLUDED_WAVEFORM_LOCATIONS = {"gardemoen"}
EXCLUDED_WAVEFORM_SESSIONS = {"300925"}


@dataclass
class EventSample:
    session: str
    location: str
    wav_path: Path
    start_time: float
    end_time: float
    duration_s: float
    gt_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare one random aircraft event per session and location."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("D:/Skatval"),
        help="Dataset root containing one folder per session.",
    )
    parser.add_argument(
        "--radius-km",
        type=float,
        default=1.0,
        help="Radius suffix used in GT file names.",
    )
    parser.add_argument(
        "--locations",
        nargs="+",
        default=["loc_1", "loc_2", "loc_3"],
        help="Locations to include.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible selection.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/exploration/waveforms"),
        help="Directory for saved figures.",
    )
    return parser.parse_args()


def is_positive_label(value: object) -> bool:
    text = str(value).strip().lower()
    if text in {"", "none", "nan", "ignore", "0", "false", "negative", "no_aircraft"}:
        return False
    try:
        return float(text) > 0
    except ValueError:
        return True


def session_sort_key(name: str) -> tuple[pd.Timestamp, int, str]:
    match = SESSION_DATE_PATTERN.match(name)
    if not match:
        return (pd.Timestamp.max, 999, name)

    dt = pd.to_datetime(match.group("date"), format="%d%m%y", errors="coerce")
    if pd.isna(dt):
        return (pd.Timestamp.max, 999, name)

    part = int(match.group("part")) if match.group("part") else 0
    return (dt, part, name)


def sort_sessions(names: Iterable[str]) -> list[str]:
    return sorted(names, key=session_sort_key)


def read_gt(path: Path) -> pd.DataFrame:
    gt = pd.read_csv(path, sep=None, engine="python")
    gt.columns = [c.strip() for c in gt.columns]
    required = ["filename", "start_time", "end_time", "class"]
    missing = [col for col in required if col not in gt.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    gt = gt.copy()
    gt["start_time"] = pd.to_numeric(gt["start_time"], errors="coerce")
    gt["end_time"] = pd.to_numeric(gt["end_time"], errors="coerce")
    gt = gt.dropna(subset=["start_time", "end_time"])
    gt = gt[gt["end_time"] > gt["start_time"]].copy()
    gt = gt[gt["class"].apply(is_positive_label)].copy()
    return gt


def parse_gt_name(path: Path) -> tuple[str, str, float]:
    match = GT_NAME_PATTERN.match(path.stem)
    if not match:
        raise ValueError(f"GT filename does not match expected format: {path.name}")
    return match.group("location"), match.group("session"), float(match.group("radius"))


def find_gt_file(session_dir: Path, location: str, session_name: str, radius_km: float) -> Path | None:
    radius_text = f"{radius_km:.1f}"
    candidates = [
        session_dir / f"{location}_{session_name}_AUTOSAVE_sphere_{radius_text}KM.csv",
        session_dir / "Newly_generated" / f"{location}_{session_name}_AUTOSAVE_sphere_{radius_text}KM.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    for candidate in sorted(session_dir.glob(f"**/{location}_{session_name}_AUTOSAVE_sphere_*KM.csv")):
        try:
            _, gt_session, radius = parse_gt_name(candidate)
        except ValueError:
            continue
        if gt_session == session_name and np.isclose(radius, radius_km):
            return candidate
    return None


def resolve_wav_path(session_dir: Path, filename: str, location: str, session_name: str) -> Path | None:
    for candidate in [session_dir / filename, session_dir / "Newly_generated" / filename, session_dir / f"{location}_{session_name}.wav"]:
        if candidate.exists():
            return candidate

    wildcard = sorted(session_dir.glob(f"{location}_*.wav"))
    return wildcard[0] if wildcard else None


def collect_random_events(
    dataset_root: Path,
    locations: list[str],
    radius_km: float,
    seed: int,
) -> tuple[list[str], dict[tuple[str, str], EventSample | None]]:
    rng = np.random.default_rng(seed)
    session_dirs = sorted(
        [
            path
            for path in dataset_root.iterdir()
            if path.is_dir() and path.name not in EXCLUDED_WAVEFORM_SESSIONS
        ],
        key=lambda p: session_sort_key(p.name),
    )
    sessions = [path.name for path in session_dirs]
    samples: dict[tuple[str, str], EventSample | None] = {}

    for session_dir in session_dirs:
        session_name = session_dir.name
        for location in locations:
            gt_path = find_gt_file(session_dir, location, session_name, radius_km)
            if gt_path is None:
                print(f"[WARN] Missing GT for {session_name} ({location}, {radius_km:.1f}KM)")
                samples[(session_name, location)] = None
                continue

            gt = read_gt(gt_path)
            if gt.empty:
                print(f"[WARN] No positive events in {gt_path.name}")
                samples[(session_name, location)] = None
                continue

            row = gt.sample(n=1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
            wav_path = resolve_wav_path(session_dir, str(row["filename"]), location, session_name)
            if wav_path is None:
                print(f"[WARN] WAV file not found for session {session_name}, location {location}")
                samples[(session_name, location)] = None
                continue

            start_time = float(row["start_time"])
            end_time = float(row["end_time"])
            samples[(session_name, location)] = EventSample(
                session=session_name,
                location=location,
                wav_path=wav_path,
                start_time=start_time,
                end_time=end_time,
                duration_s=end_time - start_time,
                gt_path=gt_path,
            )

    return sessions, samples


def extract_full_event(event: EventSample) -> tuple[np.ndarray, int]:
    info = sf.info(str(event.wav_path))
    sr = int(info.samplerate)
    start_sample = max(int(np.floor(event.start_time * sr)), 0)
    end_sample = max(int(np.ceil(event.end_time * sr)), start_sample + 1)
    signal, _ = sf.read(str(event.wav_path), start=start_sample, stop=end_sample, dtype="float32")
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)
    return signal, sr


def get_common_duration(samples: Iterable[EventSample]) -> float:
    durations = [sample.duration_s for sample in samples if sample is not None]
    return max(durations) if durations else 0.0


def get_common_ylim(samples: Iterable[EventSample]) -> tuple[float, float]:
    """Compute y-axis limits based on all signals in samples."""
    y_min, y_max = 0.0, 0.0
    for sample in samples:
        if sample is not None:
            signal, _ = extract_full_event(sample)
            y_min = min(y_min, np.min(signal))
            y_max = max(y_max, np.max(signal))
    # Add padding
    padding = (y_max - y_min) * 0.1 if y_max > y_min else 0.1
    return y_min - padding, y_max + padding


def plot_event_grid(
    sessions: list[str],
    locations: list[str],
    samples: dict[tuple[str, str], EventSample | None],
    output_dir: Path,
    radius_km: float,
) -> None:
    if not sessions:
        print("No sessions found.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    selected_samples = [sample for sample in samples.values() if sample is not None]
    common_duration = get_common_duration(selected_samples)
    fig, axes = plt.subplots(
        nrows=len(sessions),
        ncols=len(locations),
        figsize=(5 * len(locations), max(2.6 * len(sessions), 6)),
        squeeze=False,
    )

    for row_idx, session in enumerate(sessions):
        for col_idx, location in enumerate(locations):
            ax = axes[row_idx][col_idx]
            event = samples.get((session, location))

            if event is None:
                ax.text(0.5, 0.5, "No event", ha="center", va="center", fontsize=11)
                ax.set_xlim(0, 1)
                ax.set_ylim(-1, 1)
            else:
                signal, sr = extract_full_event(event)
                time = np.arange(signal.shape[0]) / sr
                ax.plot(time, signal, linewidth=0.9)
                if common_duration > 0:
                    ax.set_xlim(0, common_duration)
                session_title = SESSION_TITLES.get(session, session)
                ax.set_title(
                    session_title,
                    fontsize=9,
                )
                ax.set_xlabel("Time from event start (s)")
                ax.set_ylabel("Amp")

            if row_idx == len(sessions) - 1 and event is None:
                ax.set_xlabel("Time from event start (s)")
            if col_idx == 0:
                ax.set_ylabel("Amp")

    fig.suptitle(f"Random full aircraft events by session and location ({radius_km:.1f}KM)", y=0.995)
    plt.tight_layout()
    out_path = output_dir / f"random_event_grid_{radius_km:.1f}KM.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")
    print("Selected events:")
    for session in sessions:
        for location in locations:
            event = samples.get((session, location))
            if event is None:
                print(f"- {session} | {location}: no event")
            else:
                print(
                    f"- {session} | {location}: {event.wav_path.name} [{event.start_time:.2f}, {event.end_time:.2f}] s"
                )


def plot_one_figure_per_location(
    sessions: list[str],
    locations: list[str],
    samples: dict[tuple[str, str], EventSample | None],
    output_dir: Path,
    radius_km: float,
) -> None:
    # Compute global y-limits across all locations and sessions
    all_samples = [samples.get((session, location)) 
                   for session in sessions 
                   for location in locations 
                   if samples.get((session, location)) is not None]
    global_ymin, global_ymax = get_common_ylim(all_samples)
    
    for location in locations:
        location_sessions = [
            session for session in sessions if samples.get((session, location)) is not None
        ]
        location_samples = [samples[(session, location)] for session in location_sessions]
        if not location_sessions:
            print(f"[WARN] No events found for {location}; skipping figure.")
            continue

        common_duration = get_common_duration(location_samples)
        fig, axes = plt.subplots(
            nrows=len(location_sessions),
            ncols=1,
            figsize=(14, max(2.6 * len(location_sessions), 6)),
            squeeze=False,
        )
        axes_array = axes[:, 0]

        for row_idx, session in enumerate(location_sessions):
            ax = axes_array[row_idx]
            event = samples.get((session, location))

            if event is None:
                ax.text(0.5, 0.5, "No event", ha="center", va="center", fontsize=11)
                ax.set_xlim(0, 1)
                ax.set_ylim(-1, 1)
            else:
                signal, sr = extract_full_event(event)
                time = np.arange(signal.shape[0]) / sr
                ax.plot(time, signal, linewidth=0.9)
                if common_duration > 0:
                    ax.set_xlim(0, common_duration)
                ax.set_ylim(global_ymin, global_ymax)
                session_title = SESSION_TITLES.get(session, session)
                ax.set_title(
                    session_title,
                    fontsize=9,
                )
                ax.set_xlabel("Time from event start (s)")
                ax.set_ylabel("Amp")

            if row_idx == len(location_sessions) - 1:
                ax.set_xlabel("Time from event start (s)")
            ax.set_ylabel("Amp")

        fig.suptitle(
            f"Random full aircraft events for {location} across sessions ({radius_km:.1f}KM)",
            y=0.995,
        )
        plt.tight_layout()
        out_path = output_dir / f"random_event_{location}_{radius_km:.1f}KM.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


def main() -> None:
    args = parse_args()
    excluded_requested = sorted({location for location in args.locations if location in EXCLUDED_WAVEFORM_LOCATIONS})
    if excluded_requested:
        print(f"Skipping excluded locations: {', '.join(excluded_requested)}")

    locations = [location for location in args.locations if location not in EXCLUDED_WAVEFORM_LOCATIONS]
    if not locations:
        raise ValueError("No valid locations left after excluding gardemoen.")

    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    excluded_sessions_present = sorted(
        [
            path.name
            for path in dataset_root.iterdir()
            if path.is_dir() and path.name in EXCLUDED_WAVEFORM_SESSIONS
        ]
    )
    if excluded_sessions_present:
        print(f"Skipping excluded sessions: {', '.join(excluded_sessions_present)}")

    sessions, samples = collect_random_events(
        dataset_root=dataset_root,
        locations=locations,
        radius_km=args.radius_km,
        seed=args.seed,
    )
    plot_event_grid(
        sessions=sessions,
        locations=locations,
        samples=samples,
        output_dir=args.output_dir,
        radius_km=args.radius_km,
    )
    plot_one_figure_per_location(
        sessions=sessions,
        locations=locations,
        samples=samples,
        output_dir=args.output_dir,
        radius_km=args.radius_km,
    )


if __name__ == "__main__":
    main()