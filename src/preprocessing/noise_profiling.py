import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

import functions


@dataclass
class CsvSource:
    """One location/date CSV source with audio folders."""

    gt_csv: str
    audio_folders: List[str]
    session_id: str
    location_id: str
    dataset: str = "norwegian"


class NoiseProfiler:
    """Compute and store average noise profiles per session."""

    def __init__(self, no_aircraft_value: int = 0):
        self.no_aircraft_value = int(no_aircraft_value)

    @staticmethod
    def infer_session_location(gt_csv: str) -> tuple[str, str]:
        """
        Infer session and location from names like:
        - loc_1_280126_AUTOSAVE_sphere_1.0KM.csv
        - loc_2_230226.csv
        """
        name = os.path.basename(gt_csv)
        m = re.search(r"loc_(\d+)_(\d{6})", name)
        if not m:
            raise ValueError(
                f"Could not infer location/session from filename: {name}. "
                "Expected pattern like 'loc_1_280126_...csv'."
            )
        location_id, session_id = m.group(1), m.group(2)
        return session_id, location_id

    @classmethod
    def from_csv_paths(
        cls,
        csv_paths: Iterable[str],
        audio_folders: List[str],
        dataset: str = "norwegian",
        no_aircraft_value: int = 0,
    ) -> tuple["NoiseProfiler", List[CsvSource]]:
        """Build sources directly from per-location/date csv files."""
        profiler = cls(no_aircraft_value=no_aircraft_value)
        sources: List[CsvSource] = []
        for csv_path in csv_paths:
            session_id, location_id = cls.infer_session_location(csv_path)
            sources.append(
                CsvSource(
                    gt_csv=csv_path,
                    audio_folders=audio_folders,
                    session_id=session_id,
                    location_id=location_id,
                    dataset=dataset,
                )
            )
        return profiler, sources

    @staticmethod
    def _segment_mean_spectra(X: np.ndarray) -> np.ndarray:
        """
        Convert segment tensors into per-segment spectra.

        Expected common shapes:
        - (N, T, M)
        - (N, C, T, M)

        Returns:
            shape (N, M) where M is mel bins.
        """
        if X.ndim < 3:
            raise ValueError(f"Expected X with >=3 dims, got shape {X.shape}")

        mel_axis = X.ndim - 1
        reduce_axes = tuple(ax for ax in range(1, X.ndim) if ax != mel_axis)
        return np.mean(X, axis=reduce_axes)

    def compute_session_profile(
        self,
        sources: List[CsvSource],
        force_reload: bool = False,
        apply_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        For one session:
        1. Load all location/date CSV datasets in `sources`.
        2. Keep no-aircraft segments (label == self.no_aircraft_value).
        3. Compute mean/std/median/p5/p95 spectra across segments.
        """
        if not sources:
            raise ValueError("compute_session_profile received an empty source list")

        session_id = sources[0].session_id
        per_segment_spectra = []
        n_by_location: Dict[str, int] = {}

        for src in sources:
            data_dict = {src.gt_csv: src.audio_folders}
            X, y, _ = functions.get_data_from_dict(
                data_dict,
                force_reload=force_reload,
                apply_filter=apply_filter,
            )

            if len(X) == 0:
                n_by_location[src.location_id] = 0
                continue

            y_flat = np.asarray(y).reshape(-1)
            mask_no_aircraft = y_flat == self.no_aircraft_value
            X_no_aircraft = X[mask_no_aircraft]

            n_by_location[src.location_id] = int(len(X_no_aircraft))

            if len(X_no_aircraft) == 0:
                continue

            per_segment_spectra.append(self._segment_mean_spectra(X_no_aircraft))

        if not per_segment_spectra:
            return {
                "session_id": session_id,
                "mean": None,
                "std": None,
                "median": None,
                "p5": None,
                "p95": None,
                "n_segments": 0,
                "n_by_location": n_by_location,
            }

        S = np.concatenate(per_segment_spectra, axis=0)

        return {
            "session_id": session_id,
            "mean": np.mean(S, axis=0),
            "std": np.std(S, axis=0),
            "median": np.median(S, axis=0),
            "p5": np.percentile(S, 5, axis=0),
            "p95": np.percentile(S, 95, axis=0),
            "n_segments": int(S.shape[0]),
            "n_by_location": n_by_location,
        }

    def compute_all_profiles(
        self,
        csv_sources: List[CsvSource],
        force_reload: bool = False,
        apply_filter: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Compute profiles for all sessions from per-location/date csv sources."""
        by_session: Dict[str, List[CsvSource]] = {}
        for src in csv_sources:
            by_session.setdefault(src.session_id, []).append(src)

        profiles: Dict[str, Dict[str, Any]] = {}
        for session_id, sources in sorted(by_session.items()):
            profiles[session_id] = self.compute_session_profile(
                sources,
                force_reload=force_reload,
                apply_filter=apply_filter,
            )
        return profiles

    def save_profiles(self, profiles: Dict[str, Dict[str, Any]], output_dir: str) -> None:
        """Save each session profile to NPZ and summary metadata to JSON."""
        os.makedirs(output_dir, exist_ok=True)

        summary = {
            "sessions": [],
            "n_segments_by_session": {},
            "n_segments_by_location": {},
        }

        for session_id, profile in sorted(profiles.items()):
            summary["sessions"].append(session_id)
            summary["n_segments_by_session"][session_id] = int(profile["n_segments"])
            summary["n_segments_by_location"][session_id] = profile.get(
                "n_by_location", {}
            )

            npz_path = os.path.join(output_dir, f"{session_id}_noise_profile.npz")
            if profile["n_segments"] == 0:
                np.savez(
                    npz_path,
                    session_id=session_id,
                    n_segments=0,
                )
            else:
                np.savez(
                    npz_path,
                    session_id=session_id,
                    mean=profile["mean"],
                    std=profile["std"],
                    median=profile["median"],
                    p5=profile["p5"],
                    p95=profile["p95"],
                    n_segments=profile["n_segments"],
                )

        summary_path = os.path.join(output_dir, "noise_profiles_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
