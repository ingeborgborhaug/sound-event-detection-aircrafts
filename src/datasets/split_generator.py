import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class FoldStats:
    n_train: int
    n_val: int
    n_test: int
    train_aircraft_pct: float
    val_aircraft_pct: float
    test_aircraft_pct: float


class LOSOSplitGenerator:
    """Generate leakage-safe baseline and leave-one-session-out splits."""

    DEFAULT_SESSION_WEATHER = {
        "session1": "clear",
        "session2": "wind",
        "session3": "rain",
        "session4": "snow",
        "session5": "wind_rain",
    }

    # Support either A/B/C naming or loc_1/loc_2/loc_3 naming.
    DEFAULT_TRAIN_LOCATIONS = {"a", "b", "loc_1", "loc_2", "1", "2"}
    DEFAULT_VAL_LOCATIONS = {"c", "loc_3", "3"}

    def __init__(
        self,
        seed: int = 42,
        aerosonic_val_size: float = 0.15,
        dataset_col: str = "dataset",
        session_col: str = "session",
        location_col: str = "location",
        label_col: str = "label",
        aerosonic_name: str = "aerosonic",
        norwegian_name: str = "norwegian",
        session_weather_map: Optional[Dict[str, str]] = None,
        train_locations: Optional[Iterable[str]] = None,
        val_locations: Optional[Iterable[str]] = None,
    ):
        self.seed = int(seed)
        self.aerosonic_val_size = float(aerosonic_val_size)
        self.dataset_col = dataset_col
        self.session_col = session_col
        self.location_col = location_col
        self.label_col = label_col
        self.aerosonic_name = aerosonic_name
        self.norwegian_name = norwegian_name
        self.session_weather_map = dict(session_weather_map or self.DEFAULT_SESSION_WEATHER)
        self.train_locations = {
            self._normalize_location(v)
            for v in (train_locations or self.DEFAULT_TRAIN_LOCATIONS)
        }
        self.val_locations = {
            self._normalize_location(v)
            for v in (val_locations or self.DEFAULT_VAL_LOCATIONS)
        }

    @staticmethod
    def _normalize_location(value: object) -> str:
        s = str(value).strip().lower()
        if s in {"loc_01", "location_a"}:
            return "loc_1"
        if s in {"loc_02", "location_b"}:
            return "loc_2"
        if s in {"loc_03", "location_c"}:
            return "loc_3"
        return s

    def _require_columns(self, manifest_df: pd.DataFrame) -> None:
        required = {
            self.dataset_col,
            self.session_col,
            self.location_col,
            self.label_col,
        }
        missing = [c for c in required if c not in manifest_df.columns]
        if missing:
            raise ValueError(f"Manifest is missing required columns: {missing}")

    def _aircraft_pct(self, indices: List[int], labels: np.ndarray) -> float:
        if len(indices) == 0:
            return 0.0
        y = labels[np.asarray(indices, dtype=np.int64)]
        return float(np.mean(y == 1) * 100.0)

    def _stats(self, fold: Dict[str, object], labels: np.ndarray) -> FoldStats:
        train_idx = fold["train_indices"]
        val_idx = fold["val_indices"]
        test_idx = fold["test_indices"]
        return FoldStats(
            n_train=len(train_idx),
            n_val=len(val_idx),
            n_test=len(test_idx),
            train_aircraft_pct=self._aircraft_pct(train_idx, labels),
            val_aircraft_pct=self._aircraft_pct(val_idx, labels),
            test_aircraft_pct=self._aircraft_pct(test_idx, labels),
        )

    def _split_aerosonic(self, manifest_df: pd.DataFrame) -> tuple[List[int], List[int]]:
        aero_mask = manifest_df[self.dataset_col].astype(str).str.lower() == self.aerosonic_name
        aero_indices = manifest_df.index[aero_mask].to_numpy(dtype=np.int64)

        if len(aero_indices) == 0:
            return [], []

        aero_labels = manifest_df.loc[aero_indices, self.label_col].to_numpy()
        unique_labels = np.unique(aero_labels)
        stratify = aero_labels if len(unique_labels) > 1 else None

        train_idx, val_idx = train_test_split(
            aero_indices,
            test_size=self.aerosonic_val_size,
            random_state=self.seed,
            stratify=stratify,
        )
        return train_idx.astype(np.int64).tolist(), val_idx.astype(np.int64).tolist()

    def generate_all_folds(self, manifest_df: pd.DataFrame) -> List[Dict[str, object]]:
        """Generate baseline and LOSO folds from a manifest DataFrame."""
        self._require_columns(manifest_df)

        df = manifest_df.reset_index(drop=True).copy()
        labels = df[self.label_col].to_numpy()

        aero_train_idx, aero_val_idx = self._split_aerosonic(df)

        nor_mask = df[self.dataset_col].astype(str).str.lower() == self.norwegian_name
        nor_indices = df.index[nor_mask].to_numpy(dtype=np.int64)

        session_values = (
            df.loc[nor_indices, self.session_col].astype(str).sort_values().unique().tolist()
        )

        folds: List[Dict[str, object]] = []

        baseline = {
            "fold_name": "baseline_aerosonic_only",
            "test_session": "all_norwegian",
            "test_weather": "all",
            "train_indices": sorted(map(int, aero_train_idx)),
            "val_indices": sorted(map(int, aero_val_idx)),
            "test_indices": sorted(map(int, nor_indices.tolist())),
            "train_sessions": [],
            "val_strategy": "AeroSonicDB holdout only",
        }
        baseline_stats = self._stats(baseline, labels)
        baseline.update(
            {
                "n_train": baseline_stats.n_train,
                "n_val": baseline_stats.n_val,
                "n_test": baseline_stats.n_test,
                "train_aircraft_pct": baseline_stats.train_aircraft_pct,
                "val_aircraft_pct": baseline_stats.val_aircraft_pct,
                "test_aircraft_pct": baseline_stats.test_aircraft_pct,
            }
        )
        folds.append(baseline)

        # Build LOSO folds only from Norwegian sessions present in the manifest.
        for heldout_session in session_values:
            heldout_mask = nor_mask & (df[self.session_col].astype(str) == heldout_session)
            other_nor_mask = nor_mask & (~heldout_mask)

            other_df = df.loc[other_nor_mask, [self.location_col]].copy()
            other_loc_norm = other_df[self.location_col].map(self._normalize_location)

            train_nor_idx = other_df.index[other_loc_norm.isin(self.train_locations)].tolist()
            val_nor_idx = other_df.index[other_loc_norm.isin(self.val_locations)].tolist()
            test_idx = df.index[heldout_mask].tolist()

            # Any other location naming that is not recognized goes to val by default.
            unknown_loc_idx = other_df.index[
                ~(other_loc_norm.isin(self.train_locations) | other_loc_norm.isin(self.val_locations))
            ].tolist()
            if unknown_loc_idx:
                val_nor_idx.extend(unknown_loc_idx)

            train_idx = sorted(set(map(int, aero_train_idx)).union(map(int, train_nor_idx)))
            val_idx = sorted(set(map(int, aero_val_idx)).union(map(int, val_nor_idx)))
            test_idx = sorted(map(int, test_idx))

            weather = self.session_weather_map.get(str(heldout_session), "unknown")
            fold_name = f"loso_{heldout_session}_{weather}".replace(" ", "_")

            train_sessions = sorted(
                set(df.loc[train_nor_idx, self.session_col].astype(str).tolist())
            )

            fold = {
                "fold_name": fold_name,
                "test_session": str(heldout_session),
                "test_weather": weather,
                "train_indices": train_idx,
                "val_indices": val_idx,
                "test_indices": test_idx,
                "train_sessions": train_sessions,
                "val_strategy": "Other sessions location-C + fixed AeroSonic holdout",
            }
            stats = self._stats(fold, labels)
            fold.update(
                {
                    "n_train": stats.n_train,
                    "n_val": stats.n_val,
                    "n_test": stats.n_test,
                    "train_aircraft_pct": stats.train_aircraft_pct,
                    "val_aircraft_pct": stats.val_aircraft_pct,
                    "test_aircraft_pct": stats.test_aircraft_pct,
                }
            )
            folds.append(fold)

        return folds

    def save_folds(self, folds: List[Dict[str, object]], output_dir: str) -> None:
        """Save each fold as JSON."""
        os.makedirs(output_dir, exist_ok=True)

        for fold in folds:
            serializable = dict(fold)
            for key in ("train_indices", "val_indices", "test_indices"):
                serializable[key] = [int(v) for v in serializable[key]]

            out_path = os.path.join(output_dir, f"{fold['fold_name']}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2)

    def print_fold_summary(self, folds: List[Dict[str, object]], manifest_df: pd.DataFrame) -> pd.DataFrame:
        """Return and print a compact summary table."""
        _ = manifest_df  # kept for interface compatibility with the phase plan
        summary = pd.DataFrame(
            [
                {
                    "fold_name": f["fold_name"],
                    "n_train": f["n_train"],
                    "n_val": f["n_val"],
                    "n_test": f["n_test"],
                    "train_aircraft_pct": round(float(f["train_aircraft_pct"]), 3),
                    "val_aircraft_pct": round(float(f["val_aircraft_pct"]), 3),
                    "test_aircraft_pct": round(float(f["test_aircraft_pct"]), 3),
                }
                for f in folds
            ]
        )
        print(summary.to_string(index=False))
        return summary
