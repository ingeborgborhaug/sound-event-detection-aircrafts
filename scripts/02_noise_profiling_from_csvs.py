import glob
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing.noise_profiling import NoiseProfiler


# Update these paths for your machine.
DATA_ROOT = r"C:\Users\kampfly\Documents\Ingeborg\Prosjektoppgave\sound-event-detection-aircrafts\dataset\Skatval"
SESSIONS = ["280126", "230226"]
AUDIO_FOLDERS_BY_SESSION = {
    "280126": [os.path.join(DATA_ROOT, "280126")],
    "230226": [os.path.join(DATA_ROOT, "230226")],
}
OUTPUT_DIR = os.path.join("history", "noise_profiles")


def build_csv_sources() -> list[tuple[str, list[str]]]:
    pairs: list[tuple[str, list[str]]] = []
    for session in SESSIONS:
        session_dir = os.path.join(DATA_ROOT, session)
        csv_paths = sorted(glob.glob(os.path.join(session_dir, "loc_*_*AUTOSAVE*.csv")))
        audio_folders = AUDIO_FOLDERS_BY_SESSION.get(session, [])

        for csv_path in csv_paths:
            if not audio_folders:
                raise ValueError(f"No audio folders configured for session {session}")
            pairs.append((csv_path, audio_folders))

    return pairs


def main() -> None:
    pairs = build_csv_sources()
    csv_paths = [p[0] for p in pairs]

    if not csv_paths:
        raise RuntimeError("No CSV files found. Check DATA_ROOT and SESSIONS.")

    # Assumes same audio folder list for all CSVs. If needed, build CsvSource manually.
    profiler, sources = NoiseProfiler.from_csv_paths(
        csv_paths=csv_paths,
        audio_folders=AUDIO_FOLDERS_BY_SESSION[SESSIONS[0]],
        no_aircraft_value=0,
    )

    # Replace per-source audio folders if sessions differ.
    for src in sources:
        src.audio_folders = AUDIO_FOLDERS_BY_SESSION[src.session_id]

    profiles = profiler.compute_all_profiles(sources, force_reload=False, apply_filter=None)
    profiler.save_profiles(profiles, OUTPUT_DIR)

    print(f"Saved noise profiles to: {OUTPUT_DIR}")
    for session_id, profile in sorted(profiles.items()):
        print(
            f"Session {session_id}: {profile['n_segments']} no-aircraft segments "
            f"from locations {profile['n_by_location']}"
        )


if __name__ == "__main__":
    main()
