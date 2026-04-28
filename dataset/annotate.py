import argparse
import os
import time
from datetime import datetime, timezone
import logging
from pathlib import Path
import requests
from math import radians, sin, cos, sqrt, atan2
from typing import Optional, Dict, Any
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np
from pyproj import Transformer

from keras_yamnet.params import PATCH_WINDOW_SECONDS

from fr24sdk.client import Client
from fr24sdk.models.flight import FlightSummaryLight
import httpx
from fr24sdk.exceptions import ApiError
import argparse


API_KEY = '019bc710-7eef-7304-a61d-4e32f6213fdc|KxVxZabieXcmsvRMnrSHz06hyZO7YNOpy6TAjz6q2b8c7e93'

""" MICROPHONE_LOCATIONS = ['loc_2', 'loc_3']  # 'loc_1' is the quietest, 'gardemoen' is the busiest
session = '260326_part1'  # '280126', '230226', '030326', '300925', '260326_part1', '260326_part2'
 """

parser = argparse.ArgumentParser()
parser.add_argument("--session", required=True)
parser.add_argument("--locations", nargs="+", required=True)
args = parser.parse_args()

session = args.session
MICROPHONE_LOCATIONS = args.locations

MAX_RADIUS_KM = 15.0  # Only used for bounding API calls

N = 40
tripod_height_m = 1.2

base_folder = f"C:\\Users\\kampfly\\Documents\\Ingeborg\\Prosjektoppgave\\sound-event-detection-aircrafts\\dataset\\Skatval\\{session}\\Newly_generated"
os.makedirs(base_folder, exist_ok=True)

def get_location_config(microphone_loc: str) -> tuple[float, float, float]:
    if microphone_loc == 'loc_1':
        return (63.472832, 10.814295, 2.6 + tripod_height_m + N)
    elif microphone_loc == 'loc_2':
        return (63.49094, 10.86972, 56.0 + tripod_height_m + N)
    elif microphone_loc == 'loc_3':
        return (63.51608, 10.84220, 126.6 + tripod_height_m + N)
    elif microphone_loc == 'gardemoen':
        gardemoen_transformer = Transformer.from_crs("EPSG:32633", "EPSG:4326", always_xy=True)
        easting, northing = 284511.186, 6684706.776  # UTM coordinates for Gardermoen
        long, lat = gardemoen_transformer.transform(easting, northing)
        return (lat, long, 199.6 + tripod_height_m + N)
    else:
        raise ValueError(f"Invalid microphone location: {microphone_loc}")

CENTER_LAT = None
CENTER_LON = None
CENTER_ALT_HAE = None
AUDIO_NAME = None

def get_session_timestamps(session_id: str) -> tuple[int, int]:
    if session_id == '280126':
        dt_local_start = datetime(2026, 1, 28, 12, 43, 13, tzinfo=ZoneInfo("Europe/Oslo"))
        dt_local_end   = datetime(2026, 1, 28, 14, 55,  6, tzinfo=ZoneInfo("Europe/Oslo"))
        timestamp_start = int(dt_local_start.astimezone(ZoneInfo("UTC")).timestamp())
        timestamp_end   = int(dt_local_end.astimezone(ZoneInfo("UTC")).timestamp())

    elif session_id == '230226':
        dt_local_start = datetime(2026, 2, 23, 13, 27, 23, tzinfo=ZoneInfo("Europe/Oslo"))
        dt_local_end   = datetime(2026, 2, 23, 15, 28, 17, tzinfo=ZoneInfo("Europe/Oslo"))
        timestamp_start = int(dt_local_start.astimezone(ZoneInfo("UTC")).timestamp())
        timestamp_end   = int(dt_local_end.astimezone(ZoneInfo("UTC")).timestamp())

    elif session_id == '030326' and 'loc_1' not in MICROPHONE_LOCATIONS:
        dt_local_start = datetime(2026, 3, 3, 18, 11,  6, tzinfo=ZoneInfo("Europe/Oslo"))
        dt_local_end   = datetime(2026, 3, 3, 20, 25, 35, tzinfo=ZoneInfo("Europe/Oslo"))
        timestamp_start = int(dt_local_start.astimezone(ZoneInfo("UTC")).timestamp())
        timestamp_end   = int(dt_local_end.astimezone(ZoneInfo("UTC")).timestamp())

    elif session_id == '030326' and MICROPHONE_LOCATIONS == ['loc_1']:
        dt_local_start = datetime(2026, 3, 3, 18, 11,  6, tzinfo=ZoneInfo("Europe/Oslo"))
        dt_local_end   = datetime(2026, 3, 3, 18, 19, 42, tzinfo=ZoneInfo("Europe/Oslo"))
        timestamp_start = int(dt_local_start.astimezone(ZoneInfo("UTC")).timestamp())
        timestamp_end   = int(dt_local_end.astimezone(ZoneInfo("UTC")).timestamp())

    elif session_id == '260326_part1':
        dt_local_start = datetime(2026, 3, 26, 13, 31, 47, tzinfo=ZoneInfo("Europe/Oslo")) # ORIGINAL
        dt_local_end   = datetime(2026, 3, 26, 16, 59, 59, tzinfo=ZoneInfo("Europe/Oslo"))
        timestamp_start = int(dt_local_start.astimezone(ZoneInfo("UTC")).timestamp())
        timestamp_end   = int(dt_local_end.astimezone(ZoneInfo("UTC")).timestamp())
        #timestamp_start = 1774533696.4402142
    elif session_id == '260326_part2':
        dt_local_start = datetime(2026, 3, 26, 16, 59, 59, tzinfo=ZoneInfo("Europe/Oslo"))
        dt_local_end   = datetime(2026, 3, 26, 20, 29, 22, tzinfo=ZoneInfo("Europe/Oslo"))
        timestamp_start = int(dt_local_start.astimezone(ZoneInfo("UTC")).timestamp())
        #timestamp_start = 1774544869.4001617
        timestamp_end   = int(dt_local_end.astimezone(ZoneInfo("UTC")).timestamp())

    elif session_id == '300925':
        dt_local_start = datetime(2025, 9, 30, 12, 26, 30, tzinfo=ZoneInfo("Europe/Oslo"))
        dt_local_end   = datetime(2025, 9, 30, 14, 23, 31, tzinfo=ZoneInfo("Europe/Oslo"))
        timestamp_start = int(dt_local_start.astimezone(ZoneInfo("UTC")).timestamp())
        timestamp_end   = int(dt_local_end.astimezone(ZoneInfo("UTC")).timestamp())

    else:
        raise ValueError(f"Invalid session specified: {session_id}")

    return timestamp_start, timestamp_end

POLLING_SECONDS = PATCH_WINDOW_SECONDS
SLEEP_SECONDS = 2

start, end = get_session_timestamps(session)
print(f'Duration of session {session}: {(end - start)} seconds')

# --- Geometry ---
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def ft_to_m(feet: float) -> float:
    return feet * 0.3048

transformer = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)

def distance_3d_m(lat1, lon1, alt1_m, lat2, lon2, alt2_m) -> float:
    x1, y1, z1 = transformer.transform(float(lon1), float(lat1), float(alt1_m))
    x2, y2, z2 = transformer.transform(float(lon2), float(lat2), float(alt2_m))
    return float(np.linalg.norm([x2 - x1, y2 - y1, z2 - z1]))

def make_bounds(center_lat, center_lon, lat_delta=0.2, lon_delta=0.45) -> str:
    north = center_lat + lat_delta
    south = center_lat - lat_delta
    west  = center_lon - lon_delta
    east  = center_lon + lon_delta
    return f"{north:.6f},{south:.6f},{west:.6f},{east:.6f}"

BOUNDS = None


# ---------------------------------------------------------------------------
# Step 1: collect raw flight positions — one row per flight per timestep
# ---------------------------------------------------------------------------

def collect_raw_positions(
    audio_name: str,
    bounds: str,
    timestamp_start: int,
    timestamp_end: int,
    center_lat: float,
    center_lon: float,
    center_alt_hae: float,
) -> pd.DataFrame:
    """
    Poll the FR24 API every POLLING_SECONDS and save every flight within
    MAX_RADIUS_KM as a raw row.  No radius decision is made here.

    Returns a DataFrame with one row per (timestep, flight).
    """
    client = Client(api_token=API_KEY)

    audio_ts = 0
    local_ts = timestamp_start
    max_attempts = 3
    attempt = 0
    reliable = {500, 502, 503, 504}

    raw_cols = [
        "timestamp", "audio_ts",
        "fr24_id", "callsign",
        "lat", "lon", "alt_ft_moh", "alt_m_hae",
        "distance_m",
    ]
    rows = []

    autosave_path = os.path.join(base_folder, f"{audio_name}_raw_positions.csv")

    try:
        while local_ts < timestamp_end:
            try:
                dt_local = datetime.fromtimestamp(local_ts, tz=timezone.utc).astimezone(
                    ZoneInfo("Europe/Oslo")
                )
                logging.info(f"[{audio_name}] Polling at {dt_local}  (audio_ts={audio_ts:.2f}s)")

                flights = client.historic.flight_positions.get_light(
                    timestamp=local_ts, bounds=bounds, altitude_ranges=["1-150000"]
                )
                attempt = 0
                logging.info(f"  Flights returned: {len(flights.data)}")

                for flight in flights.data:
                    try:
                        lat    = float(flight.lat)
                        lon    = float(flight.lon)
                        alt_ft_moh = float(flight.alt)
                        alt_m_hae  = ft_to_m(alt_ft_moh) + N   # HAE
                        dist_m = distance_3d_m(lat, lon, alt_m_hae,
                                               center_lat, center_lon, center_alt_hae)

                        # Pre-filter: only keep flights within MAX_RADIUS_KM
                        if dist_m > MAX_RADIUS_KM * 1000:
                            continue

                        rows.append({
                            "timestamp":  local_ts,
                            "audio_ts":   audio_ts,
                            "fr24_id":    flight.fr24_id,
                            "callsign":   flight.callsign,
                            "lat":        lat,
                            "lon":        lon,
                            "alt_ft_moh": alt_ft_moh,
                            "alt_m_hae":  alt_m_hae,
                            "distance_m": dist_m,
                        })
                    except (TypeError, ValueError) as ex:
                        logging.warning(f"  Skipping flight {getattr(flight, 'fr24_id', '?')}: {ex}")
                        continue

                local_ts += POLLING_SECONDS
                audio_ts += POLLING_SECONDS
                time.sleep(SLEEP_SECONDS)

            except (httpx.HTTPStatusError, ApiError) as e:
                status = None
                if isinstance(e, httpx.HTTPStatusError):
                    status = e.response.status_code
                else:
                    for code in reliable:
                        if str(code) in str(e):
                            status = code
                            break

                if status in reliable:
                    attempt += 1
                    logging.warning(
                        f"FR24 server error {status} at ts={local_ts}. "
                        f"Retry {attempt}/{max_attempts}."
                    )
                    if attempt >= max_attempts:
                        raise RuntimeError(
                            f"Too many {status} errors at timestamp {local_ts}"
                        ) from e
                    time.sleep(SLEEP_SECONDS * attempt)
                    continue
                raise

    except Exception:
        # Save whatever we have before re-raising
        pd.DataFrame(rows, columns=raw_cols).to_csv(
            autosave_path.replace(".csv", "_PARTIAL.csv"), sep="\t", index=False
        )
        logging.exception("Error during collection. Partial data saved.")
        raise

    finally:
        df = pd.DataFrame(rows, columns=raw_cols)
        df.to_csv(autosave_path, sep="\t", index=False)
        logging.info(
            f"Raw positions saved to {autosave_path}  ({len(df)} rows)"
        )

    return df


# ---------------------------------------------------------------------------
# Step 2: compute ground truth from the raw positions CSV
# ---------------------------------------------------------------------------

def compute_ground_truth(
    raw_positions_path: str,
    audio_name: str,
    radius_km: float,
    session_duration_s: float,
    output_dir: str = None,
) -> pd.DataFrame:
    """
    Read the raw positions CSV and produce a ground truth annotation file
    with start/end times for each aircraft event inside `radius_km`.

    Parameters
    ----------
    raw_positions_path : path to the CSV produced by collect_raw_positions()
    audio_name         : base name of the audio file (without .wav)
    radius_km          : detection radius in km (must be <= MAX_RADIUS_KM)
    session_duration_s : total duration of the recording in seconds
    output_dir         : where to save the GT csv (defaults to base_folder)

    Returns
    -------
    DataFrame with columns: filename, start_time, end_time, class, callsign,
                             event_id, distance, fr24_id
    """
    if output_dir is None:
        output_dir = base_folder

    os.makedirs(output_dir, exist_ok=True)

    df_raw = pd.read_csv(raw_positions_path, sep="\t")

    # Keep only flights within the chosen radius
    inside = df_raw[df_raw["distance_m"] <= radius_km * 1000].copy()

    cols = ["filename", "start_time", "end_time", "class",
            "callsign", "event_id", "distance", "fr24_id"]

    if inside.empty:
        logging.info(
            f"No flights within {radius_km} km — annotating all as negative."
        )
        gt = pd.DataFrame([{
            "filename":   f"{audio_name}.wav",
            "start_time": 0,
            "end_time":   session_duration_s,
            "class":      0,
            "callsign":   None,
            "event_id":   None,
            "distance":   None,
            "fr24_id":    None,
        }])
    else:
        # Group by fr24_id and merge consecutive timesteps into events
        records = []
        for fr24_id, grp in inside.groupby("fr24_id"):
            grp = grp.sort_values("audio_ts")
            callsign = grp["callsign"].iloc[0]

            # Split into separate events when gap > 2 × POLLING_SECONDS
            gap_mask = grp["audio_ts"].diff() > 2 * POLLING_SECONDS
            event_id_local = gap_mask.cumsum()

            for ev_idx, ev_grp in grp.groupby(event_id_local):
                records.append({
                    "filename":   f"{audio_name}.wav",
                    "start_time": float(ev_grp["audio_ts"].min()),
                    "end_time":   float(ev_grp["audio_ts"].max()),
                    "class":      1,
                    "callsign":   callsign,
                    "event_id":   f"{callsign}_{fr24_id}_{ev_idx}",
                    "distance":   float(ev_grp["distance_m"].min()),  # closest approach
                    "fr24_id":    fr24_id,
                })

        gt = pd.DataFrame(records, columns=cols)

    radius_str = str(radius_km)
    out_path = os.path.join(output_dir, f"{audio_name}_AUTOSAVE_sphere_{radius_str}KM.csv")
    gt.to_csv(out_path, sep="\t", index=False)
    logging.info(f"Ground truth saved to {out_path}")

    return gt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    for microphone_loc in MICROPHONE_LOCATIONS:
        print(f"\n{'='*60}")
        print(f"Processing location: {microphone_loc}")
        print(f"{'='*60}\n")

        center_lat, center_lon, center_alt_hae = get_location_config(microphone_loc)
        print(f"Center coordinates: lat={center_lat}, lon={center_lon}, alt_hae={center_alt_hae}m")
        audio_name = f"{microphone_loc}_{session}"
        bounds = make_bounds(center_lat, center_lon)
        ts_start, ts_end = get_session_timestamps(session)
        session_duration = ts_end - ts_start

        # --- Step 1: collect once ---
        """ raw_df = collect_raw_positions(
            audio_name=audio_name,
            bounds=bounds,
            timestamp_start=ts_start,
            timestamp_end=ts_end,
            center_lat=center_lat,
            center_lon=center_lon,
            center_alt_hae=center_alt_hae,
        ) """

        raw_path = os.path.join(f'C:\\Users\\kampfly\\Documents\\Ingeborg\\Prosjektoppgave\\sound-event-detection-aircrafts\\dataset\\Skatval\\{session}', f"{audio_name}_raw_positions.csv")

        # --- Step 2: compute GT for any radius you like (can rerun without API calls) ---
        for radius_km in np.arange(1.0, float(MAX_RADIUS_KM) + 1.0, 1.0):
            compute_ground_truth(
                raw_positions_path=raw_path,
                audio_name=audio_name,
                radius_km=radius_km,
                session_duration_s=session_duration,
                output_dir=base_folder,
            )

        print(f"\nCompleted {microphone_loc}.\n")