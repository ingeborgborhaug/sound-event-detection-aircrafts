import os
import time
from datetime import datetime, timezone
import logging
from pathlib import Path
import requests
#import folium
from math import radians, sin, cos, sqrt, atan2
from typing import Optional, Dict, Any
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np
from pyproj import Transformer

from keras_yamnet.params import PATCH_WINDOW_SECONDS

# --- SDK import ---
from fr24sdk.client import Client
from fr24sdk.models.flight import FlightSummaryLight
import httpx
from fr24sdk.exceptions import ApiError

# Masteroppgave key: 
API_KEY = '019bc710-7eef-7304-a61d-4e32f6213fdc|KxVxZabieXcmsvRMnrSHz06hyZO7YNOpy6TAjz6q2b8c7e93'
# Sandbox key:
#API_KEY = '01994cdc-690a-71e4-8387-0cc69b23a4df|m9SbuJYMcyedgkuP8zbtRPIl34IxqkqWdRMyEO2y4a01f6bc'

# --- Config ---
MICROPHONE_LOCATIONS = ['loc_1', 'loc_2', 'loc_3']  # Locations to process
session = '230226' # '280126' or '230226' or '030326'
RADII_KM = sorted(set([k / 2 for k in range(2, 31)] + [1.3]))  # 1.0, 1.3, 1.5, ..., 15.0
print(f"Monitoring for radii (km): {RADII_KM}")

N = 40
tripod_height_m = 1.2


# Settings
base_folder = f"C:\\Users\\kampfly\\Documents\\Ingeborg\\Masteroppgave\\{session}"
os.makedirs(base_folder, exist_ok=True)  # Create folder if it doesn't exist

def get_location_config(microphone_loc: str) -> tuple[float, float, float]:
    """Return (CENTER_LAT, CENTER_LON, CENTER_ALT_HAE) for given location."""
    if microphone_loc == 'loc_1':
        return (63.472832, 10.814295, 2.6 + tripod_height_m + N)
    elif microphone_loc == 'loc_2':
        return (63.49094, 10.86972, 56.0 + tripod_height_m + N)
    elif microphone_loc == 'loc_3':
        return (63.51608, 10.84220, 126.6 + tripod_height_m + N)
    else:
        raise ValueError(f"Invalid microphone location: {microphone_loc}")

# Will be set per location in the main loop
CENTER_LAT = None
CENTER_LON = None
CENTER_ALT_HAE = None
AUDIO_NAME = None

# --- Time setup for detection period ---
def get_session_timestamps(session_id: str) -> tuple[int, int]:
    """Return (TIMESTAMP_START, TIMESTAMP_END) for given session."""
    if session_id == '280126':
        dt_local_start = datetime(2026, 1, 28, 12, 43, 13, tzinfo=ZoneInfo("Europe/Oslo"))
        dt_local_end = datetime(2026, 1, 28, 14, 55, 6, tzinfo=ZoneInfo("Europe/Oslo"))
    elif session_id == '230226':
        dt_local_start = datetime(2026, 2, 23, 13, 27, 23, tzinfo=ZoneInfo("Europe/Oslo"))
        dt_local_end = datetime(2026, 2, 23, 15, 28, 17, tzinfo=ZoneInfo("Europe/Oslo"))
    elif session_id == '030326' and 'loc_1' not in MICROPHONE_LOCATIONS:
        dt_local_start = datetime(2026, 3, 3, 18, 11, 6, tzinfo=ZoneInfo("Europe/Oslo"))
        dt_local_end = datetime(2026, 3, 3, 20, 25, 35, tzinfo=ZoneInfo("Europe/Oslo"))
    else:
        raise ValueError(f"Invalid session specified: {session_id}")
    
    dt_utc_start = dt_local_start.astimezone(ZoneInfo("UTC"))
    timestamp_start = int(dt_utc_start.timestamp())
    
    dt_utc_end = dt_local_end.astimezone(ZoneInfo("UTC"))
    timestamp_end = int(dt_utc_end.timestamp())
    
    return timestamp_start, timestamp_end


# Paramerters for geofence and polling
POLLING_SECONDS = PATCH_WINDOW_SECONDS # Frequency of polling the API
SLEEP_SECONDS = 2 # Sleep time between API calls to avoid rate limiting

start, end = get_session_timestamps(session)
print(f'Duration of session {session}: {(end - start)} seconds')

# --- Geometry ---
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def ft_to_m(feet: float) -> float:
    return feet * 0.3048

# WGS84 geodetic -> Earth-Centered Earth-Fixed (ECEF)
transformer = Transformer.from_crs(
    "EPSG:4979",   # lat, lon, hae (WGS84)
    "EPSG:4978",   # ECEF XYZ
    always_xy=True
)

def distance_3d_m(lat1: float, lon1: float, alt1_m: float, lat2: float, lon2: float, alt2_m: float) -> float:
    """Calculate 3D Euclidean distance between two points in ECEF coordinates (meters)."""
    x1, y1, z1 = transformer.transform(lon1, lat1, alt1_m)
    x2, y2, z2 = transformer.transform(lon2, lat2, alt2_m)
    return np.linalg.norm([x2 - x1, y2 - y1, z2 - z1])

def is_inside_sphere(flat: float, flon: float, falt_hae: float = 0, center_lat = CENTER_LAT, center_lon = CENTER_LON, center_alt_hae = CENTER_ALT_HAE, radius_km: float = None) -> bool:
    """Check if aircraft is within the specified radius using 3D Euclidean distance."""
    distance_m = distance_3d_m(flat, flon, falt_hae, center_lat, center_lon, center_alt_hae)
    return distance_m <= radius_km * 1000  # Convert radius to meters

def flights_in_circle(flights: Optional[list[FlightSummaryLight]], radius_km: float = None) -> FlightSummaryLight:
    for flight in flights:
        lat = flight.lat
        lon = flight.lon
        alt_moh = ft_to_m(flight.alt)
        alt_hae = alt_moh + N
        if is_inside_sphere(flat=lat, flon=lon, falt_hae=alt_hae, radius_km=radius_km):
            yield flight

# --- Bounds helper ---
def make_bounds(center_lat: float, center_lon: float, lat_delta: float = 0.2, lon_delta: float = 0.45) -> str:
    north = center_lat + lat_delta
    south = center_lat - lat_delta
    west  = center_lon - lon_delta
    east  = center_lon + lon_delta
    print(f"Bounds north: {haversine_km(north, CENTER_LON, CENTER_LAT, CENTER_LON)} km")
    print(f"Bounds south: {haversine_km(south, CENTER_LON, CENTER_LAT, CENTER_LON)} km")
    print(f"Bounds east: {haversine_km(CENTER_LAT, east, CENTER_LAT, CENTER_LON)} km")
    print(f"Bounds west: {haversine_km(CENTER_LAT, west, CENTER_LAT, CENTER_LON)} km")

    return f"{north:.6f},{south:.6f},{west:.6f},{east:.6f}"

# Will be set per location in the main loop
BOUNDS = None

""" flights = client.historic.flight_positions.get_light(
                    timestamp=local_ts,
                    bounds=bounds
                )

                attempt = 0 # reset retry count on success

                logging.info(f"Flights returned: {len(flights.data)}")

                inside_flights_15KM = list(flights_in_circle(flights.data, radius_km=RADIUS_KM))
                inside_flights_10KM = list(flights_in_circle(flights.data, radius_km=10))
                inside_flights_5KM = list(flights_in_circle(flights.data, radius_km=5))
                inside_flights_1_3KM = list(flights_in_circle(flights.data, radius_km=1.3))



                logging.info(f"Flights inside geofence of {RADIUS_KM} km: {len(inside_flights_15KM)}")
                logging.info(f"Flights inside geofence of 10 km: {len(inside_flights_10KM)}")
                logging.info(f"Flights inside geofence of 5 km: {len(inside_flights_5KM)}")
                logging.info(f"Flights inside geofence of 1.3 km: {len(inside_flights_1_3KM)}")
                for flight in inside_flights_15KM: """

# --- Main loop ---

def monitor_detections(microphone_loc: str, audio_name: str, bounds: str, timestamp_start: int, timestamp_end: int) -> pd.DataFrame:
    client = Client(api_token=API_KEY)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    audio_ts = 0
    local_ts = timestamp_start

    max_attempts = 3
    attempt = 0
    reliable = {500, 502, 503, 504}

    autosave_path = os.path.join(base_folder, f"{audio_name}_AUTOSAVE_sphere.csv")

    cols = ["filename", "start_time", "end_time", "class", "callsign", "event_id", "distance", "fr24_id"]
    dfs_by_radius = {r: pd.DataFrame(columns=cols) for r in RADII_KM}

    try:
        while local_ts < timestamp_end:
            try:
                dt_utc = datetime.fromtimestamp(local_ts, tz=timezone.utc)
                dt_local = dt_utc.astimezone(ZoneInfo("Europe/Oslo"))
                logging.info(f"Checking flights at {dt_local}")

                flights = client.historic.flight_positions.get_light(timestamp=local_ts, bounds=bounds)

                attempt = 0

                logging.info(f"Flights returned: {len(flights.data)}")


                inside_by_radius = {
                    r: list(flights_in_circle(flights.data, radius_km=float(r)))
                    for r in RADII_KM
                }

                for r in RADII_KM:
                    logging.info(f"Flights inside geofence of {r} km: {len(inside_by_radius[r])}")

                for r, inside_flights in inside_by_radius.items():
                    df = dfs_by_radius[r]

                    for flight in inside_flights:
                        mask = df["fr24_id"] == flight.fr24_id

                        if mask.any():
                            df.loc[mask, "end_time"] = audio_ts
                        else:
                            lat = flight.lat
                            lon = flight.lon
                            alt_moh = ft_to_m(flight.alt)
                            alt_hae = alt_moh + N
                            dist_m = distance_3d_m(lat, lon, alt_hae, CENTER_LAT, CENTER_LON, CENTER_ALT_HAE)

                            new_entry = {
                                "filename": f"{audio_name}.wav",
                                "start_time": audio_ts,
                                "end_time": audio_ts + POLLING_SECONDS,
                                "class": 1,
                                "callsign": flight.callsign,
                                "fr24_id": flight.fr24_id,
                                "event_id": f"{flight.callsign}_{local_ts}",
                                "distance": dist_m,
                            }
                            df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)

                    dfs_by_radius[r] = df

                local_ts += POLLING_SECONDS
                audio_ts += POLLING_SECONDS
                time.sleep(SLEEP_SECONDS)

            except (httpx.HTTPStatusError, ApiError) as e:
                status = None
                if isinstance(e, httpx.HTTPStatusError):
                    status = e.response.status_code
                else:
                    msg = str(e)
                    if "500" in msg:
                        status = 500
                    elif "502" in msg:
                        status = 502
                    elif "503" in msg:
                        status = 503
                    elif "504" in msg:
                        status = 504

                if status in reliable:
                    attempt += 1
                    logging.warning(f"FR24 server error {status} at timestamp {local_ts}. Retry {attempt}/{max_attempts} in 2 seconds.")
                    if attempt >= max_attempts:
                        raise RuntimeError(f"Too many {status} errors at timestamp {local_ts}") from e
                    time.sleep(SLEEP_SECONDS * attempt)
                    continue

                raise

    except Exception:
        error_path = os.path.join(base_folder, f"{audio_name}_ERROR_AT_{local_ts}.csv")
        for r, df in dfs_by_radius.items():
            df.to_csv(error_path.replace(".csv", f"_{r}KM.csv"), sep="\t", index=False)
        logging.exception(f"Error occurred during monitoring. Partial results saved to {error_path}")
        raise

    finally:
        for r, df in dfs_by_radius.items():
            if df.empty:
                logging.info(f"No detections for radius {r} km. Annotating all windows as negative.")
                # Create a DataFrame with all windows annotated as negative
                df = pd.DataFrame({
                    "filename": [f"{audio_name}.wav"],
                    "start_time": 0,
                    "end_time": timestamp_end - timestamp_start,
                    "class": [0] ,
                    "callsign": None,
                    "fr24_id": None,
                    "event_id": None,
                    "distance": None
                })
            df.to_csv(autosave_path.replace(".csv", f"_{r}KM.csv"), sep="\t", index=False)
        logging.info(f"Autosave written to disk: {autosave_path}")

    return dfs_by_radius[15]

# ...existing code...


if __name__ == "__main__":
    
    for microphone_loc in MICROPHONE_LOCATIONS:
        print(f"\n{'='*60}")
        print(f"Processing location: {microphone_loc}")
        print(f"{'='*60}\n")
        
        # Set location-specific config
        CENTER_LAT, CENTER_LON, CENTER_ALT_HAE = get_location_config(microphone_loc)
        AUDIO_NAME = f"{microphone_loc}_{session}"
        BOUNDS = make_bounds(CENTER_LAT, CENTER_LON)
        TIMESTAMP_START, TIMESTAMP_END = get_session_timestamps(session)
        
        # Run detection for this location
        aircraft_df = monitor_detections(microphone_loc, AUDIO_NAME, BOUNDS, TIMESTAMP_START, TIMESTAMP_END)
        print(f"\nCompleted {microphone_loc}. Results saved.\n")

