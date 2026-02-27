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
MICROPHONE_LOC = 'loc_1'  # 'loc_1', 'loc_2', or 'loc_3'
session = '230226' # '280126' or '230226'


# Settings
base_folder = f"C:\\Users\\kampfly\\Documents\\Ingeborg\\Masteroppgave\\{session}\\Clipped"
os.makedirs(base_folder, exist_ok=True)  # Create folder if it doesn't exist
AUDIO_NAME = f"{MICROPHONE_LOC}_{session}" 

if MICROPHONE_LOC == 'loc_1':
    CENTER_LAT, CENTER_LON = (63.472832, 10.814295) # (latitude, longitude) 
    CENTER_ALT_MOH = 2.6
elif MICROPHONE_LOC == 'loc_2':
    CENTER_LAT, CENTER_LON = (63.49094, 10.86972) # (latitude, longitude) 
    CENTER_ALT_MOH = 56.0
elif MICROPHONE_LOC == 'loc_3':
    CENTER_LAT, CENTER_LON = (63.51608, 10.84220) # (latitude, longitude) 
    CENTER_ALT_MOH = 126.6
else:
    print("Invalid microphone location specified.")

# --- Time setup for detection period ---
if session == '280126':
    dt_local_start = datetime(2026, 1, 28, 12, 43, 13, tzinfo=ZoneInfo("Europe/Oslo"))
    dt_local_end = datetime(2026, 1, 28, 14, 55, 6, tzinfo=ZoneInfo("Europe/Oslo"))
elif session == '230226':
    dt_local_start = datetime(2026, 2, 23, 13, 27, 23, tzinfo=ZoneInfo("Europe/Oslo"))
    dt_local_end = datetime(2026, 2, 23, 15, 28, 17, tzinfo=ZoneInfo("Europe/Oslo"))
else:
    NameError("Invalid session specified.")

dt_utc_start = dt_local_start.astimezone(ZoneInfo("UTC"))
TIMESTAMP_START = int(dt_utc_start.timestamp())

dt_utc_end = dt_local_end.astimezone(ZoneInfo("UTC"))
TIMESTAMP_END = int(dt_utc_end.timestamp())

# Paramerters for geofence and polling
RADIUS_KM  = 15 # circle radius
POLLING_SECONDS = PATCH_WINDOW_SECONDS # Frequency of polling the API
SLEEP_SECONDS = 2 # Sleep time between API calls to avoid rate limiting

# headers = {
#         "Authorization": f"Bearer {API_KEY}",
#         "Accept": "application/json",
#         "Accept-Version": "v1"
#     }

# --- Ground truth file setup ---
GT_FILE = f"{base_folder}\\{AUDIO_NAME}.csv"


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
    "EPSG:4979",   # lat, lon, height (WGS84)
    "EPSG:4978",   # ECEF XYZ
    always_xy=True
)

def distance_3d_m(lat1: float, lon1: float, alt1_m: float, lat2: float, lon2: float, alt2_m: float) -> float:
    """Calculate 3D Euclidean distance between two points in ECEF coordinates (meters)."""
    x1, y1, z1 = transformer.transform(lon1, lat1, alt1_m)
    x2, y2, z2 = transformer.transform(lon2, lat2, alt2_m)
    return np.linalg.norm([x2 - x1, y2 - y1, z2 - z1])

def is_inside_circle(flat: float, flon: float, falt_moh: float = 0) -> bool:
    """Check if aircraft is within 15 km using 3D Euclidean distance."""
    distance_m = distance_3d_m(flat, flon, falt_moh, CENTER_LAT, CENTER_LON, CENTER_ALT_MOH)
    return distance_m <= RADIUS_KM * 1000  # Convert 15 km to meters

def flights_in_circle(flights: Optional[list[FlightSummaryLight]]) -> FlightSummaryLight:
    for flight in flights:
        lat = flight.lat
        lon = flight.lon
        alt_moh = ft_to_m(flight.alt)
        if is_inside_circle(lat, lon, alt_moh):
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

BOUNDS = make_bounds(CENTER_LAT, CENTER_LON)

# flights = client.live.flight_positions.get_light(bounds=BOUNDS)

# --- Main loop ---
def monitor_detections() -> pd.DataFrame:

    with open(GT_FILE, "w") as f:
        f.write("filename\tstart_time\tend_time\tclass\tcallsign\tevent_id\tdistance\n")

    df = pd.read_csv(GT_FILE, sep="\t")

    client = Client(api_token=API_KEY)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    audio_ts = 0
    local_ts = TIMESTAMP_START

    # Error handling parameters
    max_attempts = 3
    attempt = 0
    reliable = {500, 502, 503, 504}

    autosave_path = os.path.join(base_folder, f"{AUDIO_NAME}_AUTOSAVE_sphere.csv")

    try:
        while local_ts < TIMESTAMP_END:
            try:
                dt_utc = datetime.fromtimestamp(local_ts, tz=timezone.utc)
                dt_local = dt_utc.astimezone(ZoneInfo("Europe/Oslo"))
                logging.info(f"Checking flights at {dt_local}")

                flights = client.historic.flight_positions.get_light(
                    timestamp=local_ts,
                    bounds=BOUNDS
                )

                attempt = 0 # reset retry count on success

                logging.info(f"Flights returned: {len(flights.data)}")

                inside_flights = list(flights_in_circle(flights.data))
                logging.info(f"Flights inside geofence: {len(inside_flights)}")

                for flight in inside_flights:
                    mask = df["callsign"] == flight.callsign

                    if mask.any():
                        df.loc[mask, "end_time"] = audio_ts
                    else:
                        new_entry = {
                            "filename": f"{AUDIO_NAME}.wav",
                            "start_time": audio_ts,
                            "end_time": audio_ts + POLLING_SECONDS,
                            "class": 1,
                            "callsign": flight.callsign,
                            "fr24_id": flight.fr24_id,
                            "event_id": f"{flight.callsign}_{local_ts}",
                            "distance": None
                        }
                        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)

                local_ts += POLLING_SECONDS
                audio_ts += POLLING_SECONDS
                time.sleep(SLEEP_SECONDS)

            except (httpx.HTTPStatusError, ApiError) as e:
                status = None

                if isinstance(e, httpx.HTTPStatusError):
                    status = e.response.status_code
                else:
                    # ApiError doesn't always expose status cleanly -> parse from string
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
                    time.sleep(SLEEP_SECONDS*attempt)
                    continue

                raise

    except Exception:
        # Save partial results on fatal error
        error_path = os.path.join(base_folder, f"{AUDIO_NAME}_ERROR_AT_{local_ts}.csv")
        df.to_csv(error_path, sep="\t", index=False)
        logging.exception(f"Error occurred during monitoring. Partial results saved to {error_path}")
        raise

    finally:
        df.to_csv(autosave_path, sep="\t", index=False)
        logging.info(f"Autosave written to disk: {autosave_path}")

    return df


if __name__ == "__main__":
    
    aircraft_df = monitor_detections()

