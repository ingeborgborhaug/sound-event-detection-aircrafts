import os
import time
from datetime import datetime, timezone
import logging
import requests
import folium
from math import radians, sin, cos, sqrt, atan2
from typing import Optional, Dict, Any
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np
from pyproj import Transformer

# --- SDK import ---
from fr24sdk.client import Client
from fr24sdk.models.flight import FlightSummaryLight

# --- Config ---
# masteroppgave key: 
API_KEY = '019bc710-7eef-7304-a61d-4e32f6213fdc|KxVxZabieXcmsvRMnrSHz06hyZO7YNOpy6TAjz6q2b8c7e93'
# sandbox key:
#API_KEY = '01994cdc-690a-71e4-8387-0cc69b23a4df|m9SbuJYMcyedgkuP8zbtRPIl34IxqkqWdRMyEO2y4a01f6bc'
CENTER_LAT, CENTER_LON = (63.472493, 10.813604) # (latitude, longitude) Microphone location
RADIUS_KM  = 10       # circle radius
POLLING_SECONDS = 5
SLEEP_SECONDS = 10 # Rate limit of 10 requests per minute
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json",
        "Accept-Version": "v1"
    }

# --- Ground truth file setup ---
GT_FILE = 'dataset/Gardemoen/gt_flightradar.csv'

# --- Time setup ---
dt_local_start = datetime(2026, 1, 27, 8, 10, 0, tzinfo=ZoneInfo("Europe/Oslo"))
dt_utc_start = dt_local_start.astimezone(ZoneInfo("UTC"))
timestamp_start = int(dt_utc_start.timestamp())

dt_local_end = datetime(2026, 1, 27, 8, 12, 0, tzinfo=ZoneInfo("Europe/Oslo"))
dt_utc_end = dt_local_end.astimezone(ZoneInfo("UTC"))
timestamp_end = int(dt_utc_end.timestamp())

# --- Geometry ---
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def is_inside_circle(flat: float, flon: float) -> bool:
    return haversine_km(flat, flon, CENTER_LAT, CENTER_LON) <= RADIUS_KM

def flights_in_circle(flights: Optional[list[FlightSummaryLight]]) -> FlightSummaryLight:
    for flight in flights:
        lat = flight.lat
        lon = flight.lon
        if lat is not None and lon is not None and is_inside_circle(lat, lon):
            yield flight

def ft_to_m(feet: float) -> float:
    return feet * 0.3048

# WGS84 geodetic -> Earth-Centered Earth-Fixed (ECEF)
transformer = Transformer.from_crs(
    "EPSG:4979",   # lat, lon, height (WGS84)
    "EPSG:4978",   # ECEF XYZ
    always_xy=True
)

def distance_3d_m(lat1=None, lon1=None, alt1_m=None, lat2=None, lon2=None, alt2_m=None):
    x1, y1, z1 = transformer.transform(lon1, lat1, alt1_m)
    x2, y2, z2 = transformer.transform(lon2, lat2, alt2_m)
    return np.linalg.norm([x2 - x1, y2 - y1, z2 - z1])

# --- Bounds helper ---
def make_bounds(center_lat: float, center_lon: float, lat_delta: float = 1.0, lon_delta: float = 1.0) -> str:
    north = center_lat + lat_delta
    south = center_lat - lat_delta
    west  = center_lon - lon_delta
    east  = center_lon + lon_delta
    return f"{north},{south},{west},{east}"

BOUNDS = make_bounds(CENTER_LAT, CENTER_LON)

# flights = client.live.flight_positions.get_light(bounds=BOUNDS)

# --- Main loop ---
def monitor_detections() -> None:
    with open(GT_FILE, "w") as f:
        f.write("filename\tstart_time\tend_time\tclass\tcallsign\tevent_id\n")  # headers
    df = pd.read_csv(GT_FILE, sep="\t")

    client = Client(api_token=API_KEY)
    for ts in range(timestamp_start, timestamp_end, POLLING_SECONDS):
        logging.info(f"Checking flights at timestamp {ts}")

        flights = client.historic.flight_positions.get_light(timestamp=ts, bounds=BOUNDS)
        logging.info(f"Flights returned: {len(flights.data)}")
        
        inside_flights = list(flights_in_circle(flights.data))
        logging.info(f"Flights inside geofence: {len(inside_flights)}")

        # Correct: for flight in inside_flights:
        for flight in inside_flights:
            logging.info(f" - Flight {flight.callsign} at ({flight.lat}, {flight.lon})")
            mask = df['callsign'] == flight.callsign
            dt_utc = datetime.fromtimestamp(ts, tz=timezone.utc)
            dt_local = dt_utc.astimezone(ZoneInfo("Europe/Oslo"))
            if mask.any():
                logging.info(f"Existing flight detected: {flight.callsign}, updating end_time. \n")
                df.loc[mask, 'end_time'] = ts # Update end_time if flight is still within reach
            else:
                logging.info(f"New flight detected: {flight.callsign}, logged to GT file.")
                new_entry = {
                    'filename': f"{flight.callsign}_{dt_local}.wav",
                    'start_time': ts,
                    'end_time': ts,
                    'class': 'aircraft',
                    'callsign': flight.callsign,
                    'event_id': f"{flight.callsign}_{dt_local}"
                    # 'first_detected_at_distance_m': distance_3d_m(lat1=flight.lat, lon1=flight.lon, alt1_m=ft_to_m(flight.alt),
                    #                                             lat2=CENTER_LAT, lon2=CENTER_LON, alt2_m=0)
                }
                df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)

        time.sleep(SLEEP_SECONDS)  # To avoid hitting rate limits

    return df

def derive_no_aircraft_intervals(
    df: pd.DataFrame,
    global_start,
    global_end,
) -> pd.DataFrame:
    df_sorted = df.sort_values(by='start_time')
    no_aircraft_intervals = []
    current_time = global_start

    for _, row in df_sorted.iterrows():
        start_time = row['start_time']
        end_time = row['end_time']
        print(f"Processing aircraft interval: {start_time} to {current_time}")
        if start_time > current_time:
            no_aircraft_intervals.append({
                'start_time': current_time,
                'end_time': start_time #, 
                #'class': 'no_aircraft'
            })
        current_time = max(current_time, end_time)

    if current_time < global_end:
        no_aircraft_intervals.append({
            'start_time': current_time,
            'end_time': global_end
        })
    
    return pd.DataFrame(no_aircraft_intervals)

if __name__ == "__main__":
    
    aircraft_df = monitor_detections()
    # no_aircraft_df = derive_no_aircraft_intervals(aircraft_df, timestamp_start, timestamp_end)

    # final_df = pd.concat([aircraft_df, no_aircraft_df]).sort_values("start_time")
    aircraft_df.to_csv(GT_FILE, sep="\t", index=False)

