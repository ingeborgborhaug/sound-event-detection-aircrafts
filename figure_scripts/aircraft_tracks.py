import requests
import datetime
import folium
import math
import json
from datetime import timezone
from zoneinfo import ZoneInfo
from fr24sdk.client import Client
from datetime import datetime, timezone
from keras_yamnet.params import PATCH_WINDOW_SECONDS
from dataset.geo_utils import transformer, ft_to_m, get_location_config, get_session_timestamps, is_inside_sphere, N
import numpy as np
import h5py
import os
import pandas as pd
import sys
import pickle
sys.path.append('..')
import settings
from dataset import gt_conversion_functions as cf
import pandas as pd


""" Config """
MICROPHONE_LOCATION = 'loc_1'  # Locations to process
session = '230226' # '280126' or '230226' or '030326'
RADIUS_KM = 2          # Radius in kilometers
USE_GROUND_TRUTH = True

PREDICTIONS_H5_PATH = 'history/20260302-114508/predictions_skatval.h5'
GROUND_TRUTH_CSV_PATH = f'D:/dataset_master/230226/loc_1_230226_{RADIUS_KM}_0KM.csv'

""" Parameters and data loading (automatic generation using config) """
prediction = f'predictions_{RADIUS_KM}KM'  # Example: 'X_1KM', 'X_2KM', ..., 'X_15KM' depending on which radius data you want to load
TIMESTAMP_START, TIMESTAMP_END = get_session_timestamps(session)
CENTER_LAT, CENTER_LON, CENTER_ALT_HAE = get_location_config(MICROPHONE_LOCATION)
skatval_dataset_folder = 'D:/dataset_master'
fr24_ids_folder = skatval_dataset_folder + f"/{session}/{MICROPHONE_LOCATION}_{session}_{RADIUS_KM}_0KM.csv"
df = pd.read_csv(fr24_ids_folder, sep='\t')
flight_ids = df['fr24_id'].dropna().unique().tolist()
print(flight_ids)


def get_flight_tracks(fr24_ids, headers):
    """
    Fetch full flight trajectories using the FR24 Flight Tracks endpoint.
    """
    
    base_url = "https://fr24api.flightradar24.com/api/flight-tracks"
    
    all_tracks = []
    
    for flight_id in fr24_ids:
        params = {
            "flight_id": flight_id
        }
        response = requests.get(base_url, params=params, headers=headers)
        response.raise_for_status()
        track_data = response.json()
        # Store track info in a list (or write to DB, etc.)
        all_tracks.append(track_data)
    
    return all_tracks


def load_or_fetch_flight_tracks(fr24_ids, headers, cache_path: str):
    """Load cached flight tracks if available for the same flight IDs; otherwise fetch and save."""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_payload = json.load(f)

            cached_ids = cache_payload.get('flight_ids', [])
            cached_tracks = cache_payload.get('flight_tracks', [])

            if cached_ids == fr24_ids and isinstance(cached_tracks, list):
                print(f"Loaded {len(cached_tracks)} cached flight tracks from: {cache_path}")
                return cached_tracks

            print("Cache exists but flight IDs differ; fetching fresh tracks.")
        except Exception as e:
            print(f"Failed to read cache ({cache_path}): {type(e).__name__}: {e}. Fetching fresh tracks.")

    all_tracks = get_flight_tracks(fr24_ids, headers)

    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'flight_ids': fr24_ids,
                'flight_tracks': all_tracks,
            },
            f,
            ensure_ascii=False,
        )
    print(f"Saved {len(all_tracks)} flight tracks to cache: {cache_path}")

    return all_tracks

def plot_flight_tracks_on_map(tracks_data, output_html="all_flight_tracks.html", callsigns=None, gt_array=None, detection_array=None):
    """
    Plots flight tracks on a folium map and saves to an HTML file.
    
    Args:
        tracks_data: List of track data from API
        output_html: Output HTML filename
        callsigns: Optional list of callsigns for display purposes
        CENTER_LAT: Center latitude for area filtering (optional)
        CENTER_LON: Center longitude for area filtering (optional)
        RADIUS_KM
    : Radius in kilometers for area filtering (optional)
    """

    if detection_array is None:
        print("Warning: No detection array provided; all points will be plotted as unflagged.")

    # Center map somewhere over Europe by default (adjust as needed)
    fmap = folium.Map(location=[59.0, 18.0], zoom_start=5)
    
    # If area filtering is enabled, draw the area circle on the map
    if CENTER_LAT is not None and CENTER_LON is not None and RADIUS_KM is not None:
        folium.Circle(
            location=[CENTER_LAT, CENTER_LON],
            radius=RADIUS_KM
         * 1000,  # Convert km to meters for folium
            color='red',
            fill=True,
            fillColor='red',
            fillOpacity=0.1,
            popup=f"Area ({RADIUS_KM} km radius)",
            weight=2
        ).add_to(fmap)
        folium.CircleMarker(
            location=[CENTER_LAT, CENTER_LON],
            radius=6,
            color='red',
            fill=True,
            fillColor='red',
            fillOpacity=1.0,
            popup="Microphone (center)",
        ).add_to(fmap)

        # Radius measure line: center -> east edge
        import math as _math
        lon_edge = CENTER_LON + (RADIUS_KM / (111.32 * _math.cos(_math.radians(CENTER_LAT))))
        mid_lon = (CENTER_LON + lon_edge) / 2
        folium.PolyLine(
            [(CENTER_LAT, CENTER_LON), (CENTER_LAT, lon_edge)],
            color='black',
            weight=1.5,
            opacity=0.8,
            dash_array='6',
        ).add_to(fmap)
        folium.Marker(
            location=[CENTER_LAT + 0.002, mid_lon],  # Adjust the position slightly for better visibility
            icon=folium.DivIcon(
                html=(
                    f'<div style="font-size:9pt; color:black; white-space:nowrap; '
                    f'font-weight:bold; font-family: Times New Roman, Times, serif;">{RADIUS_KM} km</div>'
                ),
                icon_anchor=(0, 10),
            ),
        ).add_to(fmap)
    
    # Iterate through each flight's track data and plot on the map
    for idx, track in enumerate(tracks_data):
        if not track:
            continue
        
        positions = track[0].get("tracks", []) # One track per flight, get the list of positions (lat/lon/alt/timestamp)
        
        flight_id_label = track[0].get("fr24_id", "")


        # Extract (lat, lon, flagged) for each position to form a list of coordinates
        # flagged is True when gt_array indicates a detection at that timestamp
        coords = []
        for pos in positions:
            lat = pos.get("lat")
            lon = pos.get("lon")
            alt_ft = pos.get("alt")
            alt_hae = ft_to_m(alt_ft) + N
            timestamp = pos.get("timestamp")
            ts_int = int(datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp())
            audio_ts =  ts_int - TIMESTAMP_START

            # determine flag from detection array if available
            flagged = False
        
            # Only add valid points
            if lat is not None and lon is not None:
                # If area filtering is enabled, only add points within the radius
                if is_inside_sphere(lat, lon, alt_hae, CENTER_LAT, CENTER_LON, CENTER_ALT_HAE, RADIUS_KM) and is_in_detection_array(audio_ts, gt_array):
                    if is_in_detection_array(audio_ts, detection_array):
                        coords.append((lat, lon, "green"))
                    else:
                        coords.append((lat, lon, "red"))
                elif ts_int >= TIMESTAMP_START and ts_int < TIMESTAMP_END:
                    coords.append((lat, lon, "gray"))  # Optional: plot out-of-area points in gray for context

        # Add polylines for this flight, colouring each segment by flag
        if coords:
            # Get callsign from metadata if available
            callsign_label = ""
            if callsigns and idx < len(callsigns):
                callsign_label = f" ({callsigns[idx]})"
            elif "identification" in track[0]:
                callsign_label = f" ({track[0]['identification'].get('callsign', '')})"
            popup_text = f"Flight{callsign_label}"
            flight_id_label = track[0].get("fr24_id", "")

            segment_colors = [color for _, _, color in coords[1:]] if len(coords) > 1 else [coords[0][2]]
            has_green = any(c == "green" for c in segment_colors)
            has_red = any(c == "red" for c in segment_colors)
            if has_green and has_red:
                plotted_color = "mixed"
            elif has_green:
                plotted_color = "green"
            else:
                plotted_color = "red"

            if flight_id_label:
                start_lat, start_lon, _ = coords[0]
                folium.Marker(
                    location=[start_lat, start_lon],
                    icon=folium.DivIcon(
                        html=(
                            f'<div style="font-size: 10pt; color: black; '
                            f'font-weight: bold; white-space: nowrap; font-family: Times New Roman, Times, serif;">{flight_id_label}</div>'
                        )
                    ),
                    popup=f"Flight ID: {flight_id_label}{callsign_label}"
                ).add_to(fmap)

            # draw each consecutive segment so we can colour individually
            for i in range(1, len(coords)):
                lat1, lon1, _ = coords[i-1]
                lat2, lon2, color = coords[i]
                folium.PolyLine(
                    [(lat1, lon1), (lat2, lon2)],
                    color=color,
                    weight=2.5,
                    opacity=1,
                    popup=popup_text
                ).add_to(fmap)
    
    # Save the result to an HTML file
    fmap.save(output_html)
    print(f"Map with flight tracks saved to: {output_html}")

def is_in_detection_array(ts_int: int, detection_array) -> bool:
    flagged = False
    
    if detection_array is None:
        return False

    try:
        idx_win = cf.sec_to_start_index(ts_int)
        if 0 <= idx_win < len(detection_array):
            flagged = detection_array[idx_win] == 1
        #else:
            #print(f"Timestamp {ts_int} (index {idx_win}) is out of bounds for detection array of length {len(gt_array)}.")
    except Exception as e:
        print(f"Error determining flagged status from detection array: {type(e).__name__}: {e}")
        flagged = False

    return flagged

def timestamp_to_array_index(timestamp: int) -> int:
    """Convert a UNIX timestamp to the corresponding index in the detection array."""
    if timestamp < TIMESTAMP_START or timestamp >= TIMESTAMP_END:
        raise ValueError("Timestamp is out of bounds of the detection period.")
    return (timestamp - TIMESTAMP_START) // PATCH_WINDOW_SECONDS


def load_detection_array_from_ground_truth(gt_csv_path: str, num_windows: int) -> np.ndarray:
    """Build a 0/1 detection array from a ground-truth CSV with start_time/end_time/class columns."""
    gt_df = pd.read_csv(gt_csv_path, sep='\t')

    gt_array = np.zeros(num_windows, dtype=int)

    positive_events = gt_df[gt_df['class'] == 1]

    for _, row in positive_events.iterrows():
        start_time = float(row['start_time'])
        end_time = float(row['end_time'])

        start_idx = cf.sec_to_end_index(start_time)
        end_idx_exclusive = cf.sec_to_end_index(end_time)

        gt_array[start_idx:end_idx_exclusive] = 1

    return gt_array

def main():
    API_KEY = '019bc710-7eef-7304-a61d-4e32f6213fdc|KxVxZabieXcmsvRMnrSHz06hyZO7YNOpy6TAjz6q2b8c7e93' 

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json",
        "Accept-Version": "v1"
    }
    

    num_windows = cf.sec_to_end_index(TIMESTAMP_END - TIMESTAMP_START)
    FLIGHT_TRACKS_CACHE_PATH = 'history/flight_tracks_loc1_230226.json'

    gt_array = load_detection_array_from_ground_truth(GROUND_TRUTH_CSV_PATH, num_windows)

    """ with h5py.File(PREDICTIONS_H5_PATH, 'r') as f:
        predictions = f['predictions'][:]
    detection_array = (predictions > settings.PREDICTION_THRESHOLD).astype(int).flatten()
 """

    # TEMP!!:
    detection_array = gt_array.copy()  # For testing, use GT as detection to verify plotting

    if len(gt_array) != num_windows or len(detection_array) != num_windows:
        print(f"Warning: Length of gt_array ({len(gt_array)}) or detection_array ({len(detection_array)}) does not match expected number of windows ({num_windows}). Check the timestamp range and array construction.")

    print(f'\n Ground truth source: {GROUND_TRUTH_CSV_PATH}')
    print(f'\n Detection source: {PREDICTIONS_H5_PATH}')
        
    if flight_ids:
        print(f"\nFetching flight tracks for {len(flight_ids)} flights...")
        flight_tracks = load_or_fetch_flight_tracks(flight_ids, headers, FLIGHT_TRACKS_CACHE_PATH)

        if len(flight_tracks) != len(flight_ids):
            print(f"Warning: Number of flight tracks fetched ({len(flight_tracks)}) does not match number of flight IDs ({len(flight_ids)}).")
        
        plot_flight_tracks_on_map(flight_tracks, output_html="all_flight_tracks.html", callsigns=None, gt_array=gt_array, detection_array=gt_array)
    else:
        print("No flights found for the detected/specified callsigns and time period.")

if __name__ == "__main__":
    main()