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
MICROPHONE_LOCATION = 'loc_2'  # Locations to process
session = '280126' # '280126' or '230226' or '030326'
RADIUS_KM = 15.0          # Radius in kilometers
SHOW_ALL_RADIUS_CIRCLES = True  # False: only RADIUS_KM, True: draw circles from 1..15 km
SHOW_FLIGHT_IDS = False  # True: show flight_id text labels on map, False: hide labels
USE_GROUND_TRUTH = False
FIT_BOUNDS_PADDING_PX = 120  # Pixel padding around bounds when fitting map
USE_HARDCODED_PDF_BOUNDS = True  # True: use HARDCODED_PDF_BOUNDS instead of automatic bounds
# Bounds format: (south, west, north, east)
HARDCODED_PDF_BOUNDS = (63.38, 10.48, 63.57, 11.10)
MAP_FONT_FAMILY = 'Times New Roman, Times, serif'
MAP_FONT_SIZE = '15pt'

PREDICTIONS_H5_PATH = 'history/20260302-114508/predictions_skatval.h5'
GROUND_TRUTH_CSV_PATH = f'D:\\Skatval\\{session}\\{MICROPHONE_LOCATION}_{session}_AUTOSAVE_sphere_{RADIUS_KM}KM.csv'

""" Parameters and data loading (automatic generation using config) """
prediction = f'predictions_{RADIUS_KM}KM'  # Example: 'X_1KM', 'X_2KM', ..., 'X_15KM' depending on which radius data you want to load
TIMESTAMP_START, TIMESTAMP_END = get_session_timestamps(session)
CENTER_LAT, CENTER_LON, CENTER_ALT_HAE = get_location_config(MICROPHONE_LOCATION)
gt_df_for_ids = pd.read_csv(GROUND_TRUTH_CSV_PATH, sep=None, engine='python')
if 'fr24_id' not in gt_df_for_ids.columns:
    raise ValueError(f"Column 'fr24_id' not found in {GROUND_TRUTH_CSV_PATH}")
flight_ids = (
    gt_df_for_ids['fr24_id']
    .dropna()
    .astype(str)
    .str.strip()
    .replace({'': np.nan, 'None': np.nan, 'nan': np.nan})
    .dropna()
    .unique()
    .tolist()
)
print(f"Loaded {len(flight_ids)} unique flight_ids from GROUND_TRUTH_CSV_PATH")


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


def _compute_map_bounds(center_lat: float, center_lon: float, radius_km: float) -> tuple[float, float, float, float]:
    margin_factor = 1.30
    extra_margin_km = 1.5
    effective_radius_km = radius_km * margin_factor + extra_margin_km
    lat_delta = effective_radius_km / 111.32
    lon_delta = effective_radius_km / (111.32 * math.cos(math.radians(center_lat)))
    south = center_lat - lat_delta
    north = center_lat + lat_delta
    west = center_lon - lon_delta
    east = center_lon + lon_delta
    return south, west, north, east


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

    max_display_radius_km = 15.0 if SHOW_ALL_RADIUS_CIRCLES else float(RADIUS_KM)
    auto_south, auto_west, auto_north, auto_east = _compute_map_bounds(CENTER_LAT, CENTER_LON, max_display_radius_km)
    if USE_HARDCODED_PDF_BOUNDS:
        south, west, north, east = HARDCODED_PDF_BOUNDS
    else:
        south, west, north, east = auto_south, auto_west, auto_north, auto_east

    # Start centered at microphone location and fit to radius bounds
    fmap = folium.Map(location=[CENTER_LAT, CENTER_LON], zoom_start=11, tiles="OpenStreetMap")
    global_font_css = f"""
    <style>
        .leaflet-container,
        .leaflet-control,
        .leaflet-popup-content,
        .leaflet-tooltip,
        .leaflet-marker-icon,
        .leaflet-marker-icon div {{
            font-family: {MAP_FONT_FAMILY} !important;
            font-size: {MAP_FONT_SIZE} !important;
        }}
    </style>
    """
    fmap.get_root().header.add_child(folium.Element(global_font_css))
    map_js_name = fmap.get_name()
    fmap.fit_bounds(
        [[south, west], [north, east]],
        padding=(FIT_BOUNDS_PADDING_PX, FIT_BOUNDS_PADDING_PX),
    )
    map_name = map_js_name
    if not USE_HARDCODED_PDF_BOUNDS:
        recenter_script = f"""
        <script>
        {map_name}.whenReady(function() {{
            {map_name}.setView([{CENTER_LAT}, {CENTER_LON}], {map_name}.getZoom(), {{animate: false}});
        }});
        </script>
        """
        fmap.get_root().html.add_child(folium.Element(recenter_script))
    
    # If area filtering is enabled, draw the selected radius or all radius circles on the map
    if CENTER_LAT is not None and CENTER_LON is not None and RADIUS_KM is not None:
        radius_values = list(range(1, 16)) if SHOW_ALL_RADIUS_CIRCLES else [int(round(RADIUS_KM))]
        for radius_km in radius_values:
            is_selected = abs(radius_km - RADIUS_KM) < 1e-9
            circle_color = 'red' if is_selected else '#4a90e2'
            folium.Circle(
                location=[CENTER_LAT, CENTER_LON],
                radius=radius_km * 1000,
                color=circle_color,
                fill=is_selected,
                fillColor=circle_color,
                fillOpacity=0.08 if is_selected else 0.0,
                popup=f"Area ({radius_km} km radius)",
                weight=2 if is_selected else 1,
                opacity=0.9 if is_selected else 0.55,
            ).add_to(fmap)

            if SHOW_ALL_RADIUS_CIRCLES and radius_km in {1, 5, 10, 15}:
                lon_edge_label = CENTER_LON + (radius_km / (111.32 * math.cos(math.radians(CENTER_LAT))))
                folium.Marker(
                    location=[CENTER_LAT + 0.0012, lon_edge_label],
                    icon=folium.DivIcon(
                        html=(
                            f'<div style="font-size:{MAP_FONT_SIZE}; color:#2b2b2b; white-space:nowrap; '
                            f'font-weight:bold; font-family: {MAP_FONT_FAMILY};">{radius_km} km</div>'
                        ),
                        icon_anchor=(0, 10),
                    ),
                ).add_to(fmap)

        for mic_name in ["loc_1", "loc_2", "loc_3"]:
            mic_lat, mic_lon, _ = get_location_config(mic_name)
            is_center_mic = mic_name == MICROPHONE_LOCATION
            dot_color = 'red' #if is_center_mic else '#8B0000'

            folium.CircleMarker(
                location=[mic_lat, mic_lon],
                radius=4,
                color=dot_color,
                fill=True,
                fillColor=dot_color,
                fillOpacity=1.0,
                popup="Microphone (center)" if is_center_mic else None,
            ).add_to(fmap)

            mic_label = mic_name.split('_')[-1]
            folium.Marker(
                location=[mic_lat + 0.0045, mic_lon + 0.0045],
                icon=folium.DivIcon(
                    html=(
                        f'<div style="font-size:{MAP_FONT_SIZE}; color:black; white-space:nowrap; '
                        f'font-weight:bold; font-family: {MAP_FONT_FAMILY};">{mic_label}</div>'
                    ),
                    icon_anchor=(0, 10),
                ),
            ).add_to(fmap)

        # Radius measure line: center -> east edge
        import math as _math
        lon_edge = CENTER_LON + (RADIUS_KM / (111.32 * _math.cos(_math.radians(CENTER_LAT))))
        if not SHOW_ALL_RADIUS_CIRCLES:
            mid_lon = (CENTER_LON + lon_edge) / 2
            folium.PolyLine(
                [(CENTER_LAT, CENTER_LON), (CENTER_LAT, lon_edge)],
                color='black',
                weight=1.5,
                opacity=0.8,
                dash_array='6',
            ).add_to(fmap)
            folium.Marker(
                location=[CENTER_LAT + 0.0012, mid_lon],  # Adjust the position slightly for better visibility
                icon=folium.DivIcon(
                    html=(
                        f'<div style="font-size:{MAP_FONT_SIZE}; color:black; white-space:nowrap; '
                        f'font-weight:bold; font-family: {MAP_FONT_FAMILY};">{RADIUS_KM} km</div>'
                    ),
                    icon_anchor=(0, 10),
                ),
            ).add_to(fmap)

    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 40px;
        left: 40px;
        z-index: 9999;
        background-color: white;
        border: 2px solid #444;
        border-radius: 6px;
        padding: 10px 12px;
        font-family: {MAP_FONT_FAMILY};
        font-size: {MAP_FONT_SIZE};
        line-height: 1.4;
        box-shadow: 0 1px 4px rgba(0,0,0,0.3);
    ">
        <div><span style="color: green; font-weight: bold;">■</span> Detected (model=1)</div>
        <div><span style="color: red; font-weight: bold;">■</span> Not detected (model=0)</div>
        <div><span style="color: gray; font-weight: bold;">■</span> Aircraft outside geofence</div>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))
    
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

            if SHOW_FLIGHT_IDS and flight_id_label:
                start_lat, start_lon, _ = coords[0]
                folium.Marker(
                    location=[start_lat, start_lon],
                    icon=folium.DivIcon(
                        html=(
                            f'<div style="font-size: {MAP_FONT_SIZE}; color: black; '
                            f'font-weight: bold; white-space: nowrap; font-family: {MAP_FONT_FAMILY};">{flight_id_label}</div>'
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
    FLIGHT_TRACKS_CACHE_PATH = f'history/flight_tracks_{MICROPHONE_LOCATION}_{session}.json'

    if USE_GROUND_TRUTH:
        gt_array = load_detection_array_from_ground_truth(GROUND_TRUTH_CSV_PATH, num_windows)
    else:
        gt_array = np.zeros(num_windows, dtype=int)

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