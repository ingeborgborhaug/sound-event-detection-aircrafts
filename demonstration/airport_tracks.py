import requests
import datetime
import folium
import math
from datetime import timezone
from zoneinfo import ZoneInfo
from fr24sdk.client import Client
from datetime import datetime, timezone
from keras_yamnet.params import PATCH_WINDOW_SECONDS
from pyproj import Transformer
import numpy as np
import h5py
import sys
sys.path.append('..')
import settings


# Location 1
CENTER_LAT = 63.472832  # Center latitude
CENTER_LON = 10.814295  # Center longitude
CENTER_ALT_MOH = 2.6
RADIUS_KM = 15          # Radius in kilometers

dt_local_start = datetime(2026, 1, 28, 12, 43, 13, tzinfo=ZoneInfo("Europe/Oslo"))
dt_local_end = datetime(2026, 1, 28, 14, 55, 6, tzinfo=ZoneInfo("Europe/Oslo"))

dt_utc_start = dt_local_start.astimezone(ZoneInfo("UTC"))
TIMESTAMP_START = int(dt_utc_start.timestamp())

dt_utc_end = dt_local_end.astimezone(ZoneInfo("UTC"))
TIMESTAMP_END = int(dt_utc_end.timestamp())

# WGS84 geodetic -> Earth-Centered Earth-Fixed (ECEF)
transformer = Transformer.from_crs(
    "EPSG:4979",   # lat, lon, height (WGS84)
    "EPSG:4978",   # ECEF XYZ
    always_xy=True
)

def ft_to_m(feet: float) -> float:
    return feet * 0.3048

def distance_3d_m(lat1: float, lon1: float, alt1_m: float, lat2: float, lon2: float, alt2_m: float) -> float:
    """Calculate 3D Euclidean distance between two points in ECEF coordinates (meters)."""
    x1, y1, z1 = transformer.transform(lon1, lat1, alt1_m)
    x2, y2, z2 = transformer.transform(lon2, lat2, alt2_m)
    return np.linalg.norm([x2 - x1, y2 - y1, z2 - z1])

def is_inside_sphere(flat: float, flon: float, falt_moh: float) -> bool:
    """Check if aircraft is within a given radius (km) using 3D Euclidean distance."""
    distance_m = distance_3d_m(flat, flon, falt_moh, CENTER_LAT, CENTER_LON, CENTER_ALT_MOH)
    return distance_m <= RADIUS_KM* 1000  # Convert km to meters

def get_flights_for_airport_date(route: str, date_str: str, headers):
    """
    Fetch all flights on a specific route on the specified date,
    using Flight Summary Light endpoint.
    """
    base_url = "https://fr24api.flightradar24.com/api/flight-summary/light"
    
    # Build the start/end of day in UTC for the query (example uses the entire 24-hour period)
    flight_datetime_from = f"{date_str} 00:00:00"
    flight_datetime_to = f"{date_str} 23:59:59"

    params = {
        "flight_datetime_from": flight_datetime_from,
        "flight_datetime_to": flight_datetime_to,
        "routes": f"{route}",
        "limit": 50  # Adjust limit as needed.
    }
    
    response = requests.get(base_url, params=params, headers=headers)
    response.raise_for_status()
    data = response.json()
    
    # Extract flight IDs from the response to then fetch detailed track data.
    fr24_ids = []
    
    for flight in data.get("data", []):
        fr24_id = flight.get("fr24_id")
        if fr24_id:
            fr24_ids.append(fr24_id)
    
    return fr24_ids


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

def plot_flight_tracks_on_map(tracks_data, output_html="all_flight_tracks.html", callsigns=None, detection_array=None):
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
    
    for idx, track in enumerate(tracks_data):
        if not track:
            continue
        
        positions = track[0].get("tracks", [])
        
        # Extract (lat, lon, flagged) for each position to form a list of coordinates
        # flagged is True when detection_array indicates a detection at that timestamp
        coords = []
        for pos in positions:
            lat = pos.get("lat")
            lon = pos.get("lon")
            alt_ft = pos.get("alt")
            alt_moh = ft_to_m(alt_ft)
            timestamp = pos.get("timestamp")

            # convert timestamp string to unix seconds, fallback to None
            ts_int = None
            if timestamp is not None:
                try:
                    ts_int = int(timestamp)
                except (ValueError, TypeError):
                    try:
                        ts_int = int(datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp())
                    except Exception:
                        ts_int = None

            # determine flag from detection array if available
            flagged = False
        

            # Only add valid points
            if lat is not None and lon is not None:
                # If area filtering is enabled, only add points within the radius
                if CENTER_LAT is not None and CENTER_LON is not None and RADIUS_KM is not None:
                    if is_inside_sphere(lat, lon, alt_moh):
                        flagged = is_flagged(ts_int, detection_array)
                        coords.append((lat, lon, flagged))
                else:
                    coords.append((lat, lon, flagged))

        # Add polylines for this flight, colouring each segment by flag
        if coords:
            # Get callsign from metadata if available
            callsign_label = ""
            if callsigns and idx < len(callsigns):
                callsign_label = f" ({callsigns[idx]})"
            elif "identification" in track[0]:
                callsign_label = f" ({track[0]['identification'].get('callsign', '')})"
            popup_text = f"Flight{callsign_label}"

            # draw each consecutive segment so we can colour individually
            for i in range(1, len(coords)):
                lat1, lon1, _ = coords[i-1]
                lat2, lon2, flag = coords[i]
                color = "green" if flag else "red"
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

def is_flagged(ts_int: int, detection_array) -> bool:
    flagged = False
    
    if ts_int is not None and detection_array is not None:
        try:
            idx_win = int((ts_int - TIMESTAMP_START) // PATCH_WINDOW_SECONDS)
            if 0 <= idx_win < len(detection_array):
                flagged = detection_array[idx_win] == 1
        except Exception as e:
            print(f"Error determining flagged status from detection array: {type(e).__name__}: {e}")
            flagged = False

    return flagged

def timestamp_to_array_index(timestamp: int) -> int:
    """Convert a UNIX timestamp to the corresponding index in the detection array."""
    if timestamp < TIMESTAMP_START or timestamp >= TIMESTAMP_END:
        raise ValueError("Timestamp is out of bounds of the detection period.")
    return (timestamp - TIMESTAMP_START) // PATCH_WINDOW_SECONDS

def main():
    API_KEY = '019bc710-7eef-7304-a61d-4e32f6213fdc|KxVxZabieXcmsvRMnrSHz06hyZO7YNOpy6TAjz6q2b8c7e93' 

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json",
        "Accept-Version": "v1"
    }
    

    num_windows = math.ceil((TIMESTAMP_END - TIMESTAMP_START) / PATCH_WINDOW_SECONDS)
    # Load predictions from cache and threshold to create detection_array
    with h5py.File('history/20260302-114508/predictions_skatval.h5', 'r') as f:
        predictions = f['predictions'][:]
    detection_array = (predictions > settings.PREDICTION_THRESHOLD).astype(int).flatten()
    print(f'Presence of aircraft detected in {detection_array.sum()} out of {len(detection_array)} windows during the period.')
    # detection_array now contains 0 for the first half of the period and 1 for the second
        
    flight_ids = ['3e169478', '3e16b6f6', '3e16c5e8', '3e16b426', '3e16c27c', '3e16c17f', '3e16cf7f', '3e16aedc', '3e16f9c0', '3e16fa24', '3e16d478', '3e16e03e', '3e1701bc', '3e1703a9', '3e1701f2', '3e1561f7', '3e16bae6', '3e170560']
    #flight_ids = ['3e169478']

    callsigns = [None for _ in flight_ids]  # Placeholder for callsigns if not available

    if flight_ids:
        print(f"\nFetching flight tracks for {len(flight_ids)} flights...")
        flight_tracks = get_flight_tracks(flight_ids, headers)
        if len(flight_tracks) != len(flight_ids):
            print(f"Warning: Number of flight tracks fetched ({len(flight_tracks)}) does not match number of flight IDs ({len(flight_ids)}).")
        
        plot_flight_tracks_on_map(flight_tracks, output_html="all_flight_tracks.html", callsigns=None, detection_array=detection_array)
    else:
        print("No flights found for the detected/specified callsigns and time period.")

if __name__ == "__main__":
    main()