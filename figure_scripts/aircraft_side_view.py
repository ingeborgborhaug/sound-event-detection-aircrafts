import datetime
import json
import math
import os
import sys
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

sys.path.append('..')

import settings
from dataset import gt_conversion_functions as cf
from dataset.geo_utils import N, ft_to_m, get_location_config, get_session_timestamps, is_inside_sphere
from keras_yamnet.params import PATCH_WINDOW_SECONDS


""" Config """
MICROPHONE_LOCATION = 'loc_2'  # Locations to process
session = '280126'  # '280126' or '230226' or '030326'
RADIUS_KM = 15.0  # Radius in kilometers
SHOW_ALL_RADIUS_CIRCLES = True  # Kept for logic parity with the map script
SHOW_FLIGHT_IDS = False  # True: show flight_id text labels on plot, False: hide labels
USE_GROUND_TRUTH = False

PREDICTIONS_H5_PATH = 'history/20260302-114508/predictions_skatval.h5'
GROUND_TRUTH_CSV_PATH = f'D:\\Skatval\\{session}\\{MICROPHONE_LOCATION}_{session}_AUTOSAVE_sphere_{RADIUS_KM}KM.csv'

# Output files
OUTPUT_DIR = os.path.join('outputs', 'exploration', 'side_views')
OUTPUT_BASE_NAME = f'{MICROPHONE_LOCATION}_{session}_side_view'
OUTPUT_PDF_PATH = os.path.join(OUTPUT_DIR, f'{OUTPUT_BASE_NAME}.pdf')
OUTPUT_PNG_PATH = os.path.join(OUTPUT_DIR, f'{OUTPUT_BASE_NAME}.png')

# Shared typography
FIG_FONT_FAMILY = 'Times New Roman'
FIG_FONT_SIZE = 10
plt.rcParams.update(
    {
        'font.family': FIG_FONT_FAMILY,
        'font.size': FIG_FONT_SIZE,
        'axes.titlesize': FIG_FONT_SIZE,
        'axes.labelsize': FIG_FONT_SIZE,
        'legend.fontsize': FIG_FONT_SIZE,
        'xtick.labelsize': FIG_FONT_SIZE,
        'ytick.labelsize': FIG_FONT_SIZE,
    }
)

""" Parameters and data loading (automatic generation using config) """
prediction = f'predictions_{RADIUS_KM}KM'
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
    """Fetch full flight trajectories using the FR24 Flight Tracks endpoint."""
    base_url = 'https://fr24api.flightradar24.com/api/flight-tracks'

    all_tracks = []

    for flight_id in fr24_ids:
        params = {
            'flight_id': flight_id,
        }
        response = requests.get(base_url, params=params, headers=headers)
        response.raise_for_status()
        track_data = response.json()
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

            print('Cache exists but flight IDs differ; fetching fresh tracks.')
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


def is_in_detection_array(ts_int: int, detection_array) -> bool:
    if detection_array is None:
        return False

    try:
        idx_win = cf.sec_to_start_index(ts_int)
        return 0 <= idx_win < len(detection_array) and detection_array[idx_win] == 1
    except Exception as e:
        print(f"Error determining flagged status from detection array: {type(e).__name__}: {e}")
        return False


def timestamp_to_array_index(timestamp: int) -> int:
    """Convert a UNIX timestamp to the corresponding index in the detection array."""
    if timestamp < TIMESTAMP_START or timestamp >= TIMESTAMP_END:
        raise ValueError('Timestamp is out of bounds of the detection period.')
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


def _north_offset_km(lat: float) -> float:
    """Signed north/south offset in kilometers relative to the microphone latitude."""
    return (lat - CENTER_LAT) * 111.32


def plot_aircraft_side_view(tracks_data, output_pdf=OUTPUT_PDF_PATH, output_png=OUTPUT_PNG_PATH, callsigns=None, gt_array=None, detection_array=None):
    """Plot aircraft side view: north/south offset versus altitude relative to the microphone."""
    if detection_array is None:
        print('Warning: No detection array provided; all points will be plotted as unflagged.')

    fig, ax = plt.subplots(figsize=(11.0, 6.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    all_x = []
    all_y = []

    for idx, track in enumerate(tracks_data):
        if not track:
            continue

        positions = track[0].get('tracks', [])
        flight_id_label = track[0].get('fr24_id', '')

        coords = []
        for pos in positions:
            lat = pos.get('lat')
            lon = pos.get('lon')
            alt_ft = pos.get('alt')
            timestamp = pos.get('timestamp')

            if lat is None or lon is None or alt_ft is None or not timestamp:
                continue

            alt_hae = ft_to_m(alt_ft) + N
            alt_rel_m = alt_hae - CENTER_ALT_HAE
            ts_int = int(datetime.fromisoformat(timestamp.replace('Z', '+00:00')).timestamp())
            audio_ts = ts_int - TIMESTAMP_START

            if is_inside_sphere(lat, lon, alt_hae, CENTER_LAT, CENTER_LON, CENTER_ALT_HAE, RADIUS_KM) and is_in_detection_array(audio_ts, gt_array):
                color = 'green' if is_in_detection_array(audio_ts, detection_array) else 'red'
            elif TIMESTAMP_START <= ts_int < TIMESTAMP_END:
                color = 'gray'
            else:
                continue

            x_km = _north_offset_km(lat)
            coords.append((x_km, alt_rel_m, color))
            all_x.append(x_km)
            all_y.append(alt_rel_m)

        if not coords:
            continue

        callsign_label = ''
        if callsigns and idx < len(callsigns):
            callsign_label = f' ({callsigns[idx]})'
        elif 'identification' in track[0]:
            callsign_label = f" ({track[0]['identification'].get('callsign', '')})"

        if SHOW_FLIGHT_IDS and flight_id_label:
            x0, y0, _ = coords[0]
            ax.scatter([x0], [y0], s=16, color='black', zorder=5)
            ax.text(
                x0,
                y0,
                f'  {flight_id_label}',
                fontsize=FIG_FONT_SIZE,
                fontweight='bold',
                va='center',
                ha='left',
                color='black',
            )

        for i in range(1, len(coords)):
            x1, y1, _ = coords[i - 1]
            x2, y2, color = coords[i]
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=2.0, alpha=0.95)

        # Add a small start marker for context
        x_start, y_start, start_color = coords[0]
        ax.scatter([x_start], [y_start], s=10, color=start_color, zorder=4)

    ax.axvline(0, color='black', linewidth=1.0, linestyle='--', alpha=0.8)
    ax.axhline(0, color='black', linewidth=1.0, linestyle='--', alpha=0.8)
    ax.scatter([0], [0], s=35, color='red', zorder=6, label='Microphone')

    if all_x and all_y:
        x_min = min(all_x)
        x_max = max(all_x)
        y_min = min(all_y)
        y_max = max(all_y)
        x_pad = max(0.5, 0.08 * (x_max - x_min if x_max > x_min else 1.0))
        y_pad = max(50.0, 0.08 * (y_max - y_min if y_max > y_min else 1.0))
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    ax.set_title(
        f'Aircraft side view relative to microphone {MICROPHONE_LOCATION} — session {session}',
        fontsize=FIG_FONT_SIZE,
        fontweight='bold',
        pad=12,
    )
    ax.set_xlabel('North (+) / South (-) offset from microphone (km)', fontsize=FIG_FONT_SIZE)
    ax.set_ylabel('Altitude relative to microphone (m)', fontsize=FIG_FONT_SIZE)

    legend_handles = [
        plt.Line2D([0], [0], color='green', lw=2.0, label='Detected (model=1)'),
        plt.Line2D([0], [0], color='red', lw=2.0, label='Not detected (model=0)'),
        plt.Line2D([0], [0], color='gray', lw=2.0, label='Aircraft outside geofence'),
        plt.Line2D([0], [0], color='black', lw=1.0, linestyle='--', label='Microphone axes'),
    ]
    ax.legend(handles=legend_handles, loc='best', frameon=True, framealpha=0.95)
    ax.grid(True, linestyle=':', linewidth=0.8, alpha=0.4)

    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    fig.savefig(output_pdf, format='pdf', bbox_inches='tight')
    fig.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Side view saved to: {output_pdf}')
    print(f'Side view saved to: {output_png}')


def main():
    API_KEY = '019bc710-7eef-7304-a61d-4e32f6213fdc|KxVxZabieXcmsvRMnrSHz06hyZO7YNOpy6TAjz6q2b8c7e93'

    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Accept': 'application/json',
        'Accept-Version': 'v1',
    }

    num_windows = cf.sec_to_end_index(TIMESTAMP_END - TIMESTAMP_START)
    flight_tracks_cache_path = f'history/flight_tracks_{MICROPHONE_LOCATION}_{session}.json'

    if USE_GROUND_TRUTH:
        gt_array = load_detection_array_from_ground_truth(GROUND_TRUTH_CSV_PATH, num_windows)
    else:
        gt_array = np.zeros(num_windows, dtype=int)

    # TEMP: use GT as detection to verify plotting, matching the logic of the map script.
    detection_array = gt_array.copy()

    if len(gt_array) != num_windows or len(detection_array) != num_windows:
        print(
            f"Warning: Length of gt_array ({len(gt_array)}) or detection_array ({len(detection_array)}) does not match expected number of windows ({num_windows})."
        )

    print(f'\n Ground truth source: {GROUND_TRUTH_CSV_PATH}')
    print(f'\n Detection source: {PREDICTIONS_H5_PATH}')

    if flight_ids:
        print(f'\nFetching flight tracks for {len(flight_ids)} flights...')
        flight_tracks = load_or_fetch_flight_tracks(flight_ids, headers, flight_tracks_cache_path)

        if len(flight_tracks) != len(flight_ids):
            print(
                f"Warning: Number of flight tracks fetched ({len(flight_tracks)}) does not match number of flight IDs ({len(flight_ids)})."
            )

        plot_aircraft_side_view(
            flight_tracks,
            output_pdf=OUTPUT_PDF_PATH,
            output_png=OUTPUT_PNG_PATH,
            callsigns=None,
            gt_array=gt_array,
            detection_array=detection_array,
        )
    else:
        print('No flights found for the detected/specified callsigns and time period.')


if __name__ == '__main__':
    main()
