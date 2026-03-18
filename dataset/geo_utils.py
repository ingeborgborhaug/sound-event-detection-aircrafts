"""
Shared geometric utilities and session/location configuration.
This module has NO top-level side effects and is safe to import anywhere.
"""
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime
from zoneinfo import ZoneInfo
from pyproj import Transformer

from keras_yamnet.params import PATCH_WINDOW_SECONDS

# --- Constants ---
N = 40             # Geoid undulation offset (m) applied to barometric altitude -> HAE
tripod_height_m = 1.2

# WGS84 geodetic (lat, lon, height) -> ECEF XYZ
transformer = Transformer.from_crs(
    "EPSG:4979",
    "EPSG:4978",
    always_xy=True,
)


# --- Unit conversion ---
def ft_to_m(feet: float) -> float:
    return feet * 0.3048


# --- Geometry ---
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def distance_3d_m(
    lat1: float, lon1: float, alt1_m: float,
    lat2: float, lon2: float, alt2_m: float,
) -> float:
    """3D Euclidean distance between two WGS84 points via ECEF (metres)."""
    x1, y1, z1 = transformer.transform(float(lon1), float(lat1), float(alt1_m))
    x2, y2, z2 = transformer.transform(float(lon2), float(lat2), float(alt2_m))
    return float(np.linalg.norm([x2 - x1, y2 - y1, z2 - z1]))


def is_inside_sphere(
    lat: float, lon: float, alt_hae: float,
    center_lat: float, center_lon: float, center_alt_hae: float,
    radius_km: float,
) -> bool:
    """Return True if the point is within radius_km of the centre (3D ECEF distance)."""
    return distance_3d_m(lat, lon, alt_hae, center_lat, center_lon, center_alt_hae) <= radius_km * 1000


# --- Location configuration ---
def get_location_config(microphone_loc: str) -> tuple[float, float, float]:
    """Return (lat, lon, alt_HAE_m) for the given microphone location."""
    if microphone_loc == 'loc_1':
        return (63.472832, 10.814295, 2.6 + tripod_height_m + N)
    elif microphone_loc == 'loc_2':
        return (63.49094, 10.86972, 56.0 + tripod_height_m + N)
    elif microphone_loc == 'loc_3':
        return (63.51608, 10.84220, 126.6 + tripod_height_m + N)
    else:
        raise ValueError(f"Invalid microphone location: {microphone_loc}")


# --- Session timestamps ---
def get_session_timestamps(session_id: str, microphone_locations: list[str] | None = None) -> tuple[int, int]:
    """Return (TIMESTAMP_START, TIMESTAMP_END) as UTC Unix integers for the given session."""
    if session_id == '280126':
        dt_start = datetime(2026, 1, 28, 12, 43, 13, tzinfo=ZoneInfo("Europe/Oslo"))
        dt_end   = datetime(2026, 1, 28, 14, 55,  6, tzinfo=ZoneInfo("Europe/Oslo"))
    elif session_id == '230226':
        dt_start = datetime(2026, 2, 23, 13, 27, 23, tzinfo=ZoneInfo("Europe/Oslo"))
        dt_end   = datetime(2026, 2, 23, 15, 28, 17, tzinfo=ZoneInfo("Europe/Oslo"))
    elif session_id == '030326':
        if microphone_locations and 'loc_1' in microphone_locations:
            raise ValueError("Session '030326' does not include loc_1.")
        dt_start = datetime(2026, 3, 3, 18, 11,  6, tzinfo=ZoneInfo("Europe/Oslo"))
        dt_end   = datetime(2026, 3, 3, 20, 25, 35, tzinfo=ZoneInfo("Europe/Oslo"))
    else:
        raise ValueError(f"Invalid session specified: {session_id}")

    ts_start = int(dt_start.astimezone(ZoneInfo("UTC")).timestamp())
    ts_end   = int(dt_end.astimezone(ZoneInfo("UTC")).timestamp())
    return ts_start, ts_end
