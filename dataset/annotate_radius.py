import requests
import time
from datetime import datetime, timedelta
from typing import List
import math
import pandas as pd
import os

def haversine_km(lat1, lon1, lat2, lon2):
    # Calculate distance (km) between two lat/lon points
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def is_inside_circle(flat: float, flon: float) -> bool:
    return haversine_km(CENTER_LAT, CENTER_LON, flat, flon) <= RADIUS_KM

def flights_in_circle(data):
    filtered_data = [
        flight for flight in data
        if flight['latitude'] is not None and flight['longitude'] is not None and is_inside_circle(flight['latitude'], flight['longitude'])
    ]
    return filtered_data

def fetch_historic_flight_positions(api_token: str, start_date: datetime, end_date: datetime, interval_seconds: int = 15 * 60, **filters) -> List[dict]:
    """
    Fetches historical flight positions over a date range.

    Parameters:
        api_token (str): Your Flightradar24 API token.
        start_date (datetime): The start date and time.
        end_date (datetime): The end date and time.
        interval_seconds (int): Time interval between data points in seconds. Default is 15 minutes (900 seconds).
        **filters: Endpoint filters like bounds, flights, callsigns, registrations, limit etc.

    Returns:
        Dataframe: Ground truth dataframe with columns: filename, start_time, end_time, class, aircraft-id
    """
    api_url = 'https://fr24api.flightradar24.com/api/historic/flight-positions/full'
    headers = {
        'Accept': 'application/json',
        'Accept-Version': 'v1',
        'Authorization': f'Bearer {api_token}'
    }

    # Generate list of timestamps
    timestamps = []
    current_time = start_date
    delta = timedelta(seconds=interval_seconds)

    while current_time <= end_date:
        timestamps.append(int(current_time.timestamp()))
        current_time += delta

    with open(GT_FILE, "w") as f:
        f.write("filename\tstart_time\tend_time\tclass\taircraft-id\n")

    df = pd.read_csv(GT_FILE)

    print(f"Start: {start_date}, End: {end_date}")
    print
    print(f"timestamps: {timestamps}")

    for ts in timestamps:
        params = {'timestamp': ts}
        params.update(filters) 

        response = requests.get(api_url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json().get('data', [])
            filtered_data = flights_in_circle(data)
            for flight_data in filtered_data:
                flight = flight_data['flight']
                mask = df["aircraft-id"] == flight
                if mask.any():
                    df.loc[mask, "end_time"] = ts # Update end_time if flight is still within reach
                else:
                    new_row = {
                        "filename": "-",
                        "start_time": ts,
                        "end_time": ts,
                        "class": "Aircraft",
                        "aircraft-id": flight
                    }
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True) # Add new detection to ground truth
            print(f"Timestamp {ts}: Retrieved {len(flight_data)} records")

        elif response.status_code == 429:
            print(f"Rate limit reached. Sleeping for {response.headers.get('Retry-After', 60)} seconds.")
            time.sleep(int(response.headers.get('Retry-After', 60)))

            # Retry the request after sleeping
            response = requests.get(api_url, headers=headers, params=params)

            if response.status_code == 200:
                data = response.json().get('data', [])
                filtered_data = flights_in_circle(data)
                for flight_data in filtered_data:
                    flight = flight_data['flight']
                    mask = df["aircraft-id"] == flight
                    if mask.any():
                        df.loc[mask, "end_time"] = ts # Update end_time if flight is still within reach
                    else:
                        new_row = {
                            "filename": "-",
                            "start_time": ts,
                            "end_time": ts,
                            "label": "aircraft",
                            "aircraft-id": flight
                        }
                        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True) # Add new detection to ground truth
                print(f"Timestamp {ts}: Retrieved {len(data)} records after retry")
            else:
                print(f"Error {response.status_code} for timestamp {ts} after retry")
        else:
            print(f"Error {response.status_code} for timestamp {ts}")

    return df


# Example usage
if __name__ == '__main__':
    # Your API token
    API_TOKEN = '01994cdc-690a-71e4-8387-0cc69b23a4df|m9SbuJYMcyedgkuP8zbtRPIl34IxqkqWdRMyEO2y4a01f6bc'
    GT_FILE = 'dataset/Gardemoen/gt_flightradar.csv'
    os.makedirs(os.path.dirname(GT_FILE), exist_ok=True)

    # Define date range
    start_date = datetime(2025, 10, 1, 0, 0, 0)
    end_date = datetime(2025, 10, 1, 0, 0, 10)

    # Define parameters
    CENTER_LAT, CENTER_LON = (60.231853, 11.105390) # (latitude, longitude) Microphone location
    bounds = '60.282439,60.101712,10.959275,11.256177' # 'N, S, W, E' Rough estimate of bounds around microphone
    RADIUS_KM = 10  # [km] Radius bound around microphone to consider
    interval_seconds = 5  # [sek] (How often to sample data)

    df = fetch_historic_flight_positions(
        api_token=API_TOKEN,
        start_date=start_date,
        end_date=end_date,
        interval_seconds=interval_seconds,
        limit=1000,
        bounds=bounds
    )

    df.to_csv("gt.csv", index=False)
    print(f"Annotations saved to {GT_FILE}")
