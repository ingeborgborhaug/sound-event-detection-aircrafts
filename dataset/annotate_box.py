import requests
import time
from datetime import datetime, timedelta
from typing import List

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
        List[dict]: A list of flight position data.
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

    all_data = []

    for ts in timestamps:
        params = {'timestamp': ts}
        params.update(filters)  # Add additional filters if provided

        response = requests.get(api_url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json().get('data', [])
            all_data.extend(data)
            print(f"Timestamp {ts}: Retrieved {len(data)} records")

        elif response.status_code == 429:
            print(f"Rate limit reached. Sleeping for {response.headers.get('Retry-After', 60)} seconds.")
            time.sleep(int(response.headers.get('Retry-After', 60)))

            # Retry the request after sleeping
            response = requests.get(api_url, headers=headers, params=params)

            if response.status_code == 200:
                data = response.json().get('data', [])
                all_data.extend(data)
                print(f"Timestamp {ts}: Retrieved {len(data)} records after retry")
            else:
                print(f"Error {response.status_code} for timestamp {ts} after retry")
        else:
            print(f"Error {response.status_code} for timestamp {ts}")

    return all_data

# Example usage
if __name__ == '__main__':
    # Your API token
    API_TOKEN = 'YOUR_API_TOKEN'

    # Define date range
    start_date = datetime(2025, 10, 1, 0, 0, 0)
    end_date = datetime(2025, 10, 1, 0, 0, 10)

    # Define parameters
    CENTER_LAT, CENTER_LON = (60.231853, 11.105390) # (latitude, longitude) Microphone location
    bounds = '60.282439,60.101712,10.959275,11.256177' # 'N, S, W, E' Rough estimate of bounds around microphone

    # Define parameters
    interval_seconds = 5  # 5 seconds

    # Fetch data
    flight_data = fetch_historic_flight_positions(
        api_token=API_TOKEN,
        start_date=start_date,
        end_date=end_date,
        interval_seconds=interval_seconds,
        bounds=bounds,
        limit=1000
    )

    print(f"Total records retrieved: {len(flight_data)}")
    # Process the data as needed