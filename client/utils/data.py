import logging
import requests
import requests_cache
from geopy.geocoders import Nominatim
from utils.config import WEATHER_FILE, WEATHER_FILE_URL, TEST_FILE, TEST_FILE_URL, TRAIN_FILE, TRAIN_FILE_URL, ONE_WEEK_SEC
from cachetools import TTLCache, cached
from typing import List, Dict
from pathlib import Path
import time

import pandas as pd
# Set pandas to display all columns
pd.set_option("display.max_columns", None)

# High precision longitudes and Latitudes
pd.set_option('display.float_format', '{:.16f}'.format)

# Install persistent cache
# requests_cache.install_cache('yassir_requests_cache', expire_after=ONE_WEEK_SEC)  # Cache expires after 1 week


# Log
logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Date columns to parse
parse_dates = ['Timestamp']

dtype = {
    'Origin_lat': 'float64',
    'Origin_lon': 'float64',
    'Destination_lat': 'float64',
    'Destination_lon': 'float64',
}

dtype_weather = {
    'dewpoint_2m_temperature': 'float64',
    'maximum_2m_air_temperature': 'float64',
    'mean_2m_air_temperature': 'float64',
    'mean_sea_level_pressure': 'float64',
    'minimum_2m_air_temperature': 'float64',
    'surface_pressure': 'float64',
    'total_precipitation': 'float64',
    'u_component_of_wind_10m': 'float64',
    'v_component_of_wind_10m': 'float64',
}

# Load CSV files


# @cached(cache=TTLCache(maxsize=100000, ttl=ONE_WEEK_SEC))  # Memory # change cache library
def get_data_df(file: Path, file_url: str, parse_dates: List[str], dtype: Dict[str, str]) -> pd.DataFrame:
    df = None
    try:
        df = pd.read_csv(file_url, parse_dates=parse_dates, dtype=dtype)
    except Exception as e:
        df = pd.read_csv(file, parse_dates=parse_dates, dtype=dtype)
        logging.error(
            f"Oops, the file is not available on the url, trying a local version: {e}")
    finally:
        return df


# @cached(cache=TTLCache(maxsize=100000, ttl=ONE_WEEK_SEC))  # Memory # unhassable dict. change cache library
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [col.lower() for col in df.columns]
    return df


# Read and get cleaned data
test_df = clean_df(get_data_df(TEST_FILE, TEST_FILE_URL,
                   parse_dates=parse_dates, dtype=dtype))
train_df = clean_df(get_data_df(TRAIN_FILE, TRAIN_FILE_URL,
                    parse_dates=parse_dates, dtype=dtype))
weather_df = clean_df(get_data_df(
    WEATHER_FILE, WEATHER_FILE_URL, parse_dates=['date'], dtype=dtype_weather))


def time_sec_hms(sec: float) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(sec))

def full_time_sec_hms(sec: float) -> str:
    hours = sec // 3600
    minutes = (sec % 3600) // 60
    remaining_sec = sec % 60
    return f"{round(hours):,}h: {round(minutes)}m: {round(remaining_sec)}s"


@cached(cache=TTLCache(maxsize=100000, ttl=ONE_WEEK_SEC))  # Memory
def get_country_geojson():
    data = (
        pd.concat(
            [
                train_df[['origin_lat', 'origin_lon', ]].rename(columns={'origin_lat': 'latitude', 'origin_lon': 'longitude'}),
                train_df[['destination_lat', 'destination_lon']].rename(columns={'destination_lat': 'latitude', 'destination_lon': 'longitude'})
            ],
            ignore_index=True
        )
        .drop_duplicates()
    )
    
    # Initialize the Nominatim Geocoder
    geolocator = Nominatim(user_agent="yassirAPP")

    # Function to reverse geocode
    def reverse_geocode(lat, lon):
        location = geolocator.reverse((lat, lon), exactly_one=True)
        address = location.raw['address']
        country = address.get('country', '')
        return country

    # Apply reverse geocoding to min latitude and longitude pair and also the maximum in the DataFrame
    # Find the minimum latitude and longitude
    min_lat = data['latitude'].min()
    min_lon = data['longitude'].min()
    max_lat = data['latitude'].max()
    max_lon = data['longitude'].max()
    country_min = reverse_geocode(min_lat, min_lon)
    country_max = reverse_geocode(max_lat, max_lon)
    
    if country_min == country_max:
        country = country_min
    
    
    # Get the location for Kenya
    location = geolocator.geocode(country, exactly_one=True)

    # If the location is found
    if location:
        # Get the bounding box for Kenya
        bounding_box = location.raw['boundingbox']
        print(f"Bounding Box: {bounding_box}")

        # Nominatim API URL with query parameters
        url = "https://nominatim.openstreetmap.org/search"

        # Parameters for the request
        params = {
            'q': country, # Kenya
            'format': 'json',
            'polygon_geojson': 1  # Request GeoJSON polygons in the response
        }

        # Headers for the request
        headers = {
            'User-Agent': 'yassirAPP'
        }

        # Send the request to Nominatim with headers
        response = requests.get(url, params=params, headers=headers)

        # Check if the request was successful
        if response.status_code == 200:
            country_geojson = response.json()

            geojson = country_geojson[0]['geojson']
            
    
    return country, geojson, data
            
        
                

