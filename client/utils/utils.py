import json
import requests
import requests_cache as R
from cachetools import TTLCache, cached

from typing import Literal, Tuple, Optional

import ipyleaflet as L
from ipyleaflet import AwesomeIcon

from shiny.express import ui

from utils.config import MAPS_API_KEY, BRANDCOLORS, BASEMAPS, ONE_WEEK_SEC, LOCATIONS, TRIP_DISTANCE


# Cache expires after 1 week
# R.install_cache('yassir_requests_cache',  expire_after=ONE_WEEK_SEC)  # Sqlite


# ---------------------------------------------------------------
# Helper functions for map and location inputs on predict page
# ---------------------------------------------------------------

@cached(cache=TTLCache(maxsize=300, ttl=ONE_WEEK_SEC))  # Memory
def get_bounds(country: str) -> Tuple[float]:
    headers = {
        'User-Agent': 'Yassir ETA Shiny App/1.0 (gabriel007okuns@gmail.com)'
    }

    response = requests.get(
        f"http://nominatim.openstreetmap.org/search?q={country}&format=json", headers=headers)

    boundingbox = json.loads(response.text)[0]["boundingbox"]

    # Extract the bounds as float datatype
    lat_min, lat_max, lon_min, lon_max = (float(b) for b in boundingbox)

    return lat_min, lat_max, lon_min, lon_max


@cached(cache=TTLCache(maxsize=3000, ttl=ONE_WEEK_SEC))  # Memory
def google_maps_trip_distance(origin: tuple, destination: tuple) -> float:
    """
    The road distance calculated using Google Maps distance matrix api with the driving car is the shortest 
    or optimal road distance based on the available road data and routing algorithm.

    origin is a tuple of lat, lon
    destination is a tuple of lat, lon
    
    Returns: the calculiated trip distance or a default value
    """

    # Google Maps API URL
    url = f"https://maps.googleapis.com/maps/api/distancematrix/json?origins={origin[0]},{origin[1]}&destinations={destination[0]},{destination[1]}&key={MAPS_API_KEY}"


    # Send request
    response = requests.get(url)

    if response.status_code == 200:
        # Decode the response
        data = response.json()
    
        # Extract distance information
        if "rows" in data and len(data["rows"]) > 0:
            distance_info = data["rows"][0]["elements"][0]["distance"]
            distance = float(distance_info['value'])
        else:
            distance = TRIP_DISTANCE # Default
            back_to_nairobi()
    else:
        distance = TRIP_DISTANCE # Default
        # print(response.status_code)
        back_to_nairobi()

    return distance


def update_marker(map: L.Map, loc: tuple, on_move: object, name: str, icon: AwesomeIcon):
    remove_layer(map, name)
    m = L.Marker(location=loc, draggable=True, name=name, icon=icon)
    m.on_move(on_move)
    map.add_layer(m)


def update_line(map: L.Map, loc1: tuple, loc2: tuple):
    remove_layer(map, "line")
    map.add_layer(
        L.Polyline(locations=[loc1, loc2],
                   color=BRANDCOLORS['red'], weight=3, name="line")
    )


def update_basemap(map: L.Map, basemap: str):
    for layer in map.layers:
        if isinstance(layer, L.TileLayer):
            map.remove_layer(layer)
    map.add_layer(L.basemap_to_tiles(BASEMAPS[basemap]))


def remove_layer(map: L.Map, name: str):
    for layer in map.layers:
        if layer.name == name:
            map.remove_layer(layer)


def on_move1(**kwargs):
    return on_move("origin", **kwargs)


def on_move2(**kwargs):
    return on_move("destination", **kwargs)

# When the markers are moved, update the numeric location inputs to include the new
# location (which results in the locations() reactive value getting updated,
# which invalidates any downstream reactivity that depends on it)


def on_move(loc_type: Literal['origin', 'destination'], **kwargs):
    location = kwargs["location"]
    loc_lat, loc_lon = location

    ui.update_numeric(f"{loc_type}_lat", value=loc_lat)
    ui.update_numeric(f"{loc_type}_lon", value=loc_lon)

    # origin_lat
    # origin_lon
    # destination_lat
    # destination_lon

    # Re-center to Kenya region


def back_to_nairobi():
    ui.update_numeric("origin_lat", value=LOCATIONS["Nairobi"]['latitude'])
    ui.update_numeric(
        "origin_lon", value=LOCATIONS["Nairobi"]['longitude'])
    ui.update_numeric(
        "destination_lat", value=LOCATIONS["National Museum of Kenya"]['latitude'])
    ui.update_numeric(
        "destination_lon", value=LOCATIONS["National Museum of Kenya"]['longitude'])


def validate_inputs(origin_lat: float = None, origin_lon: float = None, destination_lat: float = None, destination_lon: float = None) -> bool:
    lat_min, lat_max, lon_min, lon_max = get_bounds(country='Kenya')

    valid = True
    
    for lat, lon in [(origin_lat, origin_lon), (destination_lat, destination_lon)]:
        if lat is not None and lon is not None:
            if (lat < lat_min or lat > lat_max) or (lon < lon_min or lon > lon_max):
                ui.notification_show(
                    "😮 Location is outside Kenya, taking you back to Nairobi", type="error")
                valid = False
                back_to_nairobi()
                break


    return valid


# Footer
footer = ui.tags.footer(
    ui.tags.div(
        "© 2024. Made with 💖",
        style=f"text-align: center; padding: 10px; color: #fff; background-color: {BRANDCOLORS['purple-dark']}; margin-top: 50px;"
    )
)
