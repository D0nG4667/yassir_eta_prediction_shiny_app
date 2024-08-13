import requests

from shiny.express import ui
from utils.utils import validate_inputs



def get_full_url(url: str) -> str:
    response = requests.get(url, allow_redirects=True)
    return response.url


def on_convert(origin_lat: float, origin_lon: float, destination_lat: float, destination_lon: float) -> bool:

    valid = validate_inputs(origin_lat, origin_lon, destination_lat, destination_lon)

    if valid:
        with ui.hold():
            ui.update_numeric(f"origin_lat", value=origin_lat)
            ui.update_numeric(f"origin_lon", value=origin_lon)
            ui.update_numeric(f"destination_lat", value=destination_lat)
            ui.update_numeric(f"destination_lon", value=destination_lon)
    
    return valid
