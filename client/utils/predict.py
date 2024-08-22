import os
import re
import sys
from dotenv import load_dotenv
from datetime import datetime
import time
import logging

import httpx
import pandas as pd

from pydantic import BaseModel, Field
from typing import List, Optional

from shiny import reactive, Inputs, Outputs, Session
from shiny.express import module, render, ui
from shinywidgets import render_widget

import ipyleaflet as L
from faicons import icon_svg
from geopy.distance import geodesic

from utils.utils import *
from utils.config import LOCATIONS, BRANDTHEMES, KENYA_LAT, KENYA_LON, HOURS, MINUTES, SECONDS, ALL_MODELS
from utils.config import HISTORY_FILE, ENV_PATH
from utils.url_to_coordinates import get_full_url, on_convert

load_dotenv(ENV_PATH)


class EtaFeatures(BaseModel):
    timestamp: List[datetime] = Field(
        description="Timestamp: Time that the trip was started")
    origin_lat: List[float] = Field(
        description="Origin_lat: Origin latitude (in degrees)")
    origin_lon: List[float] = Field(
        description="Origin_lon: Origin longitude (in degrees)")
    destination_lat: List[float] = Field(
        description="Destination_lat: Destination latitude (in degrees)")
    destination_lon: List[float] = Field(
        description="Destination_lon: Destination longitude (in degrees)")
    trip_distance: List[float] = Field(
        description="Trip_distance: Distance in meters on a driving route")


# Log
logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')


lat_min, lat_max, lon_min, lon_max = get_bounds(country='Kenya')


async def endpoint(model_name: str) -> str:
    api_url = os.getenv("API_URL")
    model_endpoint = f"{api_url}={model_name}"
    return model_endpoint


async def predict_eta(data: EtaFeatures, model_name: str) -> Optional[float]:
    prediction = None
    try:
        # Get model endpoint
        model_endpoint = await endpoint(model_name)

        if "pyodide" in sys.modules:
            import pyodide.http

            response = await pyodide.http.pyfetch(
                model_endpoint,
                method="POST",
                body=data,
                headers={"Content-Type": "application/json"}
            )

            # Handle the response
            if response.ok:
                # .json() parses the response as JSON and converts to dictionary.
                result = await response.json()['result']

        else:
            # Send POST request with JSON data using the json parameter
            async with httpx.AsyncClient() as client:
                response = await client.post(model_endpoint, json=data, timeout=30)
                response.raise_for_status()  # Ensure we catch any HTTP errors
            
            # print(response.json())            
            if (response.status_code == 200):
                result = response.json()['result']

        if result:
            prediction = float(result['prediction'][0])

            # Create dataframe
            df = pd.DataFrame.from_dict(data)
            df['eta_prediction'] = prediction
            df['time_of_prediction'] = pd.Timestamp(datetime.now())
            df['model_used'] = model_name

            # Save to history csv file
            df.to_csv(HISTORY_FILE, mode='a',
                      header=not (HISTORY_FILE.exists()), index=False)
    except Exception as e:
        logging.error(f"Oops, an error occured: {e} {response}")

    return prediction


@module
def predict_page(input: Inputs, output: Outputs, session: Session):
    # Disable loading spinners, use elegant pulse
    ui.busy_indicators.use(spinners=False)

    ui.panel_title(title=ui.h1(ui.strong("Eta Prediction 🔮")),
                   window_title="Eta Prediction")

    with ui.layout_sidebar():
        with ui.sidebar():
            # Cordinates features
            ui.input_numeric("origin_lat", "Origin Latitude °",
                             value=LOCATIONS["Nairobi"]['latitude'], min=lat_min, max=lat_max, step=1)
            ui.input_numeric("origin_lon", "Origin Longitude °",
                             value=LOCATIONS["Nairobi"]['longitude'], min=lon_min, max=lon_max, step=1)
            ui.input_numeric("destination_lat", "Destination Latitude °",
                             value=LOCATIONS["National Museum of Kenya"]['latitude'], min=lat_min, max=lat_max, step=1)
            ui.input_numeric("destination_lon", "Destination Longitude °",
                             value=LOCATIONS["National Museum of Kenya"]['longitude'], min=lon_min, max=lon_max, step=1)

            # Google Maps Url to Coordinates
            ui.help_text("Convert Google Maps Url to Latitude and Longitudes")
            ui.input_action_button("map_url", "Convert")

            @reactive.effect
            @reactive.event(input.map_url)
            def maps_url_modal():
                m = ui.modal(
                    ui.help_text("From Origin:"),
                    ui.input_text("origin_url", "Google Maps url:"),

                    ui.help_text("To Destination:"),
                    ui.input_text("destination_url", "Google Maps url:"),

                    ui.input_action_button("convert_url", "Convert"),

                    title="Google Maps Url to Coordinates",
                    easy_close=True,
                    footer=None,
                )
                ui.modal_show(m)

            @reactive.effect
            @reactive.event(input.convert_url, ignore_init=True)
            def update_coordinates_from_url() -> Optional[float]:
                try:
                    origin_url = get_full_url(input.origin_url())
                    destination_url = get_full_url(input.destination_url())

                    # Coordinates are yet to be known
                    origin_latitude = None
                    origin_longitude = None
                    destination_latitude = None
                    destination_longitude = None

                    # Regular expression to find coordinates in the URL
                    pattern = re.compile(r"@(-?\d+\.\d+),(-?\d+\.\d+)")
                    match = []
                    for url in [origin_url, destination_url]:
                        match.append(pattern.search(url))

                    if all(match):
                        origin_latitude = float(match[0].group(1))
                        origin_longitude = float(match[0].group(2))
                        destination_latitude = float(match[1].group(1))
                        destination_longitude = float(match[1].group(2))

                        valid.set(on_convert(origin_latitude, origin_longitude,
                                  destination_latitude, destination_longitude))

                    if valid():
                        ui.notification_show(
                            f"✅ The coordinates have been updated", duration=3, type="default")
                    else:
                        raise Exception
                except Exception as e:
                    logging.error(
                        f"Oops, update_coordinates_from_url says an error occured converting maps url to coordinates: {e}")
                    ui.notification_show(
                        f"Error: {e}", duration=3, type="error")
                    ui.notification_show(
                        "🚨 Could not convert url to coordinates. Try again!", duration=6, type="error")

                finally:
                    ui.modal_remove()

            # Rest coordinates back to Kenyan region
            ui.input_action_button(
                "reset", "Back to Nairobi", icon=icon_svg("crosshairs"))

            # Trip Distance feature
            ui.input_numeric("trip_distance", "Trip Distance (meters)",
                             value=1, min=1, max=600000, step=10)
            ui.input_switch("manual_distance",
                            "Use manual distance", False),

            # Date feature
            ui.input_date("date", "Select a Date")
            ui.help_text("Select the UTC time")
            ui.input_select("hours", "24-hour",
                            choices=HOURS, selected=HOURS[0])
            ui.input_select("minutes", "Minutes",
                            choices=MINUTES, selected=MINUTES[0])
            ui.input_select("seconds", "Seconds",
                            choices=SECONDS, selected=SECONDS[0])

            # Select model
            ui.input_selectize(
                "modelname",
                "Choose a model",
                choices=ALL_MODELS,
                selected="XGBRegressor",
            )

            # Base map
            ui.input_selectize(
                "basemap",
                "Choose a basemap",
                choices=list(BASEMAPS.keys()),
                selected="Mapnik",
            )

        # Top 3 cards
        with ui.layout_column_wrap(fill=False):
            with ui.value_box(showcase=icon_svg("route"), theme=BRANDTHEMES['purple-dark']):
                "Trip Distance"

                @render.text
                def trip_dist_km():
                    return f"{trip_distance()/1000:,.1f} km" if valid else ""

                @render.text
                def trip_dist_m():
                    return f"{trip_distance():,.1f} m" if valid and trip_distance is not None else ""

            with ui.value_box(showcase=icon_svg("egg"), theme=BRANDTHEMES['purple-dark']):
                "Geodisic Distance"

                @reactive.calc
                def geo_dist():
                    dist = geodesic(loc1xy(), loc2xy())
                    return (f"{dist.meters:,.1f} m", f"{dist.kilometers:,.1f} km") if valid and trip_distance is not None else ""

                @render.text
                def geo_dist_km():
                    return geo_dist()[1] if valid and trip_distance is not None else ""

                @render.text
                def geo_dist_m():
                    return geo_dist()[0] if valid and trip_distance is not None else ""

            with ui.value_box(showcase=icon_svg("clock"), theme=BRANDTHEMES['red']):
                "Est. time of arrival"

                @reactive.calc
                async def eta():
                    try:
                        # print(valid())
                        # print(notification_error())
                        if validate_inputs(origin_lat(), origin_lon(), destination_lat(), destination_lon()) and valid():                        
                            data: EtaFeatures = {
                                'timestamp': [datetz()],
                                'origin_lat': [origin_lat()],
                                'origin_lon': [origin_lon()],
                                'destination_lat': [destination_lat()],
                                'destination_lon': [destination_lon()],
                                'trip_distance': [trip_distance()]
                            }

                            eta_sec = await predict_eta(data, input.modelname())

                            eta_hms = time.strftime(
                                '%H:%M:%S', time.gmtime(eta_sec))

                            ui.notification_show(
                                f"⏰ ETA: {eta_hms} H:M:S", duration=6, type="default")

                            return f"{eta_sec:,.0f} s", f"{eta_hms}"
                        else:
                            raise Exception
                    except Exception as e:
                        logging.error({e})
                        ui.notification_show(
                            "🚨 Could not predict Eta. Median eta is 1000 seconds", duration=3, type="error")
                        return None

                @render.text
                async def eta_sec():
                    text = await eta()
                    return text[0] if text else ""

                @render.text
                async def eta_hms():
                    text = await eta()
                    return text[1] if text else ""

                @render.express
                def eta_info():
                    with ui.tooltip(title="Google Maps ETA"):
                        icon_svg("google")
                        f"{trip_eta():,.0f} s | {time.strftime('%H:%M:%S', time.gmtime(trip_eta()))}"
                        
                    
                    
                    
        # Map (2 indents)
        with ui.card():
            ui.card_header(
                "💡 Map (drag the markers to change locations)")

            @render_widget
            def map():
                return L.Map(zoom=9, center=(KENYA_LAT, KENYA_LON))

    ######################################################
    # Reactive values to store location information
    origin_lat = reactive.value()
    origin_lon = reactive.value()
    destination_lat = reactive.value()
    destination_lon = reactive.value()

    valid = reactive.value()

    # Reactive value to store trip_distance information
    trip_distance = reactive.value()
    trip_eta = reactive.value()

    @reactive.effect(priority=100)
    def _():
        if (
            validate_inputs(input.origin_lat(), input.origin_lon(),
                            input.destination_lat(), input.destination_lon())
            or
            validate_inputs(origin_lat(), origin_lon(),
                            destination_lat(), destination_lon())
        ):
            value = True
        else:
            value = False

        valid.set(value)

    @reactive.calc
    def datetz():
        return f"{input.date()}T{input.hours()}:{input.minutes()}:{input.seconds()}Z"

    @reactive.effect
    def _():
        origin_lat.set(input.origin_lat()
                       if valid else LOCATIONS["Nairobi"]['latitude'])
        origin_lon.set(input.origin_lon()
                       if valid else LOCATIONS["Nairobi"]['longitude'])
        destination_lat.set(input.destination_lat(
        ) if valid else LOCATIONS["National Museum of Kenya"]['latitude'])
        destination_lon.set(input.destination_lon(
        ) if valid else LOCATIONS["National Museum of Kenya"]['longitude'])

        # Automate trip distance, eta from Google Maps
        google_td, google_eta = google_maps_trip_distance_eta(loc1xy(), loc2xy())
        if isinstance(google_td, float):
            trip_distance.set(google_td)
            trip_eta.set(google_eta)
        else:
            ui.notification_show(
                            "🚨 Could not estimate trip distance. Using Geosidic distance...", duration=3, type="warning")
            trip_distance.set(geo_dist())

        # Manual
        if input.manual_distance() and input.trip_distance() not in [0, None]:
            trip_distance.set(input.trip_distance())

    @reactive.effect
    @reactive.event(trip_distance)
    def _():
        if valid():
            # Update the trip distance input with current calculated or manual trip distance
            ui.update_numeric("trip_distance", value=trip_distance())

    @reactive.calc
    def loc1xy():
        return origin_lat(), origin_lon()

    @reactive.calc
    def loc2xy():
        return destination_lat(), destination_lon()

    # Add marker for first location

    @reactive.effect
    def _():
        if valid():
            update_marker(map.widget, loc1xy(), on_move1, "origin", icon=L.AwesomeIcon(
                name='fa-map-marker', marker_color='darkpurple'))

    # Add marker for second location

    @reactive.effect
    def _():
        if valid():
            update_marker(map.widget, loc2xy(), on_move2, "destination", icon=L.AwesomeIcon(
                name='fa-map-marker', marker_color='purple'))

    # Add line and fit bounds when either marker is moved

    @reactive.effect
    def _():
        if valid():
            update_line(map.widget, loc1xy(), loc2xy())

    # If new bounds fall outside of the current view, fit the bounds if valid coordinates

    @reactive.effect
    def _():
        # if valid():
            l1 = loc1xy()
            l2 = loc2xy()

            lat_rng = [min(l1[0], l2[0]), max(l1[0], l2[0])]
            lon_rng = [min(l1[1], l2[1]), max(l1[1], l2[1])]
            new_bounds = [
                [lat_rng[0], lon_rng[0]],
                [lat_rng[1], lon_rng[1]],
            ]

            b = map.widget.bounds
            if len(b) == 0:
                map.widget.fit_bounds(new_bounds)
            elif (
                lat_min < b[0][0]
                or lat_max > b[1][0]
                or lon_min < b[0][1]
                or lon_max > b[1][1]
            ):
                map.widget.fit_bounds(new_bounds)

    # Update the basemap

    @reactive.effect(priority=-100)  # The last effect that runs
    def _():
        if valid():
            update_basemap(map.widget, input.basemap())

    @reactive.effect(priority=95)
    @reactive.event(input.reset)
    def _():
        back_to_nairobi()
