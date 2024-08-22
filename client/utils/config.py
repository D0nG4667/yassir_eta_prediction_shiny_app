import os
from dotenv import load_dotenv
from pathlib import Path
from ipyleaflet import basemaps

from shiny.express import ui


# Paths
# ENV when using standalone shiny server, shiny for python runs from the root of the project
ENV_PATH = Path("online.env")

DATA = Path(__file__).parent.parent / "data/"
TEST_FILE = DATA / "Test.csv"
TRAIN_FILE = DATA / "Train.csv"
WEATHER_FILE = DATA / "Weather.csv"
HISTORY = DATA / "history/"
HISTORY_FILE = HISTORY / "history.csv"


# Models
ALL_MODELS = [
    "AdaBoostRegressor",
    "DecisionTreeRegressor",
    "GradientBoostingRegressor",
    "HistGradientBoostingRegressor",
    "LinearRegression",
    # "RandomForestRegressor",
    "XGBRegressor",
]

BEST_MODELS = ["RandomForestRegressor", "XGBRegressor"]


# Urls
TEST_FILE_URL = "https://raw.githubusercontent.com/valiantezabuku/Yassir-ETA-Prediction-Challenge-For-Azubian-Team-Curium/main/Data/Test.csv"
TRAIN_FILE_URL = "https://raw.githubusercontent.com/valiantezabuku/Yassir-ETA-Prediction-Challenge-For-Azubian-Team-Curium/main/Data/Train.csv"
WEATHER_FILE_URL = "https://raw.githubusercontent.com/valiantezabuku/Yassir-ETA-Prediction-Challenge-For-Azubian-Team-Curium/main/Data/Weather.csv"


# Load environment variables from .env file into a dictionary
load_dotenv(ENV_PATH)


# Google Maps Directions API
# https://maps.googleapis.com/maps/api/distancematrix/
MAPS_API_KEY = os.getenv("MAPS_API_KEY")

# https://maps.app.goo.gl/Fx5rdPs1KeA6jCeB8
KENYA_LAT = 0.15456
KENYA_LON = 37.908383


BASEMAPS = {
    "DarkMatter": basemaps.CartoDB.DarkMatter,
    "Mapnik": basemaps.OpenStreetMap.Mapnik,
    "NatGeoWorldMap": basemaps.Esri.NatGeoWorldMap,
    "WorldImagery": basemaps.Esri.WorldImagery,
}

# Yassir
BRANDCOLORS = {
    "red": "#FB2576",
    "purple-light": "#6316DB",
    "purple-dark": "#08031A",
}

BRANDTHEMES = {
    "red": ui.value_box_theme(bg=BRANDCOLORS['red'], fg='white'),
    "purple-light": ui.value_box_theme(bg=BRANDCOLORS['purple-light'], fg='white'),
    "purple-dark": ui.value_box_theme(bg=BRANDCOLORS['purple-dark'], fg='white'),
}


# Nairobi, https://maps.app.goo.gl/oPbLBYHuicjrC22J9
# National Museum of Kenya, https://maps.app.goo.gl/zbmUpe71admABU9i9
# Closest location
LOCATIONS = {
    "Nairobi": {"latitude": -1.3032036, "longitude": 36.6825914},
    "National Museum of Kenya": {"latitude": -1.2739575, "longitude": 36.8118501},
    "Mombasa": {"latitude": -1.3293123, "longitude": 36.8717466},
}


HOURS = [f"{i:02}" for i in range(0, 24)]

MINUTES = [f"{i:02}" for i in range(0, 12)]

SECONDS = [f"{i:02}" for i in range(0, 60)]


ONE_MINUTE_SEC = 60

ONE_HOUR_SEC = ONE_MINUTE_SEC * 60

ONE_DAY_SEC = ONE_HOUR_SEC * 24

ONE_WEEK_SEC = ONE_DAY_SEC * 7


# Default trip distance
TRIP_DISTANCE = 30275.7
ETA = 18000
