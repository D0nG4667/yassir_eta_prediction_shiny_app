from pathlib import Path

# ENV when using standalone uvicorn server running FastAPI in api directory
ENV_PATH = Path("../env/online.env")

ONE_DAY_SEC = 24*60*60

ONE_WEEK_SEC = ONE_DAY_SEC*7

PIPELINE_FUNCTION_URL = ""

RANDOM_FOREST_URL = "https://drive.google.com/uc?export=download&id=1t0RRzAbtW7Y1lAz4ddB5iY_0SIpfdHbB"

XGBOOST_URL = "https://raw.githubusercontent.com/valiantezabuku/Yassir-ETA-Prediction-Challenge-For-Azubian-Team-Curium/main/Models/XGBRegressor.joblib"

ADABOOST_URL = "https://raw.githubusercontent.com/valiantezabuku/Yassir-ETA-Prediction-Challenge-For-Azubian-Team-Curium/main/Models/AdaBoostRegressor.joblib"

GRADIENT_BOOST_URL = "https://raw.githubusercontent.com/valiantezabuku/Yassir-ETA-Prediction-Challenge-For-Azubian-Team-Curium/main/Models/GradientBoostingRegressor.joblib"

HISTGRADIENT_BOOST_URL = "https://raw.githubusercontent.com/valiantezabuku/Yassir-ETA-Prediction-Challenge-For-Azubian-Team-Curium/main/Models/HistGradientBoostingRegressor.joblib"

DECISION_TREE_URL = "https://raw.githubusercontent.com/valiantezabuku/Yassir-ETA-Prediction-Challenge-For-Azubian-Team-Curium/main/Models/DecisionTreeRegressor.joblib"

LINEAR_REG_URL = "https://raw.githubusercontent.com/valiantezabuku/Yassir-ETA-Prediction-Challenge-For-Azubian-Team-Curium/main/Models/LinearRegression.joblib"


ALL_MODELS = {
    "AdaBoostRegressor": ADABOOST_URL,
    "DecisionTreeRegressor": DECISION_TREE_URL,
    "GradientBoostingRegressor": GRADIENT_BOOST_URL,
    "HistGradientBoostingRegressor": HISTGRADIENT_BOOST_URL,
    "LinearRegression": LINEAR_REG_URL,
    "RandomForestRegressor": RANDOM_FOREST_URL,
    "XGBRegressor": XGBOOST_URL
}

DESCRIPTION = """
This API accurately predicts the estimated time of arrival at the dropoff point for a single Yassir journey using `Random Forest model` and `XGBoost model`.\n

The models were trained on [The Yassir Eta datasets at Zindi Africa](https://zindi.africa/competitions/yassir-eta-prediction-challenge-for-azubian/data).\n

### Features
`Timestamp:` Time that the trip was started\n
`Origin_lat:` Origin latitude (in degrees)\n
`Origin_lon:` Origin longitude (in degrees)\n
`Destination_lat:` Destination latitude (in degrees)\n
`Destination_lon:` Destination longitude (in degrees)\n
`Trip_distance:` Distance in meters on a driving route\n

#### Weather Features
Daily weather summaries, based on data from the ERA5 dataset.\n
`date:` ..\n
`dewpoint_2m_temperature:` ..\n
`maximum_2m_air_temperature:` ..\n
`mean_2m_air_temperature:` ..\n
`mean_sea_level_pressure:` ..\n
`minimum_2m_air_temperature:` ..\n
`surface_pressure:` ..\n
`total_precipitation:` ..\n
`u_component_of_wind_10m:` ..\n
`v_component_of_wind_10m:` ..\n
 
### Results 
**ETA prediction:** Estimated trip time in seconds\n


### Explore the frontend data application
To explore the fontend application (built-with shiny for python) click the link below.\n
🚗[Yassir frontend](/https://hugginface-yassir)


Made with 💖 [Team Curium](#) 
"""
