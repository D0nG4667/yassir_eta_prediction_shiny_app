import pandas as pd

import logging
from cachetools import TTLCache, cached

from shiny import Inputs, Outputs, Session
from shiny.express import module, render, ui

from .config import HISTORY_FILE, ONE_MINUTE_SEC


# Log
logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')


@cached(cache=TTLCache(maxsize=3000, ttl=ONE_MINUTE_SEC*5))  # Memory
def get_history_data():
    try:
        parse_dates = ['time_of_prediction']
        df_history = pd.read_csv(HISTORY_FILE, parse_dates=parse_dates)
    except Exception as e:
        df_history = None
        logging.error(f"Oops, an error occured in the history page: {e}")

    finally:
        return df_history


# Display History Page
@module
def history_page(input: Inputs, output: Outputs, session: Session):
    # Disable loading spinners, use elegant pulse
    ui.busy_indicators.use(spinners=False)

    ui.panel_title(title=ui.h1(ui.strong("Prediction History 🕰️")),
                   window_title="History page")

    df_history = get_history_data()

    if df_history is not None:
        with ui.card():
            with ui.card_header():
                ui.h2("Explore all past predictions")

            @render.data_frame
            def _():
                return render.DataGrid(
                    df_history,
                    selection_mode="rows",
                    filters=True,
                )

    else:
        ui.notification_show(
            "🚨 There is no history file yet. Make a prediction.", duration=6, type="warning")
