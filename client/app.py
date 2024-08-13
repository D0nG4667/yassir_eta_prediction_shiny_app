from functools import partial

from shiny.ui import page_navbar
from shiny.express import ui

from utils.utils import footer
from utils.config import BRANDCOLORS
from utils.home import home_page
from utils.dashboard import dashboard_page
from utils.predict import predict_page
from utils.history import history_page





ui.page_opts(
    title=ui.img(src="logo-yassir-forward-light.svg",
                 alt="Yassir logo", height="50px"),
    window_title="Yassir Home",
    page_fn=partial(page_navbar, id="page"),
    inverse=True,
    bg=BRANDCOLORS["purple-dark"],
    fillable=True,
    lang="en",
    footer=footer,
)

# Add Yassir favicon
ui.head_content(ui.tags.link(rel="icon", type="image/png",
                sizes="32x32", href="favicon-yassir-forward.png"))


with ui.nav_panel("Home"):
    home_page("home")


with ui.nav_panel("Dashboard"):
    dashboard_page("dashboard")


with ui.nav_panel("Predict"):
    predict_page("predict")


with ui.nav_panel("History"):
    history_page("history")


with ui.nav_control():
    # Mode Switcher
    ui.input_dark_mode(mode="light")
