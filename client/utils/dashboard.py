from .data import test_df, train_df, weather_df, time_sec_hms, full_time_sec_hms, get_country_geojson
from .config import BRANDCOLORS, BRANDTHEMES, ALL_MODELS
import time
import requests
import requests_cache
from pathlib import Path

from faicons import icon_svg

import pandas as pd
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from plotly_calplot import calplot
from plotly.subplots import make_subplots

from shiny import reactive, render, Inputs, Outputs, Session
from shiny.express import input, output, module, ui
from shinywidgets import render_plotly, render_widget


# Set pandas to display all columns
pd.set_option("display.max_columns", None)

# High precision longitudes and Latitudes
pd.set_option('display.float_format', '{:.16f}'.format)


# Yassir plotly theme
yassir_theme = pio.templates["plotly_dark"]

yassir_theme.layout.update(
    plot_bgcolor=BRANDCOLORS["purple-dark"],
    paper_bgcolor=BRANDCOLORS["purple-dark"],
    colorway=[BRANDCOLORS["red"], BRANDCOLORS["purple-light"],
              BRANDCOLORS["purple-dark"]],  # Brand colors
)

pio.templates["yassir_theme"] = yassir_theme

pio.templates.default = yassir_theme


@module
def dashboard_page(input: Inputs, output: Outputs, session: Session):
    # Disable loading spinners, use elegant pulse
    ui.busy_indicators.use(spinners=False)

    ui.panel_title(title=ui.h1(ui.strong("Dashboard 📈")),
                   window_title="Yassir Dashboard")

    # Link to the external CSS file
    ui.tags.link(rel="stylesheet", href="styles.css")

    # Add main content
    ICONS = {
        "eta": icon_svg("clock"),
        "distance": icon_svg("route"),
        "ellipsis": icon_svg("ellipsis"),
    }

    # Define the target column
    target = 'eta'

    # Columns
    columns = train_df.columns.to_list()

    # Weather columns
    w_columns = weather_df.columns.to_list()

    # Numericals
    numericals = train_df.select_dtypes(include=['number']).columns.tolist()

    # Input range
    eta_rng = (min(train_df["eta"]), max(train_df["eta"]))
    trip_distance_rng = (min(train_df["trip_distance"]),
                         max(train_df["trip_distance"]))

    with ui.layout_sidebar():

        with ui.sidebar(open="desktop"):
            ui.tags.style("""
                .shiny-input-container input[type="range"] {
                    background: linear-gradient(to right, red, #4CAF50) no-repeat;
                    height: 8px;
                }
            """),
            ui.input_slider(
                "trip_distance",
                "Trip Distance",
                min=trip_distance_rng[0],
                max=trip_distance_rng[1],
                value=trip_distance_rng,
                post=" m",
            )
            ui.input_slider(
                "eta",
                "ETA",
                min=eta_rng[0],
                max=eta_rng[1],
                value=eta_rng,
                post=" sec",
            )

            ui.input_action_button("reset", "Reset filter")

        # KPIs
        with ui.layout_column_wrap(fill=False):
            with ui.value_box(showcase=ICONS["eta"], theme=BRANDTHEMES['purple-dark']):
                "ETA (Total)"

                @reactive.calc
                def total_eta():
                    return float(train_data().eta.sum())

                @render.text
                def ts():
                    return f"{round(total_eta()):,} s"

                @render.text
                def thms():
                    return full_time_sec_hms(total_eta())

            with ui.value_box(showcase=ICONS["eta"], theme=BRANDTHEMES['purple-dark']):
                "ETA (Median)"

                @reactive.calc
                def median_eta():
                    d = train_data()
                    m_eta = None
                    if d.shape[0] > 0:
                        m_eta = d.eta.median()
                    return m_eta

                @render.text
                def ms():
                    return f"{round(median_eta()):,} s"

                @render.text
                def mhms():
                    return time_sec_hms(median_eta())

            with ui.value_box(showcase=ICONS["distance"], theme=BRANDTHEMES['purple-light']):
                "TRIP DISTANCE (Total)"

                @reactive.calc
                def total_trip():
                    return float(train_data().trip_distance.sum())

                @render.text
                def tdkm():
                    return f"{total_trip()/1000:,.1f} km"

                @render.text
                def tdm():
                    return f"{total_trip():,.1f} m"

            with ui.value_box(showcase=ICONS["distance"], theme=BRANDTHEMES['purple-light']):
                "TRIP DISTANCE (Median)"

                @reactive.calc
                def median_trip():
                    d = train_data()
                    m_trip = None
                    if d.shape[0] > 0:
                        m_trip = d.trip_distance.median()
                    return m_trip

                @render.text
                def mtdkm():
                    return f"{median_trip()/1000:,.1f} km"

                @render.text
                def mtdm():
                    return f"{median_trip():,.1f} m"

        # Dataset view
        with ui.layout_column_wrap(fill=False):
            with ui.navset_card_pill(id="data_tab"):
                with ui.nav_panel("Train data"):
                    with ui.card(full_screen=True):
                        ui.card_header("Train data")

                        @render.data_frame
                        def train_table():
                            return render.DataGrid(train_data(), filters=True)

                with ui.nav_panel("Test data"):
                    with ui.card(full_screen=True):
                        ui.card_header("Test data")

                        @render.data_frame
                        def test_table():
                            return render.DataGrid(test_df)

                with ui.nav_panel("Weather data"):
                    with ui.card(full_screen=True):
                        ui.card_header("Weather data")

                        @render.data_frame
                        def weather_table():
                            return render.DataGrid(weather_df)

            value = "Explore the visualizations"
            with ui.accordion(id="plot_acc", open=value):
                with ui.accordion_panel(title=ui.strong(value), value=value):
                    with ui.navset_card_pill(id="eda_tab"):
                        with ui.nav_panel("Train features"):
                            with ui.card(full_screen=True):
                                ui.card_header(
                                    "Correlation in the Train features")

                                @render_plotly
                                def train_features():
                                    numeric_correlation_matrix = train_data()[
                                        numericals].corr()
                                    # Create heatmap trace
                                    heatmap_trace = go.Heatmap(
                                        z=numeric_correlation_matrix.values,
                                        x=numeric_correlation_matrix.columns,
                                        y=numeric_correlation_matrix.index,
                                        colorbar=dict(title='coefficient'),
                                        colorscale="Agsunset",
                                        texttemplate='%{z:.3f}',
                                    )

                                    # Create figure
                                    fig = go.Figure(data=[heatmap_trace])

                                    return fig

                        with ui.nav_panel("Trip distance vs Eta"):
                            with ui.card(full_screen=True):
                                ui.card_header(
                                    "Relationship between Trip Distance and ETA")

                                @render_plotly
                                def scatterplot():
                                    return px.scatter(
                                        train_data(),
                                        x='trip_distance',
                                        y='eta',
                                        trendline='ols',
                                        trendline_color_override=BRANDCOLORS["purple-light"],
                                        labels={
                                            'eta': 'Eta (seconds)', 'trip_distance': 'Trip Distance (meters)'},
                                    )

                        with ui.nav_panel("Distribution"):
                            with ui.card(full_screen=True):
                                ui.card_header("Distribution per train column")
                                with ui.popover(title="Choose a train column"):
                                    ICONS["ellipsis"]
                                    ui.input_radio_buttons(
                                        "train_col",
                                        "Select:",
                                        numericals,
                                        selected="eta",
                                        inline=True,
                                    )

                                @render_plotly
                                def distribution():
                                    column = input.train_col()
                                    data = train_data()
                                    fig1 = px.violin(data, x=column, box=True)

                                    fig2 = px.histogram(data, x=column)

                                    # Create a subplot layout with 1 row and 2 columns
                                    fig = make_subplots(rows=1, cols=2)

                                    # Add traces from fig1 to the subplot
                                    for trace in fig1.data:
                                        fig.add_trace(trace, row=1, col=1)

                                    # Add traces from fig2 to the subplot
                                    for trace in fig2.data:
                                        fig.add_trace(trace, row=1, col=2)

                                    # Update layout
                                    fig.update_layout(title_text=f"Distribution in the {column} column",
                                                      showlegend=True,
                                                      legend_title_text=target
                                                      )

                                    return fig

                        with ui.nav_panel("Weather features vs Eta"):
                            with ui.card(full_screen=True):
                                ui.card_header(
                                    "Relationship between weather features and Median Eta in seconds")
                                with ui.popover(title="Choose a feature column"):
                                    ICONS["ellipsis"]
                                    ui.input_radio_buttons(
                                        "weather_col",
                                        "Select:",
                                        [col for col in w_columns if col != "date"],
                                        selected="dewpoint_2m_temperature",
                                        inline=True,
                                    )

                                @render_plotly
                                def weather_eta():
                                    column = input.weather_col()

                                    fig = px.scatter(
                                        x=daily_weather_eta_df()[column],
                                        y=daily_weather_eta_df()[target],
                                    )

                                    # Update layout
                                    fig.update_layout(
                                        title_text=f"Distribution in the {column} column",
                                        showlegend=False
                                    )
                                    
                                    fig.update_xaxes(title_text=column)  # Set x-axis title
                                    fig.update_yaxes(title_text=target)  # Set y-axis title

                                    return fig

                value = "More Visualizations..."
                with ui.accordion_panel(title=ui.strong(value), value=value):
                    with ui.navset_card_pill(id="more_visualizations"):
                        with ui.nav_panel("Weather vs Eta"):
                            with ui.card(full_screen=True):
                                ui.card_header(
                                    "Weather vs Eta Features summary")

                                @render_plotly
                                def eta_weather_summary():
                                    daily_weather_eta_correlation_matrix = daily_weather_eta_df().corr().sort_values(by='eta')

                                    # Create heatmap trace
                                    heatmap_trace = go.Heatmap(
                                        z=daily_weather_eta_correlation_matrix[[
                                            'eta']].values,
                                        x=daily_weather_eta_correlation_matrix[[
                                            'eta']].columns,
                                        y=daily_weather_eta_correlation_matrix[[
                                            'eta']].index,
                                        colorbar=dict(title='Coefficient'),
                                        colorscale="Agsunset",
                                        texttemplate='%{z:.3f}',
                                    )

                                    # Create figure
                                    fig = go.Figure(data=[heatmap_trace])

                                    return fig

                        with ui.nav_panel("Top locations"):
                            with ui.card(full_screen=True):
                                ui.card_header(
                                    "Top 10 Most Common Origin and Destination Locations")
                                with ui.popover(title="Origin or Destination?"):
                                    ICONS["ellipsis"]
                                    ui.input_radio_buttons(
                                        "location_type",
                                        "Select:",
                                        ["origin", "destination"],
                                        selected="origin",
                                        inline=True,
                                    )

                                @render_plotly
                                def top_locations():
                                    location_type = input.location_type()
                                    top_10_origin, top_10_dest = top_bottom_location()[
                                        0]

                                    data = top_10_origin if location_type == "origin" else top_10_dest
                                    # Prepare data for origin locations
                                    data.sort_values(by='count', inplace=True)
                                    data['location'] = data.sort_values(by='count').apply(
                                        lambda row: f"({row[f'{location_type}_lat']}, {row[f'{location_type}_lon']})", axis=1)

                                    fig = go.Figure()

                                    fig.add_trace(
                                        go.Bar(
                                            x=data['count'],
                                            y=data['location'],
                                            orientation='h',
                                            marker=dict(
                                                color=BRANDCOLORS['purple-light'] if location_type == "origin" else BRANDCOLORS['red']),
                                            name=f'{location_type.title()} Locations'
                                        )
                                    )

                                    # Update layout
                                    fig.update_layout(
                                        xaxis_title=f'{location_type.title()} Locations',
                                        yaxis_title='Number of Rides',
                                        showlegend=False
                                    )

                                    return fig

                        with ui.nav_panel("Trips by hour"):
                            with ui.card(full_screen=True):
                                ui.card_header(
                                    "No of trips by Hour of the day")
                                with ui.popover(title="Average or Median?"):
                                    ICONS["ellipsis"]
                                    ui.input_radio_buttons(
                                        "trip_agg",
                                        "Select:",
                                        ["average", "median"],
                                        selected="median",
                                        inline=True,
                                    )

                                @render_plotly
                                def trips_by_hour():
                                    trip_agg = input.trip_agg()

                                    # Create a DataFrame with only the Timestamp column
                                    time_df = train_data()[['timestamp']]

                                    # Extract the hour from the timestamp and add it as a new column
                                    time_df = time_df.assign(hour=time_df['timestamp'].dt.hour)

                                    # Count the number of trips for each hour of the day
                                    tps = time_df['hour'].value_counts().sort_index().reset_index().rename(
                                        columns={'hour': 'Hour', 'count': f'{trip_agg.title()} number of Trips'})

                                    # Calculate the average number of trips per hour (Note: This line might not be necessary as 'trips_per_hour' already represents counts per hour)
                                    if trip_agg == "median":
                                        agg_trips_per_hour = tps.groupby(
                                            'Hour')[f'{trip_agg.title()} number of Trips'].median().reset_index()
                                    else:
                                        agg_trips_per_hour = tps.groupby(
                                            'Hour')[f'{trip_agg.title()} number of Trips'].mean().reset_index()

                                    # Plot count ETA by hour
                                    fig = px.line(
                                        agg_trips_per_hour, x='Hour', y=f'{trip_agg.title()} number of Trips')

                                    return fig

                        with ui.nav_panel("Mapping dataset locations"):
                            with ui.card(full_screen=True):
                                ui.card_header(
                                    "Map of locations in the train dataset")
                                
                                @render_plotly
                                def location_map():
                                    country, geojson, data = get_country_geojson()
                                    data['country'] = country
                                    fig = px.scatter_geo(
                                        data,
                                        locations='country',
                                        hover_name='country',
                                        geojson=geojson,
                                        fitbounds='geojson',
                                    )

                                    # Add longitude and latitude points
                                    fig.add_scattergeo(
                                        lon=data['longitude'],
                                        lat=data['latitude'],
                                        mode='markers',
                                        marker=dict(
                                            color=BRANDCOLORS["red"],
                                        ),
                                        name='Locations in dataset'
                                    )

                                    # Add annotation to the map
                                    fig.add_annotation(
                                        text=f"{country}",
                                        showarrow=False,
                                        font=dict(size=18),
                                        align="center"
                                    )

                                    fig.update_layout(
                                        title=f'Dataset locations in {country}',
                                        geo_scope='africa'
                                    )

                                    return fig

                
                value = "Model Explainer"
                with ui.accordion_panel(title=ui.strong(value), value=value):
                    with ui.navset_card_pill(id="model_explainer"):
                        with ui.nav_panel("Model Explainer..."):
                            with ui.card(full_screen=True):
                                ui.card_header("Coming Soon...")

                                ui.h3("Models")
                                @render.ui
                                def all_models():                                    
                                    return ui.tags.ul(
                                        [ui.tags.li(item) for item in ALL_MODELS]
                                    )

    # ui.include_css("styles.css")

    # --------------------------------------------------------
    # Reactive calculations and effects
    # --------------------------------------------------------

    @reactive.calc
    def train_data():
        trip_distances = input.trip_distance()
        idx1 = train_df.trip_distance.between(
            trip_distances[0], trip_distances[1])
        eta = input.eta()
        idx2 = train_df.eta.between(eta[0], eta[1])
        return train_df[idx1 & idx2]

    @reactive.calc
    def daily_weather_eta_df():
        # Select relevant columns from the training DataFrame
        time_eta_df = train_data()[['timestamp', 'eta']]

        # Extract the date from the timestamp
        time_eta_df = time_eta_df.assign(
            date=pd.to_datetime(time_eta_df['timestamp'].dt.date))

        # Prepare daily aggregated ETA data
        daily_eta_df = (
            time_eta_df
            # Remove the 'timestamp' column as it's no longer needed
            .drop(columns=['timestamp'])
            # Set 'date' as the index for resampling
            .set_index('date')
            # Resample the data on a daily frequency
            .resample('D')
            .median()                     # Compute the median ETA for each day
            .reset_index()                # Reset the index to include 'date' as a column
        )

        # Merge the daily ETA data with the weather data
        return (
            pd.merge(daily_eta_df, weather_df, left_on='date', right_on='date')
            .drop(columns=['date'])
        )

    @reactive.calc
    def top_bottom_location():
        # Group by origin locations and count occurrences
        origin_counts = train_data().groupby(['origin_lat', 'origin_lon'])[
            'origin_lon'].count().reset_index(name='count')

        # Sort by count in descending order
        top_origin = origin_counts.nlargest(10, columns=['count'])
        bottom_origin = origin_counts.nsmallest(10, columns=['count'])

        # Group by destination locations and count occurrences
        destination_counts = train_data().groupby(['destination_lat', 'destination_lon'])[
            'destination_lon'].count().reset_index(name='count')

        # Sort by count in descending order
        top_dest = destination_counts.nlargest(10, columns=['count'])
        bottom_dest = destination_counts.nsmallest(10, columns=['count'])

        return [(top_origin, top_dest), (bottom_origin, bottom_dest)]

    @reactive.effect
    @reactive.event(input.reset)
    def _():
        ui.update_slider("trip_distance", value=trip_distance_rng)
        ui.update_slider("eta", value=eta_rng)
