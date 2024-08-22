from pathlib import Path

from shiny import experimental, Inputs, Outputs, Session
from shiny.express import ui, module

from .config import BRANDCOLORS


@module
def home_page(input: Inputs, output: Outputs, session: Session):    
    dir = Path(__file__).resolve().parent.parent
    
    ui.panel_title(title=ui.h2(ui.strong("Welcome to Yassir Data App!")),
                window_title="Yassir Home",)

    
    with ui.layout_columns(col_widths=(4, 8)):
        with ui.card():
            ui.tags.p(
                {"class": "dynamic-paragraph"},
                ui.span({"class": "dropcap"}, "Y"),
                ui.strong("assir"),
                " is the leading super App in the Maghreb region set to changing the way daily services are provided. It currently operates in 45 cities across Algeria, Morocco and Tunisia with recent expansions into France, Canada and ",
                ui.strong("Sub-Saharan Africa"),
                ". It is backed (~$200M in funding) by VCs from Silicon Valley, Europe and other parts of the world. Offering on-demand services such as ",
                ui.strong("ride-hailing"),
                " and last-mile delivery."
            )
            ui.tags.style(f"""
                .dynamic-paragraph {{
                    font-size: calc(1vw + 1vh); /* Adjust this value to control the text size */
                    width: 100%;
                    height: 100%;
                    overflow: hidden;
                }}
                .dropcap {{
                    float: left;
                    font-size: 4em;
                    line-height: 0.8;
                    margin-right: 8px;
                    font-weight: bold;
                }}
                .red-text {{
                    color: {BRANDCOLORS['red']};
                    font-weight: bold;
                }}
                .purple-text {{
                    color: {BRANDCOLORS['purple-light']};
                    font-weight: bold;
                }}
                .no-decoration a {{
                    display: inline-block;
                    padding: 10px 20px;
                    margin: 10px 0;
                    background-color: {BRANDCOLORS['purple-light']}; 
                    font-size: calc(1vw + 1vh);
                    color: inherit;
                    text-decoration: none; 
                    font-weight: bold; 
                    border-radius: 5px;
                    transition: background-color 0.3s, transform 0.3s;
                }}
            """)

        experimental.ui.card_image(file=dir / "www/ride-hailing-hero.webp")
    

    with ui.layout_columns(col_widths=(8, 4)):
        with ui.card():
            ui.tags.p(
                {"class": "dynamic-paragraph"},
                """In the fast-paced world of ride-hailing, accuracy and reliability are key. 
                    For companies like Yassir, the ability to 
                """,
                ui.strong("predict the estimated time of arrival (ETA)"),
                " for trips in ",
                ui.strong("real-time"),
                " is crucial. Our mission is to enhance the Yassir experience by leveraging ",
                ui.strong("data"),
                " and ",
                ui.strong("advanced machine learning algorithms"),
                """ to deliver more accurate ETAs, making our services more dependable and 
                    attractive to both customers and business partners.
                """,
                ui.br(),
                """By improving ETA predictions, Yassir not only boosts customer
                    satisfaction but also optimizes operational efficiency, allowing the company
                    to allocate resources more effectively and reduce costs. 
                """,
                ui.br(),
                """This data app is a step forward in revolutionizing how Yassir operates, 
                    ensuring that every journey is smooth and timely. It contributes to this vision by 
                    predicting the ETA at the drop-off point for one Yassir journey at a time. 
                """,
                ui.br(),
                ui.span(
                    {"class": "purple-text"},
                    """Join us in shaping the future of ride-hailing services with our innovative data application!"""
                )
            )
            ui.card_footer(
                ui.strong("Going somewere? "),
                ui.span(
                    {"class": "red-text"},
                    ui.span(
                        {"class": "no-decoration"},
                        ui.a("Yassir", href="https://yassir.com/ride-hailing", target="_blank")
                    ),
                    """ will take you there!"""                    
                )
            )

        experimental.ui.card_image(file=dir / "www/image-2-1.webp")
