from dash import dcc, html
import dash_bootstrap_components as dbc


def populate_layout(pre_df):
    navbar = dbc.Navbar(
                        children=[
                            html.A(
                                dbc.Row(
                                    [
                                        dbc.Col(html.Img(src='./assets/web.png', height="55px"), style={"marginLeft": "15px"}),
                                        dbc.Col(
                                            dbc.NavbarBrand("Multilingual NLP classifier", className="ml-2"),
                                            style={"marginLeft": "5px"}
                                        ),
                                    ],
                                    align="center",
                                ),
                                href="https://plot.ly",
                            )
                        ],
                        sticky="top",
                        dark=True,
                        color="orange",
                        class_name="navbar navbar-expand-lg bg-primary"
                    )
    persentage_of_data_set = dbc.Card(
                    [dbc.CardBody(
                            dbc.Row(
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.Label("Training data in processing", className="lead"),
                                            html.P(
                                                "(Lesser -- quicker, but accurancy also will be affected)",
                                                style={"fontSize": 10, "fontWeight": "lighter"},
                                            ),
                                            dcc.Slider(
                                                id="n-selection-slider",
                                                min=1,
                                                max=100,
                                                step=1,
                                                marks={
                                                    0: "0%",
                                                    10: "",
                                                    20: "20%",
                                                    30: "",
                                                    40: "40%",
                                                    50: "",
                                                    60: "60%",
                                                    70: "",
                                                    80: "80%",
                                                    90: "",
                                                    100: "100%",
                                                },
                                                value=20,
                                            ),
                                        ]
                                    ),
                                    width={"size": 12},
                                ),
                                style={"marginTop": 30, "marginBottom": "20px"},
                            ),
                    )],
                    style={"marginBottom": "20px", "marginTop": "20px"} 
                )
    pie_chart_card = dbc.Card(
        [
            dbc.CardHeader(html.H5("Extract classes of the data")),
            dbc.CardBody(
                [
                    dbc.Label("Select a Class:"),
                    dbc.Select(
                        id="class-dropdown",
                        options=[
                            {"label": "Author", "value": "owns_by"},
                            {"label": "Message type", "value": "message_class"},
                        ],
                        value="owns_by",  # Default selection
                        
                    ),
                    dbc.Row(
                        
                        dcc.Graph(id="pie-chart"),
                        style={"marginTop": "20px"} 
                    ),
                    
                ],
                style={"marginBottom": "20px"} 
            ),
        ],
        style={"marginBottom": "20px"}  # Add space between cards
    )
    topics_chart = dbc.Card(
        [
            dbc.CardHeader(html.H5("Extract topics from dataset")),
            dbc.CardBody(
                [ 
                    dbc.Row(
                        [
                            dbc.Col(
                            [   dbc.Label("Enter the desired topic count"),
                                dbc.Input(id="topic-amount-input", placeholder="Ввід...", type="number", value = 10),
                            ]), 
                            html.Br(),
                            dbc.Col(
                            [   dbc.Label("Specify the features  to be extracted for topic"),
                                dbc.Input(id="feature-amount-input", placeholder="Ввід...", type="number", value = 5),
                            ]), 
                        ]
                    ),
                    dbc.Row(
                        [dbc.Col(dcc.Graph(id="topics_features")),
                        # dbc.Col(dcc.Graph(id="topics_heatmap"))
                        ],
                        style={"marginTop": "20px"} 
                    ),
                    
                ],
                style={"marginBottom": "20px"} 
            ),
        ],
        style={"marginBottom": "20px"}  # Add space between cards
    )
    compare_authors_by_bigrams = dbc.Card(
                            [
                                dbc.CardHeader(html.H5("Most important features of the authors")),
                                dbc.CardBody(
                                    [
                                        dcc.Loading(
                                            id="loading-bigrams-comps",
                                            children=[
                                                dbc.Alert(
                                                    "Something's gone wrong! Give us a moment, but try loading this page again if problem persists.",
                                                    id="no-data-alert-bigrams_comp",
                                                    color="warning",
                                                    style={"display": "none"},
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Col(html.P("Оберіть авторів для порівняння:"), md=12),
                                                        dbc.Col(
                                                            [
                                                                dcc.Dropdown(
                                                                    id="bigrams-comp_1",
                                                                    options=[
                                                                        {"label": i, "value": i}
                                                                        for i in pre_df['owns_by'].unique()
                                                                    ],
                                                                    value = 1,
                                                                )
                                                            ],
                                                            md=6,
                                                            style = {"marginBottom": "20px"}
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                dcc.Dropdown(
                                                                    id="bigrams-comp_2",
                                                                    options=[
                                                                        {"label": i, "value": i}
                                                                        for i in pre_df['owns_by'].unique()
                                                                    ],
                                                                    value = 0,
                                                                )
                                                            ],
                                                            md=6,
                                                            style = {"marginBottom": "20px"}
                                                        ),
                                                    ],
                                                    align = "center",
                                                ),
                                                dcc.Graph(id="bigrams-comps"),
                                            ],
                                            type="default",
                                        )
                                    ],
                                    style={"marginTop": 0, "marginBottom": 0},
                                ),
                            ],
                            style={"marginBottom": "20px"}

                        )
    layout = html.Div([
            navbar,
            dbc.Container([
                persentage_of_data_set,
                pie_chart_card,
                topics_chart,
                dbc.Row(
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(html.H5("Words that are most frequantly used")),
                                dbc.Alert(
                                    "Not enough data to render these plots, please adjust the filters",
                                    id="no-data-alert",
                                    color="warning",
                                    style={"display": "none"},
                                ),
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dcc.Loading(
                                                        id="loading-frequencies",
                                                        children=[dcc.Graph(id="frequency_figure")],
                                                        type="default",
                                                    )
                                                ),
                                                dbc.Col(
                                                    [
                                                        dcc.Tabs(
                                                            id="tabs",
                                                            children=[
                                                            
                                                                        dcc.Tab(
                                                                        label="Words Map",
                                                                        children=[
                                                                            dcc.Loading(
                                                                            id="loading-treemap",
                                                                            children = [dcc.Graph(id="bank-treemap")],
                                                                            type="default")]
                                                                    
                                                                ),
                                                                dcc.Tab(
                                                                    label="Word Cloud",
                                                                    children=[
                                                                        dcc.Loading(
                                                                            id="loading-wordcloud",
                                                                            children=[
                                                                                dcc.Graph(id="bank-wordcloud")
                                                                            ],
                                                                            type="default",
                                                                        )
                                                                    ],
                                                                ),
                                                            ],
                                                        )
                                                    ],
                                                    md=8
                                                ),
                                            ]
                                        )
                                    ]
                                ),
                            ],
                            style={"marginBottom": "20px"}
                        )
                    )
                ),
                # ngrams_per_author,
                compare_authors_by_bigrams,
            ],
            className="mt-12",
            )
        ]
    )
    return layout