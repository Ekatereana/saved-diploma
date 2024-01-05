import dash
from dash import dcc, html, Input, Output, callback
import matplotlib.colors as mcolors
import dash_bootstrap_components as dbc
from sklearn.manifold import TSNE
from helpers import ngrams_info, preprocess_data, stop_words, author_dict, extract_topics, CHAT_FILE_PATH, JAVA_KPI_FILE_PATH, build_ngrams_per_author
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud
import plotly.graph_objs as go
import re
from datetime import datetime
from dateutil import relativedelta
import itertools
import dash_cytoscape as cyto

col_pal = px.colors.sequential.Pinkyl
# vects_df = ngrams_info(pre_df['text'], n=2) 
pre_df = preprocess_data(JAVA_KPI_FILE_PATH)
unique_owns_by_values = pre_df['owns_by'].unique()


# Initialize an empty list to store the results
bigrams_per_author = []

# Call ngrams_info for each unique 'owns_by' value
for author in unique_owns_by_values:
    author_df = pre_df[pre_df['owns_by'] == author]
    bigrams_info = ngrams_info(author_df['text'], n=2)
    bigrams_per_author.append((author, bigrams_info)) 

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SUPERHERO])

navbar = dbc.Navbar(
                    children=[
                        html.A(
                            dbc.Row(
                                [
                                    dbc.Col(html.Img(src='./assets/web.png', height="55px"), style={"marginLeft": "15px"}),
                                    dbc.Col(
                                        dbc.NavbarBrand("Мультилінгвістичний NLP класифікатор", className="ml-2"),
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
                                        html.Label("Відсоток даних, що включено в навчання", className="lead"),
                                        html.P(
                                            "(Чим менше тим швидше. Вищий відсоток буде мати більшу точність)",
                                            style={"fontSize": 10, "font-weight": "lighter"},
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
        dbc.CardHeader(html.H5("Кругова діаграма за классами")),
        dbc.CardBody(
            [
                dbc.Label("Select a Class:"),
                dbc.Select(
                    id="class-dropdown",
                    options=[
                        {"label": "Автор", "value": "owns_by"},
                        {"label": "Клас повідомлення", "value": "message_class"},
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
        dbc.CardHeader(html.H5("Розподіл тем по датасету")),
        dbc.CardBody(
            [ 
                dbc.Row(
                    [
                        dbc.Col(
                        [   dbc.Label("Введіть бажану кількість топіків"),
                            dbc.Input(id="topic-amount-input", placeholder="Ввід...", type="number", value = 10),
                        ]), 
                        html.Br(),
                        dbc.Col(
                        [   dbc.Label("Введіть кількість визначних ознак для теми"),
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
# ngrams_per_author = dbc.Card(
#     [
#         dbc.CardHeader(html.H5("Визначні ngram-и за автором")),
#         dbc.CardBody(
#                         [
#                             dcc.Loading(
#                                 id="loading-ngrams-per-author",
#                                 children=[
#                                     dbc.Row(
#                                             [
                                                
#                                                  dbc.Col(
#                                                         [
#                                                             dbc.Label(html.P("Оберіть авторa для порівняння:")),
#                                                             dcc.Dropdown(
#                                                                 id="author",
#                                                                 options=[
#                                                                     {"label": i, "value": i}
#                                                                     for i in pre_df['owns_by'].unique()
#                                                                 ],
#                                                                 value = 1,
#                                                             )
#                                                         ],
#                                                         md=6,
#                                                         style = {"marginBottom": "20px"}
#                                                     ),
#                                                     dbc.Col(
#                                                         [
#                                                             dbc.Label(html.P("Введіть розряд ngram-и:")),
#                                                             dbc.Input(
#                                                                 id="ngram",
#                                                                 type="number",
#                                                                 value = 2,
#                                                             )
#                                                         ],
#                                                         md=6,
#                                                         style = {"marginBottom": "20px"}
#                                                     ),
#                                                 ],

#                                                 align = "center",
#                                             ),
#                                             dcc.Graph(id="ngrams_per_author"),

#                                 ])])
#     ],
#     style={"marginBottom": "20px"}
#     )
compare_authors_by_bigrams = dbc.Card(
                        [
                            dbc.CardHeader(html.H5("Порівняльна діаграма біграм характерних для авторів")),
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

app.layout = html.Div([
        navbar,
        dbc.Container([
            persentage_of_data_set,
            pie_chart_card,
            topics_chart,
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                         [
                            dbc.CardHeader(html.H5("Найбільш часто використовувані слова")),
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
                                                                    label="Карта слів",
                                                                    children=[
                                                                        dcc.Loading(
                                                                        id="loading-treemap",
                                                                        children = [dcc.Graph(id="bank-treemap")],
                                                                        type="default")]
                                                                
                                                            ),
                                                            dcc.Tab(
                                                                label="Хмара слів",
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
        

@app.callback(
    Output("pie-chart", "figure"),
    Input("class-dropdown", "value")
)
def update_pie_chart(selected_class):
    anonimized = {}
    for author, category in author_dict.items():
        anonimized[category] = f"Автор {category}"

    if selected_class == 'owns_by':
        expl_lables = anonimized
        label = 'Автор'
    else:
        expl_lables = {
            0: "plain message, only text", 
            1: "sticker",
            2: "annimation",
            3: "video_message", 
            4: "voice_message",
            5: "video_file",
            6:"photo",
            1000: "unknown type"
        }
        label = 'Тип повідомлення'

    
    data = pre_df[selected_class].value_counts()
    sum_value = data.sum()
    data = data.T   
    
    labels=[]
    fracs=[]
    explode=[]
    
    for index,value in data.items():
        labels.append('{0}.{1} - {2:.2f} % ({3} екземплярів)'.format(index,expl_lables[index],(value/sum_value*100),value))
        fracs.append(value)
        explode.append(0.1)


    fig = px.pie(
        names=labels,
        values=fracs,
        title=f"Діаграма розподілу на класи за {label}"
    )
    fig.update_traces(textposition='inside', marker=dict(colors=col_pal))
    fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
    
    return fig


def plotly_wordcloud(data_frame):
    """A wonderful function that returns figure data for three equally
    wonderful plots: wordcloud, frequency histogram and treemap"""
    complaints_text = list(data_frame["text"].dropna().values)

    if len(complaints_text) < 1:
        return {}, {}, {}

    # join all documents in corpus
    text = " ".join(list(complaints_text))
    text=re.sub(r"""['’"`�]""", '', text)
    text=re.sub(r"""[A-z]""", ' ', text)
    text=re.sub(r"""([0-9])([\u0400-\u04FF]|[A-z])""", ' ', text)
    text=re.sub(r"""([\u0400-\u04FF]|[A-z])([0-9])""", ' ', text)
    text=re.sub(r"""[\-.,:+*/_]""", ' ', text)

    word_cloud = WordCloud(stopwords=set(stop_words), max_words=100, max_font_size=90)
    word_cloud.generate(text)

    word_list = []
    freq_list = []
    fontsize_list = []
    position_list = []
    orientation_list = []

    for (word, freq), fontsize, position, orientation, color in word_cloud.layout_:
        word_list.append(word)
        freq_list.append(freq)
        fontsize_list.append(fontsize)
        position_list.append(position)
        orientation_list.append(orientation)

    # get the positions
    x_arr = []
    y_arr = []
    for i in position_list:
        x_arr.append(i[0])
        y_arr.append(i[1])

    # get the relative occurence frequencies
    new_freq_list = []
    for i in freq_list:
        new_freq_list.append(i * 80)

    trace = go.Scatter(
        x=x_arr,
        y=y_arr,
        textfont=dict(size=new_freq_list, color=col_pal),
        hoverinfo="text",
        textposition="top center",
        hovertext=["{0} - {1}".format(w, f) for w, f in zip(word_list, freq_list)],
        mode="text",
        text=word_list
    )

    layout = go.Layout(
        {
            "xaxis": {
                "showgrid": False,
                "showticklabels": False,
                "zeroline": False,
                "automargin": True,
                "range": [-100, 250],
            },
            "yaxis": {
                "showgrid": False,
                "showticklabels": False,
                "zeroline": False,
                "automargin": True,
                "range": [-100, 450],
            },
            "margin": dict(t=20, b=20, l=10, r=10, pad=4),
            "hovermode": "closest",
        }
    )

    wordcloud_figure_data = {"data": [trace], "layout": layout}
    word_list_top = word_list[:25]
    word_list_top.reverse()
    freq_list_top = freq_list[:25]
    freq_list_top.reverse()

    frequency_figure_data = {
        "data": [
            {
                "y": word_list_top,
                "x": freq_list_top,
                "type": "bar",
                "name": "",
                "orientation": "h",
                "marker": dict(colors=col_pal)
            }
        ],
        "layout": {"height": "550", "margin": dict(t=20, b=20, l=100, r=20, pad=4)},
    }
    treemap_trace = go.Treemap(
        labels=word_list_top, parents=[""] * len(word_list_top), values=freq_list_top,
        marker_colors = col_pal
    )
    treemap_layout = go.Layout({"margin": dict(t=10, b=10, l=5, r=5, pad=4)})
    treemap_figure = {"data": [treemap_trace], "layout": treemap_layout}
    
    return wordcloud_figure_data, frequency_figure_data, treemap_figure

@callback(
   
        Output("bank-wordcloud", "figure"),
        Output("frequency_figure", "figure"),
        Output("bank-treemap", "figure"),
        Output("no-data-alert", "style"),
        [Input("n-selection-slider", "value")],
)
def update_wordcloud_plot(value):
    """ Callback to rerender wordcloud plot """
    # local_df = make_local_df(value_drop, time_values, n_selection)
    wordcloud, frequency_figure, treemap = plotly_wordcloud(pre_df)
    alert_style = {"display": "none"}
    if (wordcloud == {}) or (frequency_figure == {}) or (treemap == {}):
        alert_style = {"display": "block"}
    print("redrawing bank-wordcloud...done")
    return (wordcloud, frequency_figure, treemap, alert_style)

def vector_to_string(vector):
    return '_'.join(map(str, vector))


# @app.callback(
#     Output("ngrams_per_author", "figure"),
#     [Input("ngram", "value"), Input("author", "value")],
# )
# def comp_ngrams_per_author(ngram, author):
#     ngrams_df = build_ngrams_per_author(author, ngram);
#     print(ngrams_df.head(10))
#     fig = px.line(ngrams_df, x="ngram", y="count", color='author')
#     return fig


@app.callback(
    Output("bigrams-comps", "figure"),
    [Input("bigrams-comp_1", "value"), Input("bigrams-comp_2", "value")],
)
def comp_bigram_comparisons(comp_first, comp_second):
    comp_list = [comp_first, comp_second]
    # Filter and concatenate bigrams for the selected authors
    selected_bigrams =  pd.DataFrame(columns=['owns_by','ngram', 'count'])
    for author, bigrams_info in bigrams_per_author:
        if author in comp_list:
            bigrams_info['owns_by'] = author
            selected_bigrams = pd.concat([selected_bigrams, bigrams_info], ignore_index=True)
   
    # Create a DataFrame from the selected bigrams
    temp_df = pd.DataFrame(selected_bigrams, columns=['owns_by','ngram', 'count'])
    # Negate the "count" for comp_second to show a comparison
    temp_df.loc[temp_df['owns_by'] == comp_second, 'count'] = -temp_df[temp_df['owns_by'] == comp_second]['count'].values
    
    temp_df['ngram'] = temp_df['ngram'].apply(vector_to_string)

    fig = px.bar(
        temp_df,
        title="Comparison: Автор " + str(comp_first) + " | Автор " + str(comp_second),
        x="ngram",
        y="count",
        color="owns_by",
        template="plotly_white",
        color_discrete_sequence=col_pal,
        labels={"owns_by": "Автор:", "ngram": "N-Gram"},
    )
    fig.update_layout(legend=dict(x=0.1, y=1.1), legend_orientation="h")
    fig.update_yaxes(title="", showticklabels=False)
    fig.data[0]["hovertemplate"] = fig.data[0]["hovertemplate"][:-14]
    return fig

def topic_heatmap(topics_df, n_topics):

    user_topic_counts = pd.pivot_table(data=topics_df, 
                                   values='text', 
                                   index='from', 
                                   columns='topic', 
                                   aggfunc='count',
                                   fill_value=0)

    user_topic_counts.columns = ['Topic {}'.format(i) for i in range(n_topics)]

        # add column to sum total topics 
    user_topic_counts['total_topics'] =  user_topic_counts.sum(axis=1)

    # convert topic counts to percentages for each news source 
    user_topic_counts_pct =  user_topic_counts.apply(lambda x: (x / user_topic_counts['total_topics']))
    user_topic_counts_pct = user_topic_counts_pct.drop(columns=['total_topics'])

    # store value z-values
    z_usr = user_topic_counts_pct.values.tolist()

    # create list of hover text template strings for each z-value in matrix 
    topic_names=topics_df.topic.tolist()
    hovertext_usr = []
    for yi, yy in enumerate(user_topic_counts_pct.index.tolist()):
        hovertext_usr.append(list())
        for xi, xx in enumerate(topic_names):
            hovertext_usr[-1].append('<b>Topic:</b> {}<br />'
                                    '<b>User:</b> {}<br />'
                                    '<b>Message Proportion:</b> {}'.format(xx, yy, z_usr[yi][xi]))

        # plot heatmap
    fig = px.imshow(user_topic_counts_pct, 
                    color_continuous_scale="bluyl",
                    width=650,
                    height=500,
                    aspect="auto")
            
    fig.update_layout(
        margin=dict(l=20, 
                    r=0,  
                    b=20, 
                    t=10,
                    pad=3),
        coloraxis=dict(colorbar=dict(thickness=15,
                                    xpad=2)))

    fig.update_traces(
        hovertemplate=None, # set this to None in order to use custom hover text       
        text=hovertext_usr,
        hoverinfo="text") 

    return fig

    

@app.callback(
    Output("topics_features", "figure"),
    # Output("topics_heatmap", "figure"),
    [Input("topic-amount-input", "value"),
     Input("feature-amount-input", "value"),
    ],
)
def comp_topic_chart(topics, features):
    TOPIC, FEATURES = extract_topics(pre_df, topics, features)
    data = TOPIC['topic'].value_counts()
    expl_lables = [f'Topic {i}' for i in range(topics)]
    sum_value = data.sum()
    data = data.T   
    
    labels=[]
    fracs=[]
    explode=[]
    
    for index,value in data.items():
        labels.append('{0}.{1} - {2:.2f} % ({3} екземплярів)'.format(index,expl_lables[index],(value/sum_value*100),value))
        fracs.append(value)
        explode.append(0.1)


    fig = px.pie(
        names=labels,
        values=fracs,
        title=f"Діаграма розподілу на класи за Топіком"
    )
    fig.update_traces(textposition='inside', marker=dict(colors=col_pal))
    fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')

    # heatmap = topic_heatmap(TOPIC, topics)
    
    return fig

if __name__ == "__main__":
    app.run_server(debug=True)