from dash import Input, Output
from helpers import ngrams_info, STOP_WORDS, author_dict, extract_topics
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import plotly.graph_objs as go
import re

col_pal = px.colors.sequential.Pinkyl

def register_callbacks(app, pre_df):

    def populate_bigrams_per_author(pre_df):
        # Initialize an empty list to store the results
        bigrams_per_author = []

        unique_owns_by_values = pre_df['owns_by'].unique()

        # Call ngrams_info for each unique 'owns_by' value
        for author in unique_owns_by_values:
            author_df = pre_df[pre_df['owns_by'] == author]
            bigrams_info = ngrams_info(author_df['text'], n=2)
            bigrams_per_author.append((author, bigrams_info)) 
        return bigrams_per_author

    bigrams_per_author = populate_bigrams_per_author(pre_df)
    
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

        word_cloud = WordCloud(stopwords=set(STOP_WORDS), max_words=100, max_font_size=90)
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
                    "color": "yellow",
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

    @app.callback(
    
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
