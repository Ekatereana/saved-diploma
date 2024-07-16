import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
from sklearn.manifold import TSNE
from helpers import CHAT_FILE_PATH
from components import populate_layout, register_callbacks
from ml import preprocess_data

pre_df = preprocess_data(CHAT_FILE_PATH)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SUPERHERO])     
app.layout = populate_layout(pre_df)
register_callbacks(app, pre_df)

if __name__ == "__main__":
    app.run_server(debug=True)