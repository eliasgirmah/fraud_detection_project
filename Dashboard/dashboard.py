import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import requests
import plotly.express as px

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Fraud Insights Dashboard"),

    html.Div(id='summary', style={'margin-bottom': '20px'}),
    
    dcc.Graph(id='fraud-trend-chart'),
    
    dcc.Graph(id='device-fraud-chart'),
    
    dcc.Graph(id='browser-fraud-chart'),
])

@app.callback(
    Output('summary', 'children'),
    Output('fraud-trend-chart', 'figure'),
    Output('device-fraud-chart', 'figure'),
    Output('browser-fraud-chart', 'figure'),
    Input('summary', 'children')
)
def update_dashboard(_):
    # Fetch summary statistics
    summary_response = requests.get('http://127.0.0.1:5000/api/summary').json()
    summary = f"Total Transactions: {summary_response['total_transactions']}, Total Fraud Cases: {summary_response['total_fraud_cases']}, Fraud Percentage: {summary_response['fraud_percentage']:.2f}%"

    # Fetch fraud trends
    trends_response = requests.get('http://127.0.0.1:5000/api/fraud_trends').json()
    trends_df = pd.DataFrame(trends_response)
    fraud_trend_fig = px.line(trends_df, x='date', y='fraud_cases', title='Fraud Cases Over Time')

    # Fetch device fraud data
    device_response = requests.get('http://127.0.0.1:5000/api/device_fraud').json()
    device_df = pd.DataFrame(device_response)
    device_fraud_fig = px.bar(device_df, x='device', y='fraud_cases', title='Fraud Cases by Device')

    # Fetch browser fraud data
    browser_response = requests.get('http://127.0.0.1:5000/api/browser_fraud').json()
    browser_df = pd.DataFrame(browser_response)
    browser_fraud_fig = px.bar(browser_df, x='browser', y='fraud_cases', title='Fraud Cases by Browser')

    return