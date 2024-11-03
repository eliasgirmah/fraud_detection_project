from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the dataset
data = pd.read_csv('./data/Fraud_Data.csv')
data.columns = data.columns.str.strip()  # Remove leading/trailing spaces

# Convert purchase_time to datetime
data['purchase_time'] = pd.to_datetime(data['purchase_time'], errors='coerce')

@app.route('/api/summary', methods=['GET'])
def summary():
    total_transactions = len(data)
    total_fraud_cases = data[data['class'] == 1].shape[0]  # Assuming 'class' 1 indicates fraud
    fraud_percentage = (total_fraud_cases / total_transactions) * 100 if total_transactions > 0 else 0

    return jsonify({
        'total_transactions': total_transactions,
        'total_fraud_cases': total_fraud_cases,
        'fraud_percentage': fraud_percentage
    })

@app.route('/api/fraud_trends', methods=['GET'])
def fraud_trends():
    # Group by purchase_time and count fraud cases
    fraud_trend = data[data['class'] == 1].groupby(data['purchase_time'].dt.date)['class'].count().reset_index()
    fraud_trend.columns = ['date', 'fraud_cases']
    return fraud_trend.to_json(orient='records')

@app.route('/api/device_fraud', methods=['GET'])
def device_fraud():
    device_fraud_counts = data[data['class'] == 1]['device_id'].value_counts().reset_index()
    device_fraud_counts.columns = ['device_id', 'fraud_cases']
    return device_fraud_counts.to_json(orient='records')

@app.route('/api/browser_fraud', methods=['GET'])
def browser_fraud():
    browser_fraud_counts = data[data['class'] == 1]['browser'].value_counts().reset_index()
    browser_fraud_counts.columns = ['browser', 'fraud_cases']
    return browser_fraud_counts.to_json(orient='records')

if __name__ == '__main__':
    app.run(debug=True)