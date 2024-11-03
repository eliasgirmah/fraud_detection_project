from flask import Flask, request, jsonify, render_template
import joblib  # Add this line to import joblib

app = Flask(__name__)

import joblib
import os

# Example of setting a specific path
model_path = os.path.join(os.path.dirname(__file__), 'best_fraud_detection_model.joblib')
model = joblib.load(model_path)


@app.route('/')
def home():
    return render_template('index.html')  # Ensure 'index.html' is in a 'templates' folder

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Example feature extraction
    features = [data['feature1'], data['feature2'], data['feature3']], data['feature4']
    prediction = model.predict([features])[0]
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
