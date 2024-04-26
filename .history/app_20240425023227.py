from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('food_demand.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Extract features from the request data
    features = [data['homepage_featured'], data['emailer_for_promotion'], data['meal_id'], data['center_id']]
    # Convert features to numpy array
    features_array = np.array(features).reshape(1, -1)
    # Make prediction
    prediction = model.predict(features_array)
    # Return prediction
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
