from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('food_demand.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = [data.get('homepage_featured', None), 
                data.get('emailer_for_promotion', None), 
                data.get('meal_id', None), 
                data.get('center_id', None)]
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
