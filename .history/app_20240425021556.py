from flask import Flask, request, jsonify
import numpy as np
import joblib  # Import joblib if you saved your model using joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('food_demand.pkl')  # Replace 'your_trained_model.joblib' with the path to your saved model file

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array([data['homepage_featured'], data['emailer_for_promotion']]).reshape(1, -1)  # Assuming these are the features required for prediction
    prediction = model.predict(features)
    return jsonify(prediction.tolist())

if __name__ == '__main__':
    app.run(debug=True)
