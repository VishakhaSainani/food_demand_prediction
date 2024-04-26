from flask import Flask, request, jsonify,render_template
# from flask render_template

import numpy as np
import joblib  # Import joblib if you saved your model using joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('food_demand.pkl')  # Replace 'your_trained_model.joblib' with the path to your saved model file
# @app.route('/')
# def index():
#     return render_template('./index.html')
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        homepage_featured = int(data.get('homepage', 0))  # default to 0 if not provided
        emailer_for_promotion = int(data.get('emailpromo', 0))  # default to 0 if not provided
        meal_id = data.get('mealid', '')
        center_id = data.get('centerid', '')

        # Validate inputs
        if not meal_id or not center_id:
            return jsonify({'error': 'Meal ID and Center ID are required.'}), 400

        # Prepare features
        features = np.array([homepage_featured, emailer_for_promotion]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)

        return jsonify({'predicted_orders': prediction.tolist()}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
