import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MultiLabelBinarizer
from xgboost import XGBClassifier

from utils import RobustLocationImputer, clean_and_clip, feature_engineer

# Initialize the Flask application
app = Flask(__name__)

# --- Load Machine Learning Assets ---
try:
    # Load the pre-trained models and the preprocessor pipeline
    preprocessor = joblib.load("preprocessing_pipeline.joblib")
    model = joblib.load("best_model_XGBClassifier.joblib")
    mlb = joblib.load("MLB.joblib")
    print("Models and pipeline loaded successfully!")
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    print("Please ensure the .joblib files are in the correct directory.")
    preprocessor, model, mlb = None, None, None

# Define the expected numeric columns for type conversion
NUMERIC_COLS = [
    'N_kg_per_ha', 'P_kg_per_ha', 'K_kg_per_ha', 'pH', 'soil_temp_c',
    'soil_humidity_percent', 'env_temp_c', 'env_humidity_percent',
    'env_pollution_ppm', 'env_gasses_co2_ppm', 'altitude_m',
    'light_intensity_lux', 'pressure_hpa'
]

# --- Application Routes ---


@app.route('/')
def home():
    """Renders the main page of the application."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request from the user."""
    if not all([preprocessor, model, mlb]):
        return jsonify({"error": "Server is not ready. Models not loaded."}), 500

    try:
        # Get data from the POST request
        data = request.get_json()

        # Convert dictionary to a DataFrame
        features_df = pd.DataFrame([data])

        # Ensure numeric columns are of float type, handling potential errors
        for col in NUMERIC_COLS:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce')

        # Use the preprocessing pipeline to transform the input data
        transformed_data = preprocessor.transform(features_df)

        # Make a prediction
        prediction_encoded = model.predict(transformed_data)

        # Decode the prediction back to crop names
        prediction_labels = mlb.inverse_transform(prediction_encoded)

        # Flatten the list of tuples into a simple list of crop names
        final_prediction = [
            crop for crops_tuple in prediction_labels for crop in crops_tuple]

        # Return the prediction as a JSON response
        return jsonify({'prediction': final_prediction})

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return jsonify({"error": "An error occurred during prediction."}), 400


# --- Main Execution ---
# if __name__ == '__main__':
#     app.run(debug=True)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
