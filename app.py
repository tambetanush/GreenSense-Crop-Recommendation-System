import os

import joblib
import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, render_template, request

from utils import RobustLocationImputer, clean_and_clip, feature_engineer

app = Flask(__name__)


# LOAD MODELS

try:
    preprocessor = joblib.load("preprocessing_pipeline.joblib")
    ranker = joblib.load("best_model_XGBRanker.joblib")
    mlb = joblib.load("MLB.joblib")
    crop_list = list(mlb.classes_)

    print("Model + Preprocessor loaded successfully")

except Exception as e:
    print("ERROR loading model:", e)
    preprocessor, ranker, mlb, crop_list = None, None, None, None


# RANKING FUNCTION
def predict_ranked(x_transformed):
    n_labels = len(crop_list)
    X_full = []

    for j in range(n_labels):
        onehot = np.eye(n_labels)[j]
        combined = np.concatenate([x_transformed[0], onehot])
        X_full.append(combined)

    X_full = np.array(X_full, dtype=np.float32)

    scores = ranker.predict(X_full)
    ranked_idx = np.argsort(-scores)

    return ranked_idx, scores


def softmax(x):
    x = np.array(x)
    e = np.exp(x - np.max(x))
    return e / e.sum()


# ROUTES
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/schema")
def schema():
    return jsonify(list(preprocessor.feature_names_in_))


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if preprocessor is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.get_json()
        df = pd.DataFrame([data])

        for col in df.columns:
            if col not in ["district", "location", "season"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Data Cleaning & Feature Engineering
        x_trans = preprocessor.transform(df).to_numpy()

        # Prediction & Ranking
        ranked_idx, scores = predict_ranked(x_trans)

        probs = softmax(scores)

        # threshold-based recommendation
        THRESHOLD = 0.05

        recommended = [
            {
                "crop": crop_list[i],
                "score": float(scores[i]),
                "probability": float(probs[i])
            }
            for i in ranked_idx
            if probs[i] >= THRESHOLD
        ]

        return jsonify({
            "recommendations": recommended
        })

    except Exception as e:
        print("PREDICTION ERROR:", e)
        return jsonify({"error": "Prediction failed", "details": str(e)}), 400


# GOOGLE SHEETS ENDPOINT
GOOGLE_SHEET_API = "https://script.google.com/macros/s/AKfycbyTpysfpeQIB3wNvym2Gk8cx_dPQtLW0cB48RO07K9LpPoWe2hl_iRPpjvWdeVWgmk/exec"


@app.route("/latest-readings", methods=["GET"])
def latest_readings():
    try:
        device_id = request.args.get("ID", "1")
        response = requests.get(GOOGLE_SHEET_API, params={"ID": device_id})

        if response.status_code != 200:
            return jsonify({"error": "Google Script error"}), 500

        return jsonify({
            "device_id": device_id,
            "records": response.json()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# RUN SERVER
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
