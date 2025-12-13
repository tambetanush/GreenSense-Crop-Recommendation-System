# 🌱 GreenSense

### Intelligent Crop Recommendation System (2024–2025)

---

GreenSense is an **end-to-end intelligent crop recommendation system** that integrates **IoT sensor data**, **machine learning ranking models**, and a **production-ready Flask API** to recommend the most suitable crops for a given soil and environmental condition.

Unlike traditional classification systems, GreenSense formulates **crop selection as a ranking problem**, enabling more realistic and flexible recommendations for farmers and agronomists.

---

## 🚀 Key Highlights

* 📡 **Real-time IoT sensing** using ESP32 and environmental sensors
* 🧠 **Learning-to-Rank ML approach** using `XGBoost Ranker`
* 🔬 Combines **synthetic + real sensor data**
* ⚙️ **Robust feature engineering & preprocessing pipeline**
* 🌐 **Flask-based inference API** with real-time recommendations
* 📊 Evaluation using **Hit@K** and **NDCG**
* 🔗 Live integration with **Google Sheets API** for sensor readings

---

## 🧠 Problem Formulation

Most crop recommendation systems treat the problem as **multi-class classification**, forcing a single “best” crop.

**GreenSense instead models this as a ranking problem**, allowing:

* Multiple suitable crops per condition
* Confidence-aware recommendations
* Better real-world decision support

This is achieved using **XGBRanker**, where each crop is ranked against others based on predicted suitability.

---

### Pipeline Overview

```
IoT Sensors (ESP32)
        ↓
Google Sheets (Live Logging)
        ↓
Flask API
        ↓
Preprocessing Pipeline
        ↓
XGBRanker (Learning-to-Rank)
        ↓
Ranked Crop Recommendations
```

---

## 🛠️ Tech Stack

### Hardware

* ESP32 Microcontroller
* Soil Moisture Sensor
* Temperature & Humidity Sensors
* Environmental Sensors

### Backend & ML

* Python
* Flask
* Scikit-Learn
* XGBoost (XGBRanker)
* NumPy, Pandas
* Joblib

### Data & Integration

* Google Sheets API
* REST APIs
* JSON-based inference

---

## 📂 Project Structure

```
GreenSense/
│
├── app.py                         # Flask application (API + inference)
├── utils.py                       # Preprocessing & feature engineering utilities
├── test.py                        # Local testing script
├── backup.txt                     # Backup / experiment notes
├── tempCodeRunnerFile.py          # VS Code temporary file
│
├── requirements.txt               # Python dependencies
├── Procfile                       # Deployment configuration (Heroku-compatible)
├── LICENSE                        # Project license
├── README.md                      # Project documentation
│
├── best_model_XGBRanker.joblib    # Trained XGBRanker model
├── preprocessing_pipeline.joblib  # Saved sklearn preprocessing pipeline
├── MLB.joblib                     # MultiLabelBinarizer for crop labels
│
├── synthetic_crop_data.csv        # Synthetic dataset for training
│
├── greensense-crop-recommendation.ipynb  # EDA & recommendation experiments
├── greensense-ml-pipeline.ipynb          # End-to-end ML pipeline notebook
│
├── Old Models/                           # Archived / experimental models
│   ├── best_model_XGBClassifier.joblib
│   ├── best_model_XGBRanker.joblib
│   ├── preprocessing_pipeline.joblib
│   ├── MLB.joblib
│   └── index.html
│
├── templates/
│   └── index.html                 # Frontend HTML template
│
├── static/
│   └── style.css                  # Frontend styling
│
└── __pycache__/
    └── utils.cpython-314.pyc      # Python cache files

```

---

## ⚙️ Machine Learning Approach

### 1️⃣ Feature Engineering

* Sensor value clipping and normalization
* Robust imputation for missing location data
* Categorical encoding for:

  * District
  * Season
  * Location
* Domain-driven feature transformations

### 2️⃣ Learning-to-Rank Formulation

Each input sample is expanded as:

```
(Input Features + One-Hot Crop Vector)
```

The model predicts a **score per crop**, which is later sorted to generate rankings.

### 3️⃣ Ranking Model

* **Model**: `XGBRanker`
* **Objective**: Pairwise ranking
* **Evaluation Metrics**:

  * Hit@K
  * NDCG@K

---

## 📈 Evaluation Metrics

| Metric | Description                             |
| ------ | --------------------------------------- |
| Hit@K  | Whether true crop appears in top-K      |
| NDCG@K | Ranking quality with position weighting |

This ensures **ranking quality**, not just accuracy.

---

## 🌐 Flask API Endpoints

### 🔹 Home

```
GET /
```

Serves the frontend UI.

---

### 🔹 Feature Schema

```
GET /schema
```

Returns the expected input feature list.

---

### 🔹 Predict Crops

```
POST /predict
```

#### Sample Request

```json
{
  "district": "Pune",
  "season": "Kharif",
  "soil_moisture": 32.5,
  "temperature": 29.1,
  "humidity": 68.0,
  "ph": 6.7
}
```

#### Sample Response

```json
{
  "recommendations": [
    {
      "crop": "Rice",
      "score": 1.92,
      "probability": 0.41
    },
    {
      "crop": "Maize",
      "score": 1.21,
      "probability": 0.23
    }
  ]
}
```

---

### 🔹 Live Sensor Readings

```
GET /latest-readings?ID=1
```

Fetches latest IoT readings via Google Sheets API.

---

## 🧪 Threshold-Based Recommendation Logic

Only crops with probability ≥ **5%** are recommended, preventing noisy or irrelevant outputs.

```python
THRESHOLD = 0.05
```

---

## 🚀 Running Locally

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Start Server

```bash
python app.py
```

Server runs at:

```
http://localhost:5000
```

---

## 🔬 Research & Academic Value

* Demonstrates **learning-to-rank in agriculture**
* Suitable for **journal or conference submission**
* Can be extended with:

  * Weather forecasts
  * Satellite imagery
  * Reinforcement learning

---

## 📌 Future Enhancements

* 📊 Temporal modeling with LSTMs
* 🌦️ Weather API integration
* 📱 Mobile app interface
* 🧠 Personalized farmer profiles
* ☁️ Cloud deployment (AWS/GCP)

---

## 👨‍💻 Author

**Tanush Sudheer Tambe**

Final Year Engineering Student

Specialization: IoT + Machine Learning + Data Science

---

## 📜 License

This project is licensed under the **MIT License**.

See `LICENSE` file for details.

---
