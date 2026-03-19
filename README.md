# 🌱 GreenSense

### Intelligent Hybrid Crop Recommendation System (2024–2025)

---

GreenSense is an **end-to-end intelligent crop recommendation system** that integrates **IoT sensor data**, **machine learning ranking models**, and a **three-signal hybrid recommendation engine** to recommend the most suitable crops for a given soil and environmental condition — including crops the ML model has never seen during training.

Unlike traditional classification systems, GreenSense formulates **crop selection as a ranking problem** and extends it with a **content-based path for cold-start crops**, making it suitable for real-world agricultural deployment where new crop varieties need to be evaluated without retraining.

---

## 🚀 Key Highlights

- 📡 **Real-time IoT sensing** using ESP32 and environmental sensors
- 🧠 **Learning-to-Rank ML approach** using `CatBoostRanker` (tuned), with XGBoostRanker and LGBMRanker benchmarked
- 🔀 **Three-signal hybrid recommender** combining ML ranking, cosine similarity, and agronomic range compatibility
- 🌿 **Cold-start crop support** — register new crops via API without retraining the model
- ⚙️ **Robust feature engineering & preprocessing pipeline** with location-aware imputation
- 🌐 **Flask-based inference API** with SQLite persistence for crop registry and recommendation history
- 📊 Evaluation using **Hit@K**, **NDCG@K**, and **LRAP**
- 🔗 Live integration with **Google Sheets API** for sensor readings
- 🖥️ **Single-page dashboard** with light/dark theme, live sensor fetch, explainability panel, and history browser

---

## 🧠 Problem Formulation

Most crop recommendation systems treat the problem as **multi-class classification**, forcing a single "best" crop prediction.

**GreenSense models this as a listwise ranking problem**, enabling:

- Multiple suitable crops per condition, ranked by suitability
- Confidence-aware recommendations via RRF-fused scores
- Support for crops outside the training distribution (cold-start)
- Better real-world decision support with per-dimension explainability

---

## 🔀 Hybrid Recommendation Architecture

GreenSense v2 introduces a **three-signal hybrid engine** that fuses an ML ranker with two content-based signals using **Reciprocal Rank Fusion (RRF)** — a rank-based merging technique that is immune to score-scale differences between signals.

### Signal Paths

```
                    Raw Sensor Input
                          │
          ┌───────────────┼────────────────┐
          ▼               ▼                ▼
   [Signal 1]       [Signal 2]        [Signal 3]
   ML Ranker     Cosine Similarity   Range Compat.
  (CatBoost)    (shared 16-dim       (per-dimension
  Known crops    vector space)        agronomic
  only                                window check)
          │               │                │
          └──── Known ────┘                │
          RRF(ML, Cosine)                  │
               │                           │
               │           New crops only  │
               │      RRF(Cosine, Range)───┘
               │               │
               └───────────────┘
               RRF(known_fused, new_fused)
                          │
                    Final Ranked List
               (tagged: ml+cosine / cosine+range)
```

### Why Three Signals?

| Signal                   | Purpose                                                                  | Why Needed                                                                                                                                           |
| ------------------------ | ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ML Ranker (CatBoost)** | Listwise ranking trained on historical sensor-crop data                  | Best accuracy for known crops; captures nonlinear feature interactions                                                                               |
| **Cosine Similarity**    | Geometric closeness in shared 16-dim vector space                        | Captures inter-crop relationships; both crop profiles and sensor readings live in the same space                                                     |
| **Range Compatibility**  | Per-dimension agronomic window check (pH, temp, moisture, NPK, sunlight) | Hard constraint checker — catches cases where cosine similarity scores a crop highly despite one critical dimension (e.g. pH) being completely wrong |

### Why RRF Instead of Weighted Sum?

A weighted sum like `0.8 × ML + 0.2 × CB` requires score normalisation per inference call, making the same crop's score unstable across batches. RRF converts all signals to **rank positions** first, then combines:

```
RRF(crop) = Σ  1 / (k + rank_i)   for each signal i
```

With `k=60`, this makes the fusion robust to the ML ranker outputting raw floats (e.g. `-1.07`) while cosine outputs values in `[0,1]` — they never need to be on the same scale.

### Cold-Start: New Crops Without Retraining

Any crop not in the ML training set can be registered via the `/register_crop` API endpoint by providing agronomic requirement ranges (pH, sunlight in lux, NPK, moisture, temperature). The crop immediately enters the **cosine+range path** and competes in the final RRF merge. Once enough data is collected, the model can be retrained to absorb the new crop into the ML path — no manual catalog switching required.

---

### Full Pipeline Overview

```
IoT Sensors (ESP32)
        ↓
Google Sheets (Live Logging)
        ↓
Flask API  ←→  SQLite DB (crop registry + history)
        ↓
normalise_input() — field alias resolution + type coercion
        ↓
Preprocessing Pipeline
(clean_and_clip → RobustLocationImputer → feature_engineer → ColumnTransformer)
        ↓
┌──────────────────────────────────────────────────┐
│         HybridCropRecommenderV2                  │
│  ┌─────────────┐ ┌────────────┐ ┌─────────────┐  │
│  │ ML Ranker   │ │  Cosine    │ │   Range     │  │
│  │ (CatBoost)  │ │ Similarity │ │ Compat.     │  │
│  └──────┬──────┘ └─────┬──────┘ └──────┬──────┘  │
│         └──────RRF─────┘               │         │
│              known_fused     new_fused─┘         │
│                    └────RRF────┘                 │
└──────────────────────────────────────────────────┘
        ↓
Ranked Crop Recommendations
(with rank, RRF score, source tag, optional explain)
```

---

## 🛠️ Tech Stack

### Hardware

- ESP32 Microcontroller
- Soil Moisture Sensor (ZTS-3002)
- DHT Temperature & Humidity Sensor
- BMP Pressure / Altitude Sensor
- MQ135 Gas Sensor (CO₂, CO, NH₃)
- LDR Light Intensity Sensor

### Backend & ML

- Python 3.11+
- Flask + SQLite (WAL mode)
- Scikit-Learn (Pipeline, ColumnTransformer, StandardScaler)
- CatBoostRanker, XGBRanker, LGBMRanker
- NumPy, Pandas, Joblib
- sklearn.metrics.pairwise (cosine_similarity)

### Data & Integration

- Google Sheets API (live sensor logging)
- REST API (JSON inference)
- SQLite (crop registry, recommendation history)

---

# Diagrams & Images:

<h2>Soil + Environment Analyzer Diagram</h2>

<img src="Soil-Environment Analyzer/ESP32 S3.png" width="700">

<h3>Breadboard Implementations</h3>

<table align="center">
  <tr>
    <td align="center" valign="top">
      <img 
        src="Soil-Environment Analyzer/IMG20251225093429.jpg"
        width="350"
        style="display:block;"
      />
      <div style="width:350px; text-align:center; margin-top:6px;">
        <b>Breadboard Implementation – View 1</b>
      </div>
    </td>
    <td align="center" valign="top">
      <img 
        src="Soil-Environment Analyzer/IMG20251225093358.jpg"
        width="300"
        style="display:block;"
      />
      <div style="width:300px; text-align:center; margin-top:6px;">
        <b>Breadboard Implementation – View 2</b>
      </div>
    </td>
  </tr>

  <tr>
    <td align="center" valign="top">
      <img
        src="Soil-Environment Analyzer/IMG20251225093336.jpg"
        width="300"
        style="display:block;"
      />
      <div style="width:300px; text-align:center; margin-top:6px;">
        <b>Final Assembly</b>
      </div>
    </td>
    <td align="center" valign="top">
      <img
        src="Soil-Environment Analyzer/IMG20251225093404.jpg"
        width="300"
        style="display:block;"
      />
      <div style="width:300px; text-align:center; margin-top:6px;">
        <b>ZTS-3002 Soil Sensor</b>
      </div>
    </td>
  </tr>
</table>

<h3>Final Assembly Testing</h3>
<img src="Soil-Environment Analyzer/IMG-20251031-WA0018.jpg" width="700">

---

## 📂 Project Structure

```
GreenSense/
│
├── app.py                           # Flask application — all API routes
├── utils.py                         # Preprocessing, feature engineering, normalise_input()
├── hybrid_recommender_v2.py         # Hybrid recommender class + RRF fusion engine
├── test.py                          # Local testing script
│
├── requirements.txt                 # Python dependencies
├── LICENSE                          # Project license
├── README.md                        # Project documentation
│
├── hybrid_recommender_v2.joblib     # Serialised HybridCropRecommenderV2 (contains
│                                    #   ranker + preprocessor + mlb + scaler)
├── best_ranker.joblib               # Trained CatBoostRanker (best performing)
├── best_model_XGBRanker.joblib      # Trained XGBRanker (fallback)
├── preprocessing_pipeline.joblib   # Sklearn preprocessing pipeline (standalone backup)
├── MLB.joblib                       # MultiLabelBinarizer for crop labels (backup)
│
├── greensense.db                    # SQLite database
│                                    #   ├── registered_crops  (new crop catalog)
│                                    #   └── recommendation_history
│
├── crop_catalog_updates.json        # JSON backup of registered crops
├── synthetic_crop_data.csv          # Synthetic dataset (6,326 rows, 5 location profiles)
│
├── Ranker Notebook/
│   └── greensense-crop-recommendation.ipynb   # Main ML notebook
│                                              # (data gen → EDA → pipeline →
│                                              #  ranker training → HPT → hybrid)
│
├── Old Models/
│   └── greensense-ml-pipeline.ipynb           # Archived classification approach
│
├── Soil-Environment Analyzer/       # IoT hardware & implementation assets
│   ├── ESP32 S3.png
│   ├── IMG-20251031-WA0018.jpg
│   ├── IMG20251225093336.jpg
│   ├── IMG20251225093358.jpg
│   ├── IMG20251225093404.jpg
│   ├── IMG20251225093429.jpg
│   └── soil code.ino                # ESP32 firmware
│
├── templates/
│   └── index.html                   # Single-page dashboard (4 tabs)
│
├── static/
│   └── style.css
│
└── __pycache__/
```

---

## ⚙️ Machine Learning Approach

### 1️⃣ Synthetic Data Generation

Real standardised soil + environment datasets for Maharashtra are scarce, so a synthetic dataset of **6,326 rows** was generated using historically-backed parameters for five location profiles across Raigad and Nashik districts. The generator applies:

- Seasonal temperature simulation using sinusoidal day-of-year modelling
- Weighted crop assignment conditioned on temperature, moisture, pH, nutrients, altitude, and season
- Realistic noise (5%), random nulls (5%), sensor anomalies (2%), and extreme outliers (0.5%)

### 2️⃣ Feature Engineering

- Sensor value clipping and IQR-based outlier removal
- Three-level location-aware imputation: `(district, location, season)` → `district` → global median
- Derived features: nutrient ratios (`N_plus_P`, `P_over_N`, `K_over_N`), temperature delta, moisture delta, log-transformed light intensity, altitude bucket
- Categorical encoding: one-hot for district/location/season/alt_bucket, ordinal for season

### 3️⃣ Learning-to-Rank Formulation

Multi-label crop recommendation is reframed as a **listwise ranking problem**. Each crop competes within a query defined by a single sample:

```
(Preprocessed Features ‖ One-Hot Crop Identity Vector)  →  Relevance Score
```

The model predicts a score per crop; crops are sorted descending to produce the ranked recommendation list.

### 4️⃣ Model Comparison & Selection

Three rankers were benchmarked with default hyperparameters, then the top two were tuned via random search over 90 configurations each:

| Model                    | Hit@5      | NDCG@5     | LRAP       |
| ------------------------ | ---------- | ---------- | ---------- |
| **Tuned CatBoostRanker** | **0.7954** | **0.5083** | 0.4629     |
| CatBoostRanker (default) | 0.7883     | 0.5074     | **0.4655** |
| Tuned XGBRanker          | 0.7954     | 0.5029     | 0.4576     |
| LGBMRanker (default)     | 0.7820     | 0.4963     | 0.4531     |
| Tuned LGBMRanker         | 0.7907     | 0.4941     | 0.4457     |
| XGBRanker (default)      | 0.7796     | 0.4831     | 0.4387     |

**CatBoostRanker** with `YetiRank` loss consistently outperforms XGBoost and LightGBM across all ranking metrics, achieving the highest NDCG@5 — meaning it places relevant crops earlier in the list, which is critical when only top-ranked recommendations are acted upon.

Best CatBoost hyperparameters:

```python
{
    "loss_function": "YetiRank",
    "depth": 6,
    "learning_rate": 0.03,
    "iterations": 800,
    "l2_leaf_reg": 5,
    "random_strength": 1,
    "bagging_temperature": 0.5
}
```

### 5️⃣ Hybrid Extension (v2)

The trained CatBoostRanker is wrapped in `HybridCropRecommenderV2`, which adds two content-based signals and fuses all three via RRF. The shared 16-dimensional vector space uses:

```
[ph_mid, ph_range, log1p(sun_mid), N, P, K, moisture_mid, temp_mid,
 season_summer, season_monsoon, season_winter,
 alt_very_low, alt_low, alt_med, alt_high, alt_very_high]
```

This space is StandardScaler-fitted on training sensor readings, ensuring both crop profiles and live readings land in comparable positions for cosine comparison.

---

## 📈 Evaluation Metrics

| Metric     | Formula               | Interpretation                                                        |
| ---------- | --------------------- | --------------------------------------------------------------------- |
| **Hit@K**  | `1/N Σ 𝟙(T∩R(K) ≠ ∅)` | At least one relevant crop appears in top-K                           |
| **NDCG@K** | `DCG@K / IDCG@K`      | Ranking quality; penalises relevant crops appearing lower in the list |
| **LRAP**   | —                     | Label ranking average precision; global label ordering quality        |

---

## 🌐 Flask API Endpoints

### Pages

```
GET  /                   → Single-page dashboard (index.html)
```

### Inference

```
POST /predict            → Hybrid crop recommendation
POST /explain            → Per-dimension signal breakdown for one crop
GET  /schema             → Feature names expected by preprocessor
```

### Crop Registry

```
POST   /register_crop       → Add a new crop (content-based path, persisted to SQLite)
GET    /crops               → List all registered crops (?search=, ?limit=, ?offset=)
DELETE /crops/<crop_name>   → Remove a registered crop
```

### Recommendation History

```
GET    /history             → Paginated history (?from=, ?to=, ?limit=, ?offset=)
GET    /history/<id>        → Single recommendation record
DELETE /history             → Clear all history (requires X-Confirm: yes header)
```

### Monitoring

```
GET  /health             → Recommender + DB status check
GET  /latest-readings    → Proxy to Google Sheets sensor API
```

### Sample Predict Request

```json
{
    "N": 198.03,
    "P": 9.17,
    "K": 175.78,
    "pH": 6.96,
    "Soil_Temp": 29.15,
    "Soil_Moisture": 51.87,
    "Env_Temp": 31.8,
    "DHT_Humidity": 68.54,
    "MQ135_CO2": 416.15,
    "MQ135_CO": 2.35,
    "MQ135_NH3": 3.92,
    "Light": 111924.6,
    "BMP_Altitude": 136.63,
    "BMP_Pressure": 1008.62,
    "district": "Raigad",
    "location": "Karjat",
    "season": "Monsoon"
}
```

### Sample Predict Response

```json
{
    "status": "ok",
    "record_id": 42,
    "recommendations": [
        {
            "rank": 1,
            "crop": "Rice",
            "rrf_score": 0.02844,
            "source": "ml+cosine"
        },
        {
            "rank": 2,
            "crop": "Vari",
            "rrf_score": 0.02743,
            "source": "ml+cosine"
        },
        {
            "rank": 3,
            "crop": "Coconut",
            "rrf_score": 0.02718,
            "source": "ml+cosine"
        },
        {
            "rank": 4,
            "crop": "Quinoa",
            "rrf_score": 0.02695,
            "source": "cosine+range"
        },
        {
            "rank": 5,
            "crop": "Banana",
            "rrf_score": 0.02672,
            "source": "ml+cosine"
        }
    ]
}
```

### Sample Register Crop Request

```json
{
    "crop_name": "Quinoa",
    "ph_min": 6.0,
    "ph_max": 7.5,
    "sun_min": 50000,
    "sun_max": 90000,
    "N_req": 140,
    "P_req": 30,
    "K_req": 120,
    "moisture_min": 35,
    "moisture_max": 60,
    "temp_min": 15,
    "temp_max": 25,
    "alt_bucket": "med",
    "seasons": "winter,summer"
}
```

### Sample Explain Response (excerpt)

```json
{
    "crop": "Grapes",
    "source": "ml+cosine",
    "ml_score": -1.0171,
    "cosine_score": 0.1744,
    "range_score": 0.6062,
    "dimensions": {
        "pH": { "env": null, "range": [6.5, 8.0], "score": 0.5 },
        "Temp": { "env": 36.56, "range": [15, 35], "score": 0.922 },
        "Moisture": { "env": 75.23, "range": [30, 55], "score": 0.191 },
        "N": { "env": 293.61, "range": [140, 260], "score": 0.72 }
    }
}
```

---

## 🗄️ SQLite Persistence

Two tables are auto-created on first run:

**`registered_crops`** — stores every crop added via `/register_crop`, replayed into the recommender on each server startup so the DB is the single source of truth.

**`recommendation_history`** — stores every prediction as a JSON blob (input features + ranked output), queryable by date range. Supports full history clear with a confirmation header guard (`X-Confirm: yes`).

---

## 🖥️ Dashboard

The single-page frontend (`index.html`) has four tabs:

| Tab               | Function                                                                                                                                                      |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Recommend**     | Sensor fetch, manual input form with 🎲 randomiser, Top-K selector, recommendation cards with RRF scores, source badges (ML / CB), and per-crop explain panel |
| **Register Crop** | Form for adding new crops to the content-based catalog (all fields with validation)                                                                           |
| **Registry**      | Live search over registered crops, metadata display, remove button                                                                                            |
| **History**       | Date-filtered recommendation history with expandable input feature view and colour-coded source tags                                                          |

Light/dark theme toggle is persistent via `localStorage`.

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

The server auto-initialises `greensense.db`, loads the hybrid recommender from `hybrid_recommender_v2.joblib`, and syncs any previously registered crops from SQLite back into the in-memory catalog.

---

## 🔬 Research & Academic Value

- Demonstrates **learning-to-rank in precision agriculture**
- Novel application of **Reciprocal Rank Fusion** for hybrid ML + content-based crop recommendation
- Addresses the **cold-start problem** in agricultural recommender systems without retraining
- Explainable recommendations via per-dimension agronomic signal breakdown
- Suitable for **journal or conference submission** (ML in agriculture, recommender systems, IoT)

---

## 📌 Future Enhancements

- 📊 Temporal modelling with LSTMs for seasonal trend capture
- 🌦️ Weather API integration for forecast-aware recommendations
- 📱 Mobile app interface with offline sensor support
- 🧠 Personalized farmer profiles with preference learning
- ☁️ Cloud deployment (AWS/GCP) with auto-scaling
- 🔁 Active learning loop: use registered crop data to trigger automatic retraining

---

<!-- ## 👨‍💻 Author

**Tanush Sudheer Tambe**

Final Year Computer Engineering Student — University of Mumbai

Specialization: IoT · Machine Learning · Data Science -->

---

## 📜 License

This project is licensed under the **MIT License**.

See `LICENSE` file for details.

---
