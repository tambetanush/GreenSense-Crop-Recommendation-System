# resave_joblib.py — run once, then delete
from sklearn.model_selection import train_test_split
from hybrid_recommender_v2 import (CROP_CATALOG, HybridCropRecommenderV2,
                                   load_registered_crops, save_recommender)
import pandas as pd
import joblib
import sys

sys.path.insert(0, r"E:\Projects\GreenSense")


# ── 1. Load the individual artifacts (these have no __main__ issue) ──
print("Loading individual artifacts...")
preprocessor = joblib.load("preprocessing_pipeline.joblib")
mlb = joblib.load("MLB.joblib")

# Use the best ranker you have — try best_ranker.joblib first,
# fall back to best_model_XGBRanker.joblib
try:
    ranker = joblib.load("best_ranker.joblib")
    print("  ranker: best_ranker.joblib")
except Exception:
    ranker = joblib.load("best_model_XGBRanker.joblib")
    print("  ranker: best_model_XGBRanker.joblib")

# ── 2. Reload any crops registered in previous sessions ─────────
catalog = load_registered_crops(CROP_CATALOG.copy())

# ── 3. Rebuild the recommender cleanly ──────────────────────────
print("Rebuilding HybridCropRecommenderV2...")
rec = HybridCropRecommenderV2(
    ranker=ranker,
    preprocessor=preprocessor,
    mlb=mlb,
    catalog=catalog,
    rrf_k=60,
)

# ── 4. Fit the shared-space scaler on training data ──────────────
# We need X_train to fit the StandardScaler that lives inside the recommender.
# Load the synthetic CSV and re-split exactly as the notebook did.
print("Fitting vector scaler on training data...")

df = pd.read_csv("synthetic_crop_data.csv")
X = df.drop(columns=["target_crops"])
X_train, _ = train_test_split(X, test_size=0.2, random_state=42)

rec.fit_scaler_and_vectors(X_train)

# ── 5. Save with correct module path ────────────────────────────
save_recommender(rec, "hybrid_recommender_v2.joblib")
print(f"\nDone.")
print(f"  Known ML labels : {len(rec.known_crops)}")
print(f"  Crops in catalog: {len(rec.catalog)}")
print(f"  Saved to         hybrid_recommender_v2.joblib")
