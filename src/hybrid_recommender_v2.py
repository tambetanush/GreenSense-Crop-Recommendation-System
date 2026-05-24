# =============================================================================
# hybrid_recommender_v2.py — GreenSense standalone module
# =============================================================================
# Extracted from notebook cells so app.py can import from it directly.
# Place this file alongside app.py, utils.py, and the .joblib artifacts.
#
# Public API used by app.py:
#   register_new_crop(...)        → adds a crop to the in-memory catalog
#   load_registered_crops(...)    → reloads crops from JSON (legacy; DB takes
#                                   precedence in app.py)
#   HybridCropRecommenderV2       → the recommender class
#   save_recommender(rec, path)   → joblib.dump wrapper
#   load_recommender(path)        → joblib.load wrapper
#   CROP_CATALOG                  → module-level catalog DataFrame
# =============================================================================

import json
import os
import warnings
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# =============================================================================
# SECTION 1 — CROP CATALOG
# =============================================================================

CROP_CATALOG = pd.DataFrame([
    # ── Cereals & Millets ───────────────────────────────────────────────────
    {"crop": "Rice",
     "ph_min": 5.0, "ph_max": 7.0, "sun_min": 25000, "sun_max": 80000,
     "N_req": 250, "P_req": 12,  "K_req": 200,
     "moisture_min": 75, "moisture_max": 100, "temp_min": 20, "temp_max": 35,
     "alt_bucket": "low", "seasons": "monsoon"},

    {"crop": "Wheat",
     "ph_min": 6.0, "ph_max": 7.5, "sun_min": 50000, "sun_max": 100000,
     "N_req": 250, "P_req": 20, "K_req": 200,
     "moisture_min": 30, "moisture_max": 60, "temp_min": 12, "temp_max": 25,
     "alt_bucket": "med", "seasons": "winter"},

    {"crop": "Maize",
     "ph_min": 5.8, "ph_max": 7.0, "sun_min": 60000, "sun_max": 100000,
     "N_req": 230, "P_req": 27, "K_req": 200,
     "moisture_min": 25, "moisture_max": 60, "temp_min": 18, "temp_max": 32,
     "alt_bucket": "med", "seasons": "monsoon,summer"},

    {"crop": "Bajra",
     "ph_min": 6.0, "ph_max": 8.0, "sun_min": 70000, "sun_max": 110000,
     "N_req": 150, "P_req": 12, "K_req": 150,
     "moisture_min": 15, "moisture_max": 40, "temp_min": 25, "temp_max": 40,
     "alt_bucket": "low", "seasons": "summer,monsoon"},

    {"crop": "Finger Millet",
     "ph_min": 5.5, "ph_max": 7.5, "sun_min": 40000, "sun_max": 90000,
     "N_req": 150, "P_req": 10, "K_req": 115,
     "moisture_min": 50, "moisture_max": 90, "temp_min": 15, "temp_max": 30,
     "alt_bucket": "med", "seasons": "monsoon"},

    {"crop": "Ragi",
     "ph_min": 5.5, "ph_max": 7.5, "sun_min": 40000, "sun_max": 90000,
     "N_req": 150, "P_req": 10, "K_req": 115,
     "moisture_min": 50, "moisture_max": 90, "temp_min": 15, "temp_max": 30,
     "alt_bucket": "med", "seasons": "monsoon"},

    {"crop": "Vari",
     "ph_min": 5.0, "ph_max": 7.0, "sun_min": 30000, "sun_max": 80000,
     "N_req": 120, "P_req": 10, "K_req": 115,
     "moisture_min": 60, "moisture_max": 95, "temp_min": 20, "temp_max": 35,
     "alt_bucket": "low", "seasons": "monsoon"},

    # ── Pulses & Oilseeds ───────────────────────────────────────────────────
    {"crop": "Green Gram",
     "ph_min": 6.0, "ph_max": 7.5, "sun_min": 60000, "sun_max": 100000,
     "N_req": 50,  "P_req": 20, "K_req": 100,
     "moisture_min": 30, "moisture_max": 60, "temp_min": 25, "temp_max": 38,
     "alt_bucket": "low", "seasons": "summer,monsoon"},

    {"crop": "Groundnut",
     "ph_min": 6.0, "ph_max": 7.0, "sun_min": 70000, "sun_max": 110000,
     "N_req": 50,  "P_req": 30, "K_req": 115,
     "moisture_min": 20, "moisture_max": 45, "temp_min": 25, "temp_max": 35,
     "alt_bucket": "low", "seasons": "summer,monsoon"},

    {"crop": "Soybean",
     "ph_min": 6.0, "ph_max": 7.0, "sun_min": 55000, "sun_max": 100000,
     "N_req": 50,  "P_req": 37, "K_req": 130,
     "moisture_min": 40, "moisture_max": 70, "temp_min": 20, "temp_max": 30,
     "alt_bucket": "med", "seasons": "monsoon"},

    # ── Horticulture / Cash Crops ───────────────────────────────────────────
    {"crop": "Sugarcane",
     "ph_min": 6.0, "ph_max": 7.5, "sun_min": 70000, "sun_max": 110000,
     "N_req": 275, "P_req": 40, "K_req": 275,
     "moisture_min": 60, "moisture_max": 90, "temp_min": 24, "temp_max": 38,
     "alt_bucket": "low", "seasons": "summer,monsoon"},

    {"crop": "Onion",
     "ph_min": 6.5, "ph_max": 7.5, "sun_min": 55000, "sun_max": 95000,
     "N_req": 150, "P_req": 45, "K_req": 150,
     "moisture_min": 30, "moisture_max": 55, "temp_min": 13, "temp_max": 24,
     "alt_bucket": "med", "seasons": "winter"},

    {"crop": "Tomato",
     "ph_min": 6.0, "ph_max": 7.5, "sun_min": 55000, "sun_max": 100000,
     "N_req": 150, "P_req": 45, "K_req": 200,
     "moisture_min": 40, "moisture_max": 65, "temp_min": 18, "temp_max": 28,
     "alt_bucket": "med", "seasons": "winter,summer"},

    {"crop": "Grapes",
     "ph_min": 6.5, "ph_max": 8.0, "sun_min": 60000, "sun_max": 110000,
     "N_req": 200, "P_req": 35, "K_req": 250,
     "moisture_min": 30, "moisture_max": 55, "temp_min": 15, "temp_max": 35,
     "alt_bucket": "med", "seasons": "winter,summer"},

    {"crop": "Pomegranate",
     "ph_min": 6.5, "ph_max": 7.5, "sun_min": 70000, "sun_max": 110000,
     "N_req": 150, "P_req": 27, "K_req": 200,
     "moisture_min": 25, "moisture_max": 50, "temp_min": 25, "temp_max": 38,
     "alt_bucket": "med", "seasons": "summer"},

    {"crop": "Vegetables",
     "ph_min": 5.5, "ph_max": 7.0, "sun_min": 40000, "sun_max": 90000,
     "N_req": 175, "P_req": 40, "K_req": 150,
     "moisture_min": 50, "moisture_max": 75, "temp_min": 15, "temp_max": 30,
     "alt_bucket": "high", "seasons": "winter,monsoon"},

    # ── Plantation / Perennial ──────────────────────────────────────────────
    {"crop": "Coconut",
     "ph_min": 5.0, "ph_max": 8.0, "sun_min": 50000, "sun_max": 100000,
     "N_req": 250, "P_req": 20, "K_req": 250,
     "moisture_min": 70, "moisture_max": 95, "temp_min": 25, "temp_max": 35,
     "alt_bucket": "very_low", "seasons": "monsoon,summer"},

    {"crop": "Arecanut",
     "ph_min": 5.5, "ph_max": 7.0, "sun_min": 30000, "sun_max": 70000,
     "N_req": 230, "P_req": 17, "K_req": 200,
     "moisture_min": 75, "moisture_max": 95, "temp_min": 22, "temp_max": 34,
     "alt_bucket": "very_low", "seasons": "monsoon"},

    {"crop": "Mango",
     "ph_min": 5.5, "ph_max": 7.5, "sun_min": 65000, "sun_max": 110000,
     "N_req": 215, "P_req": 20, "K_req": 200,
     "moisture_min": 35, "moisture_max": 65, "temp_min": 24, "temp_max": 38,
     "alt_bucket": "low", "seasons": "summer"},

    {"crop": "Banana",
     "ph_min": 5.5, "ph_max": 7.0, "sun_min": 50000, "sun_max": 100000,
     "N_req": 275, "P_req": 27, "K_req": 325,
     "moisture_min": 65, "moisture_max": 90, "temp_min": 24, "temp_max": 38,
     "alt_bucket": "very_low", "seasons": "monsoon,summer"},

    {"crop": "Cashew",
     "ph_min": 4.5, "ph_max": 6.5, "sun_min": 60000, "sun_max": 110000,
     "N_req": 150, "P_req": 12, "K_req": 150,
     "moisture_min": 40, "moisture_max": 75, "temp_min": 24, "temp_max": 36,
     "alt_bucket": "low", "seasons": "summer,monsoon"},
])


# =============================================================================
# SECTION 2 — SHARED VECTOR SPACE
# =============================================================================

ALT_LABELS = ["very_low", "low", "med", "high", "very_high"]
VECTOR_DIM = 16   # 8 numeric + 3 season + 5 alt


def _alt_from_meters(alt_m: float) -> str:
    if alt_m <= 100:
        return "very_low"
    if alt_m <= 300:
        return "low"
    if alt_m <= 600:
        return "med"
    if alt_m <= 1000:
        return "high"
    return "very_high"


def _alt_onehot(label: str) -> List[float]:
    return [1.0 if label == l else 0.0 for l in ALT_LABELS]


def _season_onehot(season_str: str) -> Tuple[float, float, float]:
    s = season_str.lower()
    return (
        1.0 if "summer" in s else 0.0,
        1.0 if "monsoon" in s else 0.0,
        1.0 if "winter" in s else 0.0,
    )


def crop_to_vector(row: pd.Series) -> np.ndarray:
    ph_mid = (row["ph_min"] + row["ph_max"]) / 2.0
    ph_range = (row["ph_max"] - row["ph_min"])
    sun_mid = (row["sun_min"] + row["sun_max"]) / 2.0
    sun_log = np.log1p(sun_mid)
    N = row["N_req"]
    P = row["P_req"]
    K = row["K_req"]
    m_mid = (row["moisture_min"] + row["moisture_max"]) / 2.0
    t_mid = (row["temp_min"] + row["temp_max"]) / 2.0
    ss, sm, sw = _season_onehot(str(row.get("seasons", "")))
    alt_vec = _alt_onehot(str(row.get("alt_bucket", "low")))
    return np.array(
        [ph_mid, ph_range, sun_log, N, P, K, m_mid, t_mid,
         ss, sm, sw] + alt_vec, dtype=float
    )


def sensor_to_vector(s: pd.Series) -> np.ndarray:
    ph_mid = float(s.get("pH",            np.nan))
    ph_range = 0.0
    sun_log = np.log1p(float(s.get("Light", 0)))
    N = float(s.get("N",             np.nan))
    P = float(s.get("P",             np.nan))
    K = float(s.get("K",             np.nan))
    m_mid = float(s.get("Soil_Moisture",  np.nan))
    t_mid = float(s.get("Env_Temp",       np.nan))
    ss, sm, sw = _season_onehot(str(s.get("season", "")))
    alt_lab = _alt_from_meters(float(s.get("BMP_Altitude", 0)))
    alt_vec = _alt_onehot(alt_lab)
    return np.array(
        [ph_mid, ph_range, sun_log, N, P, K, m_mid, t_mid,
         ss, sm, sw] + alt_vec, dtype=float
    )


def _impute_matrix(M: np.ndarray) -> np.ndarray:
    col_med = np.nanmedian(M, axis=0)
    nan_idx = np.where(np.isnan(M))
    M[nan_idx] = np.take(col_med, nan_idx[1])
    return M


def build_item_matrix(catalog: pd.DataFrame,
                      scaler:  StandardScaler
                      ) -> Tuple[np.ndarray, List[str]]:
    rows, names = [], []
    for _, row in catalog.iterrows():
        names.append(row["crop"])
        rows.append(crop_to_vector(row))
    V = _impute_matrix(np.array(rows, dtype=float))
    return scaler.transform(V), names


def fit_vector_scaler(X_train_df: pd.DataFrame) -> StandardScaler:
    rows = [sensor_to_vector(row) for _, row in X_train_df.iterrows()]
    M = _impute_matrix(np.array(rows, dtype=float))
    sc = StandardScaler()
    sc.fit(M)
    return sc


# =============================================================================
# SECTION 3 — RANGE COMPATIBILITY SCORER
# =============================================================================

RANGE_WEIGHTS = {
    "pH": 1.5, "temp": 1.3, "moisture": 1.2,
    "N":  1.0, "P":    0.8, "K":        0.8, "sun": 0.7,
}


def _range_dim_score(env_val: float, opt_min: float, opt_max: float) -> float:
    if pd.isna(env_val):
        return 0.5
    if opt_min <= env_val <= opt_max:
        return 1.0
    spread = max(opt_max - opt_min, 1e-6)
    dist = min(abs(env_val - opt_min), abs(env_val - opt_max))
    return max(0.0, 1.0 - dist / spread)


def range_score_crop(sensor: dict, crop_row: pd.Series) -> float:
    dims = {
        "pH":      (_range_dim_score(sensor.get("pH"),
                                     crop_row["ph_min"], crop_row["ph_max"]),
                    RANGE_WEIGHTS["pH"]),
        "temp":    (_range_dim_score(sensor.get("Env_Temp"),
                                     crop_row["temp_min"], crop_row["temp_max"]),
                    RANGE_WEIGHTS["temp"]),
        "moisture": (_range_dim_score(sensor.get("Soil_Moisture"),
                                      crop_row["moisture_min"], crop_row["moisture_max"]),
                     RANGE_WEIGHTS["moisture"]),
        "N":       (_range_dim_score(sensor.get("N"),
                                     crop_row["N_req"] * 0.7, crop_row["N_req"] * 1.3),
                    RANGE_WEIGHTS["N"]),
        "P":       (_range_dim_score(sensor.get("P"),
                                     crop_row["P_req"] * 0.7, crop_row["P_req"] * 1.3),
                    RANGE_WEIGHTS["P"]),
        "K":       (_range_dim_score(sensor.get("K"),
                                     crop_row["K_req"] * 0.7, crop_row["K_req"] * 1.3),
                    RANGE_WEIGHTS["K"]),
        "sun":     (_range_dim_score(sensor.get("Light"),
                                     crop_row["sun_min"], crop_row["sun_max"]),
                    RANGE_WEIGHTS["sun"]),
    }
    total_w = sum(w for _, w in dims.values())
    return float(sum(s * w for s, w in dims.values()) / total_w)


def range_score_all(sensor: dict, catalog: pd.DataFrame) -> Dict[str, float]:
    return {row["crop"]: range_score_crop(sensor, row)
            for _, row in catalog.iterrows()}


# =============================================================================
# SECTION 4 — RECIPROCAL RANK FUSION
# =============================================================================

def _to_ranks(score_dict: Dict[str, float]) -> Dict[str, int]:
    ordered = sorted(score_dict, key=score_dict.get, reverse=True)
    return {crop: i + 1 for i, crop in enumerate(ordered)}


def rrf_merge(score_dicts: List[Dict[str, float]], k: int = 60) -> Dict[str, float]:
    all_crops = set().union(*[d.keys() for d in score_dicts])
    rank_lists = [_to_ranks(d) for d in score_dicts]
    worst = len(all_crops) + 1
    fused = {
        crop: sum(1.0 / (k + rl.get(crop, worst)) for rl in rank_lists)
        for crop in all_crops
    }
    return dict(sorted(fused.items(), key=lambda x: x[1], reverse=True))


# =============================================================================
# SECTION 5 — CROP REGISTRATION & PERSISTENCE
# =============================================================================

def register_new_crop(
    crop_name:    str,
    ph_min:       float,
    ph_max:       float,
    sun_min:      float,
    sun_max:      float,
    N_req:        float,
    P_req:        float,
    K_req:        float,
    moisture_min: float,
    moisture_max: float,
    temp_min:     float,
    temp_max:     float,
    alt_bucket:   str,
    seasons:      str,
    catalog:      pd.DataFrame = None,
    persist_path: str = "data/crop_catalog_updates.json",
) -> pd.DataFrame:
    """
    Add (or overwrite) a crop in the catalog DataFrame.
    Pass persist_path=None from app.py — SQLite is the source of truth there.
    Returns the updated catalog.
    """
    global CROP_CATALOG
    if catalog is None:
        catalog = CROP_CATALOG

    new_row = {
        "crop":         crop_name,
        "ph_min":       ph_min,       "ph_max":       ph_max,
        "sun_min":      sun_min,      "sun_max":      sun_max,
        "N_req":        N_req,        "P_req":        P_req,   "K_req": K_req,
        "moisture_min": moisture_min, "moisture_max": moisture_max,
        "temp_min":     temp_min,     "temp_max":     temp_max,
        "alt_bucket":   alt_bucket,   "seasons":      seasons,
    }

    catalog = catalog[catalog["crop"] != crop_name].copy()
    catalog = pd.concat([catalog, pd.DataFrame([new_row])], ignore_index=True)
    CROP_CATALOG = catalog

    if persist_path:
        existing = []
        if os.path.exists(persist_path):
            with open(persist_path, "r") as f:
                existing = json.load(f)
        existing = [r for r in existing if r["crop"] != crop_name]
        existing.append(new_row)
        with open(persist_path, "w") as f:
            json.dump(existing, f, indent=2)
        print(
            f"[OK] '{crop_name}' registered and persisted to '{persist_path}'.")
    else:
        print(
            f"[OK] '{crop_name}' registered in-memory (DB is source of truth).")

    return catalog


def load_registered_crops(
    catalog:      pd.DataFrame = None,
    persist_path: str = "data/crop_catalog_updates.json",
) -> pd.DataFrame:
    """
    Reload crops from JSON. Used in the notebook only.
    app.py uses _sync_db_crops_to_recommender() instead.
    """
    global CROP_CATALOG
    if catalog is None:
        catalog = CROP_CATALOG

    if not os.path.exists(persist_path):
        return catalog

    with open(persist_path, "r") as f:
        saved = json.load(f)

    for row in saved:
        catalog = catalog[catalog["crop"] != row["crop"]]
        catalog = pd.concat([catalog, pd.DataFrame([row])], ignore_index=True)

    CROP_CATALOG = catalog
    print(f"[OK] Loaded {len(saved)} crop(s) from '{persist_path}'.")
    return catalog


# =============================================================================
# SECTION 6 — HYBRID RECOMMENDER CLASS
# =============================================================================

class HybridCropRecommenderV2:
    """
    Three-signal hybrid recommender with RRF fusion.

    Known crops  → RRF( ML ranker score,  cosine similarity )
    New crops    → RRF( cosine similarity, range compatibility )
    Final list   → RRF( known_fused,       new_fused )
    """

    def __init__(self,
                 ranker,
                 preprocessor,
                 mlb,
                 catalog: pd.DataFrame = None,
                 rrf_k:   int = 60):
        self.ranker = ranker
        self.preprocessor = preprocessor
        self.mlb = mlb
        self.known_crops = list(mlb.classes_)
        self.catalog = catalog if catalog is not None else CROP_CATALOG
        self.rrf_k = rrf_k
        self._scaler = None
        self._item_matrix = None
        self._item_names = None

    # ── Setup ────────────────────────────────────────────────────────────────

    def fit_scaler_and_vectors(self, X_train_df: pd.DataFrame
                               ) -> "HybridCropRecommenderV2":
        self._scaler = fit_vector_scaler(X_train_df)
        self._rebuild_vectors()
        print(f"[OK] Scaler fitted. Item matrix: {self._item_matrix.shape}")
        return self

    def _rebuild_vectors(self):
        if self._scaler is None:
            raise RuntimeError("Call fit_scaler_and_vectors() first.")
        self._item_matrix, self._item_names = build_item_matrix(
            self.catalog, self._scaler
        )

    def update_catalog(self, new_catalog: pd.DataFrame):
        self.catalog = new_catalog
        self._rebuild_vectors()
        known_set = set(self.known_crops)
        new_crops = [c for c in self._item_names if c not in known_set]
        print(f"[OK] Catalog updated — {len(self._item_names)} total, "
              f"{len(new_crops)} new (CB-only).")

    # ── Signal 1: ML ranker ───────────────────────────────────────────────────

    def _ml_score_dict(self, X_trans_np: np.ndarray) -> Dict[str, float]:
        n_known = len(self.known_crops)
        X_full = np.array(
            [np.concatenate([X_trans_np[0], np.eye(n_known)[j]])
             for j in range(n_known)],
            dtype=np.float32,
        )
        raw = self.ranker.predict(X_full)
        return {crop: float(raw[j]) for j, crop in enumerate(self.known_crops)}

    # ── Signal 2: Cosine similarity ───────────────────────────────────────────

    def _cosine_score_dict(self, sensor_raw: dict) -> Dict[str, float]:
        s = pd.Series(sensor_raw)
        q = sensor_to_vector(s).reshape(1, -1)
        nan_mask = np.isnan(q)
        if nan_mask.any():
            q[nan_mask] = self._scaler.mean_[np.where(nan_mask)[1]]
        q_scaled = self._scaler.transform(q)
        sims = cosine_similarity(q_scaled, self._item_matrix).ravel()
        sims_norm = np.clip((sims + 1.0) / 2.0, 0.0, 1.0)
        return {name: float(sims_norm[i])
                for i, name in enumerate(self._item_names)}

    # ── Signal 3: Range compatibility ────────────────────────────────────────

    def _range_score_dict(self, sensor_raw: dict) -> Dict[str, float]:
        return range_score_all(sensor_raw, self.catalog)

    # ── Core inference ────────────────────────────────────────────────────────

    def recommend(self,
                  raw_input:      dict,
                  top_k:          int = 5,
                  include_scores: bool = False) -> list:
        if self._scaler is None:
            raise RuntimeError("Call fit_scaler_and_vectors(X_train) first.")

        input_df = pd.DataFrame([raw_input])
        try:
            X_trans = self.preprocessor.transform(input_df)
            X_trans_np = X_trans.to_numpy()
            ml_scores = self._ml_score_dict(X_trans_np)
        except Exception as e:
            print(f"[WARN] ML ranker failed ({e}). Using CB signals only.")
            ml_scores = {}

        cosine_scores = self._cosine_score_dict(raw_input)
        range_scores = self._range_score_dict(raw_input)
        known_set = set(self.known_crops)

        ml_known = {c: s for c, s in ml_scores.items() if c in known_set}
        cos_known = {c: s for c, s in cosine_scores.items() if c in known_set}
        known_fused = rrf_merge([ml_known, cos_known], k=self.rrf_k) \
            if ml_known else cos_known

        cos_new = {c: s for c, s in cosine_scores.items()
                   if c not in known_set}
        range_new = {c: s for c, s in range_scores.items()
                     if c not in known_set}
        new_fused = rrf_merge([cos_new, range_new], k=self.rrf_k) \
            if cos_new else {}

        final = rrf_merge([known_fused, new_fused], k=self.rrf_k) \
            if new_fused else known_fused

        top = list(final.items())[:top_k]

        if not include_scores:
            return [crop for crop, _ in top]

        return [
            (crop, round(score, 6),
             "ml+cosine" if crop in known_set else "cosine+range")
            for crop, score in top
        ]

    def recommend_from_row(self, df_row: pd.Series, **kwargs) -> list:
        return self.recommend(df_row.to_dict(), **kwargs)

    # ── Explainability ────────────────────────────────────────────────────────

    @staticmethod
    def _to_py(val):
        if isinstance(val, np.integer):
            return int(val)
        if isinstance(val, np.floating):
            return float(val)
        if isinstance(val, np.ndarray):
            return val.tolist()
        return val

    def explain(self, raw_input: dict, crop_name: str) -> dict:
        known_set = set(self.known_crops)
        source = "ml+cosine" if crop_name in known_set else "cosine+range"
        cosine_scores = self._cosine_score_dict(raw_input)
        cos_score = cosine_scores.get(crop_name)

        catalog_row = self.catalog[self.catalog["crop"] == crop_name]
        if catalog_row.empty:
            range_score = None
            dims = "Crop not in catalog — no range breakdown available."
        else:
            row = catalog_row.iloc[0]
            range_score = range_score_crop(raw_input, row)
            dims = {
                "pH":      {"env": raw_input.get("pH"),
                            "range": [self._to_py(row["ph_min"]),
                                      self._to_py(row["ph_max"])],
                            "score": round(_range_dim_score(
                                raw_input.get("pH"),
                                row["ph_min"], row["ph_max"]), 3)},
                "Temp":    {"env": raw_input.get("Env_Temp"),
                            "range": [self._to_py(row["temp_min"]),
                                      self._to_py(row["temp_max"])],
                            "score": round(_range_dim_score(
                                raw_input.get("Env_Temp"),
                                row["temp_min"], row["temp_max"]), 3)},
                "Moisture": {"env": raw_input.get("Soil_Moisture"),
                             "range": [self._to_py(row["moisture_min"]),
                                       self._to_py(row["moisture_max"])],
                             "score": round(_range_dim_score(
                                 raw_input.get("Soil_Moisture"),
                                 row["moisture_min"], row["moisture_max"]), 3)},
                "N":       {"env": raw_input.get("N"),
                            "range": [round(self._to_py(row["N_req"]) * 0.7, 2),
                                      round(self._to_py(row["N_req"]) * 1.3, 2)],
                            "score": round(_range_dim_score(
                                raw_input.get("N"),
                                row["N_req"] * 0.7, row["N_req"] * 1.3), 3)},
                "P":       {"env": raw_input.get("P"),
                            "range": [round(self._to_py(row["P_req"]) * 0.7, 2),
                                      round(self._to_py(row["P_req"]) * 1.3, 2)],
                            "score": round(_range_dim_score(
                                raw_input.get("P"),
                                row["P_req"] * 0.7, row["P_req"] * 1.3), 3)},
                "K":       {"env": raw_input.get("K"),
                            "range": [round(self._to_py(row["K_req"]) * 0.7, 2),
                                      round(self._to_py(row["K_req"]) * 1.3, 2)],
                            "score": round(_range_dim_score(
                                raw_input.get("K"),
                                row["K_req"] * 0.7, row["K_req"] * 1.3), 3)},
                "Sunlight": {"env": raw_input.get("Light"),
                             "range": [self._to_py(row["sun_min"]),
                                       self._to_py(row["sun_max"])],
                             "score": round(_range_dim_score(
                                 raw_input.get("Light"),
                                 row["sun_min"], row["sun_max"]), 3)},
            }

        ml_score = None
        if crop_name in known_set:
            try:
                X_trans = self.preprocessor.transform(
                    pd.DataFrame([raw_input]))
                ml_dict = self._ml_score_dict(X_trans.to_numpy())
                ml_score = round(float(ml_dict.get(crop_name, 0)), 4)
            except Exception:
                pass

        return {
            "crop":         crop_name,
            "source":       source,
            "ml_score":     ml_score,
            "cosine_score": round(float(cos_score), 4) if cos_score is not None else None,
            "range_score":  round(float(range_score), 4) if range_score is not None else None,
            "dimensions":   dims,
        }


# =============================================================================
# SECTION 7 — SAVE / LOAD
# =============================================================================

def save_recommender(rec: HybridCropRecommenderV2,
                     path: str = "models/hybrid_recommender_v2.joblib") -> None:
    joblib.dump(rec, path)
    print(f"[OK] Saved to '{path}'.")


def load_recommender(path: str = "models/hybrid_recommender_v2.joblib"
                     ) -> HybridCropRecommenderV2:
    rec = joblib.load(path)
    print(f"[OK] Loaded from '{path}'.")
    return rec
