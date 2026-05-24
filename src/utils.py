# =============================================================================
# utils.py — GreenSense shared utilities
# =============================================================================
# Column names match the new notebook pipeline (Env_Temp, N, P, K, etc.)
# The old column names (N_kg_per_ha, env_temp_c, …) are no longer used;
# if you have a legacy preprocessing_pipeline.joblib built on old names,
# retrain from the notebook first.
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# ---------------------------------------------------------------------------
# 1. COLUMN DEFINITIONS
# ---------------------------------------------------------------------------

# Raw sensor columns expected by the preprocessing pipeline and hybrid recommender
NUMERIC_COLS = [
    "Env_Temp",       # environmental temperature °C
    "BMP_Pressure",   # atmospheric pressure hPa
    "BMP_Altitude",   # altitude m
    "Light",          # light intensity lux
    "DHT_Humidity",   # environmental humidity %
    "MQ135_CO2",      # CO₂ ppm
    "MQ135_CO",       # CO ppm
    "MQ135_NH3",      # NH₃ ppm
    "Soil_Moisture",  # soil moisture %
    "Soil_Temp",      # soil temperature °C
    "pH",             # soil pH
    "N",              # nitrogen kg/ha
    "P",              # phosphorus kg/ha
    "K",              # potassium kg/ha
]

CATEGORICAL_COLS = ["district", "location", "season"]

ALL_INPUT_COLS = NUMERIC_COLS + CATEGORICAL_COLS


# ---------------------------------------------------------------------------
# 2. DATA CLEANING
# ---------------------------------------------------------------------------

def clean_and_clip(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Sanity-check raw sensor values and replace invalid readings with NaN.
    Applies IQR-based outlier removal on all numeric columns.
    """
    df = df_in.copy()

    # Hard physical bounds
    _null_where(df, "N",
                df["N"] < 0 if "N" in df.columns else pd.Series(dtype=bool))
    _null_where(df, "P",
                df["P"] < 0 if "P" in df.columns else pd.Series(dtype=bool))
    _null_where(df, "K",
                df["K"] < 0 if "K" in df.columns else pd.Series(dtype=bool))
    _null_where(df, "Env_Temp",     df["Env_Temp"] <
                0 if "Env_Temp" in df.columns else pd.Series(dtype=bool))
    _null_where(df, "DHT_Humidity", df["DHT_Humidity"] >
                100 if "DHT_Humidity" in df.columns else pd.Series(dtype=bool))
    _null_where(df, "Soil_Moisture", df["Soil_Moisture"] >
                100 if "Soil_Moisture" in df.columns else pd.Series(dtype=bool))
    _null_where(df, "Light",
                df["Light"] < 0 if "Light" in df.columns else pd.Series(dtype=bool))

    # IQR outlier removal
    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df.loc[(df[col] < lower) | (df[col] > upper), col] = np.nan

    return df


def _null_where(df: pd.DataFrame, col: str, mask: pd.Series) -> None:
    """Helper: set df[col] = NaN where mask is True, if col exists."""
    if col in df.columns and len(mask) > 0:
        df.loc[mask, col] = np.nan


# ---------------------------------------------------------------------------
# 3. ROBUST LOCATION-AWARE IMPUTER
# ---------------------------------------------------------------------------

class RobustLocationImputer(BaseEstimator, TransformerMixin):
    """
    Fills missing numeric values using a three-level fallback:
      1. (district, location, season) group median
      2. district median
      3. global median
    """

    def __init__(self, numeric_cols=None, min_group_size: int = 5,
                 verbose: bool = False):
        self.numeric_cols = numeric_cols
        self.min_group_size = min_group_size
        self.verbose = verbose

    # sklearn Pipeline compatibility
    def set_output(self, transform=None):
        return self

    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()
        if self.numeric_cols is None:
            self.numeric_cols = df.select_dtypes(
                include=[np.number]).columns.tolist()

        self.group_medians_ = (
            df.groupby(["district", "location", "season"])[self.numeric_cols]
            .median(numeric_only=True).dropna(how="all")
        )
        self.district_medians_ = (
            df.groupby("district")[self.numeric_cols]
            .median(numeric_only=True).dropna(how="all")
        )
        self.global_medians_ = df[self.numeric_cols].median(numeric_only=True)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for idx, row in df.iterrows():
            d = row.get("district")
            l = row.get("location")
            s = row.get("season")
            group_key = (d, l, s)

            group_valid = (
                group_key in self.group_medians_.index
                and len(df[
                    (df["district"] == d) &
                    (df["location"] == l) &
                    (df["season"] == s)
                ]) >= self.min_group_size
            )

            for col in self.numeric_cols:
                if pd.isna(row[col]):
                    if group_valid and col in self.group_medians_.columns \
                            and not pd.isna(self.group_medians_.loc[group_key, col]):
                        df.at[idx, col] = self.group_medians_.loc[group_key, col]
                    elif d in self.district_medians_.index \
                            and col in self.district_medians_.columns \
                            and not pd.isna(self.district_medians_.loc[d, col]):
                        df.at[idx, col] = self.district_medians_.loc[d, col]
                    else:
                        df.at[idx, col] = self.global_medians_.get(col, np.nan)
        return df


# ---------------------------------------------------------------------------
# 4. FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def feature_engineer(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Derives engineered features on top of raw sensor columns.
    Matches exactly what the notebook pipeline expects.
    """
    df = df_in.copy()

    # Nutrient ratios
    df["N_plus_P"] = df["N"] + df["P"]
    df["P_over_N"] = df["P"] / (df["N"] + 1e-6)
    df["K_over_N"] = df["K"] / (df["N"] + 1e-6)

    # Temperature and moisture deltas
    df["env_minus_soil_temp"] = df["Env_Temp"] - df["Soil_Temp"]
    df["env_minus_soil_humidity"] = df["DHT_Humidity"] - df["Soil_Moisture"]

    # Log transforms
    df["light_log"] = np.log1p(df["Light"])

    # Altitude bucket (categorical)
    df["alt_bucket"] = pd.cut(
        df["BMP_Altitude"],
        bins=[-1, 100, 300, 600, 1000, 5000],
        labels=["very_low", "low", "med", "high", "very_high"]
    )

    # Ordinal season encoding
    season_map = {s.lower(): i for i, s in enumerate(
        ["summer", "monsoon", "winter"])}
    df["season_enc"] = df["season"].astype(str).str.lower().map(season_map)

    return df


# ---------------------------------------------------------------------------
# 5. INPUT NORMALISATION HELPER (used by Flask before calling recommender)
# ---------------------------------------------------------------------------

# Maps any legacy / alternative field names a client might send → canonical names
_FIELD_ALIASES: dict = {
    # old names → new canonical names
    "N_kg_per_ha":           "N",
    "P_kg_per_ha":           "P",
    "K_kg_per_ha":           "K",
    "env_temp_c":            "Env_Temp",
    "soil_temp_c":           "Soil_Temp",
    "soil_humidity_percent": "Soil_Moisture",
    "env_humidity_percent":  "DHT_Humidity",
    "env_gasses_co2_ppm":    "MQ135_CO2",
    "env_pollution_ppm":     "MQ135_CO",
    "altitude_m":            "BMP_Altitude",
    "light_intensity_lux":   "Light",
    "pressure_hpa":          "BMP_Pressure",
}


def normalise_input(raw: dict) -> dict:
    """
    Accept a JSON payload that may use either old or new field names and
    return a dict with canonical column names the pipeline expects.
    Also coerces all numeric values to float and fills missing categoricals
    with sensible defaults.
    """
    # Rename legacy fields
    out = {}
    for k, v in raw.items():
        canonical = _FIELD_ALIASES.get(k, k)
        out[canonical] = v

    # Coerce numerics
    for col in NUMERIC_COLS:
        if col in out:
            try:
                out[col] = float(out[col])
            except (TypeError, ValueError):
                out[col] = np.nan
        else:
            out[col] = np.nan

    # Default categoricals if missing
    out.setdefault("district", "Unknown")
    out.setdefault("location", "Unknown")
    out.setdefault("season",   "Monsoon")

    return out
