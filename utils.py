import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.base import BaseEstimator, TransformerMixin


def clean_and_clip(df_in):
    df = df_in.copy()

    df.loc[df['N_kg_per_ha'] < 0, 'N_kg_per_ha'] = np.nan
    df.loc[df['env_temp_c'] < 0, 'env_temp_c'] = np.nan
    df.loc[df['env_humidity_percent'] > 100, 'env_humidity_percent'] = np.nan

    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df.loc[(df[col] < lower) | (df[col] > upper), col] = np.nan

    return df


class RobustLocationImputer(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_cols=None, min_group_size=5, verbose=False):
        self.numeric_cols = numeric_cols
        self.min_group_size = min_group_size
        self.verbose = verbose

    def fit(self, X, y=None):
        df = X.copy()
        if self.numeric_cols is None:
            self.numeric_cols = df.select_dtypes(
                include=[np.number]).columns.tolist()

        # Compute medians for each level
        self.group_medians_ = (
            df.groupby(["district", "location", "season"])[self.numeric_cols]
            .median(numeric_only=True)
            .dropna(how="all")
        )
        self.district_medians_ = (
            df.groupby("district")[self.numeric_cols]
            .median(numeric_only=True)
            .dropna(how="all")
        )
        self.global_medians_ = df[self.numeric_cols].median(numeric_only=True)
        return self

    def set_output(self, transform=None):
        return self

    def transform(self, X):
        df = X.copy()
        for idx, row in df.iterrows():
            d, l, s = row["district"], row["location"], row["season"]
            group_key = (d, l, s)
            if group_key in self.group_medians_.index:
                group_valid = (
                    len(df[(df["district"] == d) & (
                        df["location"] == l) & (df["season"] == s)])
                    >= self.min_group_size
                )
            else:
                group_valid = False
            for col in self.numeric_cols:
                if pd.isna(row[col]):
                    new_val = None
                    if group_valid and not pd.isna(self.group_medians_.loc[group_key, col]):
                        new_val = self.group_medians_.loc[group_key, col]
                    elif d in self.district_medians_.index and not pd.isna(self.district_medians_.loc[d, col]):
                        new_val = self.district_medians_.loc[d, col]
                    else:
                        new_val = self.global_medians_[col]
                    df.at[idx, col] = new_val
        return df


def feature_engineer(df_in):
    """Add engineered features and return a new DataFrame copy."""
    df = df_in.copy()

    df['N_plus_P'] = df['N_kg_per_ha'] + df['P_kg_per_ha']
    df['P_over_N'] = df['P_kg_per_ha'] / (df['N_kg_per_ha'] + 1e-6)
    df['K_over_N'] = df['K_kg_per_ha'] / (df['N_kg_per_ha'] + 1e-6)

    df['env_minus_soil_temp'] = df['env_temp_c'] - df['soil_temp_c']
    df['env_minus_soil_humidity'] = df['env_humidity_percent'] - \
        df['soil_humidity_percent']

    df['env_pollution_log'] = np.log1p(df['env_pollution_ppm'])
    df['light_log'] = np.log1p(df['light_intensity_lux'])

    df['alt_bucket'] = pd.cut(df['altitude_m'], bins=[-1, 100, 300, 600, 1000, 5000],
                              labels=['very_low', 'low', 'med', 'high', 'very_high'])

    season_map = {s.lower(): i for i, s in enumerate(
        ['summer', 'monsoon', 'winter', 'spring', 'autumn'])}
    df['season_enc'] = df['season'].astype(str).str.lower().map(season_map)

    return df
