"""
predict.py
----------
Inference module for house price prediction.

Provides:
- predict_price(input_dict): Fills unprovided inputs with saved defaults
  (median for numeric, mode for categorical), runs the best trained model,
  inverses the log transform, and computes the 90% prediction interval.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Module-level cache for model and metrics
_MODEL = None
_METRICS = None


def load_model_artifacts(models_dir: str = MODELS_DIR):
    """
    Load the best trained scikit-learn model and associated metrics/defaults.
    Caches artifacts in memory for fast repeated inference.
    """
    global _MODEL, _METRICS

    model_path = os.path.join(models_dir, "best_model.joblib")
    metrics_path = os.path.join(models_dir, "metrics.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            "Please run 'python src/train.py' first."
        )

    if not os.path.exists(metrics_path):
        raise FileNotFoundError(
            f"Metrics file not found at '{metrics_path}'. "
            "Please run 'python src/train.py' first."
        )

    if _MODEL is None:
        _MODEL = joblib.load(model_path)

    if _METRICS is None:
        with open(metrics_path, "r", encoding="utf-8") as f:
            _METRICS = json.load(f)

    return _MODEL, _METRICS


def get_feature_defaults() -> dict:
    """Return a dictionary of default values for all features."""
    _, metrics = load_model_artifacts()
    return dict(metrics.get("defaults", {}))


def get_top_features() -> list[dict]:
    """Return list of top ~15 most important features with their importance score."""
    _, metrics = load_model_artifacts()
    return metrics.get("top_15_features", [])


def predict_price(input_dict: dict = None) -> tuple[float, float, float]:
    """
    Predict house price and 90% prediction interval for given inputs.

    Parameters
    ----------
    input_dict : dict, optional
        Key-value mapping of feature names to user-provided values.
        Any missing features are automatically populated using saved
        defaults (median for numerics, mode for categoricals).

    Returns
    -------
    tuple of (price, lower_bound, upper_bound)
        price : float
            Predicted house price in USD.
        lower_bound : float
            Lower bound of the 90% prediction interval.
        upper_bound : float
            Upper bound of the 90% prediction interval.

    Example
    -------
    >>> price, lower, upper = predict_price({"GrLivArea": 2000, "OverallQual": 8})
    >>> print(f"Price: ${price:,.0f} (${lower:,.0f} - ${upper:,.0f})")
    """
    model, metrics = load_model_artifacts()

    # 1. Start with full baseline defaults
    data = dict(metrics.get("defaults", {}))

    # 2. Overlay user-provided inputs
    if input_dict:
        for k, v in input_dict.items():
            if v is not None:
                data[k] = v

    # 3. Re-calculate engineered features based on active values
    # TotalSF
    bsmt_sf = float(data.get("TotalBsmtSF", 0) or 0)
    flr1_sf = float(data.get("1stFlrSF", 0) or 0)
    flr2_sf = float(data.get("2ndFlrSF", 0) or 0)
    data["TotalSF"] = bsmt_sf + flr1_sf + flr2_sf

    # HouseAge and RemodAge
    yr_sold = int(data.get("YrSold", 2010) or 2010)
    yr_built = int(data.get("YearBuilt", 1970) or 1970)
    yr_remod = int(data.get("YearRemodAdd", yr_built) or yr_built)
    data["HouseAge"] = max(0, yr_sold - yr_built)
    data["RemodAge"] = max(0, yr_sold - yr_remod)

    # TotalBath
    fbath = float(data.get("FullBath", 0) or 0)
    hbath = float(data.get("HalfBath", 0) or 0)
    bfbath = float(data.get("BsmtFullBath", 0) or 0)
    bhbath = float(data.get("BsmtHalfBath", 0) or 0)
    data["TotalBath"] = fbath + 0.5 * hbath + bfbath + 0.5 * bhbath

    # Binary flags
    garage_area = float(data.get("GarageArea", 0) or 0)
    pool_area = float(data.get("PoolArea", 0) or 0)
    fireplaces = float(data.get("Fireplaces", 0) or 0)
    data["HasGarage"] = 1 if garage_area > 0 else 0
    data["HasPool"] = 1 if pool_area > 0 else 0
    data["HasFireplace"] = 1 if fireplaces > 0 else 0

    # 4. Construct DataFrame aligned with training feature columns
    feature_columns = metrics.get("feature_columns", list(data.keys()))
    row = {col: [data.get(col, metrics["defaults"].get(col, 0))] for col in feature_columns}
    input_df = pd.DataFrame(row)

    # Ensure numeric columns are numeric and categoricals are strings
    num_cols = metrics.get("numeric_features", [])
    cat_cols = metrics.get("categorical_features", [])
    for col in num_cols:
        if col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors="coerce").fillna(metrics["defaults"].get(col, 0.0))
    for col in cat_cols:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(str)

    # 5. Model prediction in log space
    pred_log = float(model.predict(input_df)[0])

    # 6. Inverse transform log1p -> expm1
    price = float(np.expm1(pred_log))

    # 7. Apply 90% prediction interval quantiles
    pi = metrics.get("prediction_interval_90", {})
    lower_q = float(pi.get("lower_quantile", -0.15))
    upper_q = float(pi.get("upper_quantile", 0.15))

    lower_bound = float(np.expm1(pred_log + lower_q))
    upper_bound = float(np.expm1(pred_log + upper_q))

    # Sanity bounds check
    if lower_bound > upper_bound:
        lower_bound, upper_bound = upper_bound, lower_bound
    if price < lower_bound:
        lower_bound = price * 0.90
    if price > upper_bound:
        upper_bound = price * 1.10

    return price, lower_bound, upper_bound


if __name__ == "__main__":
    # Self-test when run directly
    print("Testing predict.py inference...")
    try:
        sample_input = {
            "GrLivArea": 2200,
            "TotalBsmtSF": 1100,
            "OverallQual": 8,
            "OverallCond": 6,
            "YearBuilt": 2005,
            "GarageCars": 2,
            "GarageArea": 600,
            "FullBath": 2,
            "BedroomAbvGr": 3,
        }
        price, lower, upper = predict_price(sample_input)
        print(f"Predicted Price : ${price:,.2f}")
        print(f"90% Prediction Interval: [${lower:,.2f}, ${upper:,.2f}]")
    except FileNotFoundError as err:
        print(f"Note: {err}")
