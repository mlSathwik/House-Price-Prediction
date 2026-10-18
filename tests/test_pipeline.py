"""
test_pipeline.py
----------------
Unit and integration tests for Ames Housing Machine Learning Pipeline:
1. Data cleaning and NA handling per data_description.txt.
2. Feature engineering: TotalSF, HouseAge, RemodAge, TotalBath, and binary flags.
3. Prediction inference: positive price and valid 90% confidence interval ordering.
4. Robust handling of missing, empty, and partial inputs.
5. Verification of saved model artifacts.
"""

import os
import json
import numpy as np
import pandas as pd
import pytest

import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_processing import (
    remove_outliers,
    clean_missing_values,
    engineer_features,
    load_and_clean,
    NONE_FILL_CATEGORICAL,
    ZERO_FILL_NUMERIC,
)
from predict import predict_price, load_model_artifacts


# -----------------------------------------------------------------------------
# 1. Data Cleaning & Outlier Removal Tests
# -----------------------------------------------------------------------------
def test_remove_outliers():
    """Verify that GrLivArea > 4000 with SalePrice < 300,000 are removed."""
    df = pd.DataFrame({
        "GrLivArea": [2000, 4500, 4200, 3500],
        "SalePrice": [250000, 180000, 750000, 300000],
    })
    cleaned = remove_outliers(df)
    assert len(cleaned) == 3
    assert 180000 not in cleaned["SalePrice"].values
    assert 750000 in cleaned["SalePrice"].values
    assert 250000 in cleaned["SalePrice"].values


def test_data_cleaning_and_na_handling():
    """
    Verify NA handling per data_description.txt:
    - Categorical features where NA means 'feature not present' -> filled with 'None'
    - Numeric features where NA means absence of facility -> filled with 0
    - Non-null values are preserved unchanged
    """
    sample_data = {
        "PoolQC": [np.nan, "Ex", np.nan],
        "Alley": [np.nan, "Grvl", np.nan],
        "Fence": [np.nan, "GdPrv", np.nan],
        "FireplaceQu": [np.nan, "TA", np.nan],
        "GarageType": [np.nan, "Attchd", np.nan],
        "BsmtQual": [np.nan, "Gd", np.nan],
        "GarageArea": [np.nan, 450.0, np.nan],
        "TotalBsmtSF": [np.nan, 1000.0, np.nan],
        "MasVnrArea": [np.nan, 200.0, np.nan],
        "OverallQual": [6, 7, 8],
    }
    df = pd.DataFrame(sample_data)
    cleaned = clean_missing_values(df)

    # Check that categoricals are filled with 'None'
    for cat_col in ["PoolQC", "Alley", "Fence", "FireplaceQu", "GarageType", "BsmtQual"]:
        assert cleaned[cat_col].iloc[0] == "None"
        assert cleaned[cat_col].iloc[1] == sample_data[cat_col][1]
        assert cleaned[cat_col].iloc[2] == "None"

    # Check that numerics are filled with 0
    for num_col in ["GarageArea", "TotalBsmtSF", "MasVnrArea"]:
        assert cleaned[num_col].iloc[0] == 0.0
        assert cleaned[num_col].iloc[1] == sample_data[num_col][1]
        assert cleaned[num_col].iloc[2] == 0.0

    # Ensure unmodified columns remain intact
    assert cleaned["OverallQual"].tolist() == [6, 7, 8]


# -----------------------------------------------------------------------------
# 2. Feature Engineering Tests (TotalSF, HouseAge, TotalBath)
# -----------------------------------------------------------------------------
def test_feature_engineering_total_sf():
    """Verify TotalSF calculation: TotalBsmtSF + 1stFlrSF + 2ndFlrSF."""
    df = pd.DataFrame({
        "TotalBsmtSF": [800, 0, 1200],
        "1stFlrSF": [1000, 1100, 1400],
        "2ndFlrSF": [600, 0, 900],
    })
    eng = engineer_features(df)
    assert eng["TotalSF"].tolist() == [2400, 1100, 3500]


def test_feature_engineering_house_age():
    """Verify HouseAge calculation: YrSold - YearBuilt with non-negative clamping."""
    df = pd.DataFrame({
        "YrSold": [2010, 2008, 2005],
        "YearBuilt": [2000, 1950, 2006],  # 3rd row: sold before built (edge case)
        "YearRemodAdd": [2005, 1990, 2006],
    })
    eng = engineer_features(df)
    assert eng["HouseAge"].iloc[0] == 10
    assert eng["HouseAge"].iloc[1] == 58
    assert eng["HouseAge"].iloc[2] == 0  # Clamped to >= 0


def test_feature_engineering_total_bath():
    """Verify TotalBath calculation: FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath."""
    df = pd.DataFrame({
        "FullBath": [2, 1, 3],
        "HalfBath": [1, 0, 2],
        "BsmtFullBath": [1, 0, 1],
        "BsmtHalfBath": [0, 1, 1],
    })
    eng = engineer_features(df)
    assert eng["TotalBath"].iloc[0] == 2 + 0.5 + 1 + 0.0  # 3.5
    assert eng["TotalBath"].iloc[1] == 1 + 0.0 + 0 + 0.5  # 1.5
    assert eng["TotalBath"].iloc[2] == 3 + 1.0 + 1 + 0.5  # 5.5


def test_feature_engineering_binary_flags():
    """Verify binary indicators: HasGarage, HasPool, HasFireplace."""
    df = pd.DataFrame({
        "GarageArea": [400, 0],
        "PoolArea": [0, 250],
        "Fireplaces": [1, 0],
    })
    eng = engineer_features(df)
    assert eng["HasGarage"].tolist() == [1, 0]
    assert eng["HasPool"].tolist() == [0, 1]
    assert eng["HasFireplace"].tolist() == [1, 0]


# -----------------------------------------------------------------------------
# 3. Predict Price Invariants (Positive Price & Confidence Interval Ordering)
# -----------------------------------------------------------------------------
def test_predict_price_returns_positive_and_ordered_bounds():
    """
    Verify that predict_price returns:
    1. A strictly positive price (> 0).
    2. Lower bound and upper bound ordered as: lower_bound <= price <= upper_bound.
    3. All return values are valid floats.
    """
    test_inputs = {
        "GrLivArea": 2100,
        "TotalBsmtSF": 1050,
        "OverallQual": 8,
        "OverallCond": 6,
        "YearBuilt": 2004,
        "Neighborhood": "CollgCr",
        "GarageCars": 2,
        "FullBath": 2,
    }
    price, lower_bound, upper_bound = predict_price(test_inputs)

    assert isinstance(price, float)
    assert isinstance(lower_bound, float)
    assert isinstance(upper_bound, float)

    # 1. Strictly positive
    assert price > 0.0
    assert lower_bound > 0.0
    assert upper_bound > 0.0

    # 2. Confidence interval ordering invariant
    assert lower_bound <= price <= upper_bound

    # Realistic price sanity range for an Ames home with these specs
    assert 100000 <= price <= 500000


# -----------------------------------------------------------------------------
# 4. Missing, Empty, and Partial Input Robustness Tests
# -----------------------------------------------------------------------------
def test_predict_price_empty_dict():
    """Verify predict_price({}) successfully falls back to saved defaults."""
    price, lower, upper = predict_price({})
    assert isinstance(price, float)
    assert price > 0.0
    assert lower <= price <= upper


def test_predict_price_none_input():
    """Verify predict_price(None) successfully falls back to saved defaults."""
    price, lower, upper = predict_price(None)
    assert isinstance(price, float)
    assert price > 0.0
    assert lower <= price <= upper


def test_predict_price_partial_input():
    """
    Verify predict_price with a single specified feature fills all other
    features with defaults and correctly computes valuation.
    """
    # High-quality home should be valued higher than a low-quality home
    high_qual_price, _, _ = predict_price({"OverallQual": 10})
    low_qual_price, _, _ = predict_price({"OverallQual": 3})

    assert high_qual_price > low_qual_price


def test_predict_price_handles_unexpected_keys():
    """Verify predict_price silently ignores unrecognized keys without raising errors."""
    price, lower, upper = predict_price({
        "ArbitraryNonExistentColumn": 99999,
        "AnotherRandomField": "XYZ",
        "GrLivArea": 1600,
    })
    assert isinstance(price, float)
    assert price > 0.0
    assert lower <= price <= upper


# -----------------------------------------------------------------------------
# 5. Saved Model Artifacts Verification
# -----------------------------------------------------------------------------
def test_model_artifacts_integrity():
    """Verify that all required Phase 1-3 artifacts exist and contain valid structures."""
    models_dir = os.path.join(BASE_DIR, "models")
    best_model_path = os.path.join(models_dir, "best_model.joblib")
    metrics_path = os.path.join(models_dir, "metrics.json")
    holdout_path = os.path.join(models_dir, "holdout_predictions.csv")

    assert os.path.exists(best_model_path), "best_model.joblib missing"
    assert os.path.exists(metrics_path), "metrics.json missing"
    assert os.path.exists(holdout_path), "holdout_predictions.csv missing"

    with open(metrics_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert "best_model_name" in meta
    assert "models" in meta
    assert len(meta["models"]) >= 6  # At least 6 models compared
    assert "prediction_interval_90" in meta
    assert "top_15_features" in meta
    assert len(meta["top_15_features"]) == 15
    assert "defaults" in meta
    assert len(meta["defaults"]) > 50

    holdout_df = pd.read_csv(holdout_path)
    assert "Actual" in holdout_df.columns
    assert "Predicted" in holdout_df.columns
    assert "Residual" in holdout_df.columns
    assert len(holdout_df) > 100
