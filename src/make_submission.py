"""
make_submission.py
------------------
Generates Kaggle competition submission file from test.csv using the
best trained model pipeline.

Workflow:
1. Load test.csv and extract 'Id'.
2. Clean missing values according to data_description.txt rules.
3. Compute engineered features (TotalSF, HouseAge, RemodAge, TotalBath, etc.).
4. Load saved best_model.joblib and metadata from models/.
5. Run inference in log-scale and back-transform via expm1.
6. Write submission.csv formatted as (Id, SalePrice) matching sample_submission.csv.

Usage:
  python src/make_submission.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
MODELS_DIR = os.path.join(BASE_DIR, "models")

import sys
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from data_processing import clean_missing_values, engineer_features


def generate_submission(
    test_csv_path: str = None,
    output_csv_path: str = None,
    models_dir: str = MODELS_DIR,
) -> str:
    """
    Generate submission.csv from test.csv using the best trained pipeline.
    """
    if test_csv_path is None:
        test_csv_path = os.path.join(BASE_DIR, "test.csv")
    if output_csv_path is None:
        output_csv_path = os.path.join(BASE_DIR, "submission.csv")

    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(
            f"Test file not found at '{test_csv_path}'. "
            "Please ensure test.csv is placed in the project directory."
        )

    model_path = os.path.join(models_dir, "best_model.joblib")
    metrics_path = os.path.join(models_dir, "metrics.json")

    if not os.path.exists(model_path) or not os.path.exists(metrics_path):
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            "Please run 'python src/train.py' before generating submissions."
        )

    print("=" * 70)
    print(" KAGGLE AMES HOUSING - SUBMISSION GENERATOR")
    print("=" * 70)

    # 1. Load test dataset
    print(f"\n[1/5] Loading test data from: {test_csv_path}")
    test_df = pd.read_csv(test_csv_path)
    print(f"  Rows to predict: {len(test_df)}")

    if "Id" not in test_df.columns:
        raise ValueError("test.csv must contain an 'Id' column.")
    ids = test_df["Id"].copy()

    # 2. Clean missing values and engineer features
    print("\n[2/5] Cleaning missing values and engineering features...")
    cleaned_df = clean_missing_values(test_df)
    eng_df = engineer_features(cleaned_df)

    # 3. Load model and metadata
    print(f"\n[3/5] Loading trained pipeline from: {model_path}")
    model = joblib.load(model_path)
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    expected_cols = metrics.get("feature_columns", [])
    defaults = metrics.get("defaults", {})

    # Ensure all expected columns exist, filling any unexpected missing columns with defaults
    for col in expected_cols:
        if col not in eng_df.columns:
            eng_df[col] = defaults.get(col, 0)

    X_test = eng_df[expected_cols].copy()

    # Ensure correct data types
    num_cols = metrics.get("numeric_features", [])
    cat_cols = metrics.get("categorical_features", [])
    for col in num_cols:
        if col in X_test.columns:
            X_test[col] = pd.to_numeric(X_test[col], errors="coerce").fillna(defaults.get(col, 0.0))
    for col in cat_cols:
        if col in X_test.columns:
            X_test[col] = X_test[col].astype(str)

    # 4. Predict log-price and back-transform
    print("\n[4/5] Running model predictions...")
    pred_log = model.predict(X_test)
    pred_prices = np.expm1(pred_log)

    # Sanity clip to prevent negative or extreme prices
    pred_prices = np.clip(pred_prices, 10000, 1500000)

    # 5. Format and save submission.csv
    print(f"\n[5/5] Writing formatted predictions to: {output_csv_path}")
    submission_df = pd.DataFrame({
        "Id": ids,
        "SalePrice": np.round(pred_prices, 2),
    })
    submission_df.to_csv(output_csv_path, index=False)

    print("\n" + "=" * 70)
    print(f" SUBMISSION GENERATED SUCCESSFULLY ({len(submission_df)} rows)")
    print("=" * 70)
    print(f"  Output path: {output_csv_path}")
    print(f"  Min price  : ${submission_df['SalePrice'].min():,.2f}")
    print(f"  Max price  : ${submission_df['SalePrice'].max():,.2f}")
    print(f"  Mean price : ${submission_df['SalePrice'].mean():,.2f}")
    print("\nSample Preview (First 5 Rows):")
    print(submission_df.head())
    print("=" * 70)

    return output_csv_path


if __name__ == "__main__":
    generate_submission()
