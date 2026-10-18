"""
train.py
--------
Training, cross-validation, hyperparameter tuning, evaluation, and persistence
for the Ames House Price Prediction pipeline.

Models compared:
1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Random Forest Regressor
5. Gradient Boosting Regressor
6. XGBoost Regressor

Execution:
  python src/train.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib

# Suppress sklearn convergence/future warnings during cross-validation
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Try importing XGBoost
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Import data processing utilities
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from data_processing import load_and_clean

# Base project paths
BASE_DIR = os.path.dirname(CURRENT_DIR)
MODELS_DIR = os.path.join(BASE_DIR, "models")


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """
    Build scikit-learn ColumnTransformer:
    - Numeric: median imputation + StandardScaler
    - Categorical: 'None' constant imputation + OneHotEncoder (handles unseen levels)
    """
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor


def get_candidate_models(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    """
    Return dictionary of candidate models wrapped in full scikit-learn pipelines.
    """
    models = {
        "Linear Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]),
        "Ridge": Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", Ridge(alpha=10.0, random_state=42)),
        ]),
        "Lasso": Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", Lasso(alpha=0.001, max_iter=10000, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=100, random_state=42)),
        ]),
        "Gradient Boosting": Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ]),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", XGBRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=4,
                random_state=42,
                verbosity=0,
            )),
        ])

    return models


# Hyperparameter search distributions for tuning the top 2 models
PARAM_DISTRIBUTIONS = {
    "Gradient Boosting": {
        "regressor__n_estimators": [100, 150, 200],
        "regressor__learning_rate": [0.03, 0.05, 0.1],
        "regressor__max_depth": [3, 4, 5],
        "regressor__subsample": [0.8, 0.9, 1.0],
    },
    "XGBoost": {
        "regressor__n_estimators": [100, 150, 200],
        "regressor__learning_rate": [0.03, 0.05, 0.1],
        "regressor__max_depth": [3, 4, 5],
        "regressor__subsample": [0.8, 0.9, 1.0],
    },
    "Random Forest": {
        "regressor__n_estimators": [100, 200, 300],
        "regressor__max_depth": [None, 10, 15, 20],
        "regressor__min_samples_split": [2, 5],
        "regressor__min_samples_leaf": [1, 2],
    },
    "Ridge": {
        "regressor__alpha": [0.5, 1.0, 5.0, 10.0, 20.0, 50.0],
    },
    "Lasso": {
        "regressor__alpha": [0.0001, 0.0005, 0.001, 0.005, 0.01],
    },
}


def calculate_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict:
    """
    Calculate evaluation metrics:
    - MAE: Mean Absolute Error ($)
    - RMSE: Root Mean Squared Error ($)
    - R2: Coefficient of determination
    - MAPE: Mean Absolute Percentage Error (%)
    - RMSLE: Root Mean Squared Logarithmic Error (RMSE on log price)
    """
    y_true_dollars = np.expm1(y_true_log)
    y_pred_dollars = np.expm1(y_pred_log)

    mae = float(mean_absolute_error(y_true_dollars, y_pred_dollars))
    mse = float(mean_squared_error(y_true_dollars, y_pred_dollars))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true_dollars, y_pred_dollars))

    # MAPE
    non_zero = y_true_dollars != 0
    mape = float(np.mean(np.abs((y_true_dollars[non_zero] - y_pred_dollars[non_zero]) / y_true_dollars[non_zero])) * 100)

    # RMSLE is the RMSE directly on log-scale targets
    rmsle = float(np.sqrt(mean_squared_error(y_true_log, y_pred_log)))

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "MAPE": round(mape, 2),
        "RMSLE": round(rmsle, 4),
    }


def extract_top_features(pipeline: Pipeline, numeric_cols: list[str], categorical_cols: list[str], top_n: int = 15) -> list[dict]:
    """
    Extract top N most important features from the trained pipeline regressor.
    Aggregates one-hot encoded categories back to their original column names.
    """
    pre = pipeline.named_steps["preprocessor"]
    reg = pipeline.named_steps["regressor"]

    # Retrieve feature names from ColumnTransformer
    try:
        encoded_feature_names = pre.get_feature_names_out()
    except Exception:
        # Fallback if get_feature_names_out fails
        return [{"feature": col, "importance": 1.0} for col in numeric_cols[:top_n]]

    # Get raw importance values
    if hasattr(reg, "feature_importances_"):
        raw_importances = reg.feature_importances_
    elif hasattr(reg, "coef_"):
        raw_importances = np.abs(reg.coef_)
    else:
        return []

    # Aggregate importance by original column name
    aggregated = {}
    for feat_name, imp in zip(encoded_feature_names, raw_importances):
        # strip prefix like 'num__' or 'cat__'
        clean_name = feat_name.split("__", 1)[-1]
        orig_col = None
        if feat_name.startswith("cat__"):
            for c in categorical_cols:
                if clean_name.startswith(c):
                    orig_col = c
                    break
        if orig_col is None:
            orig_col = clean_name

        aggregated[orig_col] = aggregated.get(orig_col, 0.0) + float(imp)

    # Sort descending
    sorted_features = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)
    total_imp = sum(imp for _, imp in sorted_features) or 1.0

    return [
        {"feature": name, "importance": round(imp / total_imp, 4)}
        for name, imp in sorted_features[:top_n]
    ]


def train_models():
    """
    Complete training orchestrator:
    1. Loads and preprocesses dataset.
    2. Splits into 80% train and 20% holdout test.
    3. Runs 5-fold cross-validation on all 6 candidate models.
    4. Performs RandomizedSearchCV on the top 2 models.
    5. Evaluates all models on holdout test set across MAE, RMSE, R2, MAPE, RMSLE.
    6. Identifies best model and computes 90% prediction interval.
    7. Saves artifacts:
       - models/best_model.joblib
       - models/metrics.json
       - models/holdout_predictions.csv
    8. Prints a formatted metrics table.
    """
    print("=" * 80)
    print(" HOUSE PRICE PREDICTION - MODEL TRAINING & EVALUATION PIPELINE")
    print("=" * 80)

    # 1. Load data
    print("\n[Step 1/6] Loading and cleaning dataset...")
    X, y, num_cols, cat_cols, defaults = load_and_clean()
    orig_prices = np.expm1(y.values)
    print(f"  Rows after outlier removal: {len(X)}")
    print(f"  Total features: {X.shape[1]} ({len(num_cols)} numeric, {len(cat_cols)} categorical)")

    # 2. Train-test split (80/20 holdout)
    print("\n[Step 2/6] Splitting data into 80% train / 20% hold-out...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"  Train samples: {len(X_train)}")
    print(f"  Hold-out test samples: {len(X_test)}")

    # 3. Build pipelines
    preprocessor = build_preprocessor(num_cols, cat_cols)
    candidate_models = get_candidate_models(preprocessor)

    # 4. 5-Fold Cross-Validation
    print("\n[Step 3/6] Running 5-fold cross-validation on candidate models...")
    cv_scores = {}
    for name, pipe in candidate_models.items():
        scores = cross_val_score(
            pipe, X_train, y_train, cv=5, scoring="r2", n_jobs=1
        )
        mean_r2 = float(scores.mean())
        std_r2 = float(scores.std())
        cv_scores[name] = mean_r2
        print(f"  {name:20s} -> CV R2: {mean_r2:.4f} (+/- {std_r2:.4f})")

    # 5. RandomizedSearchCV on top 2 models
    top_2_names = sorted(cv_scores, key=cv_scores.get, reverse=True)[:2]
    print(f"\n[Step 4/6] Running RandomizedSearchCV hyperparameter tuning on top 2 models: {top_2_names}...")

    all_models = dict(candidate_models)

    for name in top_2_names:
        param_grid = PARAM_DISTRIBUTIONS.get(name)
        if param_grid:
            print(f"  Tuning {name} with 15 iterations (5-fold CV)...")
            search = RandomizedSearchCV(
                candidate_models[name],
                param_distributions=param_grid,
                n_iter=15,
                cv=5,
                scoring="r2",
                random_state=42,
                n_jobs=1,
            )
            search.fit(X_train, y_train)
            tuned_name = f"{name} (Tuned)"
            all_models[tuned_name] = search.best_estimator_
            tuned_cv_scores = cross_val_score(
                search.best_estimator_, X_train, y_train, cv=5, scoring="r2", n_jobs=1
            )
            cv_scores[tuned_name] = float(tuned_cv_scores.mean())
            print(f"  -> {tuned_name} CV R2: {cv_scores[tuned_name]:.4f} (Best params: {search.best_params_})")

    # 6. Evaluate all models on Hold-out Test Set
    print("\n[Step 5/6] Evaluating all models on 80/20 hold-out test set...")
    metrics_report = {}
    test_predictions_dict = {}

    for name, pipe in all_models.items():
        pipe.fit(X_train, y_train)
        pred_log = pipe.predict(X_test)
        metrics = calculate_metrics(y_test.values, pred_log)
        metrics["CV_R2"] = round(cv_scores.get(name, 0.0), 4)
        metrics_report[name] = metrics
        test_predictions_dict[name] = pred_log

    # Determine best model based on holdout R2 (and RMSE)
    best_model_name = max(metrics_report, key=lambda k: metrics_report[k]["R2"])
    best_pipe = all_models[best_model_name]
    best_pred_log = test_predictions_dict[best_model_name]

    # Calculate 90% prediction interval from residual quantiles
    residuals_log = y_test.values - best_pred_log
    lower_quantile = float(np.percentile(residuals_log, 5))
    upper_quantile = float(np.percentile(residuals_log, 95))

    # Extract top 15 most important features
    top_15_features = extract_top_features(best_pipe, num_cols, cat_cols, top_n=15)

    # 7. Save Artifacts
    print(f"\n[Step 6/6] Saving artifacts to {MODELS_DIR}...")
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save best model with joblib
    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    joblib.dump(best_pipe, model_path)
    print(f"  Saved best model ({best_model_name}) to: {model_path}")

    # Save hold-out predictions CSV
    actual_test_dollars = np.expm1(y_test.values)
    pred_test_dollars = np.expm1(best_pred_log)
    holdout_df = pd.DataFrame({
        "Actual": np.round(actual_test_dollars, 2),
        "Predicted": np.round(pred_test_dollars, 2),
        "Residual": np.round(actual_test_dollars - pred_test_dollars, 2),
        "AbsError": np.round(np.abs(actual_test_dollars - pred_test_dollars), 2),
        "PctError": np.round(np.abs((actual_test_dollars - pred_test_dollars) / actual_test_dollars) * 100, 2),
    })
    holdout_path = os.path.join(MODELS_DIR, "holdout_predictions.csv")
    holdout_df.to_csv(holdout_path, index=False)
    print(f"  Saved hold-out predictions to: {holdout_path}")

    # Build comprehensive metrics dictionary
    final_metrics = {
        "best_model_name": best_model_name,
        "models": metrics_report,
        "prediction_interval_90": {
            "confidence": 0.90,
            "lower_quantile": round(lower_quantile, 5),
            "upper_quantile": round(upper_quantile, 5),
        },
        "top_15_features": top_15_features,
        "defaults": defaults,
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "feature_columns": list(X.columns),
        "data_stats": {
            "count": int(len(X)),
            "min_price": int(orig_prices.min()),
            "max_price": int(orig_prices.max()),
            "mean_price": int(orig_prices.mean()),
        },
    }

    metrics_json_path = os.path.join(MODELS_DIR, "metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"  Saved metrics to: {metrics_json_path}")

    # Print Formatted Metrics Table
    print("\n" + "=" * 90)
    print(f"{'MODEL PERFORMANCE COMPARISON (80/20 Holdout Split)':^90}")
    print("=" * 90)
    header = f"{'Model':<26} | {'CV R2':<8} | {'MAE ($)':<11} | {'RMSE ($)':<11} | {'R2':<8} | {'MAPE (%)':<9} | {'RMSLE':<8}"
    print(header)
    print("-" * 90)
    for model_name, m in metrics_report.items():
        marker = " *" if model_name == best_model_name else ""
        row = (
            f"{model_name + marker:<26} | "
            f"{m['CV_R2']:<8.4f} | "
            f"${m['MAE']:<10,.0f} | "
            f"${m['RMSE']:<10,.0f} | "
            f"{m['R2']:<8.4f} | "
            f"{m['MAPE']:<9.2f} | "
            f"{m['RMSLE']:<8.4f}"
        )
        print(row)
    print("=" * 90)
    print(f"* Best Model: {best_model_name}")
    print(f"90% Prediction Interval: [{lower_quantile:+.4f}, {upper_quantile:+.4f}] in log-space")
    print("=" * 90 + "\n")

    return final_metrics


if __name__ == "__main__":
    train_models()