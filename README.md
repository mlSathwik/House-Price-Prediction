# Ames House Price Prediction & Real Estate Valuation Platform

An end-to-end Machine Learning web application and portfolio project for estimating residential property market values in Ames, Iowa using the Kaggle Ames Housing dataset.

Built with **scikit-learn**, **XGBoost**, **SHAP**, **Plotly**, and **Streamlit**.

---

## 📸 Application Preview

```
+---------------------------------------------------------------------------------------+
|  🏡 Ames Valuation Platform                  [ 🏠 Valuation | 🔮 What-If | 🔍 SHAP ]   |
|                                                                                       |
|  [ 📐 Size & Layout ]    [ ⭐ Quality & Age ]   [ 📍 Location ]   [ 🚗 Garage ]       |
|  +---------------------------------------------------------------------------------+  |
|  |                 ESTIMATED PROPERTY MARKET VALUE                                 |  |
|  |                           $219,551                                              |  |
|  |         90% Prediction Interval: $179,863 — $258,420                             |  |
|  +---------------------------------------------------------------------------------+  |
|                                                                                       |
|  [ Gauge: Valuation vs Neighborhood Median ]       [ 📥 Download Valuation Report ]   |
+---------------------------------------------------------------------------------------+
```

*(Placeholders for screenshots: `docs/screenshots/predict_dashboard.png`, `docs/screenshots/shap_waterfall.png`, `docs/screenshots/model_comparison.png`)*

---

## 🌟 Key Features

1. **Robust Data Pipeline & Domain Engineering**:
   - Outlier elimination for non-market transactions (`GrLivArea > 4000` with low `SalePrice`).
   - Domain-specific NA imputation according to `data_description.txt` (`PoolQC`, `Alley`, `Fence`, `FireplaceQu`, `GarageType`, `BsmtQual` NA treated as *feature not present* / `"None"`).
   - Engineered composite features: `TotalSF`, `HouseAge`, `RemodAge`, `TotalBath`, and binary facility flags (`HasGarage`, `HasPool`, `HasFireplace`).
   - Log-transformed target (`log1p(SalePrice)`) to normalize right-skewness and stabilize variance.

2. **Automated Model Benchmarking & Tuning**:
   - Encapsulated preprocessing via scikit-learn `ColumnTransformer`.
   - Compares 6 core regression algorithms: **Linear Regression**, **Ridge**, **Lasso**, **Random Forest**, **Gradient Boosting**, and **XGBoost**.
   - 5-Fold cross-validation and hyperparameter optimization using `RandomizedSearchCV` on top candidates.
   - Non-parametric **90% empirical prediction intervals** derived from hold-out residual quantiles.

3. **Modern Multi-Page Streamlit Dashboard**:
   - **🏠 Predict Price**: Interactive property valuation with grouped expanders, readable categorical dropdowns, real-time input validation, neighborhood median gauge, and downloadable CSV report.
   - **🔮 What-If Sensitivity Explorer**: Dynamic single-feature sensitivity curves holding other inputs fixed, plus side-by-side comparison of two properties (House A vs. House B) with confidence intervals.
   - **🔍 Machine Learning Explainability (SHAP)**: Global feature importance, interactive SHAP waterfall attribution for individual valuations, and plain-English factor breakdowns.
   - **📈 Interactive Data Explorer**: SalePrice distributions (raw vs. log-scale), neighborhood price box plots, Pearson correlation heatmap, and bivariate scatter plots.
   - **📊 Model Comparison**: Leaderboard table with cross-validation and holdout metrics, Plotly benchmark bar charts, actual vs. predicted scatter, and residual homoscedasticity plot.
   - **ℹ️ About & Methodology**: Detailed project documentation, dataset origins, and real-world considerations.

4. **Kaggle Submission Generator**:
   - Dedicated script (`src/make_submission.py`) to process `test.csv` and generate formatted `submission.csv` files.

---

## 🏆 Model Performance Benchmark (80/20 Holdout Split)

The models were evaluated on a 20% holdout test set ($N = 292$) with target back-transformation:

| Model | 5-Fold CV R² | Holdout MAE ($) | Holdout RMSE ($) | Holdout R² | Holdout MAPE (%) | Holdout RMSLE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Lasso (Best Model) \*** | **0.9133** | **\$14,018** | **\$19,360** | **0.9321** | **8.68%** | **0.1191** |
| **Lasso (Tuned)** | 0.9158 | \$14,130 | \$19,534 | 0.9309 | 8.65% | 0.1177 |
| **Ridge (Tuned)** | 0.9159 | \$14,594 | \$20,010 | 0.9275 | 9.00% | 0.1233 |
| **Ridge** | 0.9156 | \$14,534 | \$20,058 | 0.9272 | 8.97% | 0.1231 |
| **Gradient Boosting** | 0.9017 | \$14,353 | \$20,061 | 0.9271 | 8.85% | 0.1234 |
| **XGBoost** | 0.8924 | \$14,726 | \$20,301 | 0.9254 | 9.24% | 0.1292 |
| **Linear Regression** | 0.8895 | \$15,455 | \$21,786 | 0.9141 | 9.57% | 0.1367 |
| **Random Forest** | 0.8777 | \$16,625 | \$23,842 | 0.8971 | 10.44% | 0.1462 |

- **Best Model**: `Lasso` ($R^2 = 0.9321$, $\text{RMSE} = \$19,360$, $\text{RMSLE} = 0.1191$).
- **90% Prediction Interval**: $[-0.1994, +0.1630]$ in log-space.

---

## 📁 Project Structure

```
house-price-prediction/
├── .streamlit/
│   └── config.toml               # Custom dark theme configuration
├── models/
│   ├── best_model.joblib         # Trained scikit-learn pipeline artifact
│   ├── metrics.json              # Model metrics, defaults, top features & quantiles
│   └── holdout_predictions.csv   # Predictions and residuals on 20% holdout split
├── src/
│   ├── __init__.py
│   ├── data_processing.py        # Cleaning, outlier removal, NA handling & feature engineering
│   ├── train.py                  # Training pipeline, CV, RandomizedSearchCV, and metrics
│   ├── predict.py                # Inference function with default filling & interval computation
│   └── make_submission.py        # Kaggle test set inference and submission generator
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py          # Pytest suite covering data cleaning, features & inference
├── app.py                        # Streamlit 6-page interactive web application
├── requirements.txt              # Pinned environment dependencies
├── train.csv                     # Training dataset (Kaggle Ames Housing)
├── test.csv                      # Test dataset for competition submissions
├── sample_submission.csv         # Kaggle sample submission format
└── README.md                     # Documentation
```

---

## 🚀 Setup & Execution Guide

### 1. Environment Installation
Clone the repository, create a virtual environment, and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Model Training & Evaluation
Train all 6 models, run cross-validation and hyperparameter search, and save model artifacts:

```powershell
python src/train.py
```

### 3. Launch Web Application
Start the Streamlit dashboard in your browser:

```powershell
streamlit run app.py
```

The application will launch at `http://localhost:8501`.

### 4. Run Pytest Test Suite
Execute the unit and invariant tests:

```powershell
pytest tests/ -v
```

### 5. Generate Kaggle Submission (Optional)
Generate a `submission.csv` file from `test.csv`:

```powershell
python src/make_submission.py
```

---

## ⚠️ Real-World Limitations

1. **Geographic Localization**:
   Trained exclusively on residential properties in Ames, Iowa. Land price multipliers, local taxation, and property premiums reflect Midwestern university town dynamics and do not directly translate to major metropolitan or coastal markets.

2. **Temporal Market Drift & Inflation**:
   Transactions in the Ames dataset span 2006 to 2010. Point estimates reflect pre-2011 market prices and require indexation to account for post-pandemic real estate appreciation and macroeconomic inflation.

3. **High-End Architectural Outliers**:
   Ultra-luxury custom estates (> 4,000 sq ft living area) possess unique architectural attributes and higher residual uncertainty compared to standard tract or suburban homes.
