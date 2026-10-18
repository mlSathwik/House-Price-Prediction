# 🏠 House Price Prediction

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://house-price-prediction-js2ohw8ks6tocolxe5uta6.streamlit.app/)
&nbsp;&nbsp;**Live Demo**: [https://house-price-prediction-js2ohw8ks6tocolxe5uta6.streamlit.app/](https://house-price-prediction-js2ohw8ks6tocolxe5uta6.streamlit.app/)

An end-to-end machine learning project that predicts house prices using the
Kaggle **Ames Housing** dataset (*House Prices: Advanced Regression Techniques*),
with an interactive Streamlit web app.

## ✨ Features
- **Predict**: enter house details and get an estimated price with a 90% prediction range
- **What-if Explorer**: see how changing one feature (quality, size, year built) changes the price, and compare two houses side by side
- **Explainability**: SHAP plots show why the model predicted a given price
- **Model Comparison**: leaderboard of 6 models with MAE, RMSE, R², MAPE and RMSLE
- **Data Explorer**: interactive Plotly charts of prices, neighborhoods and correlations

## 🧠 Machine Learning Approach
- Cleaning based on `data_description.txt` (NA values treated correctly), outlier removal
- Feature engineering: TotalSF, HouseAge, RemodAge, TotalBath, HasGarage, HasPool, HasFireplace
- Log-transformed target, scikit-learn `Pipeline` + `ColumnTransformer`
- Models compared: Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost
- 5-fold cross-validation and RandomizedSearchCV hyperparameter tuning
- Best model saved with joblib

## 📊 Results
| Model | MAE | RMSE | R² | RMSLE |
|-------|-----|------|----|-------|
| **Lasso (Best Model)** | **$14,018** | **$19,360** | **0.9321** | **0.1191** |
| Lasso (Tuned) | $14,130 | $19,534 | 0.9309 | 0.1177 |
| Ridge (Tuned) | $14,594 | $20,010 | 0.9275 | 0.1233 |
| Ridge | $14,534 | $20,058 | 0.9272 | 0.1231 |
| Gradient Boosting | $14,353 | $20,061 | 0.9271 | 0.1234 |
| XGBoost | $14,726 | $20,301 | 0.9254 | 0.1292 |
| Linear Regression | $15,455 | $21,786 | 0.9141 | 0.1367 |
| Random Forest | $16,625 | $23,842 | 0.8971 | 0.1462 |

## 🛠️ Tech Stack
Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, Plotly, Streamlit, pytest

## 🌐 Live Web App
👉 **[Open Live Application](https://house-price-prediction-js2ohw8ks6tocolxe5uta6.streamlit.app/)**

## 🚀 Run Locally
```bash
git clone https://github.com/mlSathwik/House-Price-Prediction.git
cd House-Price-Prediction
pip install -r requirements.txt
python src/train.py
streamlit run app.py
```

## 📁 Project Structure
```text
House-Price-Prediction/
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
├── submission.csv                # Generated predictions on test.csv
└── README.md                     # Project documentation
```

## ⚠️ Limitations
Trained on Ames, Iowa data (2006-2010). Predictions for other cities or
current market prices are not reliable.
