"""
data_processing.py
------------------
Data loading, cleaning, feature engineering, and preprocessing utilities
for the Kaggle Ames Housing dataset.

Key steps:
1. Load dataset (train.csv only, dropping 'Id').
2. Remove extreme outliers (GrLivArea > 4000 sq ft with low SalePrice).
3. Handle missing values according to data_description.txt:
   - For categorical features where NA means "not present" (e.g., PoolQC,
     Alley, Fence, FireplaceQu, GarageType, BsmtQual), fill with 'None'.
   - For corresponding numeric features (e.g., GarageArea, BsmtFinSF1),
     fill with 0.
4. Engineer new domain-specific features:
   - TotalSF: Total square footage across basement, 1st floor, and 2nd floor
   - HouseAge: Age of the house when sold
   - RemodAge: Years since last remodel when sold
   - TotalBath: Full and half bathrooms (above and below ground) combined
   - HasGarage: Binary flag indicating garage presence
   - HasPool: Binary flag indicating pool presence
   - HasFireplace: Binary flag indicating fireplace presence
5. Log-transform the target (SalePrice) using log1p to normalize right skew.
"""

import os
import numpy as np
import pandas as pd

# List of categorical columns where NA means "feature is not present"
NONE_FILL_CATEGORICAL = [
    "PoolQC",
    "Alley",
    "Fence",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "MasVnrType",
    "MiscFeature",
]

# Numeric columns where NA indicates absence of facility (so 0 is appropriate)
ZERO_FILL_NUMERIC = [
    "TotalBsmtSF",
    "1stFlrSF",
    "2ndFlrSF",
    "GarageArea",
    "GarageCars",
    "PoolArea",
    "Fireplaces",
    "FullBath",
    "HalfBath",
    "BsmtFullBath",
    "BsmtHalfBath",
    "MasVnrArea",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "GarageYrBlt",
]


def resolve_data_path(custom_path: str = None) -> str:
    """
    Resolve the absolute path to the training dataset.
    Prioritizes train.csv, with fallback to house_prices_kaggle.csv.
    """
    if custom_path and os.path.exists(custom_path):
        return os.path.abspath(custom_path)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(base_dir, "train.csv"),
        os.path.join(base_dir, "house_prices_kaggle.csv"),
        os.path.join(os.getcwd(), "train.csv"),
        os.path.join(os.getcwd(), "house_prices_kaggle.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "Could not find train.csv or house_prices_kaggle.csv in project folder."
    )


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove influential leverage outliers from the Ames Housing dataset:
    Homes with GrLivArea > 4000 sq ft and low SalePrice (< $300,000).
    Dean De Cock (dataset author) recommends removing these unusual sales.
    """
    if "GrLivArea" in df.columns and "SalePrice" in df.columns:
        outlier_mask = (df["GrLivArea"] > 4000) & (df["SalePrice"] < 300_000)
        return df[~outlier_mask].copy()
    return df.copy()


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values per data_description.txt specifications:
    - Categorical features where NA means 'None' (e.g. no garage, no basement)
    - Numeric features where NA means 0 (e.g. 0 sq ft garage/basement)
    """
    df = df.copy()

    # Fill 'feature not present' categoricals with 'None'
    for col in NONE_FILL_CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    # Fill facility area/count numerics with 0
    for col in ZERO_FILL_NUMERIC:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create 7 domain-specific engineered features:
    1. TotalSF: Total square footage (Basement + 1st Floor + 2nd Floor)
    2. HouseAge: Age of house at time of sale (YrSold - YearBuilt)
    3. RemodAge: Years between remodel and sale (YrSold - YearRemodAdd)
    4. TotalBath: Weighted bathroom count (Full + 0.5*Half + BsmtFull + 0.5*BsmtHalf)
    5. HasGarage: 1 if GarageArea > 0, else 0
    6. HasPool: 1 if PoolArea > 0, else 0
    7. HasFireplace: 1 if Fireplaces > 0, else 0
    """
    df = df.copy()

    # Safe access with fallback to Series of default if a column is missing
    def _col(name, default=0):
        if name in df.columns:
            return df[name]
        return pd.Series(default, index=df.index)

    # 1. TotalSF
    df["TotalSF"] = _col("TotalBsmtSF") + _col("1stFlrSF") + _col("2ndFlrSF")

    # 2 & 3. HouseAge and RemodAge
    yr_sold = _col("YrSold", default=2010)
    yr_built = _col("YearBuilt", default=1970)
    yr_remod = _col("YearRemodAdd", default=yr_built)

    house_age = yr_sold - yr_built
    remod_age = yr_sold - yr_remod

    # If pandas Series, clamp to 0; otherwise max(0, val)
    if isinstance(house_age, pd.Series):
        df["HouseAge"] = house_age.clip(lower=0)
        df["RemodAge"] = remod_age.clip(lower=0)
    else:
        df["HouseAge"] = max(0, house_age)
        df["RemodAge"] = max(0, remod_age)

    # 4. TotalBath
    df["TotalBath"] = (
        _col("FullBath")
        + 0.5 * _col("HalfBath")
        + _col("BsmtFullBath")
        + 0.5 * _col("BsmtHalfBath")
    )

    # 5. HasGarage
    garage_area = _col("GarageArea")
    df["HasGarage"] = (garage_area > 0).astype(int)

    # 6. HasPool
    pool_area = _col("PoolArea")
    df["HasPool"] = (pool_area > 0).astype(int)

    # 7. HasFireplace
    fireplaces = _col("Fireplaces")
    df["HasFireplace"] = (fireplaces > 0).astype(int)

    return df


def compute_defaults(df: pd.DataFrame) -> dict:
    """
    Compute column defaults:
    - Numeric columns: median
    - Categorical columns: mode (most frequent value)
    Used by predict.py and Streamlit UI to fill unprovided inputs.
    """
    defaults = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            med = df[col].median()
            defaults[col] = float(med) if pd.notnull(med) else 0.0
        else:
            mode_series = df[col].mode()
            defaults[col] = str(mode_series.iloc[0]) if not mode_series.empty else "None"
    return defaults


def load_and_clean(csv_path: str = None) -> tuple[pd.DataFrame, pd.Series, list[str], list[str], dict]:
    """
    Full pipeline to load and prepare the Ames Housing data:
    1. Resolve file path (train.csv).
    2. Drop 'Id' column.
    3. Remove extreme outliers.
    4. Clean NA values according to data_description.txt.
    5. Engineer features (TotalSF, HouseAge, RemodAge, TotalBath, HasGarage, HasPool, HasFireplace).
    6. Compute column defaults (median for numeric, mode for categorical).
    7. Log-transform SalePrice using np.log1p.

    Returns
    -------
    X : pd.DataFrame
        Cleaned feature matrix (without SalePrice).
    y : pd.Series
        Log-transformed target (np.log1p(SalePrice)).
    numeric_cols : list[str]
        List of numeric feature column names.
    categorical_cols : list[str]
        List of categorical feature column names.
    defaults : dict
        Default values for each feature column.
    """
    path = resolve_data_path(csv_path)
    df = pd.read_csv(path)

    # 1. Drop Id column if present
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])

    # 2. Must have target SalePrice in training data
    if "SalePrice" not in df.columns:
        raise ValueError(f"Dataset at {path} does not contain 'SalePrice' column.")
    df = df.dropna(subset=["SalePrice"]).copy()

    # 3. Remove outliers
    df = remove_outliers(df)

    # 4. Clean NAs per data_description.txt
    df = clean_missing_values(df)

    # 5. Engineer features
    df = engineer_features(df)

    # 6. Separate features and target
    y = np.log1p(df["SalePrice"])
    X = df.drop(columns=["SalePrice"])

    # Separate column types
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    # 7. Compute defaults for all features
    defaults = compute_defaults(X)

    return X, y, numeric_cols, categorical_cols, defaults
