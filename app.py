"""
app.py
------
Modern Multi-Page Streamlit Application for Ames House Price Prediction.

Full Phase 3 Implementation:
1. 🏠 Predict Price: Grouped input expanders, 90% prediction interval,
   neighborhood comparison gauge, validation, and CSV export.
2. 🔮 What-If Explorer: Single-feature sensitivity curves holding other inputs fixed,
   plus side-by-side comparison of House A vs House B.
3. 🔍 Feature Explainability: Global SHAP summary and importance plots, plus an
   instance-level SHAP waterfall for the current prediction with plain-English insights.
4. 📈 Data Explorer: Interactive SalePrice distribution, neighborhood box plots,
   correlation heatmap of top features, and a 2-feature scatter colored by quality.
5. 📊 Model Comparison: Leaderboard table, Plotly RMSE/R² bar charts,
   Actual vs Predicted scatter, and holdout residual plot.
6. ℹ️ About & Methodology: Dataset context, ML methodology, and real-world limitations.
"""

import os
import json
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Paths setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SRC_DIR = os.path.join(BASE_DIR, "src")

import sys
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from predict import predict_price
from data_processing import load_and_clean

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Ames House Price Valuation Platform",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Custom Styling (Dark Modern Theme)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
/* Main title accent */
.main-header {
    font-size: 2.1rem;
    font-weight: 700;
    color: #FAFAFA;
    margin-bottom: 0.2rem;
}
.sub-header {
    font-size: 0.95rem;
    color: #9E9E9E;
    margin-bottom: 1.5rem;
}

/* Prediction Metric Card */
.prediction-card {
    background: linear-gradient(135deg, #1E88E5 0%, #1565C0 100%);
    padding: 1.6rem;
    border-radius: 12px;
    text-align: center;
    color: #FFFFFF;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4);
    margin-bottom: 1.5rem;
}
.prediction-label {
    font-size: 0.95rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    opacity: 0.9;
    margin-bottom: 0.2rem;
}
.prediction-value {
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -1px;
    margin: 0.2rem 0;
}
.prediction-range {
    font-size: 1.05rem;
    background: rgba(0, 0, 0, 0.25);
    display: inline-block;
    padding: 0.4rem 1.2rem;
    border-radius: 20px;
    margin-top: 0.4rem;
}

/* Info & Callout cards */
.info-card {
    background-color: #1C2333;
    border-left: 4px solid #1E88E5;
    padding: 1rem 1.2rem;
    border-radius: 8px;
    margin-bottom: 1rem;
    color: #E0E0E0;
}
.stat-box {
    background-color: #1C2333;
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
    border: 1px solid #2A3447;
}
.stat-box .stat-val {
    font-size: 1.5rem;
    font-weight: 700;
    color: #1E88E5;
}
.stat-box .stat-lbl {
    font-size: 0.85rem;
    color: #A0AAB8;
}

/* Comparison cards */
.compare-card {
    background-color: #1C2333;
    border-radius: 10px;
    padding: 1.2rem;
    border: 1px solid #2A3447;
    margin-bottom: 1rem;
}
.compare-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #FAFAFA;
    margin-bottom: 0.5rem;
}
.compare-price {
    font-size: 2rem;
    font-weight: 800;
    color: #1E88E5;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Mapping Dictionaries for Readable Labels
# -----------------------------------------------------------------------------
QUALITY_OPTIONS = {
    "Ex": "Excellent (Ex)",
    "Gd": "Good (Gd)",
    "TA": "Average / Typical (TA)",
    "Fa": "Fair (Fa)",
    "Po": "Poor (Po)",
}

NEIGHBORHOOD_OPTIONS = {
    "Blmngtn": "Bloomington Heights (Blmngtn)",
    "Blueste": "Bluestem (Blueste)",
    "BrDale": "Briardale (BrDale)",
    "BrkSide": "Brookside (BrkSide)",
    "ClearCr": "Clear Creek (ClearCr)",
    "CollgCr": "College Creek (CollgCr)",
    "Crawfor": "Crawford (Crawfor)",
    "Edwards": "Edwards (Edwards)",
    "Gilbert": "Gilbert (Gilbert)",
    "IDOTRR": "Iowa DOT & Rail Road (IDOTRR)",
    "MeadowV": "Meadow Village (MeadowV)",
    "Mitchel": "Mitchell (Mitchel)",
    "NAmes": "North Ames (NAmes)",
    "NPkVill": "Northpark Villa (NPkVill)",
    "NWAmes": "Northwest Ames (NWAmes)",
    "NoRidge": "Northridge (NoRidge)",
    "NridgHt": "Northridge Heights (NridgHt)",
    "OldTown": "Old Town (OldTown)",
    "SWISU": "South & West of ISU (SWISU)",
    "Sawyer": "Sawyer (Sawyer)",
    "SawyerW": "Sawyer West (SawyerW)",
    "Somerst": "Somerset (Somerst)",
    "StoneBr": "Stone Brook (StoneBr)",
    "Timber": "Timberland (Timber)",
    "Veenker": "Veenker (Veenker)",
}

HOUSE_STYLE_OPTIONS = {
    "1Story": "One story (1Story)",
    "2Story": "Two story (2Story)",
    "1.5Fin": "1.5 story - 2nd level finished (1.5Fin)",
    "1.5Unf": "1.5 story - 2nd level unfinished (1.5Unf)",
    "SLvl": "Split Level (SLvl)",
    "SFoyer": "Split Foyer (SFoyer)",
    "2.5Fin": "2.5 story - 2nd level finished (2.5Fin)",
    "2.5Unf": "2.5 story - 2nd level unfinished (2.5Unf)",
}

ZONING_OPTIONS = {
    "RL": "Residential Low Density (RL)",
    "RM": "Residential Medium Density (RM)",
    "FV": "Floating Village Residential (FV)",
    "RH": "Residential High Density (RH)",
    "C (all)": "Commercial (C)",
}

GARAGE_TYPE_OPTIONS = {
    "Attchd": "Attached to home (Attchd)",
    "Detchd": "Detached from home (Detchd)",
    "BuiltIn": "Built-in (part of house) (BuiltIn)",
    "Basment": "Basement Garage (Basment)",
    "CarPort": "Car Port (CarPort)",
    "2Types": "More than one type (2Types)",
    "None": "No Garage (None)",
}

CENTRAL_AIR_OPTIONS = {
    "Y": "Yes (Central Air Installed)",
    "N": "No (No Central Air)",
}

# -----------------------------------------------------------------------------
# Cached Resource and Data Loading
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model_and_metrics():
    """Load model pipeline and metrics metadata with caching."""
    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    if not os.path.exists(model_path) or not os.path.exists(metrics_path):
        return None, None
    model = joblib.load(model_path)
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return model, metrics


@st.cache_data
def load_holdout_data():
    """Load holdout predictions CSV for model comparison."""
    csv_path = os.path.join(MODELS_DIR, "holdout_predictions.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None


@st.cache_data
def load_raw_dataset():
    """Load raw dataset for exploratory data analysis."""
    data_path = os.path.join(BASE_DIR, "train.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join(BASE_DIR, "house_prices_kaggle.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        # Drop outliers matching training convention
        if "GrLivArea" in df.columns and "SalePrice" in df.columns:
            df = df[~((df["GrLivArea"] > 4000) & (df["SalePrice"] < 300000))].copy()
        return df
    return None


@st.cache_data
def load_neighborhood_medians():
    """Calculate and cache median sale prices by neighborhood."""
    df = load_raw_dataset()
    if df is not None and "Neighborhood" in df.columns and "SalePrice" in df.columns:
        return df.groupby("Neighborhood")["SalePrice"].median().to_dict()
    return {}


@st.cache_resource
def get_shap_explainer_and_sample():
    """
    Cache SHAP Explainer and a representative background dataset sample
    for fast explainability computation.
    """
    try:
        import shap
        model, metrics = load_model_and_metrics()
        if model is None:
            return None, None, None, None, None, None

        X, y, num_cols, cat_cols, defaults = load_and_clean()
        pre = model.named_steps["preprocessor"]
        reg = model.named_steps["regressor"]

        # Sample 100 observations to guarantee fast, responsive SHAP evaluation
        sample_X = X.sample(n=min(100, len(X)), random_state=42)
        sample_trans = pre.transform(sample_X)
        feat_names = pre.get_feature_names_out()

        explainer = shap.Explainer(reg, sample_trans, feature_names=feat_names)
        sample_shap = explainer(sample_trans)

        return explainer, pre, sample_X, sample_trans, sample_shap, feat_names
    except Exception as ex:
        print(f"Warning: could not initialize SHAP explainer: {ex}")
        return None, None, None, None, None, None


# -----------------------------------------------------------------------------
# Page 1: Predict Price
# -----------------------------------------------------------------------------
def render_predict_page(metrics, neighborhood_medians):
    st.markdown("<div class='main-header'>🏠 House Price Valuation</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Configure property details to generate estimated market price and confidence bounds.</div>",
        unsafe_allow_html=True,
    )

    defaults = metrics.get("defaults", {}) if metrics else {}

    col_left, col_right = st.columns([1.1, 0.9], gap="large")

    with col_left:
        # Group 1: Size & Layout
        with st.expander("📐 Size & Layout", expanded=True):
            s_col1, s_col2 = st.columns(2)
            with s_col1:
                gr_liv_area = st.number_input(
                    "Above Ground Living Area (sq ft)",
                    min_value=300,
                    max_value=6000,
                    value=int(defaults.get("GrLivArea", 1500)),
                    step=50,
                    help="Total square footage of above-grade living rooms, bedrooms, and kitchen.",
                )
                first_flr_sf = st.number_input(
                    "First Floor Area (sq ft)",
                    min_value=300,
                    max_value=4000,
                    value=int(defaults.get("1stFlrSF", 1087)),
                    step=50,
                )
            with s_col2:
                total_bsmt_sf = st.number_input(
                    "Total Basement Area (sq ft)",
                    min_value=0,
                    max_value=4000,
                    value=int(defaults.get("TotalBsmtSF", 991)),
                    step=50,
                )
                second_flr_sf = st.number_input(
                    "Second Floor Area (sq ft)",
                    min_value=0,
                    max_value=3000,
                    value=int(defaults.get("2ndFlrSF", 0)),
                    step=50,
                )
            lot_area = st.number_input(
                "Lot Size (sq ft)",
                min_value=1000,
                max_value=100000,
                value=int(defaults.get("LotArea", 9478)),
                step=250,
            )

        # Group 2: Quality & Age
        with st.expander("⭐ Quality & Age", expanded=True):
            q_col1, q_col2 = st.columns(2)
            with q_col1:
                overall_qual = st.slider(
                    "Overall Quality (1 = Poor, 10 = Very Excellent)",
                    min_value=1,
                    max_value=10,
                    value=int(defaults.get("OverallQual", 6)),
                )
                exter_qual_code = st.selectbox(
                    "Exterior Quality",
                    options=list(QUALITY_OPTIONS.keys()),
                    format_func=lambda x: QUALITY_OPTIONS[x],
                    index=list(QUALITY_OPTIONS.keys()).index(defaults.get("ExterQual", "TA")) if defaults.get("ExterQual", "TA") in QUALITY_OPTIONS else 2,
                )
                year_built = st.slider(
                    "Year Built",
                    min_value=1872,
                    max_value=2010,
                    value=int(defaults.get("YearBuilt", 1973)),
                )
            with q_col2:
                overall_cond = st.slider(
                    "Overall Condition (1 = Poor, 10 = Very Excellent)",
                    min_value=1,
                    max_value=10,
                    value=int(defaults.get("OverallCond", 5)),
                )
                kitchen_qual_code = st.selectbox(
                    "Kitchen Quality",
                    options=list(QUALITY_OPTIONS.keys()),
                    format_func=lambda x: QUALITY_OPTIONS[x],
                    index=list(QUALITY_OPTIONS.keys()).index(defaults.get("KitchenQual", "TA")) if defaults.get("KitchenQual", "TA") in QUALITY_OPTIONS else 2,
                )
                year_remod = st.slider(
                    "Year Remodeled / Added",
                    min_value=1950,
                    max_value=2010,
                    value=max(int(defaults.get("YearRemodAdd", 1994)), year_built),
                )

        # Group 3: Location & Style
        with st.expander("📍 Location & Style", expanded=True):
            l_col1, l_col2 = st.columns(2)
            with l_col1:
                neighborhood_code = st.selectbox(
                    "Neighborhood",
                    options=list(NEIGHBORHOOD_OPTIONS.keys()),
                    format_func=lambda x: NEIGHBORHOOD_OPTIONS[x],
                    index=list(NEIGHBORHOOD_OPTIONS.keys()).index(defaults.get("Neighborhood", "NAmes")) if defaults.get("Neighborhood", "NAmes") in NEIGHBORHOOD_OPTIONS else 12,
                )
            with l_col2:
                house_style_code = st.selectbox(
                    "House Style",
                    options=list(HOUSE_STYLE_OPTIONS.keys()),
                    format_func=lambda x: HOUSE_STYLE_OPTIONS[x],
                    index=list(HOUSE_STYLE_OPTIONS.keys()).index(defaults.get("HouseStyle", "1Story")) if defaults.get("HouseStyle", "1Story") in HOUSE_STYLE_OPTIONS else 0,
                )
            ms_zoning_code = st.selectbox(
                "Zoning Classification",
                options=list(ZONING_OPTIONS.keys()),
                format_func=lambda x: ZONING_OPTIONS[x],
                index=list(ZONING_OPTIONS.keys()).index(defaults.get("MSZoning", "RL")) if defaults.get("MSZoning", "RL") in ZONING_OPTIONS else 0,
            )

        # Group 4: Garage
        with st.expander("🚗 Garage", expanded=False):
            g_col1, g_col2 = st.columns(2)
            with g_col1:
                garage_cars = st.slider(
                    "Garage Car Capacity",
                    min_value=0,
                    max_value=4,
                    value=int(defaults.get("GarageCars", 2)),
                )
                garage_type_code = st.selectbox(
                    "Garage Type",
                    options=list(GARAGE_TYPE_OPTIONS.keys()),
                    format_func=lambda x: GARAGE_TYPE_OPTIONS[x],
                    index=list(GARAGE_TYPE_OPTIONS.keys()).index(defaults.get("GarageType", "Attchd")) if defaults.get("GarageType", "Attchd") in GARAGE_TYPE_OPTIONS else 0,
                )
            with g_col2:
                garage_area = st.number_input(
                    "Garage Area (sq ft)",
                    min_value=0,
                    max_value=1500,
                    value=int(defaults.get("GarageArea", 480)),
                    step=25,
                )

        # Group 5: Extras & Amenities
        with st.expander("✨ Extras & Amenities", expanded=False):
            e_col1, e_col2 = st.columns(2)
            with e_col1:
                full_bath = st.slider("Full Bathrooms", 0, 4, int(defaults.get("FullBath", 2)))
                bedroom_abv_gr = st.slider("Bedrooms Above Ground", 0, 8, int(defaults.get("BedroomAbvGr", 3)))
                fireplaces = st.slider("Fireplaces", 0, 4, int(defaults.get("Fireplaces", 1)))
            with e_col2:
                half_bath = st.slider("Half Bathrooms", 0, 2, int(defaults.get("HalfBath", 0)))
                central_air_code = st.selectbox(
                    "Central Air Conditioning",
                    options=list(CENTRAL_AIR_OPTIONS.keys()),
                    format_func=lambda x: CENTRAL_AIR_OPTIONS[x],
                    index=0 if defaults.get("CentralAir", "Y") == "Y" else 1,
                )

        # Input Validation Warnings
        validation_warnings = []
        if year_remod < year_built:
            validation_warnings.append(f"Remodel year ({year_remod}) cannot precede construction year ({year_built}). Auto-adjusting remodel year.")
            year_remod = year_built

        if garage_cars == 0 and garage_area > 0:
            validation_warnings.append("Garage car capacity is 0; garage area automatically adjusted to 0 sq ft.")
            garage_area = 0

        for w in validation_warnings:
            st.warning(f"⚠️ {w}")

    with col_right:
        input_data = {
            "GrLivArea": gr_liv_area,
            "1stFlrSF": first_flr_sf,
            "2ndFlrSF": second_flr_sf,
            "TotalBsmtSF": total_bsmt_sf,
            "LotArea": lot_area,
            "OverallQual": overall_qual,
            "OverallCond": overall_cond,
            "ExterQual": exter_qual_code,
            "KitchenQual": kitchen_qual_code,
            "YearBuilt": year_built,
            "YearRemodAdd": year_remod,
            "Neighborhood": neighborhood_code,
            "HouseStyle": house_style_code,
            "MSZoning": ms_zoning_code,
            "GarageCars": garage_cars,
            "GarageArea": garage_area,
            "GarageType": garage_type_code,
            "FullBath": full_bath,
            "HalfBath": half_bath,
            "BedroomAbvGr": bedroom_abv_gr,
            "CentralAir": central_air_code,
            "Fireplaces": fireplaces,
        }

        try:
            pred_price, lower_bound, upper_bound = predict_price(input_data)
        except Exception as ex:
            st.error(f"Error computing prediction: {ex}")
            pred_price, lower_bound, upper_bound = 200000.0, 170000.0, 235000.0

        # Save active inputs to session state for What-If and Explainability pages
        st.session_state["latest_prediction"] = {
            "predicted_price": pred_price,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "neighborhood": neighborhood_code,
            "neighborhood_median": neighborhood_medians.get(neighborhood_code, 180000.0),
            "inputs": input_data,
        }

        # Large Metric Card
        st.markdown(f"""
        <div class='prediction-card'>
            <div class='prediction-label'>Estimated Property Market Value</div>
            <div class='prediction-value'>${pred_price:,.0f}</div>
            <div class='prediction-range'>
                90% Prediction Interval: <strong>${lower_bound:,.0f}</strong> &mdash; <strong>${upper_bound:,.0f}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Plotly Comparison Gauge
        neigh_median = neighborhood_medians.get(neighborhood_code, 180000.0)
        gauge_max = max(pred_price * 1.35, neigh_median * 1.4, 450000.0)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred_price,
            number={"valueformat": "$,.0f", "font": {"color": "#FAFAFA", "size": 28}},
            delta={
                "reference": neigh_median,
                "position": "top",
                "valueformat": "+$,.0f",
                "prefix": "vs Median: ",
                "font": {"size": 15},
            },
            title={
                "text": f"<b>Valuation vs {neighborhood_code} Median (${neigh_median:,.0f})</b>",
                "font": {"size": 15, "color": "#FAFAFA"},
            },
            gauge={
                "axis": {"range": [0, gauge_max], "tickformat": "$,.0f", "tickcolor": "#888888"},
                "bar": {"color": "#1E88E5", "thickness": 0.3},
                "bgcolor": "#1C2333",
                "borderwidth": 1,
                "bordercolor": "#2A3447",
                "steps": [
                    {"range": [0, neigh_median * 0.85], "color": "rgba(30, 136, 229, 0.12)"},
                    {"range": [neigh_median * 0.85, neigh_median * 1.15], "color": "rgba(30, 136, 229, 0.28)"},
                    {"range": [neigh_median * 1.15, gauge_max], "color": "rgba(30, 136, 229, 0.42)"},
                ],
                "threshold": {
                    "line": {"color": "#FF7043", "width": 4},
                    "thickness": 0.8,
                    "value": neigh_median,
                },
            },
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=280,
            margin=dict(l=25, r=25, t=55, b=25),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Difference Stats
        c1, c2 = st.columns(2)
        diff_val = pred_price - neigh_median
        diff_pct = (diff_val / neigh_median) * 100
        with c1:
            st.markdown(f"""
            <div class='stat-box'>
                <div class='stat-val'>{diff_pct:+.1f}%</div>
                <div class='stat-lbl'>Difference vs Neighborhood</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class='stat-box'>
                <div class='stat-val'>${(upper_bound - lower_bound):,.0f}</div>
                <div class='stat-lbl'>Prediction Range Width</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Download Report CSV
        report_dict = {
            "Report Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Estimated Price ($)": round(pred_price, 2),
            "Lower Bound 90% ($)": round(lower_bound, 2),
            "Upper Bound 90% ($)": round(upper_bound, 2),
            "Neighborhood": neighborhood_code,
            "Neighborhood Median ($)": round(neigh_median, 2),
            "Living Area (sqft)": gr_liv_area,
            "Basement Area (sqft)": total_bsmt_sf,
            "Overall Quality": overall_qual,
            "Overall Condition": overall_cond,
            "Year Built": year_built,
            "Year Remodeled": year_remod,
            "Garage Cars": garage_cars,
            "Garage Area (sqft)": garage_area,
            "Full Bathrooms": full_bath,
            "Bedrooms": bedroom_abv_gr,
            "Kitchen Quality": kitchen_qual_code,
            "Exterior Quality": exter_qual_code,
            "Central Air": central_air_code,
        }
        report_df = pd.DataFrame([report_dict])
        csv_buffer = report_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Valuation Report (CSV)",
            data=csv_buffer,
            file_name=f"house_valuation_{neighborhood_code}_{datetime.date.today()}.csv",
            mime="text/csv",
            use_container_width=True,
        )


# -----------------------------------------------------------------------------
# Page 2: What-If Explorer (Phase 3)
# -----------------------------------------------------------------------------
def render_whatif_page(metrics):
    st.markdown("<div class='main-header'>🔮 What-If Sensitivity Explorer</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Simulate property value sensitivity to feature variations and benchmark two properties side-by-side.</div>",
        unsafe_allow_html=True,
    )

    # Retrieve baseline inputs
    defaults = metrics.get("defaults", {}) if metrics else {}
    latest = st.session_state.get("latest_prediction", {})
    base_inputs = dict(latest.get("inputs", defaults))

    tab_single, tab_compare = st.tabs(["📈 Single-Feature Sensitivity Curve", "⚖️ Property A vs Property B Comparison"])

    # 1. Single Feature Sensitivity
    with tab_single:
        st.subheader("Single Feature Valuation Sensitivity")
        st.write("Select a property attribute to vary across its realistic range. All other attributes remain locked at your current Predict inputs:")

        sweepable_features = {
            "OverallQual": {"label": "Overall Quality (1–10)", "range": list(range(1, 11)), "is_int": True},
            "GrLivArea": {"label": "Above Ground Living Area (sq ft)", "range": list(range(600, 4200, 150)), "is_int": True},
            "YearBuilt": {"label": "Year Built", "range": list(range(1900, 2011, 5)), "is_int": True},
            "TotalBsmtSF": {"label": "Total Basement Area (sq ft)", "range": list(range(0, 2600, 150)), "is_int": True},
            "GarageCars": {"label": "Garage Car Capacity", "range": [0, 1, 2, 3, 4], "is_int": True},
            "FullBath": {"label": "Full Bathrooms", "range": [1, 2, 3, 4], "is_int": True},
            "LotArea": {"label": "Lot Size (sq ft)", "range": list(range(2000, 28000, 1500)), "is_int": True},
            "Fireplaces": {"label": "Number of Fireplaces", "range": [0, 1, 2, 3, 4], "is_int": True},
        }

        col_feat, col_info = st.columns([1, 2])
        with col_feat:
            chosen_feat = st.selectbox(
                "Feature to Explore",
                options=list(sweepable_features.keys()),
                format_func=lambda x: sweepable_features[x]["label"],
            )

        feat_config = sweepable_features[chosen_feat]
        sweep_values = feat_config["range"]
        current_val = base_inputs.get(chosen_feat, sweep_values[len(sweep_values) // 2])

        # Sweep predictions
        sweep_prices = []
        for val in sweep_values:
            test_inputs = dict(base_inputs)
            test_inputs[chosen_feat] = val
            p, _, _ = predict_price(test_inputs)
            sweep_prices.append(p)

        # Plotly Line Chart
        fig_sweep = go.Figure()

        # Sensitivity Curve
        fig_sweep.add_trace(go.Scatter(
            x=sweep_values,
            y=sweep_prices,
            mode="lines+markers",
            line=dict(color="#1E88E5", width=3),
            marker=dict(size=7, color="#64B5F6"),
            name="Estimated Valuation",
            hovertemplate="<b>%{x}</b>: $%{y:,.0f}<extra></extra>",
        ))

        # Marker for current input value
        current_price, _, _ = predict_price(base_inputs)
        fig_sweep.add_trace(go.Scatter(
            x=[current_val],
            y=[current_price],
            mode="markers+text",
            marker=dict(size=14, color="#FF5252", symbol="diamond"),
            text=["Current Value"],
            textposition="top center",
            name="Current Setting",
            hovertemplate="Current %{x}: $%{y:,.0f}<extra></extra>",
        ))

        fig_sweep.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title=f"Price Trajectory vs {feat_config['label']}",
            xaxis_title=feat_config["label"],
            yaxis_title="Estimated Valuation ($)",
            yaxis=dict(tickformat="$,.0f"),
            height=420,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_sweep, use_container_width=True)

        # Calculate incremental marginal impact
        price_spread = sweep_prices[-1] - sweep_prices[0]
        val_spread = sweep_values[-1] - sweep_values[0]
        marginal_unit = price_spread / val_spread if val_spread != 0 else 0

        st.markdown(f"""
        <div class='info-card'>
            💡 <strong>Sensitivity Insight:</strong> Varying <strong>{feat_config['label']}</strong> from 
            <code>{sweep_values[0]}</code> to <code>{sweep_values[-1]}</code> shifts estimated market value from 
            <strong>${sweep_prices[0]:,.0f}</strong> to <strong>${sweep_prices[-1]:,.0f}</strong> 
            (an overall spread of <strong>${abs(price_spread):,.0f}</strong>, or approximately 
            <strong>${abs(marginal_unit):,.0f}</strong> per unit change).
        </div>
        """, unsafe_allow_html=True)

    # 2. Side-by-side Property Comparison
    with tab_compare:
        st.subheader("⚖️ Side-by-Side Property Comparison (House A vs House B)")
        st.write("Customize attributes for two properties to evaluate how architectural differences impact valuation:")

        comp_col1, comp_col2 = st.columns(2, gap="large")

        with comp_col1:
            st.markdown("<div class='compare-card'><div class='compare-title'>🏡 Property A (Baseline)</div>", unsafe_allow_html=True)
            a_qual = st.slider("House A: Overall Quality", 1, 10, int(base_inputs.get("OverallQual", 6)), key="a_qual")
            a_sqft = st.number_input("House A: Living Area (sq ft)", 400, 5000, int(base_inputs.get("GrLivArea", 1500)), 50, key="a_sqft")
            a_year = st.slider("House A: Year Built", 1900, 2010, int(base_inputs.get("YearBuilt", 1975)), key="a_year")
            a_neigh = st.selectbox("House A: Neighborhood", list(NEIGHBORHOOD_OPTIONS.keys()), format_func=lambda x: NEIGHBORHOOD_OPTIONS[x], index=list(NEIGHBORHOOD_OPTIONS.keys()).index(base_inputs.get("Neighborhood", "NAmes")), key="a_neigh")
            a_cars = st.slider("House A: Garage Cars", 0, 4, int(base_inputs.get("GarageCars", 2)), key="a_cars")
            a_bath = st.slider("House A: Full Baths", 1, 4, int(base_inputs.get("FullBath", 2)), key="a_bath")
            a_kqual = st.selectbox("House A: Kitchen Quality", list(QUALITY_OPTIONS.keys()), format_func=lambda x: QUALITY_OPTIONS[x], index=list(QUALITY_OPTIONS.keys()).index(base_inputs.get("KitchenQual", "TA")), key="a_kqual")
            st.markdown("</div>", unsafe_allow_html=True)

        with comp_col2:
            st.markdown("<div class='compare-card'><div class='compare-title'>🏡 Property B (Comparison)</div>", unsafe_allow_html=True)
            b_qual = st.slider("House B: Overall Quality", 1, 10, min(10, int(base_inputs.get("OverallQual", 6)) + 2), key="b_qual")
            b_sqft = st.number_input("House B: Living Area (sq ft)", 400, 5000, int(base_inputs.get("GrLivArea", 1500)) + 300, 50, key="b_sqft")
            b_year = st.slider("House B: Year Built", 1900, 2010, min(2010, int(base_inputs.get("YearBuilt", 1975)) + 15), key="b_year")
            b_neigh = st.selectbox("House B: Neighborhood", list(NEIGHBORHOOD_OPTIONS.keys()), format_func=lambda x: NEIGHBORHOOD_OPTIONS[x], index=list(NEIGHBORHOOD_OPTIONS.keys()).index("CollgCr") if "CollgCr" in NEIGHBORHOOD_OPTIONS else 0, key="b_neigh")
            b_cars = st.slider("House B: Garage Cars", 0, 4, min(4, int(base_inputs.get("GarageCars", 2)) + 1), key="b_cars")
            b_bath = st.slider("House B: Full Baths", 1, 4, min(4, int(base_inputs.get("FullBath", 2)) + 1), key="b_bath")
            b_kqual = st.selectbox("House B: Kitchen Quality", list(QUALITY_OPTIONS.keys()), format_func=lambda x: QUALITY_OPTIONS[x], index=list(QUALITY_OPTIONS.keys()).index("Gd") if "Gd" in QUALITY_OPTIONS else 1, key="b_kqual")
            st.markdown("</div>", unsafe_allow_html=True)

        # Assemble full inputs
        house_a_inputs = dict(base_inputs)
        house_a_inputs.update({
            "OverallQual": a_qual,
            "GrLivArea": a_sqft,
            "YearBuilt": a_year,
            "Neighborhood": a_neigh,
            "GarageCars": a_cars,
            "FullBath": a_bath,
            "KitchenQual": a_kqual,
        })

        house_b_inputs = dict(base_inputs)
        house_b_inputs.update({
            "OverallQual": b_qual,
            "GrLivArea": b_sqft,
            "YearBuilt": b_year,
            "Neighborhood": b_neigh,
            "GarageCars": b_cars,
            "FullBath": b_bath,
            "KitchenQual": b_kqual,
        })

        price_a, lower_a, upper_a = predict_price(house_a_inputs)
        price_b, lower_b, upper_b = predict_price(house_b_inputs)

        diff_ab = price_b - price_a
        pct_ab = (diff_ab / price_a) * 100 if price_a > 0 else 0

        # Side-by-side metric display
        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.markdown(f"""
            <div class='stat-box'>
                <div class='stat-lbl'>Property A Valuation</div>
                <div class='stat-val'>${price_a:,.0f}</div>
                <small style='color:#888;'>Range: ${lower_a:,.0f} – ${upper_a:,.0f}</small>
            </div>
            """, unsafe_allow_html=True)
        with res_col2:
            st.markdown(f"""
            <div class='stat-box'>
                <div class='stat-lbl'>Property B Valuation</div>
                <div class='stat-val'>${price_b:,.0f}</div>
                <small style='color:#888;'>Range: ${lower_b:,.0f} – ${upper_b:,.0f}</small>
            </div>
            """, unsafe_allow_html=True)
        with res_col3:
            diff_color = "#00C853" if diff_ab >= 0 else "#FF5252"
            st.markdown(f"""
            <div class='stat-box'>
                <div class='stat-lbl'>Valuation Difference (B - A)</div>
                <div class='stat-val' style='color:{diff_color};'>{diff_ab:+,.0f}</div>
                <small style='color:{diff_color};'>{pct_ab:+.1f}% vs Property A</small>
            </div>
            """, unsafe_allow_html=True)

        # Plotly comparison bar chart
        fig_comp = go.Figure(data=[
            go.Bar(
                name="Valuation Point Estimate",
                x=["Property A", "Property B"],
                y=[price_a, price_b],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[upper_a - price_a, upper_b - price_b],
                    arrayminus=[price_a - lower_a, price_b - lower_b],
                    color="#90CAF9",
                ),
                marker_color=["#1E88E5", "#43A047" if price_b >= price_a else "#E53935"],
                text=[f"${price_a:,.0f}", f"${price_b:,.0f}"],
                textposition="auto",
            )
        ])
        fig_comp.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title="Property A vs Property B Valuation (with 90% Confidence Interval)",
            yaxis_title="Estimated Price ($)",
            yaxis=dict(tickformat="$,.0f"),
            height=320,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_comp, use_container_width=True)


# -----------------------------------------------------------------------------
# Page 3: Feature Explainability & SHAP (Phase 3)
# -----------------------------------------------------------------------------
def render_explainability_page(metrics):
    st.markdown("<div class='main-header'>🔍 Machine Learning Explainability (SHAP)</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Understand exactly which features drive property valuation globally and for your current prediction.</div>",
        unsafe_allow_html=True,
    )

    explainer, pre, sample_X, sample_trans, sample_shap, feat_names = get_shap_explainer_and_sample()

    if explainer is None:
        st.error("SHAP explainer could not be loaded. Please ensure `shap` is installed and models are trained.")
        return

    # Section 1: Global Feature Importance
    st.subheader("🌐 Global Feature Attribution")
    st.write("Mean absolute SHAP value impact across representative houses in the Ames dataset:")

    top_15_meta = metrics.get("top_15_features", [])
    if top_15_meta:
        top_df = pd.DataFrame(top_15_meta)
        fig_imp = px.bar(
            top_df.sort_values(by="importance", ascending=True),
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Blues",
            title="Top 15 Most Important Features (Normalized Relative Contribution)",
            labels={"importance": "Relative Importance", "feature": "Feature"},
            text_auto=".2%",
        )
        fig_imp.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
            height=380,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    # Section 2: Local SHAP Explanation (Current Prediction Waterfall)
    st.subheader("🎯 Local Prediction Attribution (Waterfall)")
    st.write("Feature-by-feature breakdown of how your current inputs shifted the valuation from the baseline average:")

    # Retrieve current input
    latest = st.session_state.get("latest_prediction", {})
    defaults = metrics.get("defaults", {})
    active_inputs = dict(latest.get("inputs", defaults))
    current_predicted_price = latest.get("predicted_price")
    if current_predicted_price is None:
        current_predicted_price, _, _ = predict_price(active_inputs)

    # Format into DataFrame for ColumnTransformer
    row_df = pd.DataFrame([{col: active_inputs.get(col, defaults.get(col, 0)) for col in metrics.get("feature_columns", [])}])
    # Re-calculate engineered features
    row_df["TotalSF"] = row_df.get("TotalBsmtSF", 0) + row_df.get("1stFlrSF", 0) + row_df.get("2ndFlrSF", 0)
    row_df["HouseAge"] = (row_df.get("YrSold", 2010) - row_df.get("YearBuilt", 1970)).clip(lower=0)
    row_df["RemodAge"] = (row_df.get("YrSold", 2010) - row_df.get("YearRemodAdd", 1970)).clip(lower=0)
    row_df["TotalBath"] = row_df.get("FullBath", 0) + 0.5 * row_df.get("HalfBath", 0) + row_df.get("BsmtFullBath", 0) + 0.5 * row_df.get("BsmtHalfBath", 0)
    row_df["HasGarage"] = (row_df.get("GarageArea", 0) > 0).astype(int)
    row_df["HasPool"] = (row_df.get("PoolArea", 0) > 0).astype(int)
    row_df["HasFireplace"] = (row_df.get("Fireplaces", 0) > 0).astype(int)

    # Ensure types
    for col in metrics.get("numeric_features", []):
        if col in row_df.columns:
            row_df[col] = pd.to_numeric(row_df[col], errors="coerce").fillna(defaults.get(col, 0))
    for col in metrics.get("categorical_features", []):
        if col in row_df.columns:
            row_df[col] = row_df[col].astype(str)

    row_trans = pre.transform(row_df)
    instance_shap = explainer(row_trans)[0]

    # Aggregate SHAP contributions back to raw features
    cat_cols = metrics.get("categorical_features", [])
    raw_contribs = {}
    for name, val in zip(feat_names, instance_shap.values):
        clean_name = name.split("__", 1)[-1]
        orig_col = None
        if name.startswith("cat__"):
            for c in cat_cols:
                if clean_name.startswith(c):
                    orig_col = c
                    break
        if orig_col is None:
            orig_col = clean_name
        raw_contribs[orig_col] = raw_contribs.get(orig_col, 0.0) + float(val)

    # Sort contributions by absolute value
    sorted_contribs = sorted(raw_contribs.items(), key=lambda x: abs(x[1]), reverse=True)
    top_factors = sorted_contribs[:8]
    other_val = sum(v for _, v in sorted_contribs[8:])

    # Base price and feature contributions in dollar terms
    expected_log = float(getattr(explainer, "expected_value", 12.0))
    base_price = float(np.expm1(expected_log))

    # Approximate dollar impact per factor
    waterfall_names = ["Baseline Average"]
    waterfall_vals = [base_price]
    waterfall_measures = ["absolute"]

    insights = []
    for col, log_val in top_factors:
        dollar_impact = current_predicted_price * log_val  # First-order Taylor approximation
        waterfall_names.append(col)
        waterfall_vals.append(dollar_impact)
        waterfall_measures.append("relative")

        direction = "increased" if dollar_impact >= 0 else "reduced"
        val_disp = active_inputs.get(col, "")
        val_str = f" ({val_disp})" if val_disp != "" else ""
        insights.append(
            f"**{col}**{val_str} {direction} the estimated price by approximately **${abs(dollar_impact):,.0f}**."
        )

    waterfall_names.append("Other Features Combined")
    waterfall_vals.append(current_predicted_price * other_val)
    waterfall_measures.append("relative")

    waterfall_names.append("Final Valuation")
    waterfall_vals.append(current_predicted_price)
    waterfall_measures.append("total")

    # Plotly Waterfall Chart
    fig_waterfall = go.Figure(go.Waterfall(
        name="Valuation Impact",
        orientation="v",
        measure=waterfall_measures,
        x=waterfall_names,
        y=waterfall_vals,
        connector={"line": {"color": "#888888"}},
        increasing={"marker": {"color": "#00C853"}},
        decreasing={"marker": {"color": "#FF5252"}},
        totals={"marker": {"color": "#1E88E5"}},
        text=[f"${v:,.0f}" for v in waterfall_vals],
        textposition="outside",
    ))
    fig_waterfall.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title="Waterfall Attribution: How Attributes Build to the Final Price",
        yaxis_title="Property Valuation ($)",
        yaxis=dict(tickformat="$,.0f"),
        height=450,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_waterfall, use_container_width=True)

    # Plain English Insights (3-4 bullet points)
    st.markdown("### 📝 Plain-English Prediction Summary")
    st.markdown("""
    <div class='info-card'>
    Based on the SHAP additive attribution model for this property:
    <ul>
    """ + "".join([f"<li>{i}</li>" for i in insights[:4]]) + """
    </ul>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Page 4: Interactive Data Explorer (Phase 3)
# -----------------------------------------------------------------------------
def render_data_explorer_page():
    st.markdown("<div class='main-header'>📈 Interactive Data Explorer</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Explore historical Ames housing market patterns, distributions, and multi-variable correlations.</div>",
        unsafe_allow_html=True,
    )

    df = load_raw_dataset()
    if df is None:
        st.error("Training dataset not found. Please ensure `train.csv` is present in the project folder.")
        return

    # Section 1: SalePrice Distribution
    st.subheader("🏷️ Sale Price Distribution")
    dist_col1, dist_col2 = st.columns([3, 1])

    with dist_col2:
        use_log_scale = st.checkbox("Log-transformed Scale", value=False, help="View normalising effect of log1p(SalePrice)")
        price_bins = st.slider("Histogram Bins", 20, 100, 50)
        mean_p = df["SalePrice"].mean()
        median_p = df["SalePrice"].median()
        st.markdown(f"""
        <div class='stat-box' style='margin-bottom:0.8rem;'>
            <div class='stat-lbl'>Median Sale Price</div>
            <div class='stat-val'>${median_p:,.0f}</div>
        </div>
        <div class='stat-box'>
            <div class='stat-lbl'>Mean Sale Price</div>
            <div class='stat-val'>${mean_p:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with dist_col1:
        plot_series = np.log1p(df["SalePrice"]) if use_log_scale else df["SalePrice"]
        fig_dist = px.histogram(
            df,
            x=plot_series,
            nbins=price_bins,
            marginal="box",
            color_discrete_sequence=["#1E88E5"],
            title="Ames House Sale Price Distribution",
            labels={"x": "Log(1 + SalePrice)" if use_log_scale else "Sale Price ($)"},
        )
        if not use_log_scale:
            fig_dist.add_vline(x=median_p, line_dash="dash", line_color="#00E676", annotation_text="Median")
            fig_dist.add_vline(x=mean_p, line_dash="dot", line_color="#FF7043", annotation_text="Mean")
            fig_dist.update_layout(xaxis=dict(tickformat="$,.0f"))

        fig_dist.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=360,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    # Section 2: Price by Neighborhood (Box Plot)
    st.subheader("🏘️ Sale Price by Neighborhood")
    st.write("Neighborhoods ordered by median sale price (entry-level to affluent):")

    sorted_neigh = df.groupby("Neighborhood")["SalePrice"].median().sort_values().index.tolist()
    fig_box = px.box(
        df,
        x="Neighborhood",
        y="SalePrice",
        color="Neighborhood",
        category_orders={"Neighborhood": sorted_neigh},
        title="Price Distributions Across Ames Neighborhoods (Sorted by Median)",
        points="outliers",
    )
    fig_box.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        yaxis=dict(tickformat="$,.0f"),
        height=450,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_box, use_container_width=True)

    # Section 3: Correlation Heatmap
    st.subheader("🔥 Correlation Heatmap of Top Numeric Features")
    st.write("Pearson correlation coefficients between dominant numerical drivers and SalePrice:")

    candidate_numeric = [
        "SalePrice", "OverallQual", "GrLivArea", "TotalBsmtSF", "1stFlrSF",
        "GarageCars", "GarageArea", "FullBath", "YearBuilt", "YearRemodAdd",
        "Fireplaces", "LotArea"
    ]
    avail_numeric = [c for c in candidate_numeric if c in df.columns]
    corr_matrix = df[avail_numeric].corr()

    fig_corr = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Feature Correlation Matrix",
    )
    fig_corr.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=520,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # Section 4: 2-Feature Custom Scatter Plot
    st.subheader("🎯 Custom Bivariate Scatter Plot")
    st.write("Select any two features to explore their relationship, colored by Overall Quality:")

    sc_col1, sc_col2 = st.columns(2)
    with sc_col1:
        x_axis_col = st.selectbox("X-Axis Feature", options=avail_numeric, index=avail_numeric.index("GrLivArea") if "GrLivArea" in avail_numeric else 1)
    with sc_col2:
        y_axis_col = st.selectbox("Y-Axis Feature", options=avail_numeric, index=avail_numeric.index("SalePrice") if "SalePrice" in avail_numeric else 0)

    fig_scat = px.scatter(
        df,
        x=x_axis_col,
        y=y_axis_col,
        color="OverallQual",
        color_continuous_scale="Viridis",
        hover_data=["Neighborhood", "YearBuilt", "SalePrice"],
        opacity=0.75,
        title=f"{y_axis_col} vs {x_axis_col} (Colored by Overall Quality)",
    )
    if "Price" in y_axis_col or "SF" in y_axis_col or "Area" in y_axis_col:
        fig_scat.update_layout(yaxis=dict(tickformat="$,.0f" if "Price" in y_axis_col else ",.0f"))
    if "Price" in x_axis_col or "SF" in x_axis_col or "Area" in x_axis_col:
        fig_scat.update_layout(xaxis=dict(tickformat="$,.0f" if "Price" in x_axis_col else ",.0f"))

    fig_scat.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=440,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_scat, use_container_width=True)


# -----------------------------------------------------------------------------
# Page 5: Model Comparison
# -----------------------------------------------------------------------------
def render_model_comparison_page(metrics, holdout_df):
    st.markdown("<div class='main-header'>📊 Model Performance & Leaderboard</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Systematic comparison across 6 algorithms evaluated via 5-fold cross-validation and a 20% hold-out test set.</div>",
        unsafe_allow_html=True,
    )

    if not metrics or "models" not in metrics:
        st.error("Metrics metadata not found. Please train models first via `python src/train.py`.")
        return

    best_model_name = metrics.get("best_model_name", "Best Model")
    models_dict = metrics.get("models", {})

    # 1. Leaderboard Table
    st.subheader("🏆 Model Leaderboard")
    leaderboard_rows = []
    for m_name, m_data in models_dict.items():
        leaderboard_rows.append({
            "Model": f"⭐ {m_name}" if m_name == best_model_name else m_name,
            "5-Fold CV R²": m_data.get("CV_R2", 0.0),
            "Test MAE ($)": m_data.get("MAE", 0.0),
            "Test RMSE ($)": m_data.get("RMSE", 0.0),
            "Test R²": m_data.get("R2", 0.0),
            "Test MAPE (%)": m_data.get("MAPE", 0.0),
            "Test RMSLE": m_data.get("RMSLE", 0.0),
        })

    lb_df = pd.DataFrame(leaderboard_rows).sort_values(by="Test R²", ascending=False).reset_index(drop=True)

    st.dataframe(
        lb_df.style.format({
            "5-Fold CV R²": "{:.4f}",
            "Test MAE ($)": "${:,.0f}",
            "Test RMSE ($)": "${:,.0f}",
            "Test R²": "{:.4f}",
            "Test MAPE (%)": "{:.2f}%",
            "Test RMSLE": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(f"""
    <div class='info-card'>
        <strong>Best Performer: {best_model_name}</strong> achieved the highest hold-out R² of 
        <strong>{models_dict[best_model_name]['R2']:.4f}</strong> and a Mean Absolute Error of 
        <strong>${models_dict[best_model_name]['MAE']:,.0f}</strong>.
    </div>
    """, unsafe_allow_html=True)

    # 2. Bar Charts for RMSE and R²
    st.subheader("📈 Metric Benchmarks across Models")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        rmse_df = pd.DataFrame({
            "Model": [m for m in models_dict.keys()],
            "RMSE": [models_dict[m]["RMSE"] for m in models_dict.keys()],
            "IsBest": ["Best Model" if m == best_model_name else "Standard" for m in models_dict.keys()],
        }).sort_values(by="RMSE", ascending=True)

        fig_rmse = px.bar(
            rmse_df,
            x="RMSE",
            y="Model",
            orientation="h",
            color="IsBest",
            color_discrete_map={"Best Model": "#1E88E5", "Standard": "#546E7A"},
            title="Root Mean Squared Error (RMSE) - Lower is Better",
            text_auto="$,.0f",
        )
        fig_rmse.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            height=340,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="RMSE ($)",
            yaxis_title="",
        )
        st.plotly_chart(fig_rmse, use_container_width=True)

    with chart_col2:
        r2_df = pd.DataFrame({
            "Model": [m for m in models_dict.keys()],
            "R2": [models_dict[m]["R2"] for m in models_dict.keys()],
            "IsBest": ["Best Model" if m == best_model_name else "Standard" for m in models_dict.keys()],
        }).sort_values(by="R2", ascending=True)

        fig_r2 = px.bar(
            r2_df,
            x="R2",
            y="Model",
            orientation="h",
            color="IsBest",
            color_discrete_map={"Best Model": "#1E88E5", "Standard": "#546E7A"},
            title="Coefficient of Determination (R²) - Higher is Better",
            text_auto=".4f",
        )
        fig_r2.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            height=340,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="R² Score",
            yaxis_title="",
        )
        st.plotly_chart(fig_r2, use_container_width=True)

    # 3. Holdout Actual vs Predicted & Residual Plots
    st.subheader(f"🎯 Hold-out Evaluation for Best Model ({best_model_name})")

    if holdout_df is not None:
        plot_col1, plot_col2 = st.columns(2)

        with plot_col1:
            max_val = max(holdout_df["Actual"].max(), holdout_df["Predicted"].max()) * 1.05
            min_val = min(holdout_df["Actual"].min(), holdout_df["Predicted"].min()) * 0.95

            fig_scatter = px.scatter(
                holdout_df,
                x="Actual",
                y="Predicted",
                hover_data=["Residual", "AbsError", "PctError"],
                opacity=0.7,
                color_discrete_sequence=["#29B6F6"],
                title="Actual vs Predicted Sale Prices",
            )
            fig_scatter.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                line=dict(color="#FF7043", dash="dash", width=2),
                name="Perfect Fit (y = x)",
            ))
            fig_scatter.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Actual Price ($)",
                yaxis_title="Predicted Price ($)",
                xaxis=dict(tickformat="$,.0f"),
                yaxis=dict(tickformat="$,.0f"),
                height=380,
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with plot_col2:
            fig_res = px.scatter(
                holdout_df,
                x="Predicted",
                y="Residual",
                hover_data=["Actual", "AbsError", "PctError"],
                opacity=0.7,
                color="Residual",
                color_continuous_scale="Tealgrn",
                title="Residuals vs Predicted Values (Homoscedasticity)",
            )
            fig_res.add_hline(y=0, line_dash="dash", line_color="#FF7043", line_width=2)
            fig_res.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Predicted Price ($)",
                yaxis_title="Residual ($) [Actual - Predicted]",
                xaxis=dict(tickformat="$,.0f"),
                yaxis=dict(tickformat="$,.0f"),
                coloraxis_showscale=False,
                height=380,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_res, use_container_width=True)


# -----------------------------------------------------------------------------
# Page 6: About & Methodology
# -----------------------------------------------------------------------------
def render_about_page():
    st.markdown("<div class='main-header'>ℹ️ About this Project</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Machine Learning pipeline design, data origin, and practical deployment considerations.</div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.2, 0.8], gap="large")

    with col1:
        st.markdown("### 📚 The Ames Housing Dataset")
        st.write("""
        The **Ames Housing Dataset** was compiled by Professor Dean De Cock (Truman State University)
        as an expanded alternative to the legacy Boston Housing dataset. It records **2,930 residential
        home sales** in Ames, Iowa between 2006 and 2010.

        The dataset encompasses **79 raw explanatory variables** spanning:
        - **Physical Dimensions**: Living area, lot dimensions, basement and garage capacity, porch and deck areas.
        - **Quality Ratings**: Structural quality, condition, kitchen, basement, and exterior finishes.
        - **Temporal Markers**: Year built, year remodeled, and transaction date.
        - **Geographic & Zoning Context**: Neighborhood district, zoning code, and street topography.
        """)

        st.markdown("### ⚙️ Machine Learning Methodology")
        st.write("""
        1. **Leverage Outlier Elimination**: Removed influential outlier properties (`GrLivArea > 4,000 sq ft` with low sale prices) per De Cock's published recommendations.
        2. **Semantic NA Handling**: In features like `PoolQC`, `Alley`, `Fence`, `FireplaceQu`, `GarageType`, and `BsmtQual`, missing values (`NA`) signify that the facility is **not present**, not missing data. These are explicitly imputed as `"None"`.
        3. **Domain Feature Engineering**: Synthesized composite variables including `TotalSF` (total square footage), `HouseAge`, `RemodAge`, and `TotalBath`.
        4. **Target Log Transformation**: Applied $\\log(1 + y)$ (`np.log1p`) to normalize the right skewness of real estate prices.
        5. **Encapsulated Preprocessing**: Employed scikit-learn `ColumnTransformer` with median imputation and standardization for numerics, and constant imputation with `OneHotEncoder(handle_unknown='ignore')` for categoricals.
        6. **Model Benchmarking & Tuning**: Assessed 6 algorithms under 5-fold cross-validation, performing `RandomizedSearchCV` on top candidates.
        7. **Empirical 90% Prediction Intervals**: Derived prediction interval boundaries non-parametrically from the 5th and 95th percentiles of holdout log-residuals.
        """)

    with col2:
        st.markdown("### ⚠️ Practical Limitations")
        st.markdown("""
        <div class='info-card'>
            <strong>Geographic Specialization</strong><br>
            Trained exclusively on residential properties in Ames, Iowa. The price relationships and land values do not directly generalize to high-density metropolitan or coastal markets.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='info-card'>
            <strong>Temporal Drift & Inflation</strong><br>
            Transactions span 2006 to 2010. Nominal dollar estimates reflect pre-2011 economic conditions and do not account for subsequent macroeconomic inflation or post-pandemic real estate shifts.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='info-card'>
            <strong>Tail Risk on Luxury Estates</strong><br>
            Unusual architectural properties or extreme luxury homes (> 4,000 sq ft) exhibit higher residual uncertainty than standard suburban single-family homes.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🛠️ Tech Stack")
        st.write("""
        - **App Framework**: Streamlit
        - **Modeling & Explainability**: scikit-learn, XGBoost, SHAP
        - **Data Handling**: pandas, NumPy
        - **Interactive Visualizations**: Plotly Express & Graph Objects
        - **Artifact Storage**: joblib, JSON, CSV
        """)


# -----------------------------------------------------------------------------
# Main Application Router & Sidebar Navigation
# -----------------------------------------------------------------------------
def main():
    # Load model and data caches
    model, metrics = load_model_and_metrics()
    holdout_df = load_holdout_data()
    neighborhood_medians = load_neighborhood_medians()

    # Sidebar Brand & Navigation
    st.sidebar.markdown("""
    <div style='text-align: center; padding: 1rem 0;'>
        <h2 style='color: #1E88E5; margin-bottom: 0;'>🏡 Ames Valuation</h2>
        <p style='font-size: 0.85rem; color: #888;'>ML Real Estate Estimator</p>
    </div>
    """, unsafe_allow_html=True)

    pages = [
        "🏠 Predict Price",
        "🔮 What-If Explorer",
        "🔍 Feature Explainability",
        "📈 Data Explorer",
        "📊 Model Comparison",
        "ℹ️ About & Methodology",
    ]

    selected_page = st.sidebar.radio(
        "Navigation",
        pages,
        index=0,
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    if metrics:
        st.sidebar.markdown(f"**Best Model:** `{metrics.get('best_model_name', 'Lasso')}`")
        best_r2 = metrics.get("models", {}).get(metrics.get("best_model_name"), {}).get("R2", 0.9321)
        st.sidebar.markdown(f"**Holdout R²:** `{best_r2:.4f}`")
    st.sidebar.markdown("<p style='font-size:0.75rem; color:#666;'>Phase 3 Interactive Build</p>", unsafe_allow_html=True)

    # Route to selected page
    if selected_page == "🏠 Predict Price":
        render_predict_page(metrics, neighborhood_medians)
    elif selected_page == "🔮 What-If Explorer":
        render_whatif_page(metrics)
    elif selected_page == "🔍 Feature Explainability":
        render_explainability_page(metrics)
    elif selected_page == "📈 Data Explorer":
        render_data_explorer_page()
    elif selected_page == "📊 Model Comparison":
        render_model_comparison_page(metrics, holdout_df)
    elif selected_page == "ℹ️ About & Methodology":
        render_about_page()


if __name__ == "__main__":
    main()
