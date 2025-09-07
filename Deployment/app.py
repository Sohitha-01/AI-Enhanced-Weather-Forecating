
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from pathlib import Path

# Optional: download big artifacts at runtime to bypass 25MB repo limits
import requests

st.set_page_config(page_title="Rain Tomorrow — Predictor", page_icon="🌧️", layout="centered")
st.title("🌧️ Rain Tomorrow — Predictor")

# ---------- Config ----------
# Set these environment variables on Streamlit Cloud (or hardcode below):
#   MODEL_URL      -> direct download link to your model .pkl (e.g., HF Hub raw file / GDrive direct)
#   THRESHOLDS_URL -> (optional) thresholds_intermediate.json direct link
#   CLEAN_CSV_URL  -> (optional) clean_weather.csv direct link (for form building)
MODEL_URL = os.getenv("MODEL_URL", "").strip()
THRESHOLDS_URL = os.getenv("THRESHOLDS_URL", "").strip()
CLEAN_CSV_URL = os.getenv("CLEAN_CSV_URL", "").strip()

def download_if_missing(url: str, dest: str):
    if not url or Path(dest).exists():
        return
    try:
        st.caption(f"Downloading {dest} ...")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(dest, "wb") as f:
            f.write(r.content)
        st.caption(f"Saved: {dest}")
    except Exception as e:
        st.warning(f"Could not download {dest} from URL: {e}")

# Try to fetch artifacts if absent
download_if_missing(MODEL_URL, "model_rf.pkl")  # prefer RF name; adjust if your URL is a different model
download_if_missing(THRESHOLDS_URL, "thresholds_intermediate.json")
download_if_missing(CLEAN_CSV_URL, "clean_weather.csv")

def pick_existing(paths):
    for p in paths:
        pth = Path(p)
        if pth.exists():
            return pth
    return None

@st.cache_resource
def load_model_and_threshold():
    # Try order: RF -> XGB -> LR
    model_path = pick_existing([
        "model_rf.pkl",
        "model_rf_intermediate.pkl",
        "model_xgb_intermediate.pkl",
        "model_logreg.pkl",
        "model_logreg_intermediate.pkl",
    ])
    if model_path is None:
        st.error("No trained model file found. Place a model next to this app or set MODEL_URL env var.")
        st.stop()

    model = joblib.load(model_path)

    # Load thresholds
    thr = 0.5
    thr_map = {}
    for tf in ["thresholds_intermediate.json", "thresholds.json"]:
        p = Path(tf)
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    thr_map.update(json.load(f))
            except Exception:
                pass

    # Choose a reasonable key
    for k in ["RandomForest_fixed", "RandomForest", "XGBoost_fixed", "XGBoost", "LogisticRegression_fixed", "LogisticRegression"]:
        if k in thr_map:
            thr = float(thr_map[k])
            break

    return model, thr, model_path.name

@st.cache_resource
def load_reference_data():
    csv = Path("clean_weather.csv")
    if csv.exists():
        try:
            df = pd.read_csv(csv, parse_dates=["Date"])
            if "RainTomorrow" in df.columns:
                df = df.drop(columns=["RainTomorrow"])
            return df
        except Exception:
            return None
    return None

model, best_thr, model_name = load_model_and_threshold()
ref_df = load_reference_data()

st.caption(f"Loaded model: `{model_name}` | Decision threshold: **{best_thr:.2f}**")

# ---------- Build input form ----------
st.subheader("Enter today's weather conditions")

def defaults_from_ref(df: pd.DataFrame):
    values = {}
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    for c in num_cols:
        med = float(df[c].median()) if pd.api.types.is_numeric_dtype(df[c]) else 0.0
        values[c] = med
    for c in cat_cols:
        mode = df[c].mode()
        values[c] = (mode.iloc[0] if not mode.empty else "Unknown")
    return values, num_cols, cat_cols

user_vals = {}
if ref_df is not None:
    defaults, num_cols, cat_cols = defaults_from_ref(ref_df)
else:
    # Minimal fallback if clean_weather.csv isn't provided
    defaults = {
        "MinTemp": 10.0, "MaxTemp": 20.0, "Rainfall": 0.0,
        "Humidity9am": 60.0, "Humidity3pm": 50.0,
        "Pressure9am": 1015.0, "Pressure3pm": 1012.0,
        "Temp9am": 15.0, "Temp3pm": 20.0,
        "WindGustSpeed": 30.0, "WindSpeed9am": 15.0, "WindSpeed3pm": 20.0,
        "Cloud9am": 4.0, "Cloud3pm": 4.0,
        "Location": "Sydney", "WindDir9am": "N", "WindDir3pm": "SE", "WindGustDir": "W",
        # Common engineered features that may be required by the model (imputed if left NaN)
        "RainToday": 0.0,
        "Humidity9am_lag1": np.nan, "Humidity3pm_lag1": np.nan,
        "Pressure9am_lag1": np.nan, "Pressure3pm_lag1": np.nan,
        "Temp9am_lag1": np.nan, "Temp3pm_lag1": np.nan,
        "Rainfall_lag1": np.nan, "Sunshine": np.nan, "Evaporation": np.nan,
    }
    num_cols = [k for k, v in defaults.items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    cat_cols = [k for k, v in defaults.items() if isinstance(v, str)]

with st.form("inp"):
    cols = st.columns(2)

    # Numeric fields
    for i, c in enumerate(num_cols):
        with cols[i % 2]:
            val = st.number_input(c, value=float(defaults[c]) if pd.notna(defaults[c]) else 0.0)
            user_vals[c] = val

    # Categorical fields
    for i, c in enumerate(cat_cols):
        with cols[(i + len(num_cols)) % 2]:
            options = None
            if ref_df is not None and c in ref_df.columns:
                options = ref_df[c].dropna().astype(str).value_counts().index.tolist()[:20]
                if not options:
                    options = [str(defaults[c])]
            else:
                options = [str(defaults[c])]
            val = st.selectbox(c, options=options, index=0 if options else None)
            user_vals[c] = val

    submitted = st.form_submit_button("Predict")

# ---------- Predict ----------
if submitted:
    X = pd.DataFrame([user_vals])
    try:
        # Align X columns to the expected training schema from the pipeline
        prep = getattr(model, "named_steps", {}).get("prep", None)
        expected_cols = None
        if prep is not None and hasattr(prep, "transformers_"):
            num_cols_cfg, cat_cols_cfg = [], []
            for name, trans, cols in prep.transformers_:
                if name == "num":
                    num_cols_cfg = list(cols)
                elif name == "cat":
                    cat_cols_cfg = list(cols)
            expected_cols = (num_cols_cfg or []) + (cat_cols_cfg or [])

        if expected_cols is not None:
            # Add any missing columns with NaN so imputers can fill
            for c in expected_cols:
                if c not in X.columns:
                    X[c] = np.nan
            # Keep only expected columns and in exact order
            X = X[expected_cols]

        # Predict
        proba = model.predict_proba(X)[:, 1][0]
        pred = int(proba >= best_thr)

        st.markdown("### Prediction")
        st.metric("Rain Tomorrow Probability", f"{proba:.2%}")
        st.write("Decision:", "**Yes** ☔" if pred == 1 else "**No** 🌤️")
        st.caption("Tip: Adjust inputs and re-submit to see changes.")
    except Exception as e:
        st.error(f"Failed to run prediction: {e}")
        st.info("Ensure your model & input schema match. Keep `clean_weather.csv` next to this app "
                "so the form reflects training columns, or set CLEAN_CSV_URL to fetch it.")
