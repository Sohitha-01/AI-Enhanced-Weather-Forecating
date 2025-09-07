
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Rain Tomorrow — Predictor", page_icon="🌧️", layout="centered")
st.title("🌧️ Rain Tomorrow — Predictor")

def pick_existing(paths):
    for p in paths:
        pth = Path(p)
        if pth.exists():
            return pth
    return None

@st.cache_resource
def load_model_and_threshold():
    model_path = pick_existing([
        "model_rf.pkl",
        "model_rf_intermediate.pkl",
        "model_xgb_intermediate.pkl",
        "model_logreg.pkl",
        "model_logreg_intermediate.pkl",
    ])
    if model_path is None:
        st.error("No trained model file found. Place model_rf.pkl (or *_intermediate.pkl) next to this app.")
        st.stop()

    model = joblib.load(model_path)

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

    name_candidates = ["RandomForest_fixed", "RandomForest", "XGBoost_fixed", "XGBoost",
                       "LogisticRegression_fixed", "LogisticRegression"]
    for k in name_candidates:
        if k in thr_map:
            thr = float(thr_map[k])
            break

    return model, thr, model_path.name

@st.cache_resource
def load_reference_data():
    csv = Path("clean_weather.csv")
    if csv.exists():
        df = pd.read_csv(csv, parse_dates=["Date"])
        if "RainTomorrow" in df.columns:
            df = df.drop(columns=["RainTomorrow"])
        return df
    return None

model, best_thr, model_name = load_model_and_threshold()
ref_df = load_reference_data()

st.caption(f"Loaded model: `{model_name}` | Decision threshold: **{best_thr:.2f}**")

st.subheader("Enter today's weather conditions")

def default_inputs_from_ref(df: pd.DataFrame):
    values = {}
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    for c in num_cols:
        med = float(df[c].median()) if pd.api.types.is_numeric_dtype(df[c]) else 0.0
        values[c] = med

    for c in cat_cols:
        mode = df[c].mode()
        values[c] = (mode.iloc[0] if not mode.empty else "Unknown")

    return values, num_cols, cat_cols

user_vals = {}
if ref_df is not None:
    defaults, num_cols, cat_cols = default_inputs_from_ref(ref_df)
else:
    defaults = {
        "MinTemp": 10.0, "MaxTemp": 20.0, "Rainfall": 0.0,
        "Humidity9am": 60.0, "Humidity3pm": 50.0,
        "Pressure9am": 1015.0, "Pressure3pm": 1012.0,
        "Temp9am": 15.0, "Temp3pm": 20.0,
        "WindGustSpeed": 30.0, "WindSpeed9am": 15.0, "WindSpeed3pm": 20.0,
        "Cloud9am": 4.0, "Cloud3pm": 4.0,
        "Location": "Sydney", "WindDir9am": "N", "WindDir3pm": "SE", "WindGustDir": "W"
    }
    num_cols = [k for k,v in defaults.items() if isinstance(v, (int, float))]
    cat_cols = [k for k,v in defaults.items() if isinstance(v, str)]

with st.form("inp"):
    cols = st.columns(2)

    for i, c in enumerate(num_cols):
        with cols[i % 2]:
            val = st.number_input(c, value=float(defaults[c]))
            user_vals[c] = val

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

if submitted:
    X = pd.DataFrame([user_vals])
    try:
        # Align X columns to match the training schema expected by the pipeline
        prep = model.named_steps.get("prep", None)
        expected_cols = None
        if prep is not None and hasattr(prep, "transformers_"):
            num_cols, cat_cols = [], []
            for name, trans, cols in prep.transformers_:
                if name == "num":
                    num_cols = list(cols)
                elif name == "cat":
                    cat_cols = list(cols)
            expected_cols = num_cols + cat_cols

        if expected_cols is not None:
            # Add missing columns as NaN so imputers can handle them
            for c in expected_cols:
                if c not in X.columns:
                    X[c] = np.nan
            # Keep only expected columns and in the correct order
            X = X[expected_cols]

        # Now predict
        proba = model.predict_proba(X)[:, 1][0]
        pred = int(proba >= best_thr)

        st.markdown("### Prediction")
        st.metric("Rain Tomorrow Probability", f"{proba:.2%}")
        st.write("Decision:", "**Yes** ☔" if pred == 1 else "**No** 🌤️")
        st.caption("Tip: Adjust inputs and re-submit to see changes.")

    except Exception as e:
        st.error(f"Failed to run prediction: {e}")
        st.info("If this persists, ensure your 'clean_weather.csv' includes all training columns, "
                "or re-train the model without lag features.")

