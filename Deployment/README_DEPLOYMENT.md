# Step 7 — Deployment (Streamlit)

## Why files were too big
GitHub (web UI) blocks files >25MB. Instead of checking big binaries (models) into your repo,
this app can **download them at runtime** from a URL you control.

## How to use (local)
1) Put these files with your trained artifacts if you have them locally:
   - `model_rf.pkl` (or `model_rf_intermediate.pkl` / `model_logreg.pkl`)
   - `thresholds_intermediate.json` (optional)
   - `clean_weather.csv` (optional)
2) Install & run:
```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

## How to deploy on Streamlit Community Cloud
1) Push `app.py` and `requirements.txt` to a public GitHub repo.
2) Do **NOT** commit large model files.
3) In Streamlit Cloud, set **Secrets or Environment Variables**:
   - `MODEL_URL` : direct link to a `.pkl` (e.g., from Hugging Face Hub raw file or Google Drive direct link)
   - `THRESHOLDS_URL` : (optional) direct link to `thresholds_intermediate.json`
   - `CLEAN_CSV_URL` : (optional) direct link to `clean_weather.csv`
4) Deploy. The app will **download** the artifacts at startup if they are missing.

## Notes
- The app introspects your pipeline to match expected input columns and will auto-create missing lag features as NaN so imputers can handle them.
- Make sure your model was trained with **scikit-learn==1.6.1** (same version is pinned here).
