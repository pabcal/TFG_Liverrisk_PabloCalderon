"""
LiverRisk webapp backend.

This is a small FastAPI server with a single job: take a CSV upload
describing one or a few patients (same raw columns as train.csv /
test.csv), run them through the already-trained models in models/, and
return a risk score for each patient.

Nothing here trains or re-saves a model -- everything in models/ was
produced by scripts/train.py and is only ever *read* below.

How a prediction is scored, in plain steps:
  1. Read the uploaded CSV into a pandas DataFrame.
  2. Turn the wide per-visit columns into the same engineered feature
     table the models were trained on (build_patient_features).
  3. Ask each of the three models (Coxnet, Random Survival Forest,
     XGBoost) for a raw risk score.
  4. Convert each raw score into a percentile position within that
     model's *training-cohort* scores. This step matters: raw scores
     from different model types live on different, incomparable
     scales, so we can only combine them once they're all expressed
     as "how high is this compared to the training patients".
  5. Combine the three percentile positions with the tuned blend
     weights (models/blend_weights_*.joblib) to get one blended score
     per outcome (hepatic event, death).
  6. Report where that blended score falls, as a percentile, among the
     blended scores of the whole training cohort.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

# --------------------------------------------------------------------
# Make the "liverrisk" package (which lives at the repo root, not
# inside liverrisk-webapp/) importable from this file.
# --------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from liverrisk.features import (  # noqa: E402
    REPEATED_BASES,
    STATIC_CAT,
    STATIC_NUM,
    build_patient_features,
    load_features,
)

MODELS_DIR = REPO_ROOT / "models"
PROCESSED_DIR = REPO_ROOT / "liverrisk" / "data" / "processed"
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="LiverRisk webapp")


# --------------------------------------------------------------------
# Columns a valid upload must have. This is not the full list of
# columns build_patient_features() can use (many optional follow-up
# visits exist too) -- it's the minimum needed to compute anything
# meaningful, so we can give a clear error instead of a stack trace.
# --------------------------------------------------------------------
REQUIRED_COLUMNS = (
    ["Age_v1"]
    + STATIC_CAT
    + STATIC_NUM
    + [f"{base}_v1" for base in REPEATED_BASES]
)


# --------------------------------------------------------------------
# Load everything we need once, when the server starts. Loading is
# read-only: we open existing .joblib files, we never fit or save
# anything here.
# --------------------------------------------------------------------
def load_trained_models() -> dict:
    """Loads the six fitted model pipelines from models/*.joblib."""
    return {
        "cox_hep": joblib.load(MODELS_DIR / "cox_hep.joblib"),
        "rsf_hep": joblib.load(MODELS_DIR / "rsf_hep.joblib"),
        "xgb_hep": joblib.load(MODELS_DIR / "xgb_hep.joblib"),
        "cox_death": joblib.load(MODELS_DIR / "cox_death.joblib"),
        "rsf_death": joblib.load(MODELS_DIR / "rsf_death.joblib"),
        "xgb_death": joblib.load(MODELS_DIR / "xgb_death.joblib"),
    }


def load_training_reference_scores(models: dict) -> dict:
    """
    Runs the loaded models on the training-cohort features to get each
    model's raw score for every training patient. We need these raw
    training scores so that a new patient's raw score can be turned
    into "percentile within the training cohort" (see module docstring,
    step 4). This is plain inference (model.predict), not training.
    """
    X_hep, _, _, _ = load_features("hep", PROCESSED_DIR)
    X_death, _, _, _ = load_features("death", PROCESSED_DIR)

    return {
        "hep": {
            "cox": models["cox_hep"].predict(X_hep),
            "rsf": models["rsf_hep"].predict(X_hep),
            "xgb": models["xgb_hep"].predict(X_hep),
        },
        "death": {
            "cox": models["cox_death"].predict(X_death),
            "rsf": models["rsf_death"].predict(X_death),
            "xgb": models["xgb_death"].predict(X_death),
        },
    }


MODELS = load_trained_models()
TRAIN_RAW_SCORES = load_training_reference_scores(MODELS)

with open(MODELS_DIR / "feature_columns.json") as f:
    FEATURE_COLUMNS = json.load(f)

BLEND_WEIGHTS = {
    "hep": joblib.load(MODELS_DIR / "blend_weights_hep.joblib"),
    "death": joblib.load(MODELS_DIR / "blend_weights_death.joblib"),
}

TRAIN_BLENDED_SCORES = {
    "hep": joblib.load(MODELS_DIR / "train_scores_hepatic.joblib"),
    "death": joblib.load(MODELS_DIR / "train_scores_death.joblib"),
}


# --------------------------------------------------------------------
# Scoring helpers
# --------------------------------------------------------------------
def percentile_within_training(new_score: float, train_scores: np.ndarray) -> float:
    """
    Where does new_score fall among train_scores, as a value from 0
    (lowest) to 1 (highest)? We do this by adding new_score to the
    training scores and ranking the combined list, then reading off
    the rank of the value we just added.
    """
    combined = np.append(train_scores, new_score)
    ranks_as_percentiles = pd.Series(combined).rank(method="average", pct=True)
    return float(ranks_as_percentiles.iloc[-1])


def score_one_endpoint(patient_row: pd.DataFrame, endpoint: str) -> tuple[float, float]:
    """
    Scores a single patient (a one-row DataFrame) for one endpoint
    ("hep" or "death"). Returns (blended_risk_score, percentile).
    """
    cox_model = MODELS[f"cox_{endpoint}"]
    rsf_model = MODELS[f"rsf_{endpoint}"]
    xgb_model = MODELS[f"xgb_{endpoint}"]

    raw_cox = cox_model.predict(patient_row)[0]
    raw_rsf = rsf_model.predict(patient_row)[0]
    raw_xgb = xgb_model.predict(patient_row)[0]

    pct_cox = percentile_within_training(raw_cox, TRAIN_RAW_SCORES[endpoint]["cox"])
    pct_rsf = percentile_within_training(raw_rsf, TRAIN_RAW_SCORES[endpoint]["rsf"])
    pct_xgb = percentile_within_training(raw_xgb, TRAIN_RAW_SCORES[endpoint]["xgb"])

    w_cox, w_rsf, w_xgb = BLEND_WEIGHTS[endpoint]
    weight_sum = w_cox + w_rsf + w_xgb
    blended = (w_cox * pct_cox + w_rsf * pct_rsf + w_xgb * pct_xgb) / weight_sum

    percentile_vs_train = float((TRAIN_BLENDED_SCORES[endpoint] <= blended).mean()) * 100

    return round(float(blended), 4), round(percentile_vs_train, 1)


# --------------------------------------------------------------------
# The one API endpoint
# --------------------------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts a CSV upload (one or a few patients, same columns as
    train.csv/test.csv) and returns a risk score + percentile for
    each patient, for both the hepatic-event and death outcomes.
    """
    # Step 1: read the upload into a DataFrame.
    raw_bytes = await file.read()
    try:
        raw_df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded file as a CSV.")

    # Step 2: check the CSV has the columns we need before we try
    # anything fancier, so the error message is clear.
    missing_columns = [c for c in REQUIRED_COLUMNS if c not in raw_df.columns]
    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail=f"The uploaded CSV is missing required columns: {', '.join(missing_columns)}",
        )

    # Step 3: build the same engineered features the models were
    # trained on, then line up the columns in the exact order the
    # models expect.
    try:
        patient_features = build_patient_features(raw_df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not build features from the uploaded CSV: {e}")

    patient_features = patient_features.reindex(columns=FEATURE_COLUMNS)

    # Step 4: score every patient (row) in the upload.
    results = []
    for i in range(len(patient_features)):
        one_patient = patient_features.iloc[[i]]  # keep it as a 1-row DataFrame

        risk_hepatic_event, hepatic_percentile = score_one_endpoint(one_patient, "hep")
        risk_death, death_percentile = score_one_endpoint(one_patient, "death")

        results.append({
            "risk_hepatic_event": risk_hepatic_event,
            "hepatic_percentile": hepatic_percentile,
            "risk_death": risk_death,
            "death_percentile": death_percentile,
        })

    return results


# --------------------------------------------------------------------
# Serve the plain HTML/CSS/JS frontend from the same server, so the
# page can call /predict on its own origin with a plain fetch() call.
# This must be mounted last -- routes defined above (like /predict)
# are matched first, and anything else falls through to these static
# files.
# --------------------------------------------------------------------
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
