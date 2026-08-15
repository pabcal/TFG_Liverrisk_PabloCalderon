"""
LiverRisk webapp backend -- POST /predict route.

Accepts a CSV upload (one or a few patients, same raw columns as
train.csv / test.csv), runs them through the already-trained models
loaded in models_loader.py, and returns a risk score for each patient.
"""
from __future__ import annotations

import io

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

# models_loader must be imported before any liverrisk.* import below --
# it's the one that puts the repo root (where the liverrisk package
# lives) onto sys.path.
from models_loader import (
    AGE_COLUMNS,
    DISTRIBUTION_NOTES,
    FEATURE_COLUMNS,
    HISTOGRAMS,
    REQUIRED_COLUMNS,
    TRAIN_FIB4_SCORES,
    TRAIN_APRI_SCORES,
    TRAIN_WEIGHTED_SCORES,
    X_HEP,
    _json_safe,
    percentile_le,
    score_one_endpoint,
)

from explain import explain_patient

from liverrisk.clinical_scores import compute_fib4_apri
from liverrisk.features import build_patient_features

router = APIRouter()


# --------------------------------------------------------------------
# The one API endpoint
# --------------------------------------------------------------------
@router.post("/predict")
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

    # A fresh upload can infer different dtypes than training did (e.g.
    # int64 for a column that happens to have no missing values in this
    # particular upload, vs. float64 during training where the same
    # column had NaNs). Casting to X_HEP's dtypes keeps every column on
    # the exact dtype the fitted preprocessors expect, before any model
    # ever sees this data.
    patient_features = patient_features.astype(X_HEP.dtypes.to_dict())

    # FIB-4/APRI for every uploaded patient at once (vectorized), using
    # the same clinical_scores.compute_fib4_apri() helper the training-
    # cohort reference distribution was built from above.
    patient_fib4_apri = compute_fib4_apri(patient_features, "hep")

    # Step 4: score every patient (row) in the upload.
    results = []
    for i in range(len(patient_features)):
        one_patient = patient_features.iloc[[i]]  # keep it as a 1-row DataFrame

        risk_hepatic_event, hepatic_percentile = score_one_endpoint(one_patient, "hep")
        risk_death, death_percentile = score_one_endpoint(one_patient, "death")
        risk_explanation = explain_patient(one_patient)

        # Same formula build_submission() (scripts/train.py) uses to turn
        # a test-set patient's two blended scores into one weighted_risk.
        weighted_risk = round(0.7 * risk_hepatic_event + 0.3 * risk_death, 4)
        weighted_percentile = percentile_le(weighted_risk, TRAIN_WEIGHTED_SCORES)

        fib4_score = patient_fib4_apri["fib4_score"].iloc[i]
        apri_score = patient_fib4_apri["apri_score"].iloc[i]
        fib4_score = None if pd.isna(fib4_score) else round(float(fib4_score), 2)
        apri_score = None if pd.isna(apri_score) else round(float(apri_score), 2)
        fib4_percentile = None if fib4_score is None else percentile_le(fib4_score, TRAIN_FIB4_SCORES)
        apri_percentile = None if apri_score is None else percentile_le(apri_score, TRAIN_APRI_SCORES)

        # Age_v1 is guaranteed present (it's in REQUIRED_COLUMNS); the
        # rest of AGE_COLUMNS (Age_v2, Age_v3, ...) may not all be in a
        # smaller upload, so only count the ones actually present --
        # same visit-counting logic as select_sample_patients() in
        # sample_patients.py, just applied to whatever columns this
        # particular CSV has.
        # Used by the Rankings tab's "My active patients" scope, which
        # gets a patient's age/visits from this response rather than
        # re-parsing the CSV itself.
        present_age_columns = [c for c in AGE_COLUMNS if c in raw_df.columns]
        age_at_baseline = _json_safe(raw_df.iloc[i]["Age_v1"])
        visit_count = int(raw_df.iloc[i][present_age_columns].notna().sum())

        results.append({
            "risk_hepatic_event": risk_hepatic_event,
            "hepatic_percentile": hepatic_percentile,
            "risk_death": risk_death,
            "death_percentile": death_percentile,
            "risk_explanation": risk_explanation,
            "weighted_risk": weighted_risk,
            "weighted_percentile": weighted_percentile,
            "fib4_score": fib4_score,
            "fib4_percentile": fib4_percentile,
            "apri_score": apri_score,
            "apri_percentile": apri_percentile,
            "age_at_baseline": age_at_baseline,
            "visit_count": visit_count,
            "histograms": HISTOGRAMS,
            "distribution_notes": DISTRIBUTION_NOTES,
        })

    return results
