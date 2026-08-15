"""
LiverRisk webapp backend -- model loading and shared scoring helpers.

This module does all startup loading (models, training reference
scores, precomputed histograms) and exposes the scoring helpers shared
by more than one route module.

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

import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

# --------------------------------------------------------------------
# Make the "liverrisk" package (which lives at the repo root, not
# inside liverrisk-webapp/) importable from this file.
# --------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from liverrisk.clinical_scores import compute_fib4_apri, fib4  # noqa: E402
from liverrisk.features import (  # noqa: E402
    REPEATED_BASES,
    STATIC_CAT,
    STATIC_NUM,
    build_patient_features,
    load_features,
)
from liverrisk.models import RANDOM_STATE  # noqa: E402

MODELS_DIR = REPO_ROOT / "models"
PROCESSED_DIR = REPO_ROOT / "liverrisk" / "data" / "processed"
RAW_TEST_PATH = REPO_ROOT / "liverrisk" / "data" / "raw" / "test.csv"


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


def load_training_reference_scores(models: dict, X_hep: pd.DataFrame, X_death: pd.DataFrame) -> dict:
    """
    Runs the loaded models on the training-cohort features to get each
    model's raw score for every training patient. We need these raw
    training scores so that a new patient's raw score can be turned
    into "percentile within the training cohort" (see module docstring,
    step 4). This is plain inference (model.predict), not training.
    """
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


# X_hep/X_death are also needed (beyond load_training_reference_scores
# above) to build the FIB-4/APRI reference distribution and, further
# down, the weighted_risk reference distribution -- so they're loaded
# once here rather than inside a function that would only return the
# model scores. HEP_EVENT/HEP_TIME (whether/when each training patient
# had a hepatic event) are kept too, for the Rankings tab's "outcome"
# column further down -- both arrays are positionally aligned with
# X_HEP's rows (load_features returns them from the same saved payload).
X_HEP, _, HEP_EVENT, HEP_TIME = load_features("hep", PROCESSED_DIR)
X_DEATH, _, _, _ = load_features("death", PROCESSED_DIR)

MODELS = load_trained_models()
TRAIN_RAW_SCORES = load_training_reference_scores(MODELS, X_HEP, X_DEATH)

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
# weighted_risk reference distribution (for weighted_percentile).
#
# weighted_risk itself is just 0.7 * risk_hepatic_event + 0.3 *
# risk_death for one patient -- the same formula build_submission()
# (scripts/train.py) applies to the test-set predictions, which is
# safe there because pred_hep/pred_death are computed for the exact
# same test rows in the exact same order.
#
# The training cohort does NOT have that guarantee: the hepatic and
# death endpoints were trained on different patient subsets (death
# excludes patients without usable death-censoring data), so
# TRAIN_BLENDED_SCORES["hep"] (len 1253) and ["death"] (len 984) are
# NOT row-aligned -- death's patients are a subset of hep's, but not
# in general at the same positions. Combining them positionally would
# silently pair up scores from different patients. Instead we align by
# patient index (X_HEP.index / X_DEATH.index, which load_features
# preserves from the processed parquet files) and only build the
# weighted reference over the patients who have both scores, i.e. the
# death cohort.
_hep_scores_by_patient = pd.Series(TRAIN_BLENDED_SCORES["hep"], index=X_HEP.index)
_death_scores_by_patient = pd.Series(TRAIN_BLENDED_SCORES["death"], index=X_DEATH.index)
TRAIN_WEIGHTED_SCORES = (
    0.7 * _hep_scores_by_patient.reindex(_death_scores_by_patient.index) + 0.3 * _death_scores_by_patient
).to_numpy()

# --------------------------------------------------------------------
# FIB-4 / APRI reference distributions, computed once from the hepatic
# training cohort's already-built features (X_HEP), reusing
# clinical_scores.compute_fib4_apri() -- the exact same fib4()/apri()
# formulas Study 1's notebook used, not a reimplementation. NaN rows
# (patients missing the required last-observed labs) are dropped from
# the reference arrays so they don't distort the percentile lookup.
_train_fib4_apri = compute_fib4_apri(X_HEP, "hep")
TRAIN_FIB4_SCORES = _train_fib4_apri["fib4_score"].dropna().to_numpy()
TRAIN_APRI_SCORES = _train_fib4_apri["apri_score"].dropna().to_numpy()

# --------------------------------------------------------------------
# SHAP explainer for per-patient explanations (explain.py).
#
# rsf_hep is what drives the hepatic blend today (blend weight 1.0, see
# DISTRIBUTION_NOTES below), so it's the only model explained. It's a
# Pipeline of ("pre", ColumnTransformer) + ("rsf", RandomSurvivalForest)
# -- KernelExplainer needs a plain array-in/array-out function, so it
# wraps the "rsf" step's .predict directly and everything (background
# sample, the training cohort) is passed through "pre" first, once,
# here at startup. Only the per-patient explainer.shap_values() call in
# explain.py runs per-request.
# --------------------------------------------------------------------
print("[startup] Preparing SHAP background sample for hepatic-event explanations...")

RSF_HEP_PIPELINE = MODELS["rsf_hep"]
RSF_HEP_PREPROCESSOR = RSF_HEP_PIPELINE.named_steps["pre"]
RSF_HEP_MODEL = RSF_HEP_PIPELINE.named_steps["rsf"]

_X_HEP_TRANSFORMED = RSF_HEP_PREPROCESSOR.transform(X_HEP)
if hasattr(_X_HEP_TRANSFORMED, "toarray"):
    _X_HEP_TRANSFORMED = _X_HEP_TRANSFORMED.toarray()

RSF_HEP_FEATURE_NAMES = list(RSF_HEP_PREPROCESSOR.get_feature_names_out())
X_HEP_TRANSFORMED = pd.DataFrame(_X_HEP_TRANSFORMED, columns=RSF_HEP_FEATURE_NAMES, index=X_HEP.index)

SHAP_BACKGROUND = shap.sample(X_HEP_TRANSFORMED, 30, random_state=RANDOM_STATE)

RSF_HEP_EXPLAINER = shap.KernelExplainer(RSF_HEP_MODEL.predict, SHAP_BACKGROUND)

print(f"[startup] SHAP background ready: {SHAP_BACKGROUND.shape[0]} patients x {SHAP_BACKGROUND.shape[1]} features.")


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


def percentile_le(value: float, reference_scores: np.ndarray) -> float:
    """
    What percentage of reference_scores are <= value? Used for every
    "percentile within the training cohort" figure reported to the
    frontend (hepatic, death, weighted, FIB-4, APRI) so they're all on
    the same 0-100 scale and computed the same way.
    """
    return round(float((reference_scores <= value).mean()) * 100, 1)


def make_histogram(scores: np.ndarray, n_bins: int = 20) -> list[dict]:
    """
    Pre-bins a training-cohort reference distribution into a small,
    JSON-friendly list of {bin_start, bin_end, count} -- cheaper to
    send on every /predict response than the ~1000 raw training scores,
    and it's all the frontend needs to draw a bar-based histogram.
    """
    counts, edges = np.histogram(scores, bins=n_bins)
    return [
        {"bin_start": round(float(edges[i]), 4), "bin_end": round(float(edges[i + 1]), 4), "count": int(counts[i])}
        for i in range(len(counts))
    ]


# Computed once at startup: every /predict response reuses the same
# training-cohort histograms, only the patient's marker position changes.
HISTOGRAMS = {
    "hepatic": make_histogram(TRAIN_BLENDED_SCORES["hep"]),
    "death": make_histogram(TRAIN_BLENDED_SCORES["death"]),
    "weighted": make_histogram(TRAIN_WEIGHTED_SCORES),
}


def single_model_note(weights, threshold: float = 0.999) -> str | None:
    """
    If one blend weight is essentially 1.0 (the others ~0), the
    "blended" score for that endpoint is really just one model's
    rank-percentile alone -- which is close to uniformly distributed
    across the training cohort by construction (it's a rank/N value),
    not because of anything wrong with the histogram. Returns an
    explanatory note for that case, so the frontend can caption a flat
    histogram instead of it silently looking broken; returns None when
    the blend actually mixes models (a real, non-uniform distribution).
    """
    model_names = ["Coxnet", "Random Survival Forest", "XGBoost"]
    for name, w in zip(model_names, weights):
        if w >= threshold:
            return f"This score is the {name} model's percentile rank alone, so it's spread almost evenly across the training cohort by construction."
    return None


# hep = [0.0, 1.0, 0.0] as tuned today (pure RSF), which is why the
# hepatic histogram looks flat -- see single_model_note() above. Kept
# data-driven (reads BLEND_WEIGHTS) rather than hardcoded, so this note
# disappears on its own if the models are ever retuned to a real mix.
DISTRIBUTION_NOTES = {
    "hepatic": single_model_note(BLEND_WEIGHTS["hep"]),
    "death": single_model_note(BLEND_WEIGHTS["death"]),
}


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

    percentile_vs_train = percentile_le(blended, TRAIN_BLENDED_SCORES[endpoint])

    return round(float(blended), 4), percentile_vs_train


def _json_safe(value):
    """Turns a pandas/numpy scalar into something json-serializable, with NaN/NaT becoming None."""
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


# --------------------------------------------------------------------
# AGE_COLUMNS is used by more than one route module (predict.py,
# sample_patients.py, rankings.py), so it's derived here from the raw
# test set's columns rather than duplicated.
# --------------------------------------------------------------------
_test_columns_for_age = pd.read_csv(RAW_TEST_PATH, nrows=0).columns.tolist()
AGE_COLUMNS = [c for c in _test_columns_for_age if re.fullmatch(r"Age_v\d+", c)]
