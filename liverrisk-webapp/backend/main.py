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
import re
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

from liverrisk.clinical_scores import compute_fib4_apri, fib4  # noqa: E402
from liverrisk.features import (  # noqa: E402
    REPEATED_BASES,
    STATIC_CAT,
    STATIC_NUM,
    build_patient_features,
    load_features,
)

MODELS_DIR = REPO_ROOT / "models"
PROCESSED_DIR = REPO_ROOT / "liverrisk" / "data" / "processed"
RAW_TEST_PATH = REPO_ROOT / "liverrisk" / "data" / "raw" / "test.csv"
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
        # same visit-counting logic as select_sample_patients() below,
        # just applied to whatever columns this particular CSV has.
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


# --------------------------------------------------------------------
# "Build a test CSV" tab support: lets a user assemble a small CSV out
# of a handful of real (unlabeled) test-set patients, without needing
# their own data, to try out /predict above.
# --------------------------------------------------------------------
TEST_COLUMNS = pd.read_csv(RAW_TEST_PATH, nrows=0).columns.tolist()

# The three "NIT" (non-invasive test) repeated measures in the dataset,
# in the same order they appear as REPEATED_BASES entries in features.py.
NIT_BASES = ["fibrotest_BM_2", "aixp_aix_result_BM_3", "fibs_stiffness_med_BM_1"]
NIT_LABELS = {
    "fibrotest_BM_2": "FibroTest",
    "aixp_aix_result_BM_3": "AIx-P",
    "fibs_stiffness_med_BM_1": "Liver stiffness (FibroScan)",
}


def _visit_columns(columns: list[str], base: str) -> list[str]:
    """All `{base}_v<n>` columns for one repeated measure, e.g. all fibrotest_BM_2_v* columns."""
    return [c for c in columns if re.fullmatch(fr"{re.escape(base)}_v\d+", c)]


AGE_COLUMNS = _visit_columns(TEST_COLUMNS, "Age")
NIT_COLUMNS = {base: _visit_columns(TEST_COLUMNS, base) for base in NIT_BASES}


def select_sample_patients(raw_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Picks `n` patients out of the full (unlabeled) test set for the
    "Build a test CSV" tab, aiming for variety rather than just the
    first n rows: patients are ranked by (number of recorded visits,
    baseline FIB-4), then n picks are taken evenly spaced across that
    ranking so the sample spans low-to-high visit counts and FIB-4
    levels instead of clustering at one end.
    """
    visit_counts = raw_df[AGE_COLUMNS].notna().sum(axis=1)
    baseline_fib4 = fib4(raw_df["Age_v1"], raw_df["ast_v1"], raw_df["plt_v1"], raw_df["alt_v1"])

    ranking = pd.DataFrame({
        "visits": visit_counts,
        # Patients with no computable baseline FIB-4 sort first, so they
        # don't crowd out patients that do have a baseline value.
        "fib4": baseline_fib4.fillna(-1),
    }).sort_values(["visits", "fib4"])

    sorted_positions = np.arange(len(ranking))
    pick_positions = np.unique(np.linspace(0, len(sorted_positions) - 1, num=n).round().astype(int))

    # Rounding can collapse two target spots onto the same row for a
    # small n; top up with the next not-yet-picked rows so we still
    # return exactly n patients.
    pick_positions = set(pick_positions.tolist())
    for pos in range(len(sorted_positions)):
        if len(pick_positions) >= n:
            break
        pick_positions.add(pos)

    chosen_idx = ranking.index[sorted(pick_positions)[:n]]
    return raw_df.loc[chosen_idx]


def _json_safe(value):
    """Turns a pandas/numpy scalar into something json-serializable, with NaN/NaT becoming None."""
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


@app.get("/sample-patients")
async def sample_patients():
    """
    Returns 10 pre-selected patients from the real (unlabeled) test set,
    each with every raw column (so the frontend can reconstruct a valid
    test.csv row later) plus a few human-readable summary fields for
    display in the "Build a test CSV" tab.
    """
    raw_df = pd.read_csv(RAW_TEST_PATH)
    sample_df = select_sample_patients(raw_df, n=10)

    patients = []
    for _, row in sample_df.iterrows():
        visit_count = int(row[AGE_COLUMNS].notna().sum())

        baseline_fib4 = fib4(row["Age_v1"], row["ast_v1"], row["plt_v1"], row["alt_v1"])
        baseline_fib4 = None if pd.isna(baseline_fib4) else round(float(baseline_fib4), 2)

        # Whichever of the three NIT/lab measures has the most recorded
        # (non-null) visits for this particular patient.
        nit_counts = {base: int(row[cols].notna().sum()) for base, cols in NIT_COLUMNS.items()}
        most_complete_base = max(nit_counts, key=nit_counts.get)

        raw_values = {col: _json_safe(row[col]) for col in TEST_COLUMNS}

        patients.append({
            "trustii_id": raw_values["trustii_id"],
            "summary": {
                "age_at_baseline": raw_values["Age_v1"],
                "visit_count": visit_count,
                "baseline_fib4": baseline_fib4,
                "most_complete_nit": NIT_LABELS[most_complete_base],
                "most_complete_nit_visits": nit_counts[most_complete_base],
            },
            "raw": raw_values,
        })

    return {"columns": TEST_COLUMNS, "patients": patients}


# --------------------------------------------------------------------
# "Rankings" tab support (training-cohort scope). A read-only view of
# where every training patient falls under each of the three scoring
# methods (ml/fib4/apri) -- precomputed once at startup since none of
# it changes while the server is running. The "My active patients"
# scope has no backend counterpart at all: it's scored per patient via
# /predict and kept only in the browser (see script.js).
# --------------------------------------------------------------------
RAW_TRAIN_PATH = REPO_ROOT / "liverrisk" / "data" / "raw" / "train.csv"
_raw_train_df = pd.read_csv(RAW_TRAIN_PATH)

# NOTE on patient_id: train.csv has no trustii_id column at all --
# trustii_id only identifies rows in test.csv, the unlabeled
# competition set. Training patients are identified by
# patient_id_anon instead, which IS a column in train.csv.
#
# X_HEP.index is a *label* into _raw_train_df's rows, so this uses
# .loc rather than .iloc -- that stays correct even though, today,
# X_HEP.index happens to equal range(len(_raw_train_df)) (the hepatic
# endpoint doesn't drop any training rows; only the death endpoint
# does -- see the TRAIN_WEIGHTED_SCORES comment above -- so this join
# is currently positional in practice, but written to not depend on
# that happening to be true).
_train_patient_ids = _raw_train_df.loc[X_HEP.index, "patient_id_anon"]
_train_visit_counts = _raw_train_df.loc[X_HEP.index, AGE_COLUMNS].notna().sum(axis=1)


def build_training_ranking_rows(method: str) -> list[dict]:
    """
    Builds one training-cohort ranking table (all ~1253 X_HEP
    patients), sorted descending by `method`'s score. score=None
    patients -- missing FIB-4/APRI labs, or, for "ml", missing from the
    death-endpoint cohort the weighted score depends on (see the
    TRAIN_WEIGHTED_SCORES comment above) -- are included, not dropped,
    and always sort last regardless of direction.
    """
    if method == "ml":
        # TRAIN_WEIGHTED_SCORES only has one entry per death-cohort
        # patient (984 of X_HEP's ~1253) -- reindexing by X_DEATH.index
        # here, then looking values up per X_HEP patient below via
        # .get(), makes the ~269 patients outside that subset come back
        # as score=None instead of silently misaligning or crashing.
        scores_by_patient = pd.Series(TRAIN_WEIGHTED_SCORES, index=X_DEATH.index)
        reference_scores = TRAIN_WEIGHTED_SCORES
    elif method == "fib4":
        scores_by_patient = _train_fib4_apri["fib4_score"]
        reference_scores = TRAIN_FIB4_SCORES
    elif method == "apri":
        scores_by_patient = _train_fib4_apri["apri_score"]
        reference_scores = TRAIN_APRI_SCORES
    else:
        raise ValueError(f"Unknown ranking method: {method}")

    rows = []
    for position, patient_index in enumerate(X_HEP.index):
        raw_score = scores_by_patient.get(patient_index)  # None if patient_index isn't in this Series at all
        score = None if pd.isna(raw_score) else round(float(raw_score), 4)
        percentile = None if score is None else percentile_le(score, reference_scores)

        had_event = bool(HEP_EVENT[position])
        outcome = f"Event, {HEP_TIME[position]:.1f} yr" if had_event else None

        rows.append({
            "patient_id": _train_patient_ids.loc[patient_index],
            "score": score,
            "percentile": percentile,
            "age_at_baseline": _json_safe(_raw_train_df.loc[patient_index, "Age_v1"]),
            "visit_count": int(_train_visit_counts.loc[patient_index]),
            "outcome": outcome,
        })

    # Descending by score, with score=None patients always last: sort
    # by (is_none, -score), so the None group (True) sorts after the
    # real-score group (False) no matter what, and within the real-score
    # group ascending-by-negated-score is the same as descending-by-score.
    rows.sort(key=lambda r: (r["score"] is None, -r["score"] if r["score"] is not None else 0.0))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    return rows


TRAINING_RANKINGS = {
    "ml": build_training_ranking_rows("ml"),
    "fib4": build_training_ranking_rows("fib4"),
    "apri": build_training_ranking_rows("apri"),
}


@app.get("/rankings")
async def rankings(method: str = "ml", scope: str = "training"):
    """
    Training-cohort ranking table for one scoring method. `scope` is
    accepted (not just ignored) so the URL is self-descriptive, but
    "training" is the only scope this endpoint actually serves --
    "my active patients" lives entirely in the browser, see script.js.
    """
    if method not in TRAINING_RANKINGS:
        raise HTTPException(status_code=400, detail="method must be one of: ml, fib4, apri.")
    if scope != "training":
        raise HTTPException(status_code=400, detail="This endpoint only serves scope=training.")

    return {"method": method, "scope": scope, "rows": TRAINING_RANKINGS[method]}


# --------------------------------------------------------------------
# Serve the plain HTML/CSS/JS frontend from the same server, so the
# page can call /predict on its own origin with a plain fetch() call.
# This must be mounted last -- routes defined above (like /predict)
# are matched first, and anything else falls through to these static
# files.
# --------------------------------------------------------------------
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
