"""
LiverRisk webapp backend -- GET /rankings route.

"Rankings" tab support (training-cohort scope). A read-only view of
where every training patient falls under each of the three scoring
methods (ml/fib4/apri) -- precomputed once at startup since none of
it changes while the server is running. The "My active patients"
scope has no backend counterpart at all: it's scored per patient via
/predict and kept only in the browser (see rankings.js).
"""
from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException

from models_loader import (
    AGE_COLUMNS,
    HEP_EVENT,
    HEP_TIME,
    REPO_ROOT,
    TRAIN_APRI_SCORES,
    TRAIN_FIB4_SCORES,
    TRAIN_WEIGHTED_SCORES,
    X_DEATH,
    X_HEP,
    _json_safe,
    _train_fib4_apri,
    percentile_le,
)

router = APIRouter()

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
# does -- see the TRAIN_WEIGHTED_SCORES comment in models_loader.py --
# so this join is currently positional in practice, but written to not
# depend on that happening to be true).
_train_patient_ids = _raw_train_df.loc[X_HEP.index, "patient_id_anon"]
_train_visit_counts = _raw_train_df.loc[X_HEP.index, AGE_COLUMNS].notna().sum(axis=1)


def build_training_ranking_rows(method: str) -> list[dict]:
    """
    Builds one training-cohort ranking table (all ~1253 X_HEP
    patients), sorted descending by `method`'s score. score=None
    patients -- missing FIB-4/APRI labs, or, for "ml", missing from the
    death-endpoint cohort the weighted score depends on (see the
    TRAIN_WEIGHTED_SCORES comment in models_loader.py) -- are included,
    not dropped, and always sort last regardless of direction.
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


@router.get("/rankings")
async def rankings(method: str = "ml", scope: str = "training"):
    """
    Training-cohort ranking table for one scoring method. `scope` is
    accepted (not just ignored) so the URL is self-descriptive, but
    "training" is the only scope this endpoint actually serves --
    "my active patients" lives entirely in the browser, see rankings.js.
    """
    if method not in TRAINING_RANKINGS:
        raise HTTPException(status_code=400, detail="method must be one of: ml, fib4, apri.")
    if scope != "training":
        raise HTTPException(status_code=400, detail="This endpoint only serves scope=training.")

    return {"method": method, "scope": scope, "rows": TRAINING_RANKINGS[method]}
