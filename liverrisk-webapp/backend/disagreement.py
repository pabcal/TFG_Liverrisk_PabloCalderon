"""
LiverRisk webapp backend -- GET /disagreement route.

"Disagreement" scope of the Rankings tab: for one formula (FIB-4 or
APRI), shows training-cohort patients where the ML model's ranking and
the formula's ranking disagree most sharply -- largest |rank_gap|
first. Backed by outputs/ranking_comparison_fib4.csv and
outputs/ranking_comparison_apri.csv, produced by
notebooks/05_study1_formula_comparison.ipynb, which also documents
ml_rank/formula_rank/rank_gap's exact definitions and the
`method="min"` tie-breaking rule reused below.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

from models_loader import REPO_ROOT

router = APIRouter()

OUTPUTS_DIR = REPO_ROOT / "outputs"


def _load_disagreement_csv(path: Path) -> pd.DataFrame:
    """
    Loads one ranking_comparison_*.csv, restricted to what this route
    actually needs:

    - Only the "hepatic" endpoint rows. FIB-4/APRI are hepatic-fibrosis
      scores, and the source CSV also has a "death" endpoint subset
      (a different ml_score/ml_rank per patient) -- mixing both in would
      make the same patient_id appear twice with different numbers, and
      this table has no endpoint column to disambiguate that.
    - Rows with a non-null rank_gap only. A null rank_gap means that
      patient was missing the labs the formula needs, so there's
      nothing to compare them on.

    Sorted by |rank_gap| descending once here, since every unfiltered
    (no new_* params) response returns rows in this order.
    """
    if not path.exists():
        raise RuntimeError(
            f"{path} is missing. Regenerate it by running "
            "notebooks/05_study1_formula_comparison.ipynb."
        )

    df = pd.read_csv(path)
    df = df[df["endpoint"] == "hepatic"].dropna(subset=["rank_gap"])
    df = df.sort_values("rank_gap", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    return df


DISAGREEMENT_DATA = {
    "fib4": _load_disagreement_csv(OUTPUTS_DIR / "ranking_comparison_fib4.csv"),
    "apri": _load_disagreement_csv(OUTPUTS_DIR / "ranking_comparison_apri.csv"),
}


def _insert_and_rank(new_score: float, train_scores: pd.Series) -> int:
    """
    Where would new_score rank (1 = highest score = most urgent) if
    inserted into train_scores? Same insert-and-rank idea as
    models_loader.percentile_within_training(), adapted to return a
    1-based descending rank instead of a percentile, and using the same
    `method="min"` tie-break the training ml_rank/formula_rank columns
    were built with (see notebooks/05_study1_formula_comparison.ipynb),
    so the new patient's rank is directly comparable to theirs.
    """
    combined = pd.concat([train_scores, pd.Series([new_score])], ignore_index=True)
    ranks_descending = combined.rank(method="min", ascending=False)
    return int(ranks_descending.iloc[-1])


def _row_dict(row: pd.Series) -> dict:
    return {
        "patient_id": row["patient_id"],
        "ml_rank": int(row["ml_rank"]),
        "formula_rank": int(row["formula_rank"]),
        "rank_gap": float(row["rank_gap"]),
        "event": bool(row["event"]),
        "time": float(row["time"]),
        "is_new_patient": False,
    }


@router.get("/disagreement")
async def disagreement(
    formula: str,
    new_ml_score: float | None = None,
    new_formula_score: float | None = None,
):
    """
    Training-cohort ML-vs-formula disagreement table for one formula.

    With no new_* params: just the static training rows, already sorted
    by |rank_gap| descending. With both new_ml_score and
    new_formula_score given, inserts one extra row ("This patient") at
    its correct sorted position among them.
    """
    if formula not in DISAGREEMENT_DATA:
        raise HTTPException(status_code=400, detail="formula must be one of: fib4, apri.")

    if (new_ml_score is None) != (new_formula_score is None):
        raise HTTPException(
            status_code=400,
            detail="new_ml_score and new_formula_score must be provided together, or not at all.",
        )

    df = DISAGREEMENT_DATA[formula]
    rows = [_row_dict(row) for _, row in df.iterrows()]

    if new_ml_score is None:
        return {"formula": formula, "rows": rows}

    # This compares the new patient's PRODUCTION score (from /predict,
    # scored by models trained on the full training cohort) against the
    # training cohort's OUT-OF-FOLD scores in df (each patient scored by
    # a model that never saw them during that fold). A close
    # approximation, not a perfect match -- production and OOF models
    # saw slightly different amounts of training data.
    new_ml_rank = _insert_and_rank(new_ml_score, df["ml_score"])
    new_formula_rank = _insert_and_rank(new_formula_score, df["formula_score"])
    new_rank_gap = new_formula_rank - new_ml_rank

    new_row = {
        "patient_id": "This patient",
        "ml_rank": new_ml_rank,
        "formula_rank": new_formula_rank,
        "rank_gap": float(new_rank_gap),
        "event": None,
        "time": None,
        "is_new_patient": True,
    }

    # Insert at the correct sorted position by |rank_gap| descending,
    # rather than appending at the end.
    insert_at = len(rows)
    for i, existing_row in enumerate(rows):
        if abs(new_rank_gap) >= abs(existing_row["rank_gap"]):
            insert_at = i
            break
    rows.insert(insert_at, new_row)

    return {"formula": formula, "rows": rows}
