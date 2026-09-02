"""
LiverRisk webapp backend -- GET /sample-patients route.

"Build a test CSV" tab support: lets a user assemble a small CSV out
of a handful of real (unlabeled) test-set patients, without needing
their own data, to try out /predict.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from fastapi import APIRouter

# models_loader must be imported before any liverrisk.* import below --
# it's the one that puts the repo root (where the liverrisk package
# lives) onto sys.path.
from models_loader import AGE_COLUMNS, RAW_TEST_PATH, _json_safe

from liverrisk.clinical_scores import fib4

router = APIRouter()

TEST_COLUMNS = pd.read_csv(RAW_TEST_PATH, nrows=0).columns.tolist()

# --------------------------------------------------------------------
# The three "NIT" (non-invasive test) repeated measures shown in each
# sample patient's summary, in the same order they appear as
# REPEATED_BASES entries in features.py.
# --------------------------------------------------------------------
NIT_BASES = ["fibrotest_BM_2", "aixp_aix_result_BM_3", "fibs_stiffness_med_BM_1"]
NIT_LABELS = {
    "fibrotest_BM_2": "FibroTest",
    "aixp_aix_result_BM_3": "AIx-P",
    "fibs_stiffness_med_BM_1": "Liver stiffness (FibroScan)",
}


def _visit_columns(columns: list[str], base: str) -> list[str]:
    """All `{base}_v<n>` columns for one repeated measure, e.g. all fibrotest_BM_2_v* columns."""
    return [c for c in columns if re.fullmatch(fr"{re.escape(base)}_v\d+", c)]


NIT_COLUMNS = {base: _visit_columns(TEST_COLUMNS, base) for base in NIT_BASES}


# --------------------------------------------------------------------
# Picking a varied set of sample patients
# --------------------------------------------------------------------
def _pick_evenly_spaced(sorted_index: pd.Index, n: int) -> list:
    """
    Given an index already sorted by the ranking criterion, picks `n`
    positions evenly spaced across it (by rank) rather than just the
    first n, so the picks span the low-to-high range of that ordering
    instead of clustering at one end. Rounding can collapse two target
    spots onto the same row for a small n, so this tops up with the
    next not-yet-picked rows to still return min(n, len(sorted_index))
    picks.
    """
    total = len(sorted_index)
    n = min(n, total)
    if n <= 0:
        return []

    pick_positions = np.unique(np.linspace(0, total - 1, num=n).round().astype(int))
    pick_positions = set(pick_positions.tolist())
    for pos in range(total):
        if len(pick_positions) >= n:
            break
        pick_positions.add(pos)

    return sorted_index[sorted(pick_positions)[:n]].tolist()


def select_sample_patients(raw_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Picks `n` patients out of the full (unlabeled) test set for the
    "Build a test CSV" tab, aiming for variety rather than just the
    first n rows.

    The sample is a fixed mix rather than one evenly-spaced pick across
    everyone: 60% of `n` are patients with a computable baseline FIB-4
    (all of Age_v1/ast_v1/plt_v1/alt_v1 present) and 40% are patients
    missing it (at least one of those columns null). Both groups are
    picked evenly spaced across their own visit-count range. This keeps
    some "missing labs" patients in the sample on purpose, so the
    FIB-4/APRI "Not available (missing lab values)" UI path stays
    testable with real sample data -- an all-evenly-spaced-by-rank pick
    could otherwise end up with 10/10 patients that all have FIB-4.
    """
    visit_counts = raw_df[AGE_COLUMNS].notna().sum(axis=1)
    baseline_fib4 = fib4(raw_df["Age_v1"], raw_df["ast_v1"], raw_df["plt_v1"], raw_df["alt_v1"])

    ranking = pd.DataFrame({"visits": visit_counts, "fib4": baseline_fib4})
    has_fib4 = ranking[ranking["fib4"].notna()].sort_values("visits")
    missing_fib4 = ranking[ranking["fib4"].isna()].sort_values("visits")

    n_has = round(n * 0.6)
    n_missing = n - n_has

    has_picks = _pick_evenly_spaced(has_fib4.index, n_has)
    missing_picks = _pick_evenly_spaced(missing_fib4.index, n_missing)

    # If one group doesn't have enough candidates to fill its share, top
    # up from the other group's leftovers rather than silently returning
    # fewer than n patients.
    shortfall = (n_has - len(has_picks)) + (n_missing - len(missing_picks))
    if shortfall > 0:
        print(
            f"WARNING: select_sample_patients() wanted {n_has} has-FIB-4 / "
            f"{n_missing} missing-FIB-4 patients, but only found "
            f"{len(has_fib4)} / {len(missing_fib4)} candidates respectively. "
            f"Topping up {shortfall} pick(s) from the other group."
        )
        already_picked = set(has_picks) | set(missing_picks)
        if len(has_picks) < n_has:
            leftover = [idx for idx in missing_fib4.index if idx not in already_picked]
            top_up = leftover[: n_has - len(has_picks)]
            has_picks += top_up
            already_picked.update(top_up)
        if len(missing_picks) < n_missing:
            leftover = [idx for idx in has_fib4.index if idx not in already_picked]
            missing_picks += leftover[: n_missing - len(missing_picks)]

    chosen_idx = has_picks + missing_picks
    return raw_df.loc[chosen_idx]


# --------------------------------------------------------------------
# The one API endpoint
# --------------------------------------------------------------------
@router.get("/sample-patients")
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
