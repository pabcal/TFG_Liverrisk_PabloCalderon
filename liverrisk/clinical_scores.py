"""
Standalone clinical risk-score formulas.

fib4() is extracted from the FIB-4 calculations that used to be inline
inside build_patient_features() in the original notebook
(ANNITIA_baseline_local.ipynb) -- same formula, same inputs, now defined
once here instead of being repeated per visit and for the first/last
summaries. features.py calls this function; the numeric result is
unchanged.

apri() is NEW: it was not present anywhere in the original notebook. It's
added only because the target package structure asked for a
clinical_scores.py with fib4 + apri -- build_patient_features() has NOT
been changed to use it, so model behavior is unaffected. Flagging this so
it isn't mistaken for moved logic. APRI requires an AST upper-limit-of-
normal (ULN), which isn't in the dataset; the default of 40 U/L is a
common lab convention, not a value derived from this cohort.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_div(a, b):
    """Local copy of features.safe_div to avoid a circular import (features.py imports fib4 from here)."""
    out = a / b
    if isinstance(out, pd.Series):
        out = out.replace([np.inf, -np.inf], np.nan)
    else:
        #np.where(condition, value_if_true, value_if_false). I did it this way just in case
        #out is a numpy array.
        out = np.where(np.isfinite(out), out, np.nan) 
    return out


def fib4(age, ast, plt, alt):
    """
    FIB-4 = (age * AST) / (platelets * sqrt(ALT)).

    Same formula used for fib4_v{1..4}, fib4_first, and fib4_last in
    build_patient_features(). Accepts scalars or array-likes.
    """
    return _safe_div(age * ast, plt * np.sqrt(alt))


def apri(ast, plt, ast_uln: float = 40.0):
    """
    APRI = ((AST / AST_ULN) * 100) / platelets.

    NOT used anywhere in the current feature set (see module docstring).
    `ast_uln` defaults to 40 U/L, a common lab convention -- not a value
    derived from this cohort's data.
    """
    return _safe_div(_safe_div(ast, ast_uln) * 100.0, plt)


def compute_fib4_apri(X: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Study 1 (formula-vs-ML comparison): scores every row of an already
    built patient-feature frame (X_hep or X_death, from
    build_patient_features()/load_features()) with FIB-4 and APRI, using
    the same "last observed" labs build_patient_features() already
    computes fib4_last from: age_last_observed, ast__last, alt__last,
    plt__last.

    `name` is unused by the computation itself -- kept only so call sites
    read as `compute_fib4_apri(X_hep, "hep")` / `compute_fib4_apri(X_death,
    "death")`, matching load_features's `name` argument, for symmetry with
    the rest of the pipeline's per-endpoint calls.
    """

    #Given a patient table, first I make sure it has the four lab columns needed to calculate 
    #FIB4 and APRI. If not, return an error. If everything is fine, compute formulas for every 
    #patient at once,and return a 2 column lined up patient for patient with the results from the
    # 2 formulas. Important to remember that if patient WNNHJ7XVG0FN is row label 47, he will also be
    #in row table 47 in the new returned table, despite not having the trust id as a column.

    del name

    required = {"age_last_observed", "ast__last", "alt__last", "plt__last"}
    missing = required - set(X.columns)
    if missing:
        raise ValueError(f"compute_fib4_apri: X is missing required columns {sorted(missing)}")

    return pd.DataFrame({
        "fib4_score": fib4(X["age_last_observed"], X["ast__last"], X["plt__last"], X["alt__last"]),
        "apri_score": apri(X["ast__last"], X["plt__last"]),
    }, index=X.index)
