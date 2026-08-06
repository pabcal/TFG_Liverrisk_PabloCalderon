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
        out = np.where(np.isfinite(out), out, np.nan)
    return out


def fib4(age, ast, plt, alt):
    """
    FIB-4 = (age * AST) / (platelets * sqrt(ALT)).

    Same formula used for fib4_v{1..4}, fib4_first, and fib4_last in
    build_patient_features(). Accepts scalars or array-likes (pandas
    Series / numpy arrays broadcast the same way the inline version did).
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
