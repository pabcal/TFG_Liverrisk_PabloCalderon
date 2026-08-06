"""
Longitudinal feature engineering and survival-target preparation.

Moved verbatim (not rewritten) from ANNITIA_baseline_local.ipynb sections
4.3-4.9: get_age_cols, visit_cols, safe_div, first_non_null, last_non_null,
slope_against_age, age_of_last_measure, span_years_for_measure,
add_visit_level_features, build_patient_features, prepare_survival_target.

The only intentional behavior-preserving change: the three inline FIB-4
computations (per-visit, first, last) now call clinical_scores.fib4()
instead of repeating `safe_div(age * ast, plt * np.sqrt(alt))` -- same
formula, same result, single definition.

save_features()/load_features() and the feature_columns.json writer are
NEW, added for the notebook -> package split (01_data_exploration writes,
everything downstream reads).
"""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

from sksurv.util import Surv

from liverrisk.clinical_scores import fib4

# ---------------------------------------------------------------------
# Column families from the real train.csv
# ---------------------------------------------------------------------
STATIC_CAT = ["gender", "T2DM", "Hypertension", "Dyslipidaemia", "bariatric_surgery"]
STATIC_NUM = ["bariatric_surgery_age"]

REPEATED_BASES = [
    "BMI",
    "alt",
    "ast",
    "bilirubin",
    "chol",
    "ggt",
    "gluc_fast",
    "plt",
    "triglyc",
    "fibrotest_BM_2",
    "aixp_aix_result_BM_3",
    "fibs_stiffness_med_BM_1",
]

TARGET_COLS = [
    "evenements_hepatiques_majeurs",
    "evenements_hepatiques_age_occur",
    "death",
    "death_age_occur",
]
ID_COLS = ["patient_id_anon", "trustii_id"]


# ---------------------------------------------------------------------
# Visit column helpers
# ---------------------------------------------------------------------
def get_age_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if re.fullmatch(r"Age_v\d+", c)]
    return sorted(cols, key=lambda c: int(c.rsplit("_v", 1)[1]))


def visit_cols(df: pd.DataFrame, base: str) -> list[str]:
    cols = [c for c in df.columns if re.fullmatch(fr"{re.escape(base)}_v\d+", c)]
    return sorted(cols, key=lambda c: int(c.rsplit("_v", 1)[1]))


# ---------------------------------------------------------------------
# Basic longitudinal helpers
# ---------------------------------------------------------------------
def safe_div(a, b):
    out = a / b
    if isinstance(out, pd.Series):
        out = out.replace([np.inf, -np.inf], np.nan)
    else:
        out = np.where(np.isfinite(out), out, np.nan)
    return out


def first_non_null(block: pd.DataFrame) -> pd.Series:
    return block.bfill(axis=1).iloc[:, 0]


def last_non_null(block: pd.DataFrame) -> pd.Series:
    return block.ffill(axis=1).iloc[:, -1]


def slope_against_age(block: pd.DataFrame, age_block: pd.DataFrame) -> pd.Series:
    """
    Row-wise OLS slope: value ~ age.
    Returns NaN if fewer than 2 aligned observations.
    """
    slopes = np.full(len(block), np.nan, dtype=float)

    for pos, idx in enumerate(block.index):
        y = block.loc[idx].to_numpy(dtype=float)
        x = age_block.loc[idx, age_block.columns[: len(block.columns)]].to_numpy(dtype=float)

        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 2:
            continue

        xv = x[valid]
        yv = y[valid]
        if np.allclose(xv, xv[0]):
            continue

        xc = xv - xv.mean()
        slopes[pos] = np.sum(xc * (yv - yv.mean())) / np.sum(xc ** 2)

    return pd.Series(slopes, index=block.index)


# ---------------------------------------------------------------------
# Visit timing helpers
# ---------------------------------------------------------------------
def age_of_last_measure(block: pd.DataFrame, age_block: pd.DataFrame) -> pd.Series:
    out = np.full(len(block), np.nan, dtype=float)

    for pos, idx in enumerate(block.index):
        row = block.loc[idx]
        valid_cols = row.index[row.notna()]
        if len(valid_cols) == 0:
            continue
        last_col = valid_cols[-1]
        visit_num = int(last_col.rsplit("_v", 1)[1])
        age_col = f"Age_v{visit_num}"
        if age_col in age_block.columns:
            out[pos] = age_block.loc[idx, age_col]

    return pd.Series(out, index=block.index)


def span_years_for_measure(block: pd.DataFrame, age_block: pd.DataFrame) -> pd.Series:
    out = np.full(len(block), np.nan, dtype=float)

    for pos, idx in enumerate(block.index):
        row = block.loc[idx]
        valid_cols = row.index[row.notna()]
        if len(valid_cols) == 0:
            continue

        first_visit = int(valid_cols[0].rsplit("_v", 1)[1])
        last_visit = int(valid_cols[-1].rsplit("_v", 1)[1])

        first_age_col = f"Age_v{first_visit}"
        last_age_col = f"Age_v{last_visit}"

        if first_age_col in age_block.columns and last_age_col in age_block.columns:
            out[pos] = age_block.loc[idx, last_age_col] - age_block.loc[idx, first_age_col]

    return pd.Series(out, index=block.index)


# ---------------------------------------------------------------------
# Early visit features
# ---------------------------------------------------------------------
def add_visit_level_features(X: pd.DataFrame, df: pd.DataFrame, max_visit: int = 4) -> None:
    """
    Preserve raw dense early visits and add ratios/FIB-4 per early visit.
    Mutates X in place.
    """
    dense_bases = [
        "BMI", "alt", "ast", "bilirubin", "chol", "ggt", "gluc_fast",
        "plt", "triglyc", "fibrotest_BM_2", "aixp_aix_result_BM_3",
        "fibs_stiffness_med_BM_1",
    ]

    for v in range(1, max_visit + 1):
        age_col = f"Age_v{v}"

        for base in dense_bases:
            col = f"{base}_v{v}"
            if col in df.columns:
                X[col] = df[col]

        alt = df[f"alt_v{v}"] if f"alt_v{v}" in df.columns else np.nan
        ast = df[f"ast_v{v}"] if f"ast_v{v}" in df.columns else np.nan
        plt = df[f"plt_v{v}"] if f"plt_v{v}" in df.columns else np.nan
        ggt = df[f"ggt_v{v}"] if f"ggt_v{v}" in df.columns else np.nan
        age = df[age_col] if age_col in df.columns else np.nan

        if f"alt_v{v}" in df.columns and f"ast_v{v}" in df.columns:
            X[f"ast_alt_ratio_v{v}"] = safe_div(ast, alt)
        if f"ggt_v{v}" in df.columns and f"alt_v{v}" in df.columns:
            X[f"ggt_alt_ratio_v{v}"] = safe_div(ggt, alt)
        if f"ast_v{v}" in df.columns and f"plt_v{v}" in df.columns:
            X[f"ast_platelet_ratio_v{v}"] = safe_div(ast, plt)
        if all(c in df.columns for c in [age_col, f"ast_v{v}", f"alt_v{v}", f"plt_v{v}"]):
            X[f"fib4_v{v}"] = fib4(age, ast, plt, alt)


# ---------------------------------------------------------------------
# Patient feature engineering
# ---------------------------------------------------------------------
def build_patient_features(df: pd.DataFrame, include_followup_observation_features: bool = True) -> pd.DataFrame:
    """
    Converts wide repeated-visit columns into dense patient-level features.

    include_followup_observation_features=True is leaderboard-friendly:
    it includes age_last_observed/followup/n_visits. This can be informative
    but may capture observation-process information.
    """
    X = pd.DataFrame(index=df.index)

    age_cols = get_age_cols(df)
    if len(age_cols) == 0:
        raise ValueError("No Age_v* columns found.")

    age_block = df[age_cols]

    # Static variables
    for col in STATIC_CAT + STATIC_NUM:
        if col in df.columns:
            X[col] = df[col]

    # Global age/follow-up features
    X["age_baseline"] = df["Age_v1"]

    if include_followup_observation_features:
        X["age_last_observed"] = age_block.max(axis=1)
        X["n_visits_age"] = age_block.notna().sum(axis=1)
        X["followup_years_observed"] = age_block.max(axis=1) - age_block.min(axis=1)

        if "bariatric_surgery_age" in df.columns:
            X["years_since_bariatric_at_baseline"] = df["Age_v1"] - df["bariatric_surgery_age"]
            X["years_since_bariatric_at_last"] = X["age_last_observed"] - df["bariatric_surgery_age"]

    # Longitudinal summaries
    for base in REPEATED_BASES:
        cols = visit_cols(df, base)
        if not cols:
            continue

        block = df[cols]

        X[f"{base}__first"] = first_non_null(block)
        X[f"{base}__last"] = last_non_null(block)
        X[f"{base}__min"] = block.min(axis=1)
        X[f"{base}__max"] = block.max(axis=1)
        X[f"{base}__mean"] = block.mean(axis=1)
        X[f"{base}__median"] = block.median(axis=1)
        X[f"{base}__std"] = block.std(axis=1)
        X[f"{base}__count"] = block.notna().sum(axis=1)
        X[f"{base}__miss_frac"] = block.isna().mean(axis=1)

        X[f"{base}__delta"] = X[f"{base}__last"] - X[f"{base}__first"]
        X[f"{base}__rel_change"] = safe_div(X[f"{base}__last"], X[f"{base}__first"]) - 1.0
        X[f"{base}__slope_per_year"] = slope_against_age(block, age_block)

        if include_followup_observation_features:
            X[f"{base}__age_last_measure"] = age_of_last_measure(block, age_block)
            X[f"{base}__span_years"] = span_years_for_measure(block, age_block)

    # Clinically meaningful derived features
    required = {"ast__first", "alt__first", "ast__last", "alt__last", "ggt__last", "plt__last"}
    if required.issubset(X.columns):
        X["ast_alt_ratio_first"] = safe_div(X["ast__first"], X["alt__first"])
        X["ast_alt_ratio_last"] = safe_div(X["ast__last"], X["alt__last"])
        X["ggt_alt_ratio_last"] = safe_div(X["ggt__last"], X["alt__last"])
        X["ast_platelet_ratio_last"] = safe_div(X["ast__last"], X["plt__last"])

    if {"age_baseline", "ast__first", "plt__first", "alt__first"}.issubset(X.columns):
        X["fib4_first"] = fib4(X["age_baseline"], X["ast__first"], X["plt__first"], X["alt__first"])

    age_for_last = X["age_last_observed"] if "age_last_observed" in X.columns else X["age_baseline"]
    if {"ast__last", "plt__last", "alt__last"}.issubset(X.columns):
        X["fib4_last"] = fib4(age_for_last, X["ast__last"], X["plt__last"], X["alt__last"])

    # NIT burden / availability
    nit_last_cols = [
        c for c in [
            "fibrotest_BM_2__last",
            "aixp_aix_result_BM_3__last",
            "fibs_stiffness_med_BM_1__last",
        ]
        if c in X.columns
    ]
    if nit_last_cols:
        X["nit_available_count_last"] = X[nit_last_cols].notna().sum(axis=1)
        X["nit_last_rank_mean"] = X[nit_last_cols].rank(pct=True).mean(axis=1)

    # Metabolic interactions
    if "T2DM" in X.columns and "BMI__last" in X.columns:
        X["T2DM_x_BMI_last"] = X["T2DM"] * X["BMI__last"]
    if "T2DM" in X.columns and "gluc_fast__last" in X.columns:
        X["T2DM_x_glucose_last"] = X["T2DM"] * X["gluc_fast__last"]
    if "BMI__last" in X.columns and "triglyc__last" in X.columns:
        X["BMI_x_triglyc_last"] = X["BMI__last"] * X["triglyc__last"]

    add_visit_level_features(X, df, max_visit=4)

    # Clean infinities
    X = X.replace([np.inf, -np.inf], np.nan)

    # Keep only columns not completely empty
    X = X.dropna(axis=1, how="all")

    return X


# ---------------------------------------------------------------------
# Survival targets
# ---------------------------------------------------------------------
def prepare_survival_target(df: pd.DataFrame, outcome: str):
    df = df.copy()
    age_cols = get_age_cols(df)
    df["last_observed_age"] = df[age_cols].max(axis=1)

    if outcome == "hepatic":
        event_col = "evenements_hepatiques_majeurs"
        age_col = "evenements_hepatiques_age_occur"
        event_name = "Hepatic_event"

        invalid = (df[event_col] == 1) & df[age_col].isna()
        mask = ~invalid

    elif outcome == "death":
        event_col = "death"
        age_col = "death_age_occur"
        event_name = "Death"

        unknown = df[event_col].isna()
        invalid = (df[event_col] == 1) & df[age_col].isna()
        mask = ~unknown & ~invalid

    else:
        raise ValueError("outcome must be 'hepatic' or 'death'.")

    d = df.loc[mask].copy()
    event = (d[event_col] == 1).astype(bool).to_numpy()

    time = np.where(
        event,
        d[age_col].to_numpy(dtype=float) - d["Age_v1"].to_numpy(dtype=float),
        d["last_observed_age"].to_numpy(dtype=float) - d["Age_v1"].to_numpy(dtype=float),
    )
    time = np.maximum(time.astype(float), 1e-3)

    y = Surv.from_arrays(event=event, time=time, name_event=event_name, name_time="Time_years")
    return d, y, event, time


# ---------------------------------------------------------------------
# Persistence: data/processed/ read + write (NEW)
# ---------------------------------------------------------------------
def save_features(X: pd.DataFrame, y, event: np.ndarray, time: np.ndarray, name: str, path: str | Path) -> None:
    """
    Save one endpoint's engineered features + survival target to
    `path/{name}_X.parquet` and `path/{name}_y.pkl`, and write
    `path/feature_columns.json` with X.columns (in order) -- overwritten
    each call, so it should reflect whichever call ran last (hepatic and
    death share the same X_all column space in the current pipeline).
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    X.to_parquet(path / f"{name}_X.parquet")

    with open(path / f"{name}_y.pkl", "wb") as f:
        pickle.dump({"y": y, "event": event, "time": time}, f)

    with open(path / "feature_columns.json", "w") as f:
        json.dump(list(X.columns), f, indent=2)


def load_features(name: str, path: str | Path):
    """Inverse of save_features(). Returns (X, y, event, time)."""
    path = Path(path)

    X = pd.read_parquet(path / f"{name}_X.parquet")

    with open(path / f"{name}_y.pkl", "rb") as f:
        payload = pickle.load(f)

    return X, payload["y"], payload["event"], payload["time"]


def load_feature_columns(path: str | Path) -> list[str]:
    path = Path(path)
    with open(path / "feature_columns.json") as f:
        return json.load(f)


def save_test_features(X_test: pd.DataFrame, path: str | Path) -> None:
    """Save the (unlabeled) test-set features to `path/test_X.parquet`."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    X_test.to_parquet(path / "test_X.parquet")


def load_test_features(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_parquet(path / "test_X.parquet")
