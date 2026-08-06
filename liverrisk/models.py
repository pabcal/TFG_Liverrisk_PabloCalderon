"""
Model pipeline factories, moved verbatim from ANNITIA_baseline_local.ipynb
sections 4.10-4.13: split_feature_types, make_preprocessor,
make_coxnet_pipeline, fit_coxnet_with_alpha_cv, make_rsf_pipeline,
signed_time_label, make_xgb_pipeline.

Behavior-preserving change: make_xgb_pipeline's XGBRegressor kwargs and
fit_coxnet_with_alpha_cv's (n_alphas, n_splits) now come from
liverrisk.config by default instead of being hardcoded literals. Passing
them explicitly (as 02_grid_search.ipynb must, per the requirement that
grid search never reads config mid-search) reproduces the exact original
defaults, since config falls back to those same literals when
best_config.json doesn't exist yet -- see config.DEFAULTS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

from liverrisk import config
from liverrisk.features import STATIC_CAT

RANDOM_STATE = 27


# ---------------------------------------------------------------------
# Feature preprocessing
# ---------------------------------------------------------------------
def split_feature_types(X: pd.DataFrame):
    cat_cols = [c for c in STATIC_CAT if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return num_cols, cat_cols


def make_preprocessor(X: pd.DataFrame, sparse_threshold: float = 0.3):
    num_cols, cat_cols = split_feature_types(X)

    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median", add_indicator=True)),
        ("sc", StandardScaler(with_mean=False)),
    ])

    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value=-1)),
        ("oh", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=sparse_threshold,
    )


# ---------------------------------------------------------------------
# Coxnet
# ---------------------------------------------------------------------
def make_coxnet_pipeline(X: pd.DataFrame) -> Pipeline:
    return Pipeline([
        ("pre", make_preprocessor(X, sparse_threshold=0.3)),
        ("cox", CoxnetSurvivalAnalysis(
            l1_ratio=0.9,
            alpha_min_ratio=0.01,
            max_iter=20000,
            fit_baseline_model=True,
        )),
    ])


def fit_coxnet_with_alpha_cv(X: pd.DataFrame, y, random_state: int = RANDOM_STATE,
                              n_alphas: int | None = None, n_splits: int | None = None) -> Pipeline:
    """
    Like the baseline, first fit an alpha path, then select the best alpha.
    This usually works better than leaving the full path inside predict().

    Speed: the full alpha path from Coxnet can have ~100 alphas. Grid-searching
    all of them re-fits the (alpha-independent) preprocessing pipeline every
    time, which dominates runtime. We instead:
      1. Fit the preprocessor once and reuse the transformed features across
         all alpha/fold combinations.
      2. Evenly subsample the alpha path down to `n_alphas` candidates, which
         still spans the same regularization range at much lower cost.

    `n_alphas`/`n_splits` default to config.coxnet_alpha_search() (30/3,
    same as the original hardcoded defaults) if not passed explicitly.
    """
    if n_alphas is None or n_splits is None:
        cfg = config.coxnet_alpha_search()
        n_alphas = cfg["n_alphas"] if n_alphas is None else n_alphas
        n_splits = cfg["n_splits"] if n_splits is None else n_splits

    path_model = make_coxnet_pipeline(X)
    path_model.fit(X, y)
    alphas = path_model.named_steps["cox"].alphas_

    if len(alphas) > n_alphas:
        idx = np.linspace(0, len(alphas) - 1, n_alphas).round().astype(int)
        idx = np.unique(idx)
        alphas = alphas[idx]

    preprocessor = make_preprocessor(X, sparse_threshold=0.3)
    X_t = preprocessor.fit_transform(X)

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    grid = GridSearchCV(
        estimator=CoxnetSurvivalAnalysis(
            l1_ratio=0.9,
            max_iter=20000,
            fit_baseline_model=True,
        ),
        param_grid={"alphas": [[float(a)] for a in alphas]},
        cv=cv,
        error_score=0.5,
        n_jobs=-1,
    )
    grid.fit(X_t, y)

    return Pipeline([
        ("pre", preprocessor),
        ("cox", grid.best_estimator_),
    ])


# ---------------------------------------------------------------------
# Random Survival Forest
# ---------------------------------------------------------------------
def make_rsf_pipeline(X: pd.DataFrame, n_estimators: int = 500) -> Pipeline:
    return Pipeline([
        ("pre", make_preprocessor(X, sparse_threshold=0.3)),
        ("rsf", RandomSurvivalForest(
            n_estimators=n_estimators,
            min_samples_leaf=20,
            min_samples_split=40,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )),
    ])


# ---------------------------------------------------------------------
# XGBoost survival:cox
# ---------------------------------------------------------------------
def signed_time_label(event: np.ndarray, time: np.ndarray) -> np.ndarray:
    """
    XGBoost survival:cox convention:
    positive time = observed event, negative time = right-censored.
    """
    return np.where(event, time, -time)


def make_xgb_pipeline(X: pd.DataFrame, **hyperparam_overrides) -> Pipeline:
    """
    XGB hyperparameters come from config.xgb_hyperparams() by default (same
    literal values as the original notebook's hardcoded XGBRegressor kwargs).

    Pass explicit overrides (e.g. `make_xgb_pipeline(X, max_depth=3)`) to
    try a different hyperparameter set WITHOUT touching best_config.json --
    this is how a future XGB hyperparameter search in 02_grid_search.ipynb
    should explore candidates; only write the winner back via
    config.update_config() once the search is done.
    """
    if not HAS_XGB:
        raise ImportError("xgboost is not installed.")

    params = config.xgb_hyperparams()
    params.update(hyperparam_overrides)

    return Pipeline([
        ("pre", make_preprocessor(X, sparse_threshold=1.0)),
        ("xgb", XGBRegressor(
            objective="survival:cox",
            random_state=RANDOM_STATE,
            tree_method="hist",
            n_jobs=-1,
            **params,
        )),
    ])
