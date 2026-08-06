"""
Model pipeline factories, moved verbatim from ANNITIA_baseline_local.ipynb
sections 4.10-4.13: split_feature_types, make_preprocessor,
make_coxnet_pipeline, fit_coxnet_with_alpha_cv, make_rsf_pipeline,
signed_time_label, make_xgb_pipeline. Also home to the two hyperparameter
searches 02_grid_search.ipynb runs: search_coxnet_l1_ratio and
search_xgb_hyperparams.

fit_coxnet_with_alpha_cv's (n_alphas, n_splits) come from
liverrisk.config by default instead of being hardcoded literals. Passing
them explicitly (as 02_grid_search.ipynb must, per the requirement that
grid search never reads config mid-search) reproduces the exact original
defaults, since config falls back to those same literals when
best_config.json doesn't exist yet -- see config.DEFAULTS.

make_xgb_pipeline and make_coxnet_pipeline do NOT read config themselves.
XGB hyperparameters and Coxnet l1_ratio are tuned per endpoint
(config.xgb_hyperparams_hep()/_death(), config.coxnet_l1_ratio_hep()/
_death()), and neither factory knows which endpoint it's being called
for -- every caller (cv.py, scripts/train.py, this module's own search
functions) must explicitly pass the right endpoint's values in. This
replaced an earlier version where make_xgb_pipeline silently defaulted to
a single shared config.xgb_hyperparams(), which made it easy to lose
track of which endpoint a given fit actually used.
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
def make_coxnet_pipeline(X: pd.DataFrame, l1_ratio: float = 0.9) -> Pipeline:
    """
    `l1_ratio` defaults to the original hardcoded value (0.9) so every
    existing caller that doesn't pass it explicitly is unaffected.
    search_coxnet_l1_ratio() sweeps this argument; nothing here reads
    config.coxnet_l1_ratio_hep()/_death() automatically -- callers that
    want the tuned value must pass it in themselves.
    """
    return Pipeline([
        ("pre", make_preprocessor(X, sparse_threshold=0.3)),
        ("cox", CoxnetSurvivalAnalysis(
            l1_ratio=l1_ratio,
            alpha_min_ratio=0.01,
            max_iter=20000,
            fit_baseline_model=True,
        )),
    ])


def search_coxnet_l1_ratio(X: pd.DataFrame, y, event: np.ndarray, time: np.ndarray,
                            l1_ratios: list[float] | None = None,
                            n_repeats: int = 1) -> tuple[float, float, pd.DataFrame]:
    """
    Sweeps Coxnet's l1_ratio, scoring each candidate with cv_cindex_sksurv
    (alpha itself is left alone -- fit_coxnet_with_alpha_cv already tunes
    alpha per fit, and re-tuning it inside this search too would multiply
    the cost for no benefit).

    `time` is accepted but unused -- kept only so this search's call
    signature matches search_xgb_hyperparams's (X, y, event, time), which
    does need `time` (for signed_time_label) but not `y`.

    Returns (best_l1_ratio, best_mean_cindex, results_df), sorted
    best-first, same shape as search_blend_weights. Does not write
    anywhere -- the caller (02_grid_search.ipynb) persists the winner via
    config.update_config(coxnet_hyperparams=...).
    """
    del time

    if l1_ratios is None:
        l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]

    from liverrisk.cv import cv_cindex_sksurv  # local import: cv.py imports from this module, avoids a circular import

    rows = []
    for l1_ratio in l1_ratios:
        factory = lambda X_fold, l1_ratio=l1_ratio: make_coxnet_pipeline(X_fold, l1_ratio=l1_ratio)
        mean, std = cv_cindex_sksurv(factory, X, y, event, n_repeats=n_repeats)
        rows.append({"l1_ratio": l1_ratio, "mean": mean, "std": std})

    results_df = pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)
    best = results_df.iloc[0]

    return float(best["l1_ratio"]), float(best["mean"]), results_df


def fit_coxnet_with_alpha_cv(X: pd.DataFrame, y, random_state: int = RANDOM_STATE,
                              n_alphas: int | None = None, n_splits: int | None = None,
                              l1_ratio: float = 0.9) -> Pipeline:
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

    path_model = make_coxnet_pipeline(X, l1_ratio=l1_ratio)
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
            l1_ratio=l1_ratio,
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
    Takes XGBRegressor kwargs only via `hyperparam_overrides` -- there is no
    internal config fallback. Callers that want the tuned per-endpoint
    values must pass them explicitly, e.g.
    `make_xgb_pipeline(X_hep, **config.xgb_hyperparams_hep())`. Passing
    nothing falls through to xgboost's own defaults, not the tuned ones --
    this is deliberate, so it's never ambiguous which endpoint's
    hyperparameters a given fit used.
    """
    if not HAS_XGB:
        raise ImportError("xgboost is not installed.")

    return Pipeline([
        ("pre", make_preprocessor(X, sparse_threshold=1.0)),
        ("xgb", XGBRegressor(
            objective="survival:cox",
            random_state=RANDOM_STATE,
            tree_method="hist",
            n_jobs=-1,
            **hyperparam_overrides,
        )),
    ])


def search_xgb_hyperparams(X: pd.DataFrame, y, event: np.ndarray, time: np.ndarray,
                            n_candidates: int = 12,
                            random_state: int = RANDOM_STATE) -> tuple[dict, float, pd.DataFrame]:
    """
    Randomly samples `n_candidates` distinct combinations from a bounded
    grid (learning_rate, max_depth, n_estimators, min_child_weight,
    subsample, colsample_bytree, reg_lambda, reg_alpha) and scores each
    with cv_cindex_xgb (n_repeats=1, to keep the search affordable -- same
    trade-off search_blend_weights makes).

    `y` is accepted but unused -- kept only so this search's call signature
    matches search_coxnet_l1_ratio's (X, y, event, time), which needs `y`
    but not `time`.

    Returns (best_params, best_mean_cindex, results_df) where best_params
    is the FULL 8-key dict (never a partial one) so it can be passed
    straight to config.update_config(xgb_hyperparams_hep=...) /
    (xgb_hyperparams_death=...) without a shallow-merge accidentally
    dropping keys. Does not write anywhere itself.
    """
    del y

    if not HAS_XGB:
        raise ImportError("xgboost is not installed.")

    from liverrisk.cv import cv_cindex_xgb  # local import: cv.py imports from this module, avoids a circular import

    param_grid: dict[str, list] = {
        "learning_rate": [0.01, 0.025, 0.05, 0.1],
        "max_depth": [2, 3, 4],
        "n_estimators": [300, 600, 900],
        "min_child_weight": [5, 10, 20],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "reg_lambda": [1.0, 5.0, 10.0],
        "reg_alpha": [0.0, 0.5, 1.0],
    }
    keys = list(param_grid.keys())

    rng = np.random.RandomState(random_state)
    seen: set[tuple] = set()
    candidates: list[dict] = []
    while len(candidates) < n_candidates:
        candidate = {k: param_grid[k][rng.randint(len(param_grid[k]))] for k in keys}
        sig = tuple(candidate[k] for k in keys)
        if sig in seen:
            continue
        seen.add(sig)
        candidates.append(candidate)

    rows = []
    best_params: dict | None = None
    best_mean = -np.inf
    for candidate in candidates:
        mean, std = cv_cindex_xgb(X, event, time, n_repeats=1, xgb_params=candidate)
        rows.append({**candidate, "mean": mean, "std": std})
        if mean > best_mean:
            best_mean = mean
            best_params = candidate

    # Track the winner from `candidates` directly (native Python int/float),
    # rather than reading it back out of results_df -- a DataFrame row that
    # mixes int columns (max_depth, n_estimators, ...) with float columns
    # (mean, std, ...) upcasts everything to float64 on `.iloc[row]`, which
    # would silently turn e.g. max_depth=3 into 3.0.
    results_df = pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)

    return dict(best_params), float(best_mean), results_df
