"""
Where the 3 models get defined. The preprocessing they use, the hyperpareameters, structure

I use functions like these instead of one fixed model object because the same model gets fit
many times on different subsets of data, once per cross val fold. Each time it needs a fresh untrained 
copy. Thats why a function that builds the new pipeline each time its called allows this. 
Calling make_coxnet_pipeline(X) 15 times during cross-validation, gets me 15 independent, untrained 
pipelines, not one pipeline re fitting on top of itself. Reusing weights carries over knowledge from previous folds,
 causes data leakage that inflates your performance metrics and leads to overfitting. So this way is better
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

# xgboost has a few dependencies that dont always work on everyones machine (from what ive researched)
# in all machines ive tried its worked, but just in case i wrap it around a try
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
    #cat_cols = []
    #for c in STATIC_CAT:
        #if c in X.columns:
            #cat_cols.append(c)
    cat_cols = [c for c in STATIC_CAT if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return num_cols, cat_cols


def make_preprocessor(X: pd.DataFrame, sparse_threshold: float = 0.3):
    num_cols, cat_cols = split_feature_types(X)

    #Pipeline([("name, tool"), ("name, tool")]). Pipeline has 1 argument: a list contining tuples
    # Its like a plan, first do imputation, then do scaling
    num_pipe = Pipeline([
        #fills missing NaN values with median
        ("imp", SimpleImputer(strategy="median", add_indicator=True)),
        #rescale the numbers
        ("sc", StandardScaler(with_mean=False)),
    ])

    cat_pipe = Pipeline([
        #NaN gets replaced with -1
        ("imp", SimpleImputer(strategy="constant", fill_value=-1)),
        #Turns categories like gender: F/M into seperate 0/1 columns. ignores unknown values (if a new patient doesnt have F/M)
        ("oh", OneHotEncoder(handle_unknown="ignore")),
    ])

    # ColumnTransformer takes the whole 255 column X, but only hands X's num_cols to num_pipe and X's cat_cols
    # to cat_pipe. Completely seperate, each pipeline never sees the otger's columns. It then stiches both process 
    # back together in one combined table
    # It returns the ColumnTransformer object, but not yet run. 
    # take the full table, split it into numeric/categorical pieces, process each piece with its own separate mini-recipe, then combine the results" 
    #  and it only actually executes when something later calls .fit()/.transform() on it.
    return ColumnTransformer(
        transformers=[
            #applies num_pipe to num cols
            ("num", num_pipe, num_cols),
            #applies cat_pipe only to cat_cols
            ("cat", cat_pipe, cat_cols),
        ],
        #any column not in either list gets dropped
        remainder="drop",
        sparse_threshold=sparse_threshold,
    )


# ---------------------------------------------------------------------
# Coxnet
# Builds the pipeline for the coxnet model. This function is called fresh on every fold furing 
# cross validation.
# ---------------------------------------------------------------------
def make_coxnet_pipeline(X: pd.DataFrame, l1_ratio: float = 0.9) -> Pipeline:
    """
    `l1_ratio` defaults to the original hardcoded value (0.9) so every
    existing caller that doesn't pass it explicitly is unaffected.
    search_coxnet_l1_ratio() sweeps this argument; nothing here reads
    config.coxnet_l1_ratio_hep()/_death() automatically -- callers that
    want the tuned value must pass it in themselves.
    This way it doesnt have to guess if its death or hep

    When cox_pipeline.fit(X_train, y_train) runs, Pipeline (not make_coxnet_pipeline, not you) internally does this:

    1. Takes X_train (the data YOU gave it, right now, in this .fit() call).
    2. Runs step "pre" (the already-built ColumnTransformer)'s .fit_transform() on X_train — producing a cleaned version.
    3. Overwrites its own internal notion of "the current data" with that cleaned result.
    4. Runs step "cox"'s .fit(), using that now-cleaned data (not the original X_train).
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
    del time

    # mutable defaults (like lists) can behave unexpectedly if used directly as a default value
    # which is why i default it to none in the arguments and build it here

    if l1_ratios is None:
        l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]

    from liverrisk.cv import cv_cindex_coxnet  # local import: cv.py imports from this module, avoids a circular import

    rows = []
    for l1_ratio in l1_ratios:
        mean, std = cv_cindex_coxnet(X, y, event, n_repeats=n_repeats, l1_ratio=l1_ratio)
        rows.append({"l1_ratio": l1_ratio, "mean": mean, "std": std})

    results_df = pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)
    best = results_df.iloc[0]

    return float(best["l1_ratio"]), float(best["mean"]), results_df


def fit_coxnet_with_alpha_cv(X: pd.DataFrame, y, random_state: int = RANDOM_STATE,
                              n_alphas: int | None = None, n_splits: int | None = None,
                              l1_ratio: float = 0.9) -> Pipeline:

    #loads from best config in case the function is called without one of the params
    if n_alphas is None or n_splits is None:
        cfg = config.coxnet_alpha_search()
        n_alphas = cfg["n_alphas"] if n_alphas is None else n_alphas
        n_splits = cfg["n_splits"] if n_splits is None else n_splits

    # fits the model becaause after fitting the model, we get the 100 (or around there) alpha values
    # because when coxnet fits, it doesnt compute one single answer, but tries a whole range of alpha strengths.
    # and each computes a seperate set of feature weights.
    path_model = make_coxnet_pipeline(X, l1_ratio=l1_ratio)
    path_model.fit(X, y)
    #grabs the actual CoxSurvivalAnalysis object frin the cox part of the pipeline, and from that object
    # grabs the list of alpha values (cox_) coxnet has computed internally, which could maybe be 100 of them
    alphas = path_model.named_steps["cox"].alphas_

    # if the coxSurvivalAnalysis returned more alphas than we want, (100 > 30), we create evenly spaced
    # positions betweenn 0 and the last index. For example, if alpha returned 100 alphas and we wanted 30, we create 
    # 30 evenly spaced alphas from 0 to 100.
    if len(alphas) > n_alphas:
        idx = np.linspace(0, len(alphas) - 1, n_alphas).round().astype(int)
        idx = np.unique(idx)
        #select just those subsampled alphas
        alphas = alphas[idx]

    preprocessor = make_preprocessor(X, sparse_threshold=0.3)
    #X_t now has the full clean table
    X_t = preprocessor.fit_transform(X)
    #not a stratifiedKFold to save time. We are only looking for the best alpha, no need to do it with
    # the same rigor as a model evaluation.
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

    # The Pipeline wrapper is what GUARANTEES the raw-data-cleaning step happens automatically before Coxnet ever 
    # sees anything, every time .predict() gets called on the bundle
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

    My y has event and time. But the XGBoost doesnt accept that format.
    it expects the target as one plain array of numbers. But it still needs to know 
    both things, when and if it happened. The array now becomes:
    Positive time → "this patient had the event, at this time"
    Negative time → "this patient was censored (event didn't happen, or unknown), observed for at least this long"
    so event = true, time = 3.5 becomes +3.5. Event = false time = 8.0 becomes -8 and so on
    """
    return np.where(event, time, -time)


def make_xgb_pipeline(X: pd.DataFrame, **hyperparam_overrides) -> Pipeline:

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
    #the 8 hyperparameter names as a list
    keys = list(param_grid.keys())

    rng = np.random.RandomState(random_state)
    seen = set()
    candidates = []
    while len(candidates) < n_candidates:
        #candidate = {k: param_grid[k][rng.randint(len(param_grid[k]))] for k in keys}
        candidate = {}
        #loops through all the keys and randomely finds a value for each key, when the loop ends
        # you would have found one possible combination 
        for k in keys:
            #grabs the value from the param_grid for the current param, for example if the current param is max_depth, grabs [2,3,4]
            options_for_current_param = param_grid[k]
            # random number from 0 to the length of the param, on the above scenario it would be random number from 0 to 2
            random_idx = rng.randint(len(options_for_current_param))
            # chooses one of the options randomely
            chosen_value = options_for_current_param[random_idx]
            #becomes the candidate 
            candidate[k] = chosen_value
        # candidate could now be something like {"learning_rate": 0.05, "max_depth": 3, "n_estimators": 900, "min_child_weight": 10, "subsample": 0.85, "colsample_bytree": 0.7, "reg_lambda": 5.0, "reg_alpha": 0.5}
        # sig becomes (0.05, 3, 900, 10, 0.85, 0.7, 5.0, 0.5)
        sig = tuple(candidate[k] for k in keys)
        # checks if i have seen this exact combination before, if i have dont take this repeated one into account
        if sig in seen:
            continue
        # if not seen, add it
        seen.add(sig)
        candidates.append(candidate)

    rows = []
    best_params = None
    best_mean = -np.inf
    #go through all the candidates built earlier
    for candidate in candidates:
        mean, std = cv_cindex_xgb(X, event, time, n_repeats=1, xgb_params=candidate)
        # builds one row of the results table. The 8 hyperparameters plus the mean and std
        rows.append({**candidate, "mean": mean, "std": std})
        #winner tracker
        if mean > best_mean:
            best_mean = mean
            best_params = candidate
            
    results_df = pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)

    return dict(best_params), float(best_mean), results_df
