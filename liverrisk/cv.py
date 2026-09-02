from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold

from sksurv.metrics import concordance_index_censored

from liverrisk.models import (
    HAS_XGB,
    RANDOM_STATE,
    make_coxnet_pipeline,
    make_rsf_pipeline,
    make_xgb_pipeline,
    signed_time_label,
)

"""
Runs full cross-validation for Coxnet, returning the average C-index and
its spread across folds.
"""

def cv_cindex_coxnet(X: pd.DataFrame, y, event: np.ndarray,
                      n_splits: int = 5, n_repeats: int = 3,
                      l1_ratio: float = 0.9) -> tuple[float, float]:
    #builds the fold splitting tool. 5 folds, 3 repeats
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
    )
    #gets the field names out of y
    event_name, time_name = y.dtype.names
    scores = []

    #tr is the training subset and va the validation subset
    for tr, va in cv.split(X, event.astype(int)):
        #builds a fresh, untrained pipeline for this exact round of training and val
        model = make_coxnet_pipeline(X.iloc[tr], l1_ratio=l1_ratio)
        # trains the model on this rounds training patients
        model.fit(X.iloc[tr], y[tr])
        # predict on the held out patients, the validation ones
        pred = model.predict(X.iloc[va])
        # grade pred against the true event/time for those same held-out patients — this round's C-index score.
        scores.append(concordance_index_censored(y[event_name][va], y[time_name][va], pred)[0])

    #after all 15 rounds, average the scores and measure their spread.
    return float(np.mean(scores)), float(np.std(scores))


"""
Runs full cross-validation for RSF, returning the average C-index and its
spread across folds.
"""

def cv_cindex_rsf(X: pd.DataFrame, y, event: np.ndarray,
                   n_splits: int = 5, n_repeats: int = 3,
                   n_estimators: int = 500) -> tuple[float, float]:
    #builds the fold splitting tool. 5 folds, 3 repeats
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
    )
    #gets the field names out of y
    event_name, time_name = y.dtype.names
    scores = []

    #tr is the training subset and va the validation subset
    for tr, va in cv.split(X, event.astype(int)):
        #builds a fresh, untrained pipeline for this exact round of training and val
        model = make_rsf_pipeline(X.iloc[tr], n_estimators=n_estimators)
        # trains the model on this rounds training patients
        model.fit(X.iloc[tr], y[tr])
        # predict on the held out patients, the validation ones
        pred = model.predict(X.iloc[va])
        # grade pred against the true event/time for those same held-out patients — this round's C-index score.
        scores.append(concordance_index_censored(y[event_name][va], y[time_name][va], pred)[0])

    #after all 15 rounds, average the scores and measure their spread.
    return float(np.mean(scores)), float(np.std(scores))

# Since xgboost doesnt share sksurv model itnerface, it needs a different function, this time with time encoded
def cv_cindex_xgb(X: pd.DataFrame, event: np.ndarray, time: np.ndarray,
                  n_splits: int = 5, n_repeats: int = 3,
                  xgb_params: dict | None = None) -> tuple[float, float]:
    if not HAS_XGB:
        return np.nan, np.nan

    if xgb_params is None:
        raise ValueError(
            "cv_cindex_xgb requires explicit xgb_params (e.g. config.xgb_hyperparams_hep() "
            "or config.xgb_hyperparams_death()) -- there is no implicit default."
        )

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
    )

    scores = []
    #XGboost specific encoding trick, onlu applied to the training fold.
    y_signed = signed_time_label(event, time)

    for tr, va in cv.split(X, event.astype(int)):
        # **xgb_params unpacks the dictionary. 
        # unpacks a dictionary back out into individual named arguments. 
        # So if xgb_params = {"learning_rate": 0.025, "max_depth": 2, ...}, then make_xgb_pipeline(X.iloc[tr], **xgb_params) 
        # is exactly equivalent to writing make_xgb_pipeline(X.iloc[tr], learning_rate=0.025, max_depth=2, ...)
        model = make_xgb_pipeline(X.iloc[tr], **xgb_params)
        model.fit(X.iloc[tr], y_signed[tr])
        pred = model.predict(X.iloc[va])
        scores.append(concordance_index_censored(event[va], time[va], pred)[0])

    return float(np.mean(scores)), float(np.std(scores))


def cv_cindex_blend(X: pd.DataFrame, y, event: np.ndarray, time: np.ndarray,
                    n_splits: int = 5, n_repeats: int = 1,
                    weights: list[float] | None = None,
                    rsf_n_estimators: int = 150,
                    xgb_params: dict | None = None) -> tuple[float, float]:
    from liverrisk.blend import blend_predictions  # local import: avoids a circular import with blend.py

    if weights is None:
        raise ValueError(
            "cv_cindex_blend requires explicit weights (e.g. config.blend_weights_hep() "
            "or config.blend_weights_death()) -- there is no implicit default."
        )

    if HAS_XGB and xgb_params is None:
        raise ValueError(
            "cv_cindex_blend requires explicit xgb_params when xgboost is installed "
            "(e.g. config.xgb_hyperparams_hep() or config.xgb_hyperparams_death()) -- "
            "there is no implicit default."
        )

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
    )

    event_name, time_name = y.dtype.names
    y_signed = signed_time_label(event, time) if HAS_XGB else None
    scores = []

    for tr, va in cv.split(X, event.astype(int)):
        X_tr, X_va = X.iloc[tr], X.iloc[va]

        preds = []

        cox = make_coxnet_pipeline(X_tr)
        cox.fit(X_tr, y[tr])
        preds.append(cox.predict(X_va))

        rsf = make_rsf_pipeline(X_tr, n_estimators=rsf_n_estimators)
        rsf.fit(X_tr, y[tr])
        preds.append(rsf.predict(X_va))

        if HAS_XGB:
            xgb = make_xgb_pipeline(X_tr, **xgb_params)
            xgb.fit(X_tr, y_signed[tr])
            preds.append(xgb.predict(X_va))

        blended = blend_predictions(preds, weights)
        scores.append(concordance_index_censored(y[event_name][va], y[time_name][va], blended)[0])

    return float(np.mean(scores)), float(np.std(scores))


def cv_cindex_formula(X: pd.DataFrame, y, event: np.ndarray, score_array: np.ndarray,
                       n_splits: int = 5, n_repeats: int = 3) -> tuple[float, float]:
   
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
    )

    event_name, time_name = y.dtype.names
    score_array = np.asarray(score_array)
    scores = []
    #no training here, only validation
    for _, va in cv.split(X, event.astype(int)):
        #score_array[va]: grab this fold's formula scores (positions va only), example: the FIB-4 values for the held out patients
        #np.isfinite(score_array[va]) — check each of those: is it a real, usable number? Gives back True/False.
        #va[np.isfinite(score_array[va])]: filters by true or false, so only keeps the valid positions (where the formula score is actually usable)
        valid = va[np.isfinite(score_array[va])]
        # uses valid so that the patients with an undefined FIB4/APRI score dont contribute to the score 
        scores.append(concordance_index_censored(y[event_name][valid], y[time_name][valid], score_array[valid])[0])

    return float(np.mean(scores)), float(np.std(scores))


def bootstrap_cindex_diff(y, event: np.ndarray, time: np.ndarray,
                           score_a: np.ndarray, score_b: np.ndarray,
                           n_boot: int = 1000, random_state: int = RANDOM_STATE) -> tuple[np.ndarray, float, float]:

    """
    the function does:

    After computing C indez for ML and Fib say, ML = 0.80, FIB-4 = 0.66. The gap looks like ML wins by 0.14. 
    But is that gap REAL, or could it just be luck — bad luck for FIB-4, good luck for ML, purely because of which specific 
    patients happened to be in the dataset. This is a real worry as there are only 47 hepatic events. 
    It basically asks if I had gotten slightly different patients, how much would this 0.14 gap have wobbled around?

    I cant get new patients to test this, but i can fake having new patients by randomely resampling from the existing ones,
    some show up twice or more, some dont show up. I do this 1000 times, each time i build a pretend dataset compute MLS and FIB4s C index
    and record the difference between the 2.

    After doing this 1000 times, you have 1000 different versions of "the gap" — maybe: +0.08, +0.15, +0.11, +0.03, +0.19, ... — sometimes bigger than 
    the original 0.14, sometimes smaller, because each pretend dataset was slightly different.

    The output: 

    hepatic's [-0.110, +0.067] (crosses zero → honestly, a tie) versus death's [+0.204, +0.415] (clearly above zero → a real, confident win). Without this function, 
    you'd have had no way to know that hepatic's apparent difference was actually just noise

    [-0.110, +0.067] is a range that contains the middle 95% of your 1000 resampled differences — throwing away the most extreme 2.5% on the bottom and the most extreme 2.5% on the top.

    If the ENTIRE middle-95% range sits above zero (like death's [+0.204, +0.415]) — that means even the "unlucky, low end" of your 1000 resamples still showed ML winning.

    If it crosses 0, it is a tie
    """

    event = np.asarray(event)
    time = np.asarray(time)
    #score a could be the ML models risk score for each of the 1253 hepatic patients
    # score b could be FIB4-risk score for the same 1253 patients in the same order
    score_a = np.asarray(score_a)
    score_b = np.asarray(score_b)
    n = len(event)

    #random number generator
    rng = np.random.RandomState(random_state)
    diffs = []

    while len(diffs) < n_boot:
        # generate n random whole numbers between 0 and n-1
        # Some patient positions will show up multiple times in idx; others won't show up at all.
        idx = rng.randint(0, n, size=n)

        valid = idx[np.isfinite(score_a[idx]) & np.isfinite(score_b[idx])]
        if len(valid) < 2 or event[valid].sum() == 0:
            continue

        c_a = concordance_index_censored(event[valid], time[valid], score_a[valid])[0]
        c_b = concordance_index_censored(event[valid], time[valid], score_b[valid])[0]
        diffs.append(c_a - c_b)

    diffs = np.asarray(diffs)
    ci_lower, ci_upper = np.percentile(diffs, [2.5, 97.5])

    return diffs, float(ci_lower), float(ci_upper)
