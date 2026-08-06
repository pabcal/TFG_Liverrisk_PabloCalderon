"""
Cross-validation scoring, moved verbatim from ANNITIA_baseline_local.ipynb
section 4.14: cindex_from_y, cv_cindex_sksurv, cv_cindex_xgb, cv_cindex_blend.

cv_cindex_blend still requires explicit `weights` (raises ValueError if
None) and still accepts `rsf_n_estimators` for cheap CV during search --
unchanged from the notebook version.

Note: cv_cindex_blend imports blend_predictions from liverrisk.blend
*inside* the function body rather than at module level, because
liverrisk.blend.search_blend_weights imports cv_cindex_blend from this
module -- a top-level import either direction would be circular.
"""
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


def cindex_from_y(y, pred: np.ndarray) -> float:
    event_name, time_name = y.dtype.names
    return float(concordance_index_censored(y[event_name], y[time_name], pred)[0])


def cv_cindex_sksurv(model_factory, X: pd.DataFrame, y, event: np.ndarray,
                    n_splits: int = 5, n_repeats: int = 3) -> tuple[float, float]:
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
    )

    event_name, time_name = y.dtype.names
    scores = []

    for tr, va in cv.split(X, event.astype(int)):
        model = model_factory(X.iloc[tr])
        model.fit(X.iloc[tr], y[tr])
        pred = model.predict(X.iloc[va])
        scores.append(concordance_index_censored(y[event_name][va], y[time_name][va], pred)[0])

    return float(np.mean(scores)), float(np.std(scores))


def cv_cindex_xgb(X: pd.DataFrame, event: np.ndarray, time: np.ndarray,
                  n_splits: int = 5, n_repeats: int = 3,
                  xgb_params: dict | None = None) -> tuple[float, float]:
    """
    `xgb_params` is required (no implicit default) -- cv_cindex_xgb is
    endpoint-agnostic (it just scores whatever X/event/time it's given), so
    it can't guess whether to use config.xgb_hyperparams_hep() or
    config.xgb_hyperparams_death(); the caller must pass the right one (or
    a candidate dict, from search_xgb_hyperparams).
    """
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
    y_signed = signed_time_label(event, time)

    for tr, va in cv.split(X, event.astype(int)):
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
    """
    Fold-honest score for the actual blended ensemble used in the submission.

    Individual per-model CV scores (Coxnet/RSF/XGB) don't tell you the blend's
    score, since rank-blending can help or hurt depending on how correlated
    the models' errors are. This fits all three models inside each fold,
    rank-blends their held-out predictions, and scores that blend -- the same
    procedure used for the final submission, just cross-validated.

    Uses the plain (non alpha-tuned) Coxnet pipeline per fold to keep this
    affordable -- nesting the alpha-selection GridSearchCV inside this CV
    would multiply the cost again. Likewise, `rsf_n_estimators` defaults to a
    much smaller forest than the final fit (500): this is a CV *estimate*, and
    a 150-tree RSF ranks patients almost identically to a 500-tree one, so
    fewer trees buys most of the runtime back without moving the score.

    No implicit default weights: pass an explicit `weights` triple
    (e.g. config.blend_weights_hep() / config.blend_weights_death(), or a
    candidate from search_blend_weights). This is deliberate -- a hardcoded
    default here used to be one of two places blend weights lived, which let
    the CV-reported score and the actual submitted blend silently drift apart.

    Likewise no implicit default `xgb_params` when HAS_XGB -- pass
    config.xgb_hyperparams_hep() / config.xgb_hyperparams_death() (or a
    search_xgb_hyperparams candidate), matching whichever endpoint `X`/`y`
    belong to.
    """
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
