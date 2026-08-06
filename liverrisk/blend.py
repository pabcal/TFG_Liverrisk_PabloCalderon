"""
Prediction blending, moved verbatim from ANNITIA_baseline_local.ipynb
sections 4.14b and 4.15: search_blend_weights, rank_normalize,
blend_predictions.

search_blend_weights() itself does NOT write to best_config.json -- it
just returns (best_weights, best_mean, results_df), same as in the
notebook. Writing the winner back to disk is done by the *caller*
(02_grid_search.ipynb), via `config.update_config(blend_weights_hep=...)`
/ `update_config(blend_weights_death=...)`, one explicit call per
endpoint, right after each search. Keeping the file write out of this
module keeps it a pure/testable function and keeps the "when do we
persist tuning results" decision visible in the notebook that runs the
search, rather than hidden as a side effect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from liverrisk.cv import cv_cindex_blend
from liverrisk.models import HAS_XGB


def rank_normalize(x) -> np.ndarray:
    return pd.Series(np.asarray(x)).rank(method="average", pct=True).to_numpy()


def blend_predictions(predictions: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    if weights is None:
        weights = [1.0] * len(predictions)

    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    blended = np.zeros(len(predictions[0]), dtype=float)
    for p, w in zip(predictions, weights):
        blended += w * rank_normalize(p)

    return blended


def search_blend_weights(X: pd.DataFrame, y, event: np.ndarray, time: np.ndarray,
                          n_points: int = 6,
                          xgb_params: dict | None = None) -> tuple[tuple[float, float, float], float, pd.DataFrame]:
    """
    Grid-search rank-blend weights (w_cox, w_rsf, w_xgb) that sum to 1,
    scoring each candidate with cv_cindex_blend.

    Explicitly includes edge cases where one or two weights are 0 (e.g.
    pure RSF, or RSF+XGB with no Coxnet) -- the best blend for a given
    endpoint isn't guaranteed to use all three models.

    Every candidate is scored with n_repeats=1 and rsf_n_estimators=150
    (both passed through to cv_cindex_blend) to keep the grid affordable.
    These scores are for *ranking* candidates against each other; re-score
    the winning weights with more repeats separately for a trustworthy
    final number.

    `xgb_params` is forwarded straight to cv_cindex_blend, which requires
    it explicitly whenever xgboost is installed -- pass whichever
    endpoint's config.xgb_hyperparams_hep()/_death() matches this `X`.

    Returns (best_weights, best_mean_cindex, results_df) where results_df
    has every candidate sorted best-first, for inspection. Does not write
    anywhere -- see module docstring.
    """
    grid_vals = np.linspace(0.0, 1.0, n_points)

    candidates: set[tuple[float, float, float]] = set()
    for w_cox in grid_vals:
        for w_rsf in grid_vals:
            w_xgb = 1.0 - w_cox - w_rsf
            if w_xgb < -1e-9 or w_xgb > 1.0 + 1e-9:
                continue
            triple = (
                round(float(w_cox), 4),
                round(float(w_rsf), 4),
                round(float(max(0.0, min(1.0, w_xgb))), 4),
            )
            if abs(sum(triple) - 1.0) > 1e-6:
                continue
            if not HAS_XGB and triple[2] != 0.0:
                # No XGB model available -- only search the cox/rsf simplex.
                continue
            candidates.add(triple)

    rows = []
    for w_cox, w_rsf, w_xgb in sorted(candidates):
        mean, std = cv_cindex_blend(
            X, y, event, time,
            n_repeats=1,
            weights=[w_cox, w_rsf, w_xgb],
            rsf_n_estimators=150,
            xgb_params=xgb_params,
        )
        rows.append({"w_cox": w_cox, "w_rsf": w_rsf, "w_xgb": w_xgb, "mean": mean, "std": std})

    results_df = pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)
    best = results_df.iloc[0]
    best_weights = (float(best["w_cox"]), float(best["w_rsf"]), float(best["w_xgb"]))

    return best_weights, float(best["mean"]), results_df
