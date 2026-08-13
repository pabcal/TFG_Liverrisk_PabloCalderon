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

"""
Patient	  Raw score	Rank	Percentile
Patient A	0.3	2	2/5    = 0.4
Patient B	0.7	4	4/5    = 0.8
Patient C	0.1	1	1/5    = 0.2
Patient D	0.9	5	5/5    = 1.0
Patient E	0.5	3	3/5    = 0.6

this is what rank normalize does
"""
def rank_normalize(x) -> np.ndarray:
    return pd.Series(np.asarray(x)).rank(method="average", pct=True).to_numpy()


def blend_predictions(predictions: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """
    predictions is a list of arrays. cox_predictions = [0.3, 0.7, 0.1, 0.9, 0.5]   # Coxnet's score for each of the 5 patients
                                     rsf_predictions  = [0.2, 0.8, 0.15, 0.85, 0.4]  # RSF's score for each of the 5 patients
                                     xgb_predictions  = [0.35, 0.65, 0.05, 0.95, 0.55]  # XGB's score for each of the 5 patients
    """
    if weights is None:
        #if no weights were given, give all equal weights. The * on a python list means "repeat this N times"
        weights = [1.0] * len(predictions)

    weights = np.asarray(weights, dtype=float)
    #normalizes the weights so they all equal 1. [1.0, 1.0, 1.0] sum up to 3.0, so this would give each 0.333
    weights = weights / weights.sum()
    #an array of all zeroes, one slot per patient
    blended = np.zeros(len(predictions[0]), dtype=float)
    #zip pairs the two lists element by element, so we get (predictions[0], weights[0]) first, then (predictions[1], weights[1]) in one loop
    for p, w in zip(predictions, weights):
        #for each models predictions, rank normalize them into percentiles and multiply it bu that models weight
        # add it to the running total.

        """
        1st loop iteration: p = cox_predictions ([0.3, 0.7, 0.1, 0.9, 0.5]), w = 0.45
        2nd loop iteration: p = rsf_predictions ([0.2, 0.8, 0.15, 0.85, 0.4]), w = 0.25
        3rd loop iteration: p = xgb_predictions ([0.35, 0.65, 0.05, 0.95, 0.55]), w = 0.30

        Each time through, p is one MODEL's entire array of 5 patient scores, and w is that model's single weight.
        After all 3 loop iterations finish, blended holds 5 final numbers — one combined score per patient, each one being the weighted sum of that patient's 3 models' rank-normalized scores.

        each iteration rank-normalizes just that ONE model's predictions, scales them by that model's weight, and adds the result on top of whatever was already accumulated from the previous models

        this returns just one array with all the patients blended
"""
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

    # n_points evenly spaced numbers from 0 to 1.0
    grid_vals = np.linspace(0.0, 1.0, n_points)

    candidates: set[tuple[float, float, float]] = set()
    #for every possible cox value, try every possible rsf value
    for w_cox in grid_vals:
        for w_rsf in grid_vals:
            #the weights have to add to 1, so if w_cox = 0.2 and w_rsf = 0.1, w_xgb must = 0.7 because it must equal to 1
            w_xgb = 1.0 - w_cox - w_rsf
            if w_xgb < -1e-9 or w_xgb > 1.0 + 1e-9:
                continue
            #rounds to 4 decimal places
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
    # we now have all our potential weight candidates in candidates touple

    #this now scores every candidate, same as search_coxnet_l1_ratio
    rows = []
    for w_cox, w_rsf, w_xgb in sorted(candidates):
        #fits each model inside the fold with this candidates weights
        mean, std = cv_cindex_blend(
            X, y, event, time,
            n_repeats=1,
            weights=[w_cox, w_rsf, w_xgb],
            rsf_n_estimators=150,
            xgb_params=xgb_params,
        )
        rows.append({"w_cox": w_cox, "w_rsf": w_rsf, "w_xgb": w_xgb, "mean": mean, "std": std})

    # after building the table, sort and grab the top row.

    """
    CAMBIAR:
    Interesting: this one doesn't have the dtype-upcasting protection search_xgb_hyperparams used — but that protection was specifically needed because XGB's hyperparameters mix integers (max_depth) and floats (learning_rate). Here, all three weights are already floats to begin with, so reading them back via .iloc[0] and wrapping in float(...) is safe
    
    """
    results_df = pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)
    best = results_df.iloc[0]
    best_weights = (float(best["w_cox"]), float(best["w_rsf"]), float(best["w_xgb"]))

    return best_weights, float(best["mean"]), results_df
