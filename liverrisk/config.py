"""
Single source of truth for tuned values (per-endpoint XGB hyperparameters,
per-endpoint Coxnet l1_ratio, Coxnet alpha search settings, per-endpoint
blend weights).

Loads liverrisk/best_config.json once at import time. If the file is
missing, falls back to DEFAULTS -- the values that were hardcoded directly
in the original notebook (make_xgb_pipeline's XGBRegressor kwargs,
make_coxnet_pipeline's l1_ratio, fit_coxnet_with_alpha_cv's
n_alphas/n_splits, and the pre-tuning OLD_BLEND_WEIGHTS = (0.45, 0.25,
0.30) used for both endpoints) -- so the package works before
02_grid_search.ipynb has ever been run.

XGB hyperparameters and Coxnet l1_ratio are stored per endpoint
(xgb_hyperparams_hep/_death, coxnet_hyperparams's l1_ratio_hep/_death) --
there is deliberately no shared/unsplit fallback key for either. Every
caller that builds an XGB or l1_ratio-tuned Coxnet model must explicitly
pick the hep or death reader; make_xgb_pipeline/make_coxnet_pipeline never
guess which endpoint they're being used for.

02_grid_search.ipynb is the only notebook that calls save_config() /
update_config(); everything else (models.py, cv.py, 03_train_final,
scripts/train.py) only reads.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(__file__).resolve().parent / "best_config.json"

DEFAULTS: dict[str, Any] = {
    # Identical starting values for both endpoints -- until
    # search_xgb_hyperparams() (see models.py) overwrites one or both via
    # 02_grid_search.ipynb, hep and death use the same original hardcoded
    # XGBRegressor kwargs.
    "xgb_hyperparams_hep": {
        "n_estimators": 600,
        "learning_rate": 0.025,
        "max_depth": 2,
        "min_child_weight": 10,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 5.0,
        "reg_alpha": 0.5,
    },
    "xgb_hyperparams_death": {
        "n_estimators": 600,
        "learning_rate": 0.025,
        "max_depth": 2,
        "min_child_weight": 10,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 5.0,
        "reg_alpha": 0.5,
    },
    "coxnet_alpha_search": {
        "n_alphas": 30,
        "n_splits": 3,
    },
    # l1_ratio=0.9 is make_coxnet_pipeline's original hardcoded default,
    # for both endpoints, until search_coxnet_l1_ratio() overwrites one or
    # both.
    "coxnet_hyperparams": {
        "l1_ratio_hep": 0.9,
        "l1_ratio_death": 0.9,
    },
    # Pre-tuning weights, applied uniformly to both endpoints in the
    # original notebook. search_blend_weights() (see blend.py) replaces
    # these per-endpoint once 02_grid_search.ipynb has been run.
    "blend_weights_hep": [0.45, 0.25, 0.30],
    "blend_weights_death": [0.45, 0.25, 0.30],
}


def _load(path: Path = CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        return json.loads(json.dumps(DEFAULTS))  # deep copy

    with open(path) as f:
        on_disk = json.load(f)

    # Shallow-merge over defaults so a partially-written config file (e.g.
    # only blend weights tuned so far) doesn't lose the other defaults.
    merged = json.loads(json.dumps(DEFAULTS))
    merged.update(on_disk)
    return merged


# Loaded once at import time. Call reload_config() if best_config.json
# changes during a running process (e.g. after update_config()).
_config: dict[str, Any] = _load()


def reload_config() -> dict[str, Any]:
    global _config
    _config = _load()
    return _config


def get_config() -> dict[str, Any]:
    return _config


def update_config(**kwargs: Any) -> dict[str, Any]:
    """
    Merge `kwargs` into best_config.json and reload the in-memory config.
    Only touches the keys passed in -- unrelated keys already on disk are
    preserved. This is the only intended way to persist tuning results
    (called from 02_grid_search.ipynb after search_blend_weights()).
    """
    on_disk = json.loads(json.dumps(DEFAULTS))
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            on_disk.update(json.load(f))

    on_disk.update(kwargs)

    with open(CONFIG_PATH, "w") as f:
        json.dump(on_disk, f, indent=2)

    return reload_config()


def xgb_hyperparams_hep() -> dict[str, Any]:
    return dict(_config["xgb_hyperparams_hep"])


def xgb_hyperparams_death() -> dict[str, Any]:
    return dict(_config["xgb_hyperparams_death"])


def coxnet_alpha_search() -> dict[str, Any]:
    return dict(_config["coxnet_alpha_search"])


def coxnet_l1_ratio_hep() -> float:
    return float(_config["coxnet_hyperparams"]["l1_ratio_hep"])


def coxnet_l1_ratio_death() -> float:
    return float(_config["coxnet_hyperparams"]["l1_ratio_death"])


def blend_weights_hep() -> list[float]:
    return list(_config["blend_weights_hep"])


def blend_weights_death() -> list[float]:
    return list(_config["blend_weights_death"])
