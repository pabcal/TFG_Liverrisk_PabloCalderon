"""
its one single file that contains all the tuned values. Everything else reads from it 
so it never has its own copy, only this copy exists.

Timeline:
 1. Fresh repo, notebook 02 has not been run yet and therfore best_config.json doesnt exist. 
    anything that imports from config.py gets the DEFAULTS, which are the hardcoded values.
    this way everything works even if the gridsearches havent run yet. These values will 
    get overwritten when notebook 02 is run.
 2. Notebook 02 is runfor the first time. In it, there is a search function that calls
    config.update(....). This is where best_config.json is created, which holds whatever was tuned
 3. Every notebook and script afterwards now reads the real tuned value instead of falling back to 
    the defaults, since best_config.json was created.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#__file__ refers to config.py own path. It points to that file's own location on disk
# this basically figures out where config.py lives on disk, go up one level to the folder 
# containing it, and thats where best_config.json should be.
#I do it this way so it can produce the correct path no whatever machine is running the code.
# if i had it hardcoded, it would only work on my machine.
CONFIG_PATH = Path(__file__).resolve().parent / "best_config.json"

DEFAULTS: dict[str, Any] = {
    # Identical starting values for both endpoints -- until
    # search_xgb_hyperparams() in models.py overwrites one or both via
    # 02_grid_search.ipynb, hep and death use the same original hardcoded
    # XGBRegressor kwargs.
    "xgb_hyperparams_hep": {
        "n_estimators": 600,     #How many trees the model builds. More trees more learning capacity -> high chance of overfitting
        "learning_rate": 0.025,  #How big of a step each new tree takes toward correcting mistakes
        "max_depth": 2,          #how many questions deep each individual tree is allowed to ask before making a decision
        "min_child_weight": 10,  # it says Don't create a split if either resulting group would end up too small." If a proposed split would put, say, only 3 patients into one of the two 
                                 #resulting groups, and min_child_weight=10 says you need at least 10, that split simply isn't allowed to happen — the tree has to either try a different question, or stop splitting that branch altogether.
        "subsample": 0.85,       #what fraction of patients each individual tree gets trained on (a random 85%, different each time)
        "colsample_bytree": 0.85,#what fraction of your 255 features each individual tree is allowed to consider (a random 85%)
        "reg_lambda": 5.0,       #L2 regularization penalty strength. It is so large here because of having 255 features but only 47 hepatic events. 
                                 #A lower number would most likely lead to the model learning what seperates the 47 patients from everyone else, memorizing noise rather than a real singal 
        "reg_alpha": 0.5,        #L1 regularization penalty strength
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
        "n_alphas": 30,        #coxnet could find weights that fit the 47 hepatic patients perfectly, but those wieghts would just be memorizing coincidence in the small group, not generalizing to new patients.
                               #Alpha controls how much the model gets punished for having large weights. The higher the alpha the heavier the punishment for large weights. 
                               # Coxnet can geenrate 100 canndidate alpha values in pne pass, from very weak to very strong. Rather than test all of them, the code picks 30. Thats what n_alphas is 
        "n_splits": 3,         # split the training data 3 ways, see which alpha gives the best average score across those 3 folds, pick the winner
    },

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


#if i call this function with nothing, the path automatically becomes CONFIG_OATH
def _load(path: Path = CONFIG_PATH) -> dict[str, Any]:
    #If the file is missing, return defaults
    if not path.exists():
        return json.loads(json.dumps(DEFAULTS))  

    #only runs if best_config does exist
    with open(path) as f:
        on_disk = json.load(f)

    # only blend weights tuned so far doesn't lose the other defaults.
    # merged starts with a safe load of the defaults, so it starts with the defaults
    # it then gets updated with the on_disk. But if any key is not yet on on_disk (because it hasnt been tuned yet)
    # it stays as its safe default value
    merged = json.loads(json.dumps(DEFAULTS))
    merged.update(on_disk)
    return merged


# Loaded once at import time. Call reload_config() if best_config.json
# changes during a running process (e.g. after update_config()).
# executes: the very first time any other file does from liverrisk import config
_config = _load()


def reload_config() -> dict[str, Any]:
    #brings _config up to date with whatever is on disk right now
    global _config
    _config = _load()
    return _config


def get_config() -> dict[str, Any]:
    return _config

#**kwargs lets a function accept any number of arguments with any names decided by whoever is calling it
# config.update_config(blend_weights_hep=[0.0, 1.0, 0.0], blend_weights_death=[0.4, 0.4, 0.2])
# i can call this and it will only update the blend_weights and blend_weights_death
def update_config(**kwargs: Any) -> dict[str, Any]:
    """
    Merge `kwargs` into best_config.json and reload the in-memory config.
    Only touches the keys passed in -- unrelated keys already on disk are
    preserved. This is the only intended way to persist tuning results
    (called from 02_grid_search.ipynb after search_blend_weights()).
    """
    #on_disk now has the defaults
    on_disk = json.loads(json.dumps(DEFAULTS))
    #if best_config exists, on disk is updated with the contents from on_disk
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            on_disk.update(json.load(f))
    #I update it with whatever i passed to the function, in the case above i would update it with 
    # blend_weights_hep=[0.0, 1.0, 0.0], blend_weights_death=[0.4, 0.4, 0.2]
    on_disk.update(kwargs)

    #write it back to best_config.json
    with open(CONFIG_PATH, "w") as f:
        json.dump(on_disk, f, indent=2)
    # calls relpad_config() which calls _load() which reads the best_config that update_config has modified (reads the whole file including the changes). And returns that to reload_config() which returns it here
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
