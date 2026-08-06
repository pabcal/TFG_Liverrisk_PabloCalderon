"""
CLI entry point for final training + submission building.

Single source of truth for what "train the final models" means: both
`python scripts/train.py` and notebooks/03_train_final.ipynb call
train_final_models() / save_models() / build_submission() from here
instead of duplicating the training logic in two places (03 is a thin,
cell-by-cell walkthrough that imports this module).

Equivalent to the "Fit final models" + "Submission" sections of main()
in the original ANNITIA_baseline_local.ipynb, split out so it can run
without a notebook. Reads processed features from liverrisk/data/processed/
(written once by notebooks/01_data_exploration.ipynb) and blend weights
from liverrisk/best_config.json (written by notebooks/02_grid_search.ipynb)
-- it does NOT call build_patient_features() or search_blend_weights()
itself.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import joblib
import pandas as pd

from liverrisk import config
from liverrisk.blend import blend_predictions
from liverrisk.features import load_feature_columns, load_features, load_test_features, load_test_ids
from liverrisk.models import HAS_XGB, fit_coxnet_with_alpha_cv, make_rsf_pipeline, make_xgb_pipeline, signed_time_label

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "liverrisk" / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models"
OUTPUT_PATH = REPO_ROOT / "outputs" / "improved_submission.csv"


def train_final_models(processed_dir: str | Path = PROCESSED_DIR, verbose: bool = True) -> dict:
    """
    Loads processed features, fits the final Coxnet/RSF/XGB models per
    endpoint, and rank-blends them with config.blend_weights_hep() /
    config.blend_weights_death(). Returns a dict with the fitted models,
    test-set predictions, and the blended training-cohort scores (needed
    later for percentile-rank lookups).
    """
    def log(msg: str) -> None:
        if verbose:
            print(msg)

    X_hep, y_hep, hep_event, hep_time = load_features("hep", processed_dir)
    X_death, y_death, death_event, death_time = load_features("death", processed_dir)
    X_test = load_test_features(processed_dir)
    feature_columns = load_feature_columns(processed_dir)
    X_test = X_test.reindex(columns=feature_columns)

    log(f"Loaded processed features: hepatic n={len(X_hep)}, death n={len(X_death)}, test n={len(X_test)}")

    blend_weights_hep = config.blend_weights_hep()
    blend_weights_death = config.blend_weights_death()
    log(f"Blend weights (w_cox, w_rsf, w_xgb) -- hepatic={blend_weights_hep}, death={blend_weights_death}")

    log("Fitting Coxnet (alpha CV)...")
    cox_hep = fit_coxnet_with_alpha_cv(X_hep, y_hep, l1_ratio=config.coxnet_l1_ratio_hep())
    cox_death = fit_coxnet_with_alpha_cv(X_death, y_death, l1_ratio=config.coxnet_l1_ratio_death())

    log("Fitting RSF (500 trees)...")
    rsf_hep = make_rsf_pipeline(X_hep)
    rsf_hep.fit(X_hep, y_hep)
    rsf_death = make_rsf_pipeline(X_death)
    rsf_death.fit(X_death, y_death)

    models = {
        "cox_hep": cox_hep, "cox_death": cox_death,
        "rsf_hep": rsf_hep, "rsf_death": rsf_death,
    }

    pred_hep_list = [cox_hep.predict(X_test), rsf_hep.predict(X_test)]
    pred_death_list = [cox_death.predict(X_test), rsf_death.predict(X_test)]

    # Also score the training cohort itself with the final models, for
    # later percentile-rank lookups (e.g. "this new patient's risk score
    # sits at the 80th percentile of the training cohort").
    train_pred_hep_list = [cox_hep.predict(X_hep), rsf_hep.predict(X_hep)]
    train_pred_death_list = [cox_death.predict(X_death), rsf_death.predict(X_death)]

    if HAS_XGB:
        log("Fitting XGB (survival:cox)...")
        xgb_hep = make_xgb_pipeline(X_hep, **config.xgb_hyperparams_hep())
        xgb_hep.fit(X_hep, signed_time_label(hep_event, hep_time))
        xgb_death = make_xgb_pipeline(X_death, **config.xgb_hyperparams_death())
        xgb_death.fit(X_death, signed_time_label(death_event, death_time))

        models["xgb_hep"] = xgb_hep
        models["xgb_death"] = xgb_death

        pred_hep_list.append(xgb_hep.predict(X_test))
        pred_death_list.append(xgb_death.predict(X_test))
        train_pred_hep_list.append(xgb_hep.predict(X_hep))
        train_pred_death_list.append(xgb_death.predict(X_death))

    pred_hep = blend_predictions(pred_hep_list, blend_weights_hep)
    pred_death = blend_predictions(pred_death_list, blend_weights_death)

    train_scores_hep = blend_predictions(train_pred_hep_list, blend_weights_hep)
    train_scores_death = blend_predictions(train_pred_death_list, blend_weights_death)

    log(f"Fitted {len(models)} models.")

    return {
        "models": models,
        "pred_hep": pred_hep,
        "pred_death": pred_death,
        "train_scores_hep": train_scores_hep,
        "train_scores_death": train_scores_death,
        "blend_weights_hep": blend_weights_hep,
        "blend_weights_death": blend_weights_death,
        "X_test": X_test,
    }


def save_models(result: dict, models_dir: str | Path = MODELS_DIR, processed_dir: str | Path = PROCESSED_DIR) -> None:
    """
    Saves each fitted Pipeline as models/{name}.joblib, plus the blend
    weights used and the training cohort's own blended risk scores.
    Copies feature_columns.json (written by 01, not regenerated here)
    alongside the models so a future webpage backend has everything it
    needs in one directory.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    for name, model in result["models"].items():
        joblib.dump(model, models_dir / f"{name}.joblib")

    joblib.dump(result["blend_weights_hep"], models_dir / "blend_weights_hep.joblib")
    joblib.dump(result["blend_weights_death"], models_dir / "blend_weights_death.joblib")
    joblib.dump(result["train_scores_hep"], models_dir / "train_scores_hepatic.joblib")
    joblib.dump(result["train_scores_death"], models_dir / "train_scores_death.joblib")

    feature_columns_src = Path(processed_dir) / "feature_columns.json"
    if feature_columns_src.exists():
        shutil.copy(feature_columns_src, models_dir / "feature_columns.json")


def build_submission(result: dict, processed_dir: str | Path = PROCESSED_DIR, output_path: str | Path = OUTPUT_PATH) -> pd.DataFrame:
    """
    Loads the test-set id column from `processed_dir/test_ids.json` (written
    by 01_data_exploration's save_test_ids(), in the same row order as
    test_X.parquet) rather than re-reading raw test.csv, which could drift
    out of row-order sync with `result["pred_hep"]`/`pred_death"]` if the
    raw file or the feature-building pipeline ever reorders rows.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ids, id_col_name = load_test_ids(processed_dir)

    submission = pd.DataFrame({
        id_col_name: ids,
        "risk_hepatic_event": result["pred_hep"],
        "risk_death": result["pred_death"],
    })

    submission["weighted_risk"] = 0.7 * submission["risk_hepatic_event"] + 0.3 * submission["risk_death"]
    submission = submission.sort_values("weighted_risk", ascending=False).reset_index(drop=True)

    submission.to_csv(output_path, index=False)
    return submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Train final LiverRisk models and write the submission CSV.")
    parser.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    parser.add_argument("--models-dir", default=str(MODELS_DIR))
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    result = train_final_models(processed_dir=args.processed_dir)
    save_models(result, models_dir=args.models_dir, processed_dir=args.processed_dir)

    submission = build_submission(result, processed_dir=args.processed_dir, output_path=args.output)

    print(f"Saved {len(submission)} predictions -> {args.output}")
    print(submission.head())


if __name__ == "__main__":
    main()
