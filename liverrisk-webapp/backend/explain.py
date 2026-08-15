"""
LiverRisk webapp backend -- per-patient SHAP explanation.

Computed live, per request, for the one uploaded patient being scored
in predict.py's /predict route -- never precomputed or cached, and
never applied to the training-cohort rankings (rankings.py is
unaffected).

The background sample and the training cohort's transformed features
are built once at startup (models_loader.py); only the actual
explainer.shap_values() call below happens per-request.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from models_loader import RSF_HEP_EXPLAINER, RSF_HEP_FEATURE_NAMES, RSF_HEP_PREPROCESSOR

# --------------------------------------------------------------------
# Turning a raw engineered column name (e.g. "num__ggt__span_years",
# the ColumnTransformer's "<num|cat>__<REPEATED_BASE>__<suffix>"
# convention -- see liverrisk/features.py) into a short, plain-language
# label (e.g. "GGT trend duration").
# --------------------------------------------------------------------
_BASE_LABELS = {
    "BMI": "BMI",
    "alt": "ALT",
    "ast": "AST",
    "bilirubin": "Bilirubin",
    "chol": "Cholesterol",
    "ggt": "GGT",
    "gluc_fast": "Fasting glucose",
    "plt": "Platelet count",
    "triglyc": "Triglycerides",
    "fibrotest_BM_2": "FibroTest",
    "aixp_aix_result_BM_3": "AIX-P",
    "fibs_stiffness_med_BM_1": "Liver stiffness (FibroScan)",
}

_SUFFIX_LABELS = {
    "first": "starting value",
    "last": "most recent value",
    "mean": "average",
    "delta": "change over time",
    "rel_change": "relative change",
    "slope_per_year": "trend (per year)",
    "age_last_measure": "age at last measurement",
    "span_years": "trend duration",
}


def _clean_feature_name(raw_name: str) -> str:
    """"num__ggt__span_years" -> "GGT trend duration"."""
    name = raw_name.split("__", 1)[-1] if raw_name.startswith(("num__", "cat__")) else raw_name
    parts = name.split("__", 1)
    base = parts[0]
    suffix = parts[1] if len(parts) > 1 else None

    base_label = _BASE_LABELS.get(base, base.replace("_", " ").title())
    if suffix is None:
        return base_label

    suffix_label = _SUFFIX_LABELS.get(suffix, suffix.replace("_", " "))
    return f"{base_label} {suffix_label}"


def _build_sentence(feature_names: list[str], shap_row: np.ndarray) -> str:
    top_idx = np.argsort(-np.abs(shap_row))[:3]
    clauses = []
    for idx in top_idx:
        label = _clean_feature_name(feature_names[idx])
        direction = "increased" if shap_row[idx] > 0 else "decreased"
        clauses.append(f"{label} ({direction} risk)")
    return "Top factors behind this patient's hepatic-event risk: " + ", ".join(clauses) + "."


def explain_patient(patient_row: pd.DataFrame) -> str:
    """
    Runs SHAP (KernelExplainer over rsf_hep, the model that drives the
    hepatic blend today) for a single uploaded patient and returns a
    plain-language sentence naming their top 3 features by absolute
    SHAP impact on the hepatic-event risk score.
    """
    print("[shap] computing per-patient hepatic-risk explanation...")
    start = time.perf_counter()

    patient_transformed = RSF_HEP_PREPROCESSOR.transform(patient_row)
    if hasattr(patient_transformed, "toarray"):
        patient_transformed = patient_transformed.toarray()
    patient_transformed = pd.DataFrame(patient_transformed, columns=RSF_HEP_FEATURE_NAMES)

    shap_values = np.asarray(RSF_HEP_EXPLAINER.shap_values(patient_transformed, silent=True)).reshape(-1)

    elapsed = time.perf_counter() - start
    print(f"[shap] explanation computed in {elapsed:.2f}s")

    return _build_sentence(RSF_HEP_FEATURE_NAMES, shap_values)
