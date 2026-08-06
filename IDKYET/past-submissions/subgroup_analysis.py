import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sksurv.metrics import concordance_index_censored

from ramp2_submission import MyLiverRiskModel


def compute_cindex(event, time, risk):
    event = pd.Series(event).fillna(0).astype(bool)
    time = pd.Series(time).copy()
    risk = pd.Series(risk).astype(float)

    # Fill missing times with the maximum observed time
    time = time.fillna(time.max()).astype(float)

    mask = ~(pd.isna(event) | pd.isna(time) | pd.isna(risk))

    if mask.sum() < 2:
        return np.nan

    event = event[mask].to_numpy()
    time = time[mask].to_numpy()
    risk = risk[mask].to_numpy()

    if len(np.unique(event)) < 2:
        return np.nan

    return concordance_index_censored(event, time, risk)[0]


def evaluate_subgroup(subgroup_df, subgroup_col, y, preds):
    results = []

    values = subgroup_df[subgroup_col].dropna().unique()

    for val in sorted(values):
        mask = subgroup_df[subgroup_col] == val

        n_patients = int(mask.sum())
        n_hepatic_events = int(y.loc[mask, "evenements_hepatiques_majeurs"].fillna(0).sum())
        n_deaths = int(y.loc[mask, "death"].fillna(0).sum())

        hepatic_cindex = compute_cindex(
            y.loc[mask, "evenements_hepatiques_majeurs"],
            y.loc[mask, "evenements_hepatiques_age_occur"],
            preds.loc[mask, "risk_hepatic_event"]
        )

        death_cindex = compute_cindex(
            y.loc[mask, "death"],
            y.loc[mask, "death_age_occur"],
            preds.loc[mask, "risk_death"]
        )

        results.append({
            "subgroup_variable": subgroup_col,
            "subgroup_value": val,
            "n_patients": n_patients,
            "n_hepatic_events": n_hepatic_events,
            "n_deaths": n_deaths,
            "hepatic_cindex": hepatic_cindex,
            "death_cindex": death_cindex
        })

    return pd.DataFrame(results)


def main():
    train = pd.read_csv("liverrisk/data/train.csv")

    target_cols = [
        "evenements_hepatiques_majeurs",
        "evenements_hepatiques_age_occur",
        "death",
        "death_age_occur"
    ]

    # Original features for the model
    X = train.drop(columns=target_cols).copy()
    y = train[target_cols].copy()

    # Separate dataframe for subgroup analysis only
    subgroups = X.copy()

    categorical_vars = [
    "T2DM",
    "Hypertension",
    "Dyslipidaemia",
    "bariatric_surgery",
    "gender"
    ]

    for col in categorical_vars:
        if col in subgroups.columns:
            subgroups[col] = subgroups[col].astype("Int64").astype(str)
        
    # Add derived subgroup columns here
    subgroups["AgeGroup"] = pd.qcut(
        subgroups["Age_v1"],
        q=2,
        labels=["younger", "older"],
        duplicates="drop"
    ).astype(str)

    subgroups["BMIGroup"] = pd.qcut(
        subgroups["BMI_v1"],
        q=2,
        labels=["lower_BMI", "higher_BMI"],
        duplicates="drop"
    ).astype(str)

    # Cross-validation for out-of-fold predictions
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    oof_preds = pd.DataFrame(
        index=X.index,
        columns=["risk_hepatic_event", "risk_death"],
        dtype=float
    )

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X), 1):
        print(f"Fold {fold}")

        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_train = y.iloc[train_idx]

        model = MyLiverRiskModel()
        model.fit(X_train, y_train)

        preds = model.predict(X_valid)

        oof_preds.loc[X_valid.index, "risk_hepatic_event"] = preds["risk_hepatic_event"].values
        oof_preds.loc[X_valid.index, "risk_death"] = preds["risk_death"].values

    # Overall metrics
    overall_hepatic_cindex = compute_cindex(
        y["evenements_hepatiques_majeurs"],
        y["evenements_hepatiques_age_occur"],
        oof_preds["risk_hepatic_event"]
    )

    overall_death_cindex = compute_cindex(
        y["death"],
        y["death_age_occur"],
        oof_preds["risk_death"]
    )

    print(f"Overall hepatic C-index: {overall_hepatic_cindex:.4f}")
    print(f"Overall death C-index:   {overall_death_cindex:.4f}")

    subgroup_vars = [
        "gender",
        "T2DM",
        "Hypertension",
        "Dyslipidaemia",
        "bariatric_surgery",
        "AgeGroup",
        "BMIGroup"
    ]

    all_results = []

    overall_row = pd.DataFrame([{
        "subgroup_variable": "overall",
        "subgroup_value": "all",
        "n_patients": len(X),
        "n_hepatic_events": int(y["evenements_hepatiques_majeurs"].fillna(0).sum()),
        "n_deaths": int(y["death"].fillna(0).sum()),
        "hepatic_cindex": overall_hepatic_cindex,
        "death_cindex": overall_death_cindex
    }])

    all_results.append(overall_row)

    for subgroup_col in subgroup_vars:
        if subgroup_col not in subgroups.columns:
            print(f"Skipping {subgroup_col}: column not found")
            continue

        subgroup_result = evaluate_subgroup(subgroups, subgroup_col, y, oof_preds)
        all_results.append(subgroup_result)

    results_df = pd.concat(all_results, ignore_index=True)

    print("\nSubgroup results:")
    print(results_df)

    results_df.to_csv("subgroup_results.csv", index=False)
    print("\nSaved results to subgroup_results.csv")


if __name__ == "__main__":
    main()