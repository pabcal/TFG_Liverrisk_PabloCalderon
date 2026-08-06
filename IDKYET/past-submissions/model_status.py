import argparse
import importlib.util
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from liverrisk.model import compute_score


OUTCOME_COLS = [
    "evenements_hepatiques_majeurs",
    "evenements_hepatiques_age_occur",
    "death",
    "death_age_occur",
]


def ensure_runtime_dependencies() -> None:
    if importlib.util.find_spec("sksurv") is None:
        raise SystemExit(
            "Missing dependency: scikit-survival (`sksurv`).\n"
            "Install project dependencies with `uv sync`, then run:\n"
            "  uv run python model_status.py --data-dir liverrisk/data\n"
            "If you are not using uv, install it manually with:\n"
            "  pip install scikit-survival"
        )


def get_model_class():
    from best_grid_submission import EnsembleLiverRiskModel

    return EnsembleLiverRiskModel


def load_training_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    X = train[[c for c in train.columns if c not in OUTCOME_COLS]].copy()
    y = train[OUTCOME_COLS].copy()
    return X, y


def make_trustii_submission(data_dir: str | Path, output_path: str | Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")

    X_train = train[[c for c in train.columns if c not in OUTCOME_COLS]].copy()
    y_train = train[OUTCOME_COLS].copy()

    id_col = next((col for col in ["trustii_id", "trustii-id"] if col in test.columns), None)
    if id_col is None:
        raise ValueError("test.csv must contain `trustii_id` or `trustii-id`.")

    X_test = test.drop(columns=[id_col]).copy()

    MyLiverRiskModel = get_model_class()
    model = MyLiverRiskModel()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    submission = pd.DataFrame(
        {
            id_col: test[id_col].values,
            "risk_hepatic_event": preds["risk_hepatic_event"].values,
            "risk_death": preds["risk_death"].values,
        }
    )
    submission.to_csv(output_path, index=False, encoding="UTF-8")
    return submission


def compute_training_score(X: pd.DataFrame, y: pd.DataFrame) -> float:
    MyLiverRiskModel = get_model_class()
    model = MyLiverRiskModel()
    model.fit(X, y)
    preds = model.predict(X)
    return compute_score(y, preds)


def compute_validation_score(
    X: pd.DataFrame,
    y: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[float, list[float]]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_preds = pd.DataFrame(index=X.index, columns=["risk_hepatic_event", "risk_death"], dtype=float)
    fold_scores: list[float] = []
    MyLiverRiskModel = get_model_class()

    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X, y["evenements_hepatiques_majeurs"].fillna(0)), start=1):
        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        model = MyLiverRiskModel()
        model.fit(X_train, y_train)
        fold_preds = model.predict(X_valid)

        oof_preds.loc[X_valid.index, ["risk_hepatic_event", "risk_death"]] = fold_preds[
            ["risk_hepatic_event", "risk_death"]
        ].values

        fold_score = compute_score(y_valid, fold_preds)
        fold_scores.append(fold_score)
        print(f"Fold {fold_idx}: {fold_score:.4f}")

    return compute_score(y, oof_preds), fold_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute current training and validation scores.")
    parser.add_argument("--data-dir", type=str, default="liverrisk/data")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--submission-path", type=str, default=None)
    args = parser.parse_args()

    ensure_runtime_dependencies()
    X, y = load_training_data(args.data_dir)

    training_score = compute_training_score(X, y)
    validation_score, fold_scores = compute_validation_score(
        X,
        y,
        n_splits=args.n_splits,
        random_state=args.random_state,
    )

    print("\nModel Status")
    print(f"Training score:   {training_score:.4f}")
    print(f"Validation score: {validation_score:.4f}")
    print(f"Mean fold score:  {sum(fold_scores) / len(fold_scores):.4f}")

    if args.submission_path is not None:
        submission = make_trustii_submission(args.data_dir, args.submission_path)
        print(f"\nSaved Trustii submission to {args.submission_path}")
        print(f"Submission shape: {submission.shape}")


if __name__ == "__main__":
    main()
