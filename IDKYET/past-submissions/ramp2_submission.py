import pandas as pd
import argparse
import numpy as np
from pathlib import Path
from liverrisk.model import LiverriskModel, score_model
from sksurv.ensemble import RandomSurvivalForest

OUTCOME_COLS = [
    "evenements_hepatiques_majeurs",
    "evenements_hepatiques_age_occur",
    "death",
    "death_age_occur",
]
ID_COLS = ["trustii_id", "trustii-id"]

class MyLiverRiskModel(LiverriskModel):
    def __init__(self):
        self.feature_columns = None
        self.hepatic_model = RandomSurvivalForest(
            n_estimators=100, 
            random_state=42,
            min_samples_leaf=15,
            n_jobs = -1,
            max_features = "sqrt"
            )
        self.death_model = RandomSurvivalForest(
            n_estimators=100, 
            random_state=42,
            min_samples_leaf=15,
            n_jobs = -1,
            max_features = "sqrt"
            )

    def process_inputs(self, X):
        X_processed = X.copy()
        numeric_cols = X_processed.select_dtypes(include=['number']).columns
        categorical_cols = X_processed.select_dtypes(exclude=['number']).columns
        X_processed[numeric_cols] = X_processed[numeric_cols].fillna(X_processed[numeric_cols].median())
        X_processed[categorical_cols] = X_processed[categorical_cols].fillna("missing")
        X_processed = pd.get_dummies(X_processed, columns=categorical_cols)

        if self.feature_columns is None:
            self.feature_columns = X_processed.columns
        else:
            X_processed = X_processed.reindex(columns=self.feature_columns, fill_value=0)

        return X_processed
        

    def fit(self, X, y):
        X_processed = self.process_inputs(X)

        y_hepatic = np.array(
            list(zip(
                y["evenements_hepatiques_majeurs"].fillna(0).astype(bool),
                y["evenements_hepatiques_age_occur"].fillna(y["evenements_hepatiques_age_occur"].max())
            )),
            dtype=[("event", bool), ("time", float)]
        )
        y_death = np.array(
            list(zip(
                y["death"].fillna(0).astype(bool),
                y["death_age_occur"].fillna(y["death_age_occur"].max())
            )),
            dtype=[("event", bool), ("time", float)]
        )
        self.hepatic_model.fit(X_processed, y_hepatic)
        self.death_model.fit(X_processed, y_death)

        return self

    def predict(self, X):
        X_processed = self.process_inputs(X)

        hepatic_pred = self.hepatic_model.predict(X_processed)
        death_pred = self.death_model.predict(X_processed)

        return pd.DataFrame({
            "risk_hepatic_event": hepatic_pred, 
            "risk_death": death_pred
        }, index=X.index)  


def make_trustii_submission(data_dir, output_path="my_submission.csv"):
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")

    X_train = train.drop(columns=OUTCOME_COLS)
    y_train = train[OUTCOME_COLS]

    id_col = next((col for col in ID_COLS if col in test.columns), None)
    if id_col is None:
        raise ValueError(f"test.csv must include one of these ID columns: {ID_COLS}")

    X_test = test.drop(columns=[id_col])

    model = MyLiverRiskModel()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    submission = pd.DataFrame({
        id_col: test[id_col].values,
        "risk_hepatic_event": predictions["risk_hepatic_event"].values,
        "risk_death": predictions["risk_death"].values,
    })
    submission.to_csv(output_path, index=False, encoding="UTF-8")
    return submission
    
""""
def run_report_analysis(model, data_dir, score):
    import os
    train_path = os.path.join(data_dir, "train.csv")
    if not os.path.exists(train_path):
        return
    
    data = pd.read_csv(train_path)
    y_cols = ["evenements_hepatiques_majeurs", "evenements_hepatiques_age_occur", "death", "death_age_occur"]
    y_true = data[y_cols]
    X_eval = data.drop(columns=y_cols)
    
    y_pred = model.predict(X_eval)
    
    # Calibration
    y_pred['risk_decile'] = pd.qcut(y_pred['risk_death'], 10, labels=False, duplicates='drop')
    calibration = y_true.assign(decile=y_pred['risk_decile']).groupby('decile')['death'].mean()
    
    print("\n" + "="*30)
    print(f"FINAL SCORE: {score:.4f}")
    print("="*30)
    print("\nCALIBRATION (Death Rate per Risk Decile):")
    print(calibration)
"""
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--submission-path", type=str, default=None)
    args = parser.parse_args()

    model = MyLiverRiskModel()
    score = score_model(model, args.data_dir)
    print(f"Score: {score:.4f}")

    if args.submission_path is not None:
        submission = make_trustii_submission(args.data_dir, args.submission_path)
        print(f"Saved Trustii submission to {args.submission_path}")
        print(f"Submission shape: {submission.shape}")

    #run_report_analysis(model, args.data_dir, score)


