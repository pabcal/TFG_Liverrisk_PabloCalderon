import argparse

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sksurv.ensemble import GradientBoostingSurvivalAnalysis

from liverrisk.experiment import LongitudinalFeatureEngineer
from liverrisk.model import LiverriskModel, score_model


class BestGridLiverRiskModel(LiverriskModel):
    """Best combined model from grid_results_combined.csv."""

    def __init__(self):
        self.fe = LongitudinalFeatureEngineer()
        self.base_estimator = ExtraTreesClassifier(
            n_estimators=1000,
            max_depth=None,
            max_features=0.5,
            min_samples_leaf=5,
            class_weight=None,
            random_state=42,
            n_jobs=-1,
        )
        self.hepatic_model = clone(self.base_estimator)
        self.death_model = clone(self.base_estimator)

    def process_inputs(self, X: pd.DataFrame):
        return self.fe.transform(X)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        self.fe.fit(X, y)
        X_processed = self.process_inputs(X)

        y_hepatic = y["evenements_hepatiques_majeurs"].fillna(0).astype(int).values
        y_death = y["death"].fillna(0).astype(int).values

        self.hepatic_model.fit(X_processed, y_hepatic)
        self.death_model.fit(X_processed, y_death)
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        X_processed = self.process_inputs(X)
        hepatic_pred = self.hepatic_model.predict_proba(X_processed)[:, 1]
        death_pred = self.death_model.predict_proba(X_processed)[:, 1]
        return pd.DataFrame(
            {
                "risk_hepatic_event": hepatic_pred,
                "risk_death": death_pred,
            },
            index=X.index,
        )


class ExtendedFeatureEngineer(LongitudinalFeatureEngineer):
    """Adds instability and missingness features on top of LongitudinalFeatureEngineer.

    Extra columns per lab (13 labs × 3 = 39 new columns):
      - variance          — spread across visits; high variance signals instability
      - range             — max minus min across visits
      - missingness_frac  — fraction of visit slots with no recorded value

    Extra global columns (2):
      - followup_span     — last_age minus first_age (distinct from visit count)
      - fib4_first        — FIB-4 at first available visit (complement to fib4_last
                            already computed by the parent)
    """

    def _row_first_from(self, X: pd.DataFrame, cols: list[str]) -> np.ndarray:
        if not cols:
            return np.full(len(X), np.nan)
        mat = X[cols].values.astype(float)
        valid = ~np.isnan(mat)
        return self._row_first(mat, valid)

    def _raw(self, X: pd.DataFrame) -> np.ndarray:
        base = super()._raw(X)
        extras: list[np.ndarray] = []

        for cols in self._lab_cols.values():
            if not cols:
                continue
            mat = X[cols].values.astype(float)
            valid = ~np.isnan(mat)
            n_valid = valid.sum(axis=1).astype(float)
            n_total = mat.shape[1]

            with np.errstate(all="ignore"):
                var = np.where(n_valid > 1, np.nanvar(mat, axis=1), np.nan)
                rng = np.where(
                    n_valid > 1,
                    np.nanmax(mat, axis=1) - np.nanmin(mat, axis=1),
                    np.nan,
                )
                miss_frac = 1.0 - n_valid / n_total

            extras.append(np.column_stack([var, rng, miss_frac]))

        # Follow-up span
        age_cols = self._lab_cols.get("age", [])
        if age_cols:
            mat = X[age_cols].values.astype(float)
            valid = ~np.isnan(mat)
            span = self._row_last(mat, valid) - self._row_first(mat, valid)
            extras.append(span.reshape(-1, 1))

        # FIB-4 at first available visit
        age_f = self._row_first_from(X, self._lab_cols.get("age", []))
        ast_f = self._row_first_from(X, self._lab_cols.get("ast", []))
        plt_f = self._row_first_from(X, self._lab_cols.get("plt", []))
        alt_f = self._row_first_from(X, self._lab_cols.get("alt", []))
        with np.errstate(all="ignore"):
            fib4_first = age_f * ast_f / (plt_f * np.sqrt(np.maximum(alt_f, 1e-8)))
            fib4_first = np.where(np.isinf(fib4_first), np.nan, fib4_first)
        extras.append(fib4_first.reshape(-1, 1))

        return np.hstack([base] + extras)


class OutcomeEnsemble:
    def __init__(self, classifier_estimators, survival_estimator):
        self.classifier_estimators = classifier_estimators
        self.survival_estimator = survival_estimator

    def _make_survival_target(self, event, time, last_age=None):
        event_arr = pd.Series(event).fillna(0).astype(bool).to_numpy()
        time_s = pd.Series(time)
        if last_age is not None:
            # Use each patient's last observed age as their censoring time instead of
            # the dataset-wide max, which over-estimates follow-up for most patients.
            # Fall back to global max only for the rare patient with no age visits.
            time_arr = time_s.fillna(pd.Series(last_age)).fillna(time_s.max()).to_numpy(dtype=float)
        else:
            time_arr = time_s.fillna(time_s.max()).to_numpy(dtype=float)
        return event_arr, time_arr

    def _structured_survival_target(self, event, time, last_age=None):
        event_arr, time_arr = self._make_survival_target(event, time, last_age)
        return np.array(
            list(zip(event_arr, time_arr)),
            dtype=[("event", bool), ("time", float)]
        )

    @staticmethod
    def _rank_average(predictions: list[pd.Series]) -> pd.Series:
        ranked = [pred.rank(method="average", pct=True) for pred in predictions]
        return sum(ranked) / len(ranked)

    def fit(self, X, y_class_target, y_survival_target) -> "OutcomeEnsemble":
        self.classifiers_ = [clone(est).fit(X, y_class_target) for est in self.classifier_estimators]
        self.survival_ = clone(self.survival_estimator).fit(X, y_survival_target)
        return self

    def predict(self, X, index) -> pd.Series:
        parts = [pd.Series(m.predict_proba(X)[:, 1], index=index) for m in self.classifiers_]
        parts.append(pd.Series(self.survival_.predict(X), index=index))
        return self._rank_average(parts)


class EnsembleLiverRiskModel(LiverriskModel):
    """Lower-variance ensemble of the strongest grid-search model families."""

    def __init__(self):
        self.fe = ExtendedFeatureEngineer()
        classifier_estimators = [
            ExtraTreesClassifier(
                n_estimators=1000,
                max_depth=20,
                max_features=0.5,
                min_samples_leaf=5,
                class_weight=None,
                random_state=42,
                n_jobs=-1,
            ),
            RandomForestClassifier(
                n_estimators=1000,
                max_depth=10,
                max_features=0.5,
                min_samples_leaf=5,
                class_weight=None,
                random_state=43,
                n_jobs=-1,
            ),
            GradientBoostingClassifier(
                n_estimators=500,
                learning_rate=0.01,
                max_depth=3,
                subsample=0.7,
                random_state=44,
            ),
        ]
        survival_estimator = GradientBoostingSurvivalAnalysis(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=3,
            subsample=0.7,
            random_state=45,
        )
        self.hepatic = OutcomeEnsemble(classifier_estimators, survival_estimator)
        self.death = OutcomeEnsemble(classifier_estimators, survival_estimator)

    def process_inputs(self, X: pd.DataFrame):
        return self.fe.transform(X)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        self.fe.fit(X, y)
        X_processed = self.process_inputs(X)

        age_cols = sorted(
            [c for c in X.columns if c.startswith("Age_v")],
            key=lambda c: int(c.rsplit("_v", 1)[1]),
        )
        last_age = X[age_cols].apply(
            lambda r: r.dropna().iloc[-1] if r.notna().any() else np.nan, axis=1
        )

        y_hepatic_class = y["evenements_hepatiques_majeurs"].fillna(0).astype(int).values
        y_death_class = y["death"].fillna(0).astype(int).values
        y_hepatic_surv = self.hepatic._structured_survival_target(
            y["evenements_hepatiques_majeurs"],
            y["evenements_hepatiques_age_occur"],
            last_age=last_age,
        )
        y_death_surv = self.death._structured_survival_target(
            y["death"],
            y["death_age_occur"],
            last_age=last_age,
        )

        self.hepatic.fit(X_processed, y_hepatic_class, y_hepatic_surv)
        self.death.fit(X_processed, y_death_class, y_death_surv)
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        X_processed = self.process_inputs(X)
        return pd.DataFrame(
            {
                "risk_hepatic_event": self.hepatic.predict(X_processed, X.index),
                "risk_death": self.death.predict(X_processed, X.index),
            },
            index=X.index,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    args = parser.parse_args()

    model = BestGridLiverRiskModel()
    score = score_model(model, args.data_dir)
    print(f"Score: {score:.4f}")
