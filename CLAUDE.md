# LiverRisk (ANNITIA) — CLAUDE.md

## What this is
Survival analysis competition predicting two outcomes in MASLD patients.
Metric: `0.7 * C-index(hepatic) + 0.3 * C-index(death)`.
Train: 1253 patients (47 hepatic events, 76 deaths — heavily censored).
Test: 423 patients, no labels (scored externally via Trustii — competition is closed).

## File layout
```
liverrisk/          # competition framework — DO NOT EDIT
  model.py          # LiverriskModel base class, score_model(), compute_score()
  experiment.py     # grid search, LongitudinalFeatureEngineer, MultiOutputWrapper
  data/             # train.csv, test.csv

src/                # competition base classes — DO NOT EDIT

past-submissions/   # all editable code lives here
  best_grid_submission.py   # current best model (EnsembleLiverRiskModel)
  ramp2_submission.py       # earlier RSF-based model
  grid_results.csv          # raw grid search results
  grid_results_combined.csv # per-family summary
  COMPETITION_ANALYSIS.md   # dataset notes and high-value improvement ideas
```
