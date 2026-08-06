# Model Pipeline And Top 3 Models

## Pipeline overview IMPORTANT: This was run before the fix so scores will be lower, however it still gives a good indicator 



The current experiment pipeline is defined in [liverrisk/experiment.py]

1. Load `train.csv` and split it into:
   - `X`: all clinical and longitudinal features
   - `y`: hepatic event flag + time, death flag + time

2. Apply feature engineering.
   - The main setup uses `LongitudinalFeatureEngineer`.
   - Instead of using every visit column directly, it summarizes each biomarker trajectory into:
     - first value
     - last value
     - mean
     - max
     - visit count
     - trend per visit
   - It also derives `FIB-4` and `AST/ALT` ratio from the latest available labs.

3. Wrap each estimator in a two-head model.
   - One copy predicts hepatic risk.
   - One copy predicts death risk.
   - The final score is the competition score:
     - `0.7 * C-index(hepatic)`
     - `0.3 * C-index(death)`

4. Run 5-fold cross-validation.
   - The folds are stratified on the hepatic event label.
   - This helps class balance somewhat, but the task is still noisy because each fold only has about `9-10` hepatic positives.

5. Run grid search over many model families and hyperparameters.
   - The combined results are saved in [grid_results_combined.csv](<c:\Users\paabl\OneDrive\Documents\GitHub\mlc-sp26-Liver1\grid_results_combined.csv>).
   - The per-outcome results are saved in [grid_results.csv](<c:\Users\paabl\OneDrive\Documents\GitHub\mlc-sp26-Liver1\grid_results.csv>).

## Top 3 models

These are the top 3 **model families** from the combined grid search using the weighted competition score.

### 1. ExtraTreesClassifier

Best combined score: about `0.8694 +/- 0.0256`

Best config:

- `max_features=0.5`
- `min_samples_leaf=5`
- `n_estimators=1000`
- `max_depth=None` or `20` performed almost identically
- `class_weight=None`

Why it stands out:

- this is the strongest overall family by a clear margin
- it also dominates most of the top-ranked configurations, not just the single best row
- it is especially strong on the hepatic component, which matters most because hepatic has weight `0.7`

Per-outcome signal for the best config:

- hepatic: about `0.8791`
- death: about `0.8482`

Main takeaway:

- this is the best current default direction for the competition

### 2. RandomForestClassifier

Best combined score: about `0.8642 +/- 0.0307`

Best config:

- `max_features=0.5`
- `min_samples_leaf=5`
- `n_estimators=1000`
- `max_depth=None` or `20`
- `class_weight=None`

Why it is important:

- it is the closest competitor to ExtraTrees
- it performs very similarly on death
- it is slightly worse on hepatic, which is why it loses on the combined score

Per-outcome signal for the best combined config:

- hepatic: about `0.8734`
- death: about `0.8488`

Extra note:

- the best **death-only** model in the full results was also a RandomForest, but with `class_weight="balanced"` and shallower trees
- that suggests RandomForest may still be useful if the team wants a death-focused branch or ensemble

### 3. GradientBoostingClassifier

Best combined score: about `0.8513 +/- 0.0307`

Best config:

- `learning_rate=0.01`
- `max_depth=3`
- `n_estimators=500`
- `subsample=0.7`

Why it matters:

- it is the strongest non-forest classifier in the combined search
- it does not beat the tree ensembles, but it is still clearly competitive
- it may be worth keeping as an ensemble component because it learns in a different way from ExtraTrees and RandomForest

## Quick summary

- The current pipeline is: longitudinal feature summaries -> two-head model wrapper -> 5-fold CV -> weighted C-index scoring.
- The best model family is `ExtraTreesClassifier`.
- The second-best family is `RandomForestClassifier`.
- The third-best family is `GradientBoostingClassifier`.
- The winning pattern across the best forests is:
  - `max_features=0.5`
  - many trees
  - moderate regularization through `min_samples_leaf=5`
- The combined score is being driven mostly by hepatic performance, so small hepatic gains matter a lot.
