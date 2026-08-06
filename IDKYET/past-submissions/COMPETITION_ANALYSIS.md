# LiverRisk Competition Analysis Notes

## Metric reminder

This competition is not standard accuracy. The repo and scorer use a weighted concordance index (C-index):

- `0.7 * C-index(hepatic)`
- `0.3 * C-index(death)`

So the goal is to **rank patients correctly by risk and event time**.

## Outcome counts observed in train.csv

- `hep=0, death=0`: `873`
- `hep=0, death=1`: `65`
- `hep=1, death=0`: `35`
- `hep=1, death=1`: `11`
- `hep=0, death missing`: `268`
- `hep=1, death missing`: `1`

Totals:

- Observed hepatic events: `47`
- Observed deaths: `76`
- Rows with missing death label: `269`

### 1. Death is associated more with fibrosis severity and age than with ALT alone

Compared with `hep=0, death=0`, the `hep=0, death=1` group is:

- older at first and last observation
- higher on `fibrotest`
- higher on `FIB-4`
- higher on liver stiffness
- higher on `GGT`
- higher on fasting glucose
- lower on platelets

Practical takeaway:

- prioritize fibrosis and trajectory features over raw single ALT values
- especially watch `fibrotest`, `liver stiffness`, `FIB-4`, `GGT`, platelets, and age

### 2. The sickest subgroup is `hep=1, death=1`

Compared with `hep=0, death=1`, the `hep=1, death=1` group looks much more liver-severe:

- much higher `GGT`
- much higher liver stiffness
- much higher `fibrotest`
- much higher `FIB-4`
- lower platelets

Practical takeaway:

- the combined endpoint seems driven by a strong advanced liver disease signal
- hepatic-event modeling and death modeling should probably share many fibrosis-heavy features

### 3. Hepatic-event patients show stronger liver-specific severity than the overall death group

The strongest hepatic-event-associated features were:

- `FIB-4`
- `fibrotest`
- liver stiffness
- `AIX`
- `AST`

Practical takeaway:

- the hepatic target seems more tightly tied to direct liver severity markers than the death target
- since the competition weights hepatic risk at `0.7`, this part deserves extra attention

## Censoring and missingness

### Important distinction

There are **two different things** in this dataset:

1. **Observed non-event / right censoring**
   - patient has no observed event
   - we know they were event-free up to some age

2. **Missing death label**
   - `death` itself is `NaN`

### What the death-missing rows look like

The `death missing` rows do **not** look like the observed death group overall.

Clinically, their medians are closer to `hep=0, death=0` than to `hep=0, death=1` for:

- `GGT`
- platelets
- `FIB-4`
- liver stiffness

But they look very different administratively:

- median `visit_count` is only `2`
- median follow-up span is only `1` year
- `112 / 269` have exactly `1` visit
- around `89%` to `92%` are also missing static history fields like:
  - `T2DM`
  - `Hypertension`
  - `Dyslipidaemia`
  - `bariatric_surgery`

Interpretation:

- most `death missing` rows are more consistent with **incomplete follow-up / incomplete abstraction / administrative missingness**
- they do **not** look like a block of hidden deaths overall

### Most suspicious death-missing patients

A small subset of `death missing` rows does look clinically similar to observed deaths. Examples include:

- `C2RH2XK2PTHO`
  - high bilirubin
  - high `GGT`
  - high `fibrotest`
  - very high glucose
- `06NKWFG3MWU4`
  - high bilirubin
  - very high `GGT`
  - high liver stiffness
  - high `fibrotest`
  - very high glucose
- `9Q44DVVPGB8I`
  - older age
  - low platelets
  - high liver stiffness
  - high `FIB-4`
- `EV6XGZ7CTM09`
  - older age
  - high `GGT`
  - high `fibrotest`
  - high liver stiffness
- `T82DDL7FKNDG`
  - older age
  - high bilirubin
  - high `fibrotest`
  - high glucose

Interpretation:

- most missing-death rows should not be assumed dead
- but a small subset probably belongs to a genuinely high-risk population

### The lone `hep=1, death missing` case

There is exactly one row with:

- `hep=1`
- `death missing`

Patient id:

- `60V33LIFNCRN`

Observed pattern:

- hepatic event age = `71`
- only `1` recorded visit
- last observed age = `74`


## Timing / visit structure observations

The dataset uses **ages at visits**

Still, there are useful time-structure patterns:

- observed deaths happen about `3.8` years after the last recorded visit on average
- median death lag from last visit is about `3` years
- hepatic events often occur near the last visit, and sometimes even before the final recorded age
- visit gaps are roughly annual in many patients

Practical takeaway:

- using age trajectories and follow-up length is likely useful
- last observed age is informative, but it can also reflect administrative follow-up length rather than biology

## High-value competition tips

These are the highest-leverage ideas from the current analysis.

### 1. Do not train on fake deaths created by missing labels

This was an important bug risk in the repo:

- `NaN.astype(bool)` becomes `True`
- that means missing `death` labels can silently become deaths

This has already been fixed in:

- `liverrisk/experiment.py`
- `liverrisk/model.py`
- `ramp2_submission.py`

Practical takeaway:

- always use `fillna(0).astype(bool)` or explicit masking before boolean conversion

### 2. Treat censoring carefully in training

For survival models, the censoring time should ideally reflect the patient's **last observed age**, not a dataset-wide constant.

High-value idea:

- use `last_age` as the censoring time for patients without an observed event
- compare this against the current simpler strategy in CV

Why this matters:

- it preserves real follow-up information
- it avoids pretending every censored patient was followed equally long

### 3. Build longitudinal summary features, not just raw visit columns

Raw visit columns are very sparse.

The current `LongitudinalFeatureEngineer` direction is good because it captures:

- first value
- last value
- mean
- max
- visit count
- per-visit trend

This should be extended rather than replaced.

High-value additions:

- per-patient missingness indicators
- time-span features
- last minus first
- max minus min
- instability features such as variance or range

### 4. Prioritize fibrosis-heavy features

Most useful feature families from the analysis:

- liver stiffness
- `fibrotest`
- `FIB-4`
- platelets
- `GGT`
- bilirubin
- fasting glucose
- age

Suggested derived features:

- `FIB-4` at first and last visit
- slope of `fibrotest`
- slope of liver stiffness
- platelet decline
- `AST / ALT`
- combined fibrosis burden score

### 5. Model hepatic and death risk separately, then combine

Because:

- the competition metric is weighted
- the hepatic endpoint is more liver-specific
- death risk may include both liver severity and broader frailty / age signal

Recommended direction:

- keep two risk heads or two models
- optimize hepatic ranking especially carefully because it has weight `0.7`

### 6. Separate biological risk from follow-up missingness

`visit_count`, `followup_span`, and missing history fields are useful, but dangerous.

Why:

- they may reflect data collection patterns rather than disease severity
- they can help CV if train and test share the same artifact
- they can also hurt generalization if the missingness pattern shifts

Recommended direction:

- test models with and without these administrative features
- inspect whether gains are stable across folds
- do not rely only on short-follow-up patterns

### 7. Validate subgroup performance, not just overall score

Important subgroups:

- older vs younger
- low vs high BMI
- `T2DM`
- hypertension
- dyslipidaemia
- sparse follow-up vs dense follow-up

Why:

- the event counts are small
- overfitting can hide inside one subgroup while looking good overall

### 8. Consider survival-native models first

Most promising model families to test seriously:

- Random Survival Forest
- Gradient Boosting Survival Analysis
- Extra Survival Trees

Also worth testing:

- separate calibrated classifiers for event indicators
- ensembling survival and non-survival models

### 9. Use out-of-fold analysis for every major change

Because the event counts are low:

- tiny preprocessing mistakes can move CV a lot
- fold leakage is a serious risk

Recommended workflow:

- generate out-of-fold predictions
- compare hepatic and death C-index separately
- only trust improvements that repeat across seeds / folds

## Best current working interpretation

The strongest current interpretation is:

- liver fibrosis severity is the core signal for both endpoints
- the hepatic endpoint is the more liver-pure target
- many `death missing` rows are probably **censored or administratively incomplete**, not dead
- a small subset of missing-death patients looks genuinely high risk
- handling censoring and missing labels correctly is likely one of the easiest ways to avoid losing score

## next steps

1. Re-run cross-validation after the missing-label boolean fix.
2. Compare censoring time strategies:
   - current fill-with-global-max strategy
   - censor at `last_age`
3. Add more fibrosis trajectory features.
4. Create a separate report of:
   - high-risk `death missing` patients
   - likely administratively incomplete patients
5. Tune the hepatic model more aggressively than the death model because of the `0.7 / 0.3` weighting.
