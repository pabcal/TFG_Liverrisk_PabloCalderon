"""
Longitudinal feature engineering and survival-target preparation.

Moved verbatim (not rewritten) from ANNITIA_baseline_local.ipynb sections
4.3-4.9: get_age_cols, visit_cols, safe_div, first_non_null, last_non_null,
slope_against_age, age_of_last_measure, span_years_for_measure,
add_visit_level_features, build_patient_features, prepare_survival_target.

The only intentional behavior-preserving change: the three inline FIB-4
computations (per-visit, first, last) now call clinical_scores.fib4()
instead of repeating `safe_div(age * ast, plt * np.sqrt(alt))` -- same
formula, same result, single definition.

save_features()/load_features() and the feature_columns.json writer are
NEW, added for the notebook -> package split (01_data_exploration writes,
everything downstream reads).
"""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

from sksurv.util import Surv

from liverrisk.clinical_scores import fib4

# ---------------------------------------------------------------------
# Column families names from train.csv
# ---------------------------------------------------------------------

#Colums that dont repeat per visit. Fixed values like gender, t2dm...
STATIC_CAT = ["gender", "T2DM", "Hypertension", "Dyslipidaemia", "bariatric_surgery"]
STATIC_NUM = ["bariatric_surgery_age"]

#The prefixes of columns that do repeat per visit. For example alt_v1 alt_v2, alt_v3 etc
REPEATED_BASES = [
    "BMI",
    "alt",
    "ast",
    "bilirubin",
    "chol",
    "ggt",
    "gluc_fast",
    "plt",
    "triglyc",
    "fibrotest_BM_2",
    "aixp_aix_result_BM_3",
    "fibs_stiffness_med_BM_1",
]

TARGET_COLS = [
    "evenements_hepatiques_majeurs",
    "evenements_hepatiques_age_occur",
    "death",
    "death_age_occur",
]
ID_COLS = ["patient_id_anon", "trustii_id"]


"""
# ---------------------------------------------------------------------
# Visit column helpers. I do this because if i want to work with all of a patients
# age measurements or ALT measurements, I cant just say df["age"] or df["alt"]. 
# Its to figure out which of the 287 colums belong to that one lab across all visits.
# Sorted by visit.
# ---------------------------------------------------------------------
"""

# A helper function that when given a column like age_v22, it returns 
# just the number 22
def visit_number (column_name):
    # rsplit is like split but it splits from the right to the left. 
    # So if you have a column like age_v22, 
    # it will split it into ["age", "22"].
    # rsplit("_v", 1) means split at _v and only do it once
    parts = column_name.rsplit("_v", 1)
    number_as_text = parts[1]
    return int(number_as_text)


def get_age_cols(df: pd.DataFrame) -> list[str]:
    #loops through the df.columns one at a time. Only keep column c if meets the condition.
    #condition: refullmatch(pattern,text) -> re.fullmatch(r"Age_v\d+", c).
    # r"Age_v\d+ means match Age_v literally. Requires the string to have Age_v literally.
    # \d means any single digit, 0 - 9. + means one or more of what came before it.
    # d+ menas one or more digits in a row.

    #Important: Its just the names of the columns, no data
    cols = [c for c in df.columns if re.fullmatch(r"Age_v\d+", c)]
    return sorted(cols, key=visit_number)


def visit_cols(df: pd.DataFrame, base: str) -> list[str]:
    # same idea, but fr means f-string raw. So you can put variables in the string.
    # re.escape means to escape any special characters in the base string. 
    # just in case base has special characters like . or * or ? in it that are 
    # special in regex. re.escape will escape them so they are treated literally.
    cols = [c for c in df.columns if re.fullmatch(fr"{re.escape(base)}_v\d+", c)]
    return sorted(cols, key=visit_number)

"""
# ---------------------------------------------------------------------
# Basic longitudinal helpers
# ---------------------------------------------------------------------
"""
def safe_div(a, b):
    out = a / b
    if isinstance(out, pd.Series):
        out = out.replace([np.inf, -np.inf], np.nan)
    else:
        out = np.where(np.isfinite(out), out, np.nan)
    return out

"""
# the first non null is not always the first column as that patient could have
# skipped the first visit, but have visit 2 and 3. Same goes for last non null,
# its not always 22

# In the future, it is used like this: 
# cols = visit_cols(df, "alt")
# block = df[cols]
# X[f"alt__first"] = first_non_null(block)
# And this way we can get the first non null value for each patient for that lab.
"""

def first_non_null(block: pd.DataFrame) -> pd.Series:

    # bfill means backwards fill, which fils each NaN with whatever real value
    # comes next after it. For example, v1 = NaN v2 = NaN, v4 = 45, v4 = NaN, v5 = 52,
    # ater bfill, v1 = 45, v2 = 45, v3 = 45, v4 = 45, v5 = 52.
    # Then we just take the first column, which is now the first non null value.
    return block.bfill(axis=1).iloc[:, 0]

# Same as first_non_null, but backwards. Fills each NaN with whatever real value comes before it.
# and uses -1 to get the last column, which is now the last non null value.
def last_non_null(block: pd.DataFrame) -> pd.Series:
    return block.ffill(axis=1).iloc[:, -1]

"""
# A single lab value like ALT = 45 tells you a snapshot. We want to know how a patients
# liver is doing over time. I calulcate the slope over age and not against visit number
# because the visits are not evenly spaced. Some patients have 1 visit per year, 
# some have 1 visit every 5 years.
# a big jump between two visits 5 years apart doesn't mean the same thing as a 
# big jump between two visits 2 months apart
"""
def slope_against_age(block: pd.DataFrame, age_block: pd.DataFrame) -> pd.Series:

    # Block is one labs visit columns like ALT
    # age_block is the age columns. age_blocks's column N tells you the age at the N-th of block.

    # An array of one slot per patient, initialized to NaN. This will hold the slope for each patient.
    slopes = np.full(len(block), np.nan, dtype=float)

    # normal enumerate loop, pos is a count from 0 to len(block)-1, idx is the patients actual row label,
    #tells you which real patient you are working on.

    #block.index gives you the row labels of the block. so it will give you which patient does each row belong to,
    # it doesnt say anything about the actual values (like ALT values)

    # im workig with a table thats like:
    # row label --- alt_v1 --- alt_v2 --- alt_v3
    # 47 -----------50 --------- NaN --------- 60
    # 12 ----------- NaN ------- 45 ---------- 55
    # 5 ------------ 30 --------- 35 ---------- NaN

    # block.index returns [47, 12, 5] which are the row labels of the block.
    #idx is the address you give loc to get this specific person's data

    for pos, idx in enumerate(block.index):
        # when idx = 47 block.loc[idx] returns the row for patient 47, which is [50, NaN, 60]
        y = block.loc[idx].to_numpy(dtype=float)
        # grabs the same patients age at each of those same visits
        # From age_block, grab this one patient's row (idx), but only the 
        # first however-many columns that match how many visit-columns block has
        x = age_block.loc[idx, age_block.columns[: len(block.columns)]].to_numpy(dtype=float)

        #y is basically one patients lab values and x is the same patients age at those visits

        #checks if each age and value is a real number. & combines position by position, both need
        # to be true for the pair to be usable. Bascially the lab result (thats not NaN) must have an age (thats not NaN)
        valid = np.isfinite(x) & np.isfinite(y)
        # if the patient only has one valid pair of age and lab value, we cant calculate a slope, so skip this patient
        if valid.sum() < 2:
            continue

        #x[valid] means keep only the positions where valid was true.
        # x = [45, 46, 47, 49], valid = [True, True, False, True] → xv = [45, 46, 49] (position 2 dropped)
        # y = [30, 35, NaN, 38] → yv = [30, 35, 38] (same position dropped)
        # still matches up since the same position was dropped (which will always happen)
        
        xv = x[valid]
        yv = y[valid]
        # checks: are all the values in xv essentially equal to the very first one?.
        # cant compute slope if all the ages are the same, so skip this patient if they are.
        if np.allclose(xv, xv[0]):
            continue

        # centers so the average sits at 0
        xc = xv - xv.mean()

        # standard slope formula: slope = sum((x - x_mean) * (y - y_mean)) / sum((x - x_mean)^2)
        slopes[pos] = np.sum(xc * (yv - yv.mean())) / np.sum(xc ** 2)

    return pd.Series(slopes, index=block.index)

"""
# ---------------------------------------------------------------------
# Visit timing helpers
# ---------------------------------------------------------------------
"""

"""
#For one lab age of patient when the last non-null measurement was taken.
# I decided to do this becasue it tells me how current the data is 
"""
def age_of_last_measure(block: pd.DataFrame, age_block: pd.DataFrame) -> pd.Series:
    # one slot per patient, initialized to NaN.
    out = np.full(len(block), np.nan, dtype=float)

    for pos, idx in enumerate(block.index):
        #Patients whole row of lab values
        row = block.loc[idx]
        #row.notna() returns boolean array, true if not NaN.
        #row.index is the column names 
        # combined it goes through row.index and only keeps the enties where the 
        #corresponding position in row.notna() is True. So it returns the column names of the non-NaN values.

        #Important: remember row now has the data like this:
        # alt_v1    30.0
        # alt_v2    35.0
        # alt_v3     NaN
        # alt_v4    38.0
        #which is why .index returns ['alt_v1', 'alt_v2', 'alt_v3', 'alt_v4'] and row.notna() returns [True, True, False, True]

        valid_cols = row.index[row.notna()]
        #skip if the patient has no valid measurements for this lab
        if len(valid_cols) == 0:
            continue
        #grab last item in the list, will return something like alt_v4
        last_col = valid_cols[-1]
        #extract the visit number from the column name
        visit_num = int(last_col.rsplit("_v", 1)[1])
        # create the corresponding age column name for that visit
        age_col = f"Age_v{visit_num}"
        # check it exists
        if age_col in age_block.columns:
            # grab the patients age at that visit and store it in the output array
            out[pos] = age_block.loc[idx, age_col]

    return pd.Series(out, index=block.index)

"""
Finds first real measurement and last measurement and subtracts them
Tells me how many years did we actually get measurements for this lab

I could use age_of_last_measure but since I have to loop through the patients anyway,
 I might as well get the last measure here too
"""
def span_years_for_measure(block: pd.DataFrame, age_block: pd.DataFrame) -> pd.Series:
    out = np.full(len(block), np.nan, dtype=float)

    for pos, idx in enumerate(block.index):
        row = block.loc[idx]
        valid_cols = row.index[row.notna()]
        if len(valid_cols) == 0:
            continue

        first_visit = int(valid_cols[0].rsplit("_v", 1)[1])
        last_visit = int(valid_cols[-1].rsplit("_v", 1)[1])

        first_age_col = f"Age_v{first_visit}"
        last_age_col = f"Age_v{last_visit}"

        if first_age_col in age_block.columns and last_age_col in age_block.columns:
            #grabs value at row = idx and col = last_age_col to get the last age, same for the first
            out[pos] = age_block.loc[idx, last_age_col] - age_block.loc[idx, first_age_col]

    return pd.Series(out, index=block.index)

"""
# ---------------------------------------------------------------------
# Early visit features
# ---------------------------------------------------------------------
"""

def add_visit_level_features(X: pd.DataFrame, df: pd.DataFrame, max_visit: int = 4) -> None:
    """
    Preserve raw dense early visits and add ratios/FIB-4 per early visit.
    The first 4 visits are usually the most populated ones and clinically important for a 
    baseline assesment. This gives a way for the model to see the trends over the whole study
    as well as exact numbers from earlier visits
    Mutates X in place.
    """

    for v in range(1, max_visit + 1):
        age_col = f"Age_v{v}"

        for base in REPEATED_BASES:
            col = f"{base}_v{v}"
            if col in df.columns:
                #directly copies the raw column (e.g. alt_v1, unmodified) from the original data df 
                # straight into X, the features table being built.
                X[col] = df[col]

        #get everything we need to calculate the ratios and fib4 for this visit
        alt = df[f"alt_v{v}"] if f"alt_v{v}" in df.columns else np.nan
        ast = df[f"ast_v{v}"] if f"ast_v{v}" in df.columns else np.nan
        plt = df[f"plt_v{v}"] if f"plt_v{v}" in df.columns else np.nan
        ggt = df[f"ggt_v{v}"] if f"ggt_v{v}" in df.columns else np.nan
        age = df[age_col] if age_col in df.columns else np.nan

        #compute rations and fib4 FOR THE FIRST 4 VISITS
        if f"alt_v{v}" in df.columns and f"ast_v{v}" in df.columns:
            X[f"ast_alt_ratio_v{v}"] = safe_div(ast, alt)
        if f"ggt_v{v}" in df.columns and f"alt_v{v}" in df.columns:
            X[f"ggt_alt_ratio_v{v}"] = safe_div(ggt, alt)
        if f"ast_v{v}" in df.columns and f"plt_v{v}" in df.columns:
            X[f"ast_platelet_ratio_v{v}"] = safe_div(ast, plt)
        if all(c in df.columns for c in [age_col, f"ast_v{v}", f"alt_v{v}", f"plt_v{v}"]):
            X[f"fib4_v{v}"] = fib4(age, ast, plt, alt)


# ---------------------------------------------------------------------
# Patient feature engineering
# ---------------------------------------------------------------------
def build_patient_features(df: pd.DataFrame, include_followup_observation_features: bool = True) -> pd.DataFrame:
    """
    converts the 280+ wide columns and turns it into X, a denser table.
    This is done because models like Coxnet/RSF/XGB expect one fixed row per patient, with a fixed set of meaningful numbers.
    they can't directly understand 22 raw ALT readings, some missing.
    It does the following:
    1. Copy static info (gener, T2DM, etc) things that dont repeat per visit
    2. Add age/follow up info
    3. For every lab, compute summary features (__first, __last, __slope etc)
    4. Compute clinically meaningful derived features (AST/ALT ratio, FIB-4, etc)
    5. NIT availability 
    6. Interaction features
    7. Add early visit features from add_visit_level_features()
    8. cleanup

    Converts wide repeated-visit columns into dense patient-level features.

    include_followup_observation_features=True is leaderboard-friendly:
    it includes age_last_observed/followup/n_visits. This can be informative
    but may capture observation-process information.
    """

    #X shares the same row labels as df
    X = pd.DataFrame(index=df.index)

    age_cols = get_age_cols(df)
    if len(age_cols) == 0:
        raise ValueError("No Age_v* columns found.")

    #age_cols has the column names, but no values. We get the values from df (the argument)
    age_block = df[age_cols]

    # Static variables. Copy them into X if they exist.
    for col in STATIC_CAT + STATIC_NUM:
        if col in df.columns:
            X[col] = df[col]

    """
    Global age/follow-up features. Measures relating to the age of patient globally, not lab specific 
    """

    #Age at first visit
    X["age_baseline"] = df["Age_v1"]

    if include_followup_observation_features:
        #Find the max age across all visits for each patient. Not the as age of last measure
        #as age of last measure is the last age of a specific lab, but this is the last age across all labs.
        X["age_last_observed"] = age_block.max(axis=1)
        #counts how many visits actually have a real age recorded for each patient
        X["n_visits_age"] = age_block.notna().sum(axis=1)
        #total years the patient was actually followed for
        X["followup_years_observed"] = age_block.max(axis=1) - age_block.min(axis=1)

        if "bariatric_surgery_age" in df.columns:
            X["years_since_bariatric_at_baseline"] = df["Age_v1"] - df["bariatric_surgery_age"]
            X["years_since_bariatric_at_last"] = X["age_last_observed"] - df["bariatric_surgery_age"]

    # Longitudinal summaries
    for base in REPEATED_BASES:
        cols = visit_cols(df, base)
        if not cols:
            continue

        #grab the specific labs whole table from the passed in table
        block = df[cols]

        X[f"{base}__first"] = first_non_null(block)
        X[f"{base}__last"] = last_non_null(block)
        X[f"{base}__min"] = block.min(axis=1)
        X[f"{base}__max"] = block.max(axis=1)
        X[f"{base}__mean"] = block.mean(axis=1)
        X[f"{base}__median"] = block.median(axis=1)
        X[f"{base}__std"] = block.std(axis=1)
        X[f"{base}__count"] = block.notna().sum(axis=1)
        #patients missinginess rate
        X[f"{base}__miss_frac"] = block.isna().mean(axis=1)
        #how much did it change from first to last measurement. This is a good feature because it tells you 
        # if the patient is getting better or worse over time.
        X[f"{base}__delta"] = X[f"{base}__last"] - X[f"{base}__first"]
        #percentage of change from first to last measurement. 
        # the -1 makes it more intuitive. if last measure = 40 and first = 20, 40/20 = 2, 2-1 = 1, which is a 100% increase. 
        # if last measure = 10 and first = 20, 10/20 = 0.5, 0.5-1 = -0.5, which is a 50% decrease.
        X[f"{base}__rel_change"] = safe_div(X[f"{base}__last"], X[f"{base}__first"]) - 1.0
        #slope of the change per year
        X[f"{base}__slope_per_year"] = slope_against_age(block, age_block)

        if include_followup_observation_features:
            X[f"{base}__age_last_measure"] = age_of_last_measure(block, age_block)
            X[f"{base}__span_years"] = span_years_for_measure(block, age_block)

    # Clinically meaningful derived features
    required = {"ast__first", "alt__first", "ast__last", "alt__last", "ggt__last", "plt__last"}
    if required.issubset(X.columns):
        X["ast_alt_ratio_first"] = safe_div(X["ast__first"], X["alt__first"])
        X["ast_alt_ratio_last"] = safe_div(X["ast__last"], X["alt__last"])
        X["ggt_alt_ratio_last"] = safe_div(X["ggt__last"], X["alt__last"])
        X["ast_platelet_ratio_last"] = safe_div(X["ast__last"], X["plt__last"])

    if {"age_baseline", "ast__first", "plt__first", "alt__first"}.issubset(X.columns):
        X["fib4_first"] = fib4(X["age_baseline"], X["ast__first"], X["plt__first"], X["alt__first"])

    age_for_last = X["age_last_observed"] if "age_last_observed" in X.columns else X["age_baseline"]
    if {"ast__last", "plt__last", "alt__last"}.issubset(X.columns):
        X["fib4_last"] = fib4(age_for_last, X["ast__last"], X["plt__last"], X["alt__last"])

    # NIT burden / availability
    """
    nit_last_cols = [
        c for c in [
            "fibrotest_BM_2__last",
            "aixp_aix_result_BM_3__last",
            "fibs_stiffness_med_BM_1__last",
        ]
        if c in X.columns
    ]
    """
    possible_nit_columns = [
        "fibrotest_BM_2__last",
        "aixp_aix_result_BM_3__last",
        "fibs_stiffness_med_BM_1__last",
    ]
    nit_last_cols = []
    for column_name in possible_nit_columns:
        if column_name in X.columns:
            nit_last_cols.append(column_name)
    if nit_last_cols:
        #for each patient, how many of the NIT last measurements are not NaN. 
        #This tells you how many NITs were actually done for this patient.
        X["nit_available_count_last"] = X[nit_last_cols].notna().sum(axis=1)
        #on average, how high does this patient rank compared to other patients, across whichever NITs they do have?
        X["nit_last_rank_mean"] = X[nit_last_cols].rank(pct=True).mean(axis=1)

    # Metabolic interactions
    if "T2DM" in X.columns and "BMI__last" in X.columns:
        X["T2DM_x_BMI_last"] = X["T2DM"] * X["BMI__last"]
    if "T2DM" in X.columns and "gluc_fast__last" in X.columns:
        X["T2DM_x_glucose_last"] = X["T2DM"] * X["gluc_fast__last"]
    if "BMI__last" in X.columns and "triglyc__last" in X.columns:
        X["BMI_x_triglyc_last"] = X["BMI__last"] * X["triglyc__last"]

    add_visit_level_features(X, df, max_visit=4)

    # Clean infinities
    X = X.replace([np.inf, -np.inf], np.nan)

    # Keep only columns not completely empty
    X = X.dropna(axis=1, how="all")

    return X


# ---------------------------------------------------------------------
# Survival targets
# ---------------------------------------------------------------------
def prepare_survival_target(df: pd.DataFrame, outcome: str):
    """
    A survival model needs two things about the outcome: whether the event happened (event) 
    and how long until it happened (time) or until we lost track of them - censoring.
    This function builds that pair.

    We cant read this directly from the table because of how messy the data is.
        some rows say "event happened" but don't have an age recorded for when.
        some patients have completely unknown death status (21.5% of the training set).
    
    """
    df = df.copy()
    age_cols = get_age_cols(df)
    df["last_observed_age"] = df[age_cols].max(axis=1)

    if outcome == "hepatic":
        event_col = "evenements_hepatiques_majeurs"
        age_col = "evenements_hepatiques_age_occur"
        event_name = "Hepatic_event"

        #this finds the patients that had the event but dont have an age recorded for when it happened.
        #we cant use these patients.
        invalid = (df[event_col] == 1) & df[age_col].isna()
        #invalid marks the broken rows as true, so you want to keep the rows that are not true, hence the ~. 
        mask = ~invalid

    elif outcome == "death":
        event_col = "death"
        age_col = "death_age_occur"
        event_name = "Death"

        #this finds the patients with unknown death status
        unknown = df[event_col].isna()
        #this finds the patients that had the event but dont have an age recorded for when it happened.
        #we cant use these patients.
        invalid = (df[event_col] == 1) & df[age_col].isna()
        mask = ~unknown & ~invalid

    else:
        raise ValueError("outcome must be 'hepatic' or 'death'.")

    #filters the table with the mask
    # d is a new table containing the patients who pass the checks for the outcome requested
    d = df.loc[mask].copy()
    #d[event_col] == 1 — for every remaining patient, check "does their event column equal 1?
    #Result: event = a plain array of True/False, one per remaining patient — True meaning 
    #"this patient had the event," False meaning "censored" (no event observed).
    event = (d[event_col] == 1).astype(bool).to_numpy()
    #for everyone in the numpy array event, 
    #   if true: replace the value with how many years after baseline did the event happen
    #   if false: replace the value with how many years did we get to watch this patient before losing track of them, still event-free
    time = np.where(
        event,
        d[age_col].to_numpy(dtype=float) - d["Age_v1"].to_numpy(dtype=float),
        d["last_observed_age"].to_numpy(dtype=float) - d["Age_v1"].to_numpy(dtype=float),
    )
    #gurantees no value will be 0 as survival models need time > 0
    time = np.maximum(time.astype(float), 1e-3)

    #survival model need 2 seperate arrays like mentioned earlier, which in this case its event and time
    # the scikit-survival library expects one single object as the target not 2 seperate arguments for model.fit (X, y)

    #Surv.from_arrays is the tool that takes the two separate arrays and fuses them into one single object 
    # that still contains both pieces of information
    #y is something like: [( True, 2.5) (False, 8. ) ( True, 4.1)]
    #I dont use array of arrays because event and time are different types of data 
    # and they dont have names for each field like here where y["Hepatic_event"] y["Time_years"]
    #in an array of arrays it would be plain_version[:, 0] plain_version[:, 1] forcing you to remember 
    # where event or time is 
    y = Surv.from_arrays(event=event, time=time, name_event=event_name, name_time="Time_years")
    return d, y, event, time


# ---------------------------------------------------------------------
# Persistence: data/processed/ read + write (NEW)
# ---------------------------------------------------------------------
def save_features(X: pd.DataFrame, y, event: np.ndarray, time: np.ndarray, name: str, path: str | Path) -> None:
    """
    Save one endpoint's engineered features + survival target to
    `path/{name}_X.parquet` and `path/{name}_y.pkl`, and write
    `path/feature_columns.json` with X.columns (in order) -- overwritten
    each call, so it should reflect whichever call ran last (hepatic and
    death share the same X_all column space in the current pipeline).
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    X.to_parquet(path / f"{name}_X.parquet")

    with open(path / f"{name}_y.pkl", "wb") as f:
        pickle.dump({"y": y, "event": event, "time": time}, f)

    with open(path / "feature_columns.json", "w") as f:
        json.dump(list(X.columns), f, indent=2)


def load_features(name: str, path: str | Path):
    """Inverse of save_features(). Returns (X, y, event, time)."""
    path = Path(path)

    X = pd.read_parquet(path / f"{name}_X.parquet")

    with open(path / f"{name}_y.pkl", "rb") as f:
        payload = pickle.load(f)

    return X, payload["y"], payload["event"], payload["time"]


def load_feature_columns(path: str | Path) -> list[str]:
    path = Path(path)
    with open(path / "feature_columns.json") as f:
        return json.load(f)


def save_test_features(X_test: pd.DataFrame, path: str | Path) -> None:
    """Save the (unlabeled) test-set features to `path/test_X.parquet`."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    X_test.to_parquet(path / "test_X.parquet")


def load_test_features(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_parquet(path / "test_X.parquet")


def save_test_ids(test_df: pd.DataFrame, path: str | Path) -> None:
    """
    Save the test set's id column (`trustii_id` if present, else
    `patient_id_anon`) to `path/test_ids.json`, in the same row order as
    `save_test_features`'s `test_X.parquet` -- `test_df` must be the same
    (unfiltered, unreordered) frame passed to `build_patient_features()`.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    id_col = "trustii_id" if "trustii_id" in test_df.columns else "patient_id_anon"

    with open(path / "test_ids.json", "w") as f:
        json.dump({"id_col": id_col, "ids": test_df[id_col].tolist()}, f, indent=2)


def load_test_ids(path: str | Path) -> tuple[list, str]:
    """Inverse of save_test_ids(). Returns (ids, id_col_name)."""
    path = Path(path)
    with open(path / "test_ids.json") as f:
        payload = json.load(f)
    return payload["ids"], payload["id_col"]
