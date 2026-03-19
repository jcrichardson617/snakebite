# -*- coding: utf-8 -*-

"""

Created on Thu Feb 12 10:15:32 2026

 

@author: RicharJa

"""

 

import pandas as pd

import numpy as np

 

from datasets import load_dataset

dataset = load_dataset("electricsheepafrica/snakebite-envenomation", "district_hospital")

df = dataset["train"].to_pandas()

 

df = pd.read_csv("downloads/snakebite_district_hospital.csv")

 

df = df[(df["dry_bite"] == 0)]

 

# ==========================================

# XGBoost model for snakebite outcome (survived vs died)

# Assumes you already have: df (pandas DataFrame)

# ==========================================

import re

import numpy as np

import pandas as pd

 

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.metrics import (

    classification_report,

    roc_auc_score,

    average_precision_score,

    confusion_matrix

)

from xgboost import XGBClassifier

 

# --------------------------

# 0) Basic config

# --------------------------

TEST_SIZE = 0.2

BINARY_AS_ONEHOT = False  # set True if you insist on OHE for binaries

 

# --------------------------

# 1) Standardize column names

# --------------------------

df.columns = (

    df.columns

      .str.strip()

      .str.replace(r"[^\w]+", "_", regex=True)

      .str.lower()

      .str.strip("_")

)

 

# --------------------------

# 2) Identify and drop leakage columns

#    - Drop any 'id' column

#    - Drop hospital length-of-stay fields (e.g., hospital_days, length_of_stay)

# --------------------------

drop_exact = set([

    "id", "hospital_days"

])

 

to_drop = set()

for col in df.columns:

    if col in drop_exact:

        to_drop.add(col)

 

# Keep a short human-checkable list for transparency

print("Dropping potential leakage columns:", sorted(to_drop))

df = df.drop(columns=list(to_drop), errors="ignore")

 

# --------------------------

# 3) Collapse Outcome -> survived/died

#    - Create y: 1 = survived, 0 = died

#    - Drop unknown/ambiguous outcomes

# --------------------------

# Try common variants; if your dataset uses a different column name, edit here:

possible_outcome_cols = [c for c in df.columns if c in ("outcome", "patient_outcome", "final_outcome")]

if not possible_outcome_cols:

    raise ValueError("Couldn't find an outcome column. Please rename it to 'outcome' or set outcome_col below.")

 

outcome_col = possible_outcome_cols[0]

 

# Normalize text

df[outcome_col] = df[outcome_col].astype(str).str.strip().str.lower()

 

survive_tokens = {"survived_full_recovery", "survived_with_disability", "survived_with_sequalae"}

death_tokens   = {"died"}

 

# Keep only rows that we can map confidently

mask_keep = df[outcome_col].isin(survive_tokens | death_tokens)

df = df.loc[mask_keep].copy()

 

y = df[outcome_col].apply(lambda s: 1 if s in survive_tokens else 0)

X = df.drop(columns=[outcome_col])

 

# --------------------------

# 4) Identify columns to encode

# --------------------------

# Categorical columns explicitly requested for OHE:

requested_ohe = [

    "occupation",

    "bite_location",

    "activity_at_bite",

    "traditional_treatment_first",

    "snake_species",

    "severity",

]

 

# Keep only those present in X

requested_ohe = [c for c in requested_ohe if c in X.columns]

 

# Detect binary-like columns (object/string or category with 2 unique values)

# We'll map common yes/no variants to 1/0.

def is_binary_series(s: pd.Series) -> bool:

    if s.dtype.kind in "biu":  # already numeric integer/bool

        # consider numeric with at most two unique values (0/1)

        vals = pd.unique(s.dropna())

        return len(vals) <= 2

    if s.dtype == "bool":

        return True

    if s.dtype == "object" or pd.api.types.is_categorical_dtype(s):

        vals = pd.unique(s.dropna().astype(str).str.strip().str.lower())

        return len(vals) <= 2 and all(v in {"0", "1", "yes", "no", "true", "false", "present", "absent"} for v in vals)

    return False

 

binary_cols = [c for c in X.columns if is_binary_series(X[c])]

# Remove any binary columns that are also in requested_ohe (we won't OHE them if BINARY_AS_ONEHOT=False)

if not BINARY_AS_ONEHOT:

    requested_ohe = [c for c in requested_ohe if c not in binary_cols]

 

# Map binary-like columns to 0/1 if they are not numeric already

def map_to_binary01(s: pd.Series) -> pd.Series:

    if s.dtype.kind in "biu" or s.dtype == "bool":

        # convert True/False to 1/0, ints stay as-is

        return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)

    # string/categorical mapping

    m = {

        "yes": 1, "true": 1, "present": 1, "1": 1,

        "no": 0,  "false": 0, "absent": 0,  "0": 0

    }

    return s.astype(str).str.strip().str.lower().map(m)

 

if not BINARY_AS_ONEHOT:

    for c in binary_cols:

        X[c] = map_to_binary01(X[c])

 

# Identify numeric columns (after binary mapping) and remaining categorical to OHE

numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]

# Categorical candidates = requested_ohe + other object/category columns not already binary-mapped

cat_candidate_cols = [c for c in X.columns if c not in numeric_cols]

# Make sure requested OHE are included and exist

ohe_cols = sorted(set(requested_ohe + [c for c in cat_candidate_cols

                                       if (X[c].dtype == "object" or pd.api.types.is_categorical_dtype(X[c]))

                                       and (BINARY_AS_ONEHOT or c not in binary_cols)]))

 

# Safety: ensure ohe_cols don’t include numeric

ohe_cols = [c for c in ohe_cols if c not in numeric_cols]

 

# Update numeric_cols in case some non-binary non-OHE columns remain numeric-like

numeric_cols = [c for c in X.columns if c not in ohe_cols]

 

print("Categorical (OHE) columns:", ohe_cols)

print("Binary columns (mapped to 0/1):", binary_cols if not BINARY_AS_ONEHOT else [])

print("Numeric columns (pass-through):", numeric_cols)

 

# --------------------------

# 5) Train/Validation split (stratified)

# --------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=TEST_SIZE, stratify=y

)

 

# --------------------------

# 6) Preprocessor + Model

# --------------------------

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

preprocess = ColumnTransformer(

    transformers=[

        ("ohe", ohe, ohe_cols),

        ("num", "passthrough", numeric_cols),

    ],

    remainder="drop"

)

 

# Class imbalance handling

pos = y_train.sum()

neg = len(y_train) - pos

scale_pos_weight = float(neg) / float(pos) if pos > 0 else 1.0

 

xgb = XGBClassifier(

    n_estimators=600,

    learning_rate=0.05,

    max_depth=4,

    subsample=0.8,

    colsample_bytree=0.8,

    reg_lambda=1.0,

    n_jobs=-1,

    tree_method="hist",

    eval_metric="logloss",

    scale_pos_weight=scale_pos_weight

)

 

clf = Pipeline(steps=[

    ("pre", preprocess),

    ("xgb", xgb)

])

 

# --------------------------

# 7) Fit

# --------------------------

clf.fit(X_train, y_train)

 

# --------------------------

# 8) Evaluate

# --------------------------

proba_test = clf.predict_proba(X_test)[:, 1]

pred_test = (proba_test >= 0.5).astype(int)

 

roc_auc = roc_auc_score(y_test, proba_test)

pr_auc = average_precision_score(y_test, proba_test)

cm = confusion_matrix(y_test, pred_test)

 

print("\n=== Metrics ===")

print(f"ROC-AUC: {roc_auc:.3f}")

print(f"PR-AUC : {pr_auc:.3f}")

print("\nConfusion Matrix [tn, fp; fn, tp]:\n", cm)

print("\nClassification Report:\n", classification_report(y_test, pred_test, digits=3))

 

# --------------------------

# 9) Feature importances (aligned to real feature names)

# --------------------------

# Get transformed feature names

ohe_feature_names = []

if ohe_cols:

    # get_feature_names_out requires scikit-learn >= 1.0

    ohe_feature_names = clf.named_steps["pre"].named_transformers_["ohe"].get_feature_names_out(ohe_cols).tolist()

 

feature_names = ohe_feature_names + numeric_cols

importances = clf.named_steps["xgb"].feature_importances_

 

feat_imp = (

    pd.Series(importances, index=feature_names)

      .sort_values(ascending=False)

)

 

print("\nTop 25 features by importance:")

print(feat_imp.head(25))
