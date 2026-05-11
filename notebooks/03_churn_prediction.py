"""
03_churn_prediction.py
----------------------
Predict whether a customer will churn within their first 12 months, using
only information known at acquisition time.

Why this framing
================
A model that predicts "will this customer churn" using post-signup behavioral
features is mostly useless for the business - by the time those signals
appear, the customer is already drifting. The harder and more useful question
is: based on what we knew on day 1 (industry, region, tier, channel, discount),
can we score new customers for churn risk before they even start?

That framing also avoids data leakage: every feature in this model is locked
in at signup, so the prediction is fair.

Target definition
=================
churned_12m = 1 if the customer has an end_reason='Churn' subscription within
              12 months of signup, 0 otherwise.
We only include customers with at least 12 months of observation window, so
recent signups (where we wouldn't know yet) are excluded.

What this script does
=====================
1. Loads data, builds feature table + target
2. Splits into train/test
3. Fits a logistic regression with one-hot encoded categoricals
4. Reports accuracy, ROC AUC, classification report
5. Prints the most influential coefficients
6. Saves a feature-importance bar chart

Run from project root:
    python notebooks/03_churn_prediction.py

Dependencies: pandas, numpy, scikit-learn, matplotlib
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, confusion_matrix
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
ASSETS_DIR = os.path.join(HERE, "..", "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
customers = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"),
                        parse_dates=["signup_date"])
subs      = pd.read_csv(os.path.join(DATA_DIR, "subscriptions.csv"),
                        parse_dates=["start_date", "end_date"])

# ---------------------------------------------------------------------------
# Target: churn within 12 months of signup
# ---------------------------------------------------------------------------
DATA_END = pd.Timestamp("2025-12-31")
HORIZON_MONTHS = 12

# Only keep customers who signed up at least 12 months before data end so we
# can fairly observe whether they churned in the window.
customers["months_observed"] = (
    (DATA_END.year - customers["signup_date"].dt.year) * 12
    + (DATA_END.month - customers["signup_date"].dt.month)
)
eligible = customers[customers["months_observed"] >= HORIZON_MONTHS].copy()

# Look up the earliest churn (if any) per customer
churn_subs = subs[subs["end_reason"] == "Churn"].copy()
first_churn = (churn_subs.groupby("customer_id")["end_date"].min()
               .rename("first_churn_date").reset_index())
eligible = eligible.merge(first_churn, on="customer_id", how="left")

# Months from signup to first churn
eligible["months_to_churn"] = (
    (eligible["first_churn_date"].dt.year  - eligible["signup_date"].dt.year)  * 12
  + (eligible["first_churn_date"].dt.month - eligible["signup_date"].dt.month)
)
eligible["churned_12m"] = (
    eligible["months_to_churn"].fillna(999) <= HORIZON_MONTHS
).astype(int)

print(f"Eligible customers: {len(eligible)}")
print(f"Base 12-month churn rate: {100 * eligible['churned_12m'].mean():.2f}%")

# ---------------------------------------------------------------------------
# Features (all known at signup)
# ---------------------------------------------------------------------------
eligible["signup_year"]    = eligible["signup_date"].dt.year
eligible["signup_quarter"] = eligible["signup_date"].dt.quarter

CATEGORICAL = ["industry", "region", "sales_channel", "initial_tier"]
NUMERIC     = ["discount_pct", "signup_year", "signup_quarter"]

X = eligible[CATEGORICAL + NUMERIC]
y = eligible["churned_12m"]

# ---------------------------------------------------------------------------
# Pipeline: one-hot encode categoricals, scale numerics, fit logistic
# ---------------------------------------------------------------------------
# Handle sklearn version compatibility: newer sklearn uses 'sparse_output',
# older uses 'sparse'. Try the new param first.
try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocessor = ColumnTransformer([
    ("cat", ohe,             CATEGORICAL),
    ("num", StandardScaler(), NUMERIC),
])

model = Pipeline([
    ("prep", preprocessor),
    ("clf",  LogisticRegression(max_iter=1000, class_weight="balanced")),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n=== Evaluation on hold-out set ===")
print(f"Accuracy:   {accuracy_score(y_test, y_pred):.3f}")
print(f"ROC AUC:    {roc_auc_score(y_test, y_proba):.3f}")
print("\nConfusion matrix (rows=actual, cols=predicted):")
print(confusion_matrix(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

# ---------------------------------------------------------------------------
# Coefficient interpretation: which features push toward churn?
# ---------------------------------------------------------------------------
feature_names = (
    list(model.named_steps["prep"].named_transformers_["cat"]
         .get_feature_names_out(CATEGORICAL))
    + NUMERIC
)
coefs = model.named_steps["clf"].coef_[0]

coef_table = (pd.DataFrame({"feature": feature_names, "coef": coefs})
              .assign(abs_coef=lambda d: d["coef"].abs())
              .sort_values("abs_coef", ascending=False))

print("\nTop 15 features by absolute coefficient (positive = increases churn risk):")
print(coef_table.head(15).to_string(index=False))

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
top = coef_table.head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(10, 7))
colors = ["#d62728" if c > 0 else "#2ca02c" for c in top["coef"]]
ax.barh(top["feature"], top["coef"], color=colors)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Logistic regression coefficient  (positive = higher churn risk)")
ax.set_title("Top drivers of 12-month churn (at-signup features only)")
plt.tight_layout()
out = os.path.join(ASSETS_DIR, "churn_drivers.png")
plt.savefig(out, dpi=160, bbox_inches="tight")
print(f"\nSaved {out}")
