"""
02_cohort_analysis.py
---------------------
Cohort retention analysis with a heatmap visualization.

What this script does
=====================
Reads customers.csv and transactions.csv, builds the same monthly cohort
retention table that sql/01_cohort_retention.sql produces, then plots it as
a heatmap. Saves the figure to ../assets/cohort_retention.png.

Why a heatmap
=============
A retention table with 30+ cohort rows and 30+ tenure columns is unreadable
as a number grid. A heatmap turns the same table into something a stakeholder
absorbs in three seconds: deeper red = stronger retention, white-yellow = where
customers are dropping off. It also makes seasonal patterns visible (e.g.,
Q4 cohorts retaining better than Q1 cohorts).

Run from project root:
    python notebooks/02_cohort_analysis.py

Dependencies: pandas, numpy, matplotlib
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
ASSETS_DIR = os.path.join(HERE, "..", "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
customers = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"),
                        parse_dates=["signup_date"])
tx = pd.read_csv(os.path.join(DATA_DIR, "transactions.csv"),
                 parse_dates=["billing_month"])

# ---------------------------------------------------------------------------
# Build cohort table
# ---------------------------------------------------------------------------
# Cohort month = first day of the customer's signup month
customers["cohort_month"] = customers["signup_date"].values.astype("datetime64[M]")

tx = tx.merge(customers[["customer_id", "cohort_month"]], on="customer_id")
tx["active_month"] = tx["billing_month"].values.astype("datetime64[M]")

# Tenure month = months between cohort and activity
def months_between(later, earlier):
    return ((later.dt.year - earlier.dt.year) * 12
            + (later.dt.month - earlier.dt.month))

tx["tenure_month"] = months_between(tx["active_month"], tx["cohort_month"])

# Distinct (customer, cohort, tenure) so multiple bills in one month don't
# double-count
unique_activity = tx[["customer_id", "cohort_month", "tenure_month"]].drop_duplicates()

retained = (unique_activity
            .groupby(["cohort_month", "tenure_month"])
            .customer_id.nunique()
            .reset_index(name="active"))

# Cohort size = retained at tenure 0
cohort_size = (retained[retained["tenure_month"] == 0]
               [["cohort_month", "active"]]
               .rename(columns={"active": "cohort_size"}))

retained = retained.merge(cohort_size, on="cohort_month")
retained["retention_pct"] = (100.0 * retained["active"] / retained["cohort_size"]).round(1)

# Pivot for the heatmap
pivot = retained.pivot(index="cohort_month",
                       columns="tenure_month",
                       values="retention_pct")

# Limit to a reasonable size for plotting (first 24 tenure months)
MAX_TENURE = 24
pivot = pivot.loc[:, [c for c in pivot.columns if c <= MAX_TENURE]]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 12))

# Show retention values inside cells. Mask NaN cells so they appear white.
data = pivot.values
masked = np.ma.masked_invalid(data)

cmap = plt.cm.RdYlGn
cmap.set_bad(color="white")

im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=0, vmax=100)

# Tick labels
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns, fontsize=9)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels([d.strftime("%Y-%m") for d in pivot.index], fontsize=9)

# Numbers inside cells (only where data exists and value is meaningful)
for i in range(masked.shape[0]):
    for j in range(masked.shape[1]):
        v = masked[i, j]
        if not np.ma.is_masked(v):
            color = "white" if v < 40 or v > 80 else "black"
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    fontsize=7, color=color)

ax.set_xlabel("Tenure month (0 = signup month)", fontsize=11)
ax.set_ylabel("Signup cohort", fontsize=11)
ax.set_title("Monthly cohort retention (% of cohort still paying)", fontsize=13, pad=12)

cbar = fig.colorbar(im, ax=ax, shrink=0.7)
cbar.set_label("Retention %", fontsize=10)

plt.tight_layout()
out = os.path.join(ASSETS_DIR, "cohort_retention.png")
plt.savefig(out, dpi=160, bbox_inches="tight")
print(f"Saved {out}")

# ---------------------------------------------------------------------------
# Print headline numbers for the README / memo
# ---------------------------------------------------------------------------
# Average retention by tenure month, weighted across all cohorts that
# reached that tenure
avg_curve = (retained.groupby("tenure_month")
             .apply(lambda g: (g["active"].sum() / g["cohort_size"].sum()) * 100)
             .reset_index(name="avg_retention_pct"))

print("\nAverage retention curve (all cohorts, weighted):")
print(avg_curve.head(25).to_string(index=False))
