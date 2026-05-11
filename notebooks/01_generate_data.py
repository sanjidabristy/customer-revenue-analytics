"""
01_generate_data.py
-------------------
Synthetic B2B subscription dataset generator for the Customer Revenue & Retention
Analytics project.

Why this script exists
======================
Real B2B customer data is either proprietary or scrubbed to the point of
being useless, and the public datasets on Kaggle are over-analyzed. This
script generates a believable substitute: 8,000 customers, three subscription
tiers, three years of monthly billing, with churn/upgrade/downgrade behavior
that reflects realistic SaaS economics. The output is three CSVs that the
SQL and Python notebooks in this repo read directly.

How to run
==========
From the project root:

    python notebooks/01_generate_data.py

This writes three files into ../data/ relative to this script:
    - customers.csv       (~8,000 rows)
    - subscriptions.csv   (~10-12K rows, one per subscription period)
    - transactions.csv    (~150K rows, one per customer per billing month)

Dependencies: standard library only (csv, random, datetime, os). No pandas.
This is intentional so the script runs on any machine without setup.

Reproducibility: a fixed random seed makes the output deterministic. Re-running
this script will produce byte-identical CSVs.
"""

import csv
import os
import random
from datetime import date, timedelta

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42
random.seed(SEED)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_CUSTOMERS = 8000
START_DATE = date(2023, 1, 1)
END_DATE = date(2025, 12, 31)

# Subscription tiers and list prices (monthly USD)
TIERS = {
    "Starter":    {"price":   99, "monthly_churn": 0.035, "upgrade_prob": 0.012, "downgrade_prob": 0.000},
    "Growth":     {"price":  499, "monthly_churn": 0.018, "upgrade_prob": 0.008, "downgrade_prob": 0.003},
    "Enterprise": {"price": 2499, "monthly_churn": 0.007, "upgrade_prob": 0.000, "downgrade_prob": 0.004},
}

# Initial tier distribution (must sum to 1.0)
INITIAL_TIER_WEIGHTS = [("Starter", 0.55), ("Growth", 0.32), ("Enterprise", 0.13)]

INDUSTRIES = [
    ("Financial Services", 0.18),
    ("Technology",         0.22),
    ("Retail",             0.15),
    ("Healthcare",         0.12),
    ("Manufacturing",      0.11),
    ("Media",              0.10),
    ("Education",          0.07),
    ("Other",              0.05),
]

REGIONS = [
    ("North America", 0.52),
    ("EMEA",          0.27),
    ("APAC",          0.15),
    ("LATAM",         0.06),
]

SALES_CHANNELS = [
    ("Self-Serve", 0.48),
    ("Direct",     0.34),
    ("Partner",    0.18),
]

# Industry-level adjustments. Financial Services and Healthcare skew Enterprise
# and have lower churn (sticky, regulated buyers). Retail skews Starter and
# churns harder. These multipliers are applied to the base monthly churn rate.
INDUSTRY_CHURN_MULTIPLIER = {
    "Financial Services": 0.75,
    "Healthcare":         0.80,
    "Technology":         1.00,
    "Manufacturing":      0.95,
    "Media":              1.10,
    "Retail":             1.25,
    "Education":          1.15,
    "Other":              1.00,
}

INDUSTRY_TIER_BIAS = {
    # (Starter, Growth, Enterprise) tilt added to base weights
    "Financial Services": (-0.15,  0.00,  0.15),
    "Healthcare":         (-0.10,  0.00,  0.10),
    "Technology":         ( 0.00,  0.05, -0.05),
    "Manufacturing":      (-0.05,  0.05,  0.00),
    "Media":              ( 0.05,  0.00, -0.05),
    "Retail":             ( 0.10, -0.05, -0.05),
    "Education":          ( 0.10, -0.05, -0.05),
    "Other":              ( 0.00,  0.00,  0.00),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def weighted_choice(pairs):
    """Pick an element from [(value, weight), ...] proportional to weight."""
    total = sum(w for _, w in pairs)
    r = random.uniform(0, total)
    upto = 0
    for value, weight in pairs:
        if upto + weight >= r:
            return value
        upto += weight
    return pairs[-1][0]


def random_signup_date():
    """Signup distributed across the 3-year window with mild growth over time.
    Later quarters get slightly more signups (typical SaaS shape)."""
    total_days = (END_DATE - START_DATE).days
    # Quadratic bias toward later dates: r^0.8 puts more weight near the end.
    r = random.random() ** 0.8
    offset = int(r * total_days)
    return START_DATE + timedelta(days=offset)


def month_floor(d):
    """Return the first day of the month for a given date."""
    return date(d.year, d.month, 1)


def add_months(d, n):
    """Return a date n months after d, snapped to the first of the month."""
    m = d.month - 1 + n
    y = d.year + m // 12
    return date(y, m % 12 + 1, 1)


def pick_initial_tier(industry):
    """Apply industry bias to the base tier distribution."""
    bias = INDUSTRY_TIER_BIAS[industry]
    weights = [
        ("Starter",    max(0.01, INITIAL_TIER_WEIGHTS[0][1] + bias[0])),
        ("Growth",     max(0.01, INITIAL_TIER_WEIGHTS[1][1] + bias[1])),
        ("Enterprise", max(0.01, INITIAL_TIER_WEIGHTS[2][1] + bias[2])),
    ]
    return weighted_choice(weights)


def initial_discount(tier, channel):
    """Discount distribution depends on tier and channel."""
    if tier == "Starter":
        return random.choice([0, 0, 0, 0, 5, 10])
    if tier == "Growth":
        base = random.choice([0, 0, 5, 10, 10, 15])
        return base + (5 if channel == "Partner" else 0)
    # Enterprise: negotiated deals
    base = random.choice([5, 10, 10, 15, 15, 20, 25])
    return min(35, base + (5 if channel == "Direct" else 0))


# ---------------------------------------------------------------------------
# Step 1: Generate customers
# ---------------------------------------------------------------------------

print("Generating customers...")
customers = []
for i in range(1, NUM_CUSTOMERS + 1):
    customer_id = f"C{i:05d}"
    signup = random_signup_date()
    industry = weighted_choice(INDUSTRIES)
    region = weighted_choice(REGIONS)
    channel = weighted_choice(SALES_CHANNELS)
    tier = pick_initial_tier(industry)
    discount = initial_discount(tier, channel)

    customers.append({
        "customer_id":   customer_id,
        "signup_date":   signup.isoformat(),
        "industry":      industry,
        "region":        region,
        "sales_channel": channel,
        "initial_tier":  tier,
        "discount_pct":  discount,
    })

print(f"  -> {len(customers)} customers generated")

# ---------------------------------------------------------------------------
# Step 2: Simulate subscription lifecycle and transactions month by month
# ---------------------------------------------------------------------------
#
# For each customer, walk month by month from signup to END_DATE. Each month
# they pay (one transaction row). Each month they roll dice:
#   - churn?  -> end this subscription with end_reason=Churn, customer gone
#   - upgrade -> end this subscription with end_reason=Upgrade, start a new
#                subscription at the next tier
#   - downgrade -> end this subscription with end_reason=Downgrade, start a
#                  new subscription at the previous tier
# A subscription record captures one contiguous period at one tier.

print("Simulating subscription lifecycles and transactions...")

subscriptions = []
transactions = []
sub_counter = 0

TIER_ORDER = ["Starter", "Growth", "Enterprise"]

for cust in customers:
    signup = date.fromisoformat(cust["signup_date"])
    current_tier = cust["initial_tier"]
    discount = cust["discount_pct"]
    industry = cust["industry"]
    churn_mult = INDUSTRY_CHURN_MULTIPLIER[industry]

    sub_start = month_floor(signup)
    month_cursor = sub_start

    # Sentinel: if a customer is still active at END_DATE, end_reason=Active
    while month_cursor <= END_DATE:
        # Record a transaction for this month
        list_price = TIERS[current_tier]["price"]
        net = round(list_price * (1 - discount / 100.0), 2)
        transactions.append({
            "transaction_id": f"T{len(transactions)+1:07d}",
            "customer_id":    cust["customer_id"],
            "billing_month":  month_cursor.isoformat(),
            "tier":           current_tier,
            "list_price":     list_price,
            "discount_pct":   discount,
            "net_amount":     net,
        })

        # Decide what happens next month
        next_month = add_months(month_cursor, 1)
        if next_month > END_DATE:
            # Reached the data window edge; subscription stays active
            sub_counter += 1
            subscriptions.append({
                "subscription_id": f"S{sub_counter:06d}",
                "customer_id":     cust["customer_id"],
                "tier":            current_tier,
                "start_date":      sub_start.isoformat(),
                "end_date":        month_cursor.isoformat(),
                "end_reason":      "Active",
            })
            break

        churn_p     = TIERS[current_tier]["monthly_churn"] * churn_mult
        upgrade_p   = TIERS[current_tier]["upgrade_prob"]
        downgrade_p = TIERS[current_tier]["downgrade_prob"]

        roll = random.random()
        if roll < churn_p:
            # Churned. Close the subscription, customer exits.
            sub_counter += 1
            subscriptions.append({
                "subscription_id": f"S{sub_counter:06d}",
                "customer_id":     cust["customer_id"],
                "tier":            current_tier,
                "start_date":      sub_start.isoformat(),
                "end_date":        month_cursor.isoformat(),
                "end_reason":      "Churn",
            })
            break
        elif roll < churn_p + upgrade_p:
            # Upgraded. Close current subscription, open a new one next month.
            sub_counter += 1
            subscriptions.append({
                "subscription_id": f"S{sub_counter:06d}",
                "customer_id":     cust["customer_id"],
                "tier":            current_tier,
                "start_date":      sub_start.isoformat(),
                "end_date":        month_cursor.isoformat(),
                "end_reason":      "Upgrade",
            })
            idx = TIER_ORDER.index(current_tier)
            current_tier = TIER_ORDER[min(idx + 1, len(TIER_ORDER) - 1)]
            sub_start = next_month
        elif roll < churn_p + upgrade_p + downgrade_p:
            # Downgraded.
            sub_counter += 1
            subscriptions.append({
                "subscription_id": f"S{sub_counter:06d}",
                "customer_id":     cust["customer_id"],
                "tier":            current_tier,
                "start_date":      sub_start.isoformat(),
                "end_date":        month_cursor.isoformat(),
                "end_reason":      "Downgrade",
            })
            idx = TIER_ORDER.index(current_tier)
            current_tier = TIER_ORDER[max(idx - 1, 0)]
            sub_start = next_month

        month_cursor = next_month

print(f"  -> {len(subscriptions)} subscription periods")
print(f"  -> {len(transactions)} transactions")

# ---------------------------------------------------------------------------
# Step 3: Write CSVs
# ---------------------------------------------------------------------------

def write_csv(filename, rows, fieldnames):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> wrote {path}")


print("Writing CSVs...")
write_csv("customers.csv", customers,
          ["customer_id", "signup_date", "industry", "region",
           "sales_channel", "initial_tier", "discount_pct"])

write_csv("subscriptions.csv", subscriptions,
          ["subscription_id", "customer_id", "tier",
           "start_date", "end_date", "end_reason"])

write_csv("transactions.csv", transactions,
          ["transaction_id", "customer_id", "billing_month",
           "tier", "list_price", "discount_pct", "net_amount"])

print("Done.")
          