"""
ATM Fraud Detection - Synthetic Data Generator
=============================================
Generates realistic ATM transaction data with fraud patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Seed for reproducibility
np.random.seed(42)
random.seed(42)

# ─── Configuration ────────────────────────────────────────────────────────────
ATM_LOCATIONS = [
    "Downtown Branch",   "Airport Terminal",  "Shopping Mall",
    "University Campus", "Gas Station",       "Hotel Lobby",
    "Hospital",          "Train Station",     "Suburban Branch",
    "Remote Village",
]

CARD_TYPES     = ["Visa", "MasterCard", "Amex", "RuPay"]
FAILURE_REASON = ["Wrong PIN", "Insufficient Funds", "Card Blocked",
                  "Network Error", "Card Expired"]


def random_timestamp(start: datetime, end: datetime) -> datetime:
    delta = end - start
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return start + timedelta(seconds=random_seconds)


def generate_transaction(transaction_id: int, is_fraud: bool = False) -> dict:
    """
    Build a single ATM transaction record.

    Fraud patterns injected:
    - High withdrawal amounts
    - Odd hours (1 AM – 4 AM)
    - Multiple rapid PIN failures
    - Geographic impossibility (high velocity)
    - Unusual locations
    """
    start = datetime(2024, 1, 1)
    end   = datetime(2024, 12, 31)
    ts    = random_timestamp(start, end)
    hour  = ts.hour

    if is_fraud:
        # Pattern 1 – large night withdrawal
        if random.random() < 0.4:
            amount       = round(random.uniform(800, 5000), 2)
            hour         = random.randint(1, 4)
            pin_failures = random.randint(2, 5)
            location     = random.choice(["Airport Terminal", "Remote Village"])
            ts           = ts.replace(hour=hour)
        # Pattern 2 – rapid repeated transactions
        elif random.random() < 0.3:
            amount             = round(random.uniform(200, 500), 2)
            hour               = random.randint(0, 6)
            pin_failures       = random.randint(0, 2)
            location           = random.choice(ATM_LOCATIONS)
            ts                 = ts.replace(hour=hour)
            # Velocity: 3–10 transactions within the same hour
        # Pattern 3 – card cloning
        else:
            amount       = round(random.uniform(100, 300), 2)
            hour         = random.randint(10, 23)
            pin_failures = random.randint(1, 3)
            location     = random.choice(ATM_LOCATIONS)
            ts           = ts.replace(hour=hour)

        transaction_success = random.random() < 0.6
        velocity_1h         = random.randint(3, 15)
        velocity_24h        = random.randint(5, 30)
        distance_from_home  = round(random.uniform(200, 5000), 2)
        balance_after       = round(random.uniform(-500, 2000), 2)
        account_age_days    = random.randint(1, 365)

    else:
        amount              = round(random.uniform(20, 500), 2)
        pin_failures        = random.choices([0, 1, 2], weights=[0.85, 0.12, 0.03])[0]
        location            = random.choice(ATM_LOCATIONS)
        transaction_success = random.random() < 0.96
        velocity_1h         = random.randint(1, 3)
        velocity_24h        = random.randint(1, 8)
        distance_from_home  = round(random.uniform(0, 50), 2)
        balance_after       = round(random.uniform(100, 50000), 2)
        account_age_days    = random.randint(30, 3650)

    return {
        "transaction_id":       transaction_id,
        "timestamp":            ts,
        "hour":                 hour,
        "day_of_week":          ts.weekday(),                  # 0=Mon, 6=Sun
        "is_weekend":           int(ts.weekday() >= 5),
        "amount":               amount,
        "atm_location":         location,
        "card_type":            random.choice(CARD_TYPES),
        "pin_failures":         pin_failures,
        "transaction_success":  int(transaction_success),
        "velocity_1h":          velocity_1h,                   # txns in last 1 hr
        "velocity_24h":         velocity_24h,                  # txns in last 24 hr
        "distance_from_home_km": distance_from_home,
        "balance_after":        balance_after,
        "account_age_days":     account_age_days,
        "is_fraud":             int(is_fraud),
    }


def generate_dataset(n_legitimate: int = 9500,
                     n_fraud: int = 500) -> pd.DataFrame:
    """
    Generate a complete dataset of ATM transactions.
    Default ratio ≈ 5 % fraud (realistic real-world scenario).
    """
    print(f"[DataGen] Generating {n_legitimate} legitimate + "
          f"{n_fraud} fraudulent transactions …")

    records = []
    txn_id  = 1

    for _ in range(n_legitimate):
        records.append(generate_transaction(txn_id, is_fraud=False))
        txn_id += 1

    for _ in range(n_fraud):
        records.append(generate_transaction(txn_id, is_fraud=True))
        txn_id += 1

    df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"[DataGen] Dataset shape: {df.shape}")
    print(f"[DataGen] Fraud rate: {df['is_fraud'].mean():.2%}")
    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("atm_transactions.csv", index=False)
    print("[DataGen] Saved → atm_transactions.csv")
    print(df.head())
