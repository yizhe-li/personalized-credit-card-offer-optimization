"""
01_generate_data.py
Generates the RAW inputs of the project (synthetic replication):
  - data/users.csv               : demographics
  - data/delta_transactions.csv  : monthly point-earning rows, Delta SkyMiles
  - data/aa_transactions.csv     : monthly point-earning rows, AA AAdvantage

A latent `engagement` variable drives flight activity. It is used ONLY for
data generation and is NOT saved as a feature (it must be recovered through
the engineered features in 02_sql_features.py).
"""
import numpy as np
import pandas as pd
import os
from config import SEED, N_USERS, N_MONTHS, DATA_DIR

rng = np.random.default_rng(SEED)
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------- Users ----------------
age = np.clip(rng.normal(42, 12, N_USERS), 21, 75).round(0)
income = np.clip(rng.lognormal(11.1, 0.45, N_USERS), 25_000, 400_000).round(-2)
region = rng.choice(["Northeast", "Southeast", "Midwest", "Southwest", "West"],
                    N_USERS, p=[.22, .20, .20, .15, .23])
tenure = rng.integers(3, 96, N_USERS)
engagement = rng.beta(2, 5, N_USERS)  # latent travel engagement in [0,1]

has_delta = rng.random(N_USERS) < (0.50 + 0.30 * engagement)
has_aa = rng.random(N_USERS) < (0.45 + 0.30 * engagement)
neither = ~(has_delta | has_aa)
has_delta[neither] = True  # everyone belongs to at least one program

users = pd.DataFrame({
    "user_id": np.arange(1, N_USERS + 1),
    "age": age.astype(int),
    "income": income,
    "region": region,
    "tenure_months": tenure,
})
users.to_csv(f"{DATA_DIR}/users.csv", index=False)

# ---------------- Transactions ----------------
def gen_program_transactions(member_mask: np.ndarray, program: str) -> pd.DataFrame:
    rows = []
    member_ids = users.loc[member_mask, "user_id"].to_numpy()
    eng = engagement[member_mask]
    inc = income[member_mask]
    for month in range(1, N_MONTHS + 1):
        active = rng.random(len(member_ids)) < (0.20 + 0.65 * eng)
        if not active.any():
            continue
        ids = member_ids[active]
        e, i = eng[active], inc[active]
        flights = rng.poisson(0.5 + 3.0 * e) + 1
        miles_per_flight = np.clip(rng.normal(1400, 400, len(ids)), 400, 4000)
        fare_mult = 1.0 + 0.8 * e + 0.15 * (i / 200_000)
        points = (flights * miles_per_flight * fare_mult).round(0)
        rows.append(pd.DataFrame({
            "user_id": ids, "month": month,
            "flights": flights, "points": points,
        }))
    out = pd.concat(rows, ignore_index=True)
    out["program"] = program
    return out

delta_tx = gen_program_transactions(has_delta, "DELTA")
aa_tx = gen_program_transactions(has_aa, "AA")
delta_tx.drop(columns="program").to_csv(f"{DATA_DIR}/delta_transactions.csv", index=False)
aa_tx.drop(columns="program").to_csv(f"{DATA_DIR}/aa_transactions.csv", index=False)

print(f"users: {len(users):,} rows")
print(f"delta_transactions: {len(delta_tx):,} rows | aa_transactions: {len(aa_tx):,} rows")
print("Saved to data/.")
