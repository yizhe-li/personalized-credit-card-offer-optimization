"""
06_tableau_export.py
Resume bullet 4 (second half): produce the data layer for the Tableau
dashboard -- causal lift across loyalty tiers, velocity bands, and
demographic segments.

Outputs (open these in Tableau):
  tableau/scored_users.csv        user-level scored holdout (for free slicing)
  tableau/lift_heatmap.csv        lift by loyalty tier x velocity band
  tableau/lift_by_demographics.csv lift by age band / region / income band
  tableau/decile_calibration.csv  predicted vs actual uplift by decile
  tableau/qini_curve.csv          Qini curve points
  tableau/roi_by_threshold.csv    ROI vs targeting depth

Suggested dashboard sheets:
  1. Heatmap: tier x velocity band, color = actual lift
  2. Line: ROI vs % of base targeted (the policy curve)
  3. Bar: predicted vs actual uplift by decile (calibration)
  4. Qini curve vs random baseline
"""
import shutil
import pandas as pd
from config import DATA_DIR, TABLEAU_DIR

hold = pd.read_csv(f"{DATA_DIR}/scored_holdout.csv")

TIER_LABELS = {0: "Basic", 1: "Silver", 2: "Gold", 3: "Platinum"}
hold["loyalty_tier"] = hold["max_tier"].map(TIER_LABELS)
hold["velocity_band"] = pd.cut(hold["total_velocity"] / 1000,
                               bins=[-0.01, 2, 4, 6, 9, 100],
                               labels=["<2k", "2-4k", "4-6k", "6-9k", "9k+"])
hold["age_band"] = pd.cut(hold["age"], bins=[20, 30, 40, 50, 60, 80],
                          labels=["21-30", "31-40", "41-50", "51-60", "61+"])
hold["income_band"] = pd.cut(hold["income"] / 1000,
                             bins=[0, 50, 80, 120, 200, 1000],
                             labels=["<50k", "50-80k", "80-120k", "120-200k", "200k+"])

def lift_table(df: pd.DataFrame, dims: list[str]) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(dims, observed=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        t, c = g[g["T"] == 1], g[g["T"] == 0]
        if len(t) < 30 or len(c) < 30:      # skip unstable cells
            continue
        rows.append({**dict(zip(dims, keys)),
                     "n": len(g),
                     "conv_control": c["converted"].mean(),
                     "conv_treatment": t["converted"].mean(),
                     "actual_lift": t["converted"].mean() - c["converted"].mean(),
                     "avg_predicted_cate": g["tau_hat"].mean()})
    return pd.DataFrame(rows)

lift_table(hold, ["loyalty_tier", "velocity_band"]) \
    .to_csv(f"{TABLEAU_DIR}/lift_heatmap.csv", index=False)

demo = pd.concat([
    lift_table(hold, ["age_band"]).assign(dimension="age_band")
        .rename(columns={"age_band": "value"}),
    lift_table(hold, ["region"]).assign(dimension="region")
        .rename(columns={"region": "value"}),
    lift_table(hold, ["income_band"]).assign(dimension="income_band")
        .rename(columns={"income_band": "value"}),
])
demo.to_csv(f"{TABLEAU_DIR}/lift_by_demographics.csv", index=False)

keep = ["user_id", "age_band", "income_band", "region", "loyalty_tier",
        "velocity_band", "cross_program", "T", "converted", "tau_hat"]
hold[keep].to_csv(f"{TABLEAU_DIR}/scored_users.csv", index=False)

for f in ["decile_calibration.csv", "qini_curve.csv", "roi_by_threshold.csv"]:
    shutil.copy(f"{DATA_DIR}/{f}", f"{TABLEAU_DIR}/{f}")

print("Tableau data layer written to tableau/:")
print(" - lift_heatmap.csv (tier x velocity)")
print(" - lift_by_demographics.csv (age / region / income)")
print(" - scored_users.csv, decile_calibration.csv, qini_curve.csv, roi_by_threshold.csv")
print("\nPreview - lift by loyalty tier x velocity band:")
print(pd.read_csv(f"{TABLEAU_DIR}/lift_heatmap.csv").round(3).to_string(index=False))
