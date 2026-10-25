"""
05_roi_simulation.py
Resume bullet 4 (first half): translate CATE into a spend-allocation policy
and compute the projected ROI gain. THIS is the exact arithmetic behind the
"projected X% increase in conversion ROI" number on the resume.

Method (experimentally grounded, not model-circular):
  - The MODEL only picks WHO to target: top-k% of holdout users by tau_hat.
  - The LIFT inside the targeted group is measured from the holdout's
    ACTUAL experimental outcomes (treat-vs-control conversion gap), not
    from the model's own predictions.
  - ROI(policy) = (LTV x incremental_conversions - cost) / cost,
    cost = COST_PER_TREATED x n_targeted.
  - Baseline policy: send the status-matched offer to EVERYONE (k=100%).
"""
import numpy as np
import pandas as pd
from config import DATA_DIR, LTV_PER_CONVERSION as V, COST_PER_TREATED as C

hold = pd.read_csv(f"{DATA_DIR}/scored_holdout.csv")
n_total = len(hold)

rows = []
for k in range(10, 101, 10):
    cutoff = np.percentile(hold["tau_hat"], 100 - k)
    sel = hold[hold["tau_hat"] >= cutoff]
    n_sel = len(sel)
    # actual experimental uplift inside the targeted group
    uplift = (sel.loc[sel["T"] == 1, "converted"].mean()
              - sel.loc[sel["T"] == 0, "converted"].mean())
    inc_conv = uplift * n_sel              # if offer were sent to all selected
    value = V * inc_conv
    cost = C * n_sel
    roi = (value - cost) / cost
    rows.append({"top_k_pct": k, "n_targeted": n_sel,
                 "actual_uplift_in_group": uplift,
                 "incremental_conversions": inc_conv,
                 "value_$": value, "cost_$": cost, "ROI": roi})

roi_df = pd.DataFrame(rows)
roi_df.to_csv(f"{DATA_DIR}/roi_by_threshold.csv", index=False)

base = roi_df.loc[roi_df["top_k_pct"] == 100].iloc[0]   # untargeted
best = roi_df.loc[roi_df["ROI"].idxmax()]

print("=" * 72)
print(f"SPEND-ALLOCATION POLICY SIMULATION   (LTV=${V:.0f}, cost/offer=${C:.0f})")
print("=" * 72)
print(roi_df.round(4).to_string(index=False))
print("-" * 72)
print(f"Untargeted (send to all):  ROI = {base['ROI']:+.1%}")
print(f"Best targeted policy:      top {best['top_k_pct']:.0f}% by CATE -> "
      f"ROI = {best['ROI']:+.1%}")
print(f"ROI improvement:           {best['ROI'] - base['ROI']:+.1%} points "
      f"vs untargeted")
print("\nNOTE: this printed number is your resume's 'projected ROI gain'.")
print("It moves with LTV_PER_CONVERSION / COST_PER_TREATED in config.py --")
print("set them to your assumptions, rerun, and quote what prints here.")

# sim-only: who did the model actually pick at the best k?
cutoff = np.percentile(hold["tau_hat"], 100 - best["top_k_pct"])
picked = hold[hold["tau_hat"] >= cutoff]
print("\n[sim-only] true-segment composition of the targeted group:")
print(picked["true_segment"].value_counts(normalize=True).round(3).to_string())
