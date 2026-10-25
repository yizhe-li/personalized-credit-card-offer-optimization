"""
03_ab_test.py
Resume bullet 1: the randomized experiment.

  1. Randomly assigns each user 50/50:
        T=0  standard rewards offer (control)
        T=1  status-matched offer   (treatment)
  2. Simulates conversion outcomes from a known data-generating process
     (this is the synthetic ground truth -- in the real project this is
     where the live experiment happens).
  3. Runs the standard A/B readout: SRM check, conversion rates, ATE with
     a two-proportion z-test and 95% CI, plus a placebo (permutation) test.

Ground-truth segments baked into tau(x):
  - Persuadables   : Silver/Gold tier, high points velocity  -> large lift
  - Sure things    : Platinum + still flying                 -> tiny lift
  - Lost causes    : near-zero velocity                      -> ~no lift
  - Do-not-disturb : lapsed Platinums                        -> small NEGATIVE lift
"""
import numpy as np
import pandas as pd
from scipy import stats
from config import SEED, DATA_DIR

rng = np.random.default_rng(SEED + 1)
df = pd.read_csv(f"{DATA_DIR}/features.csv")

# ---------------- 1. Randomized assignment ----------------
# complete randomization: exactly 50/50, shuffled
assign = np.repeat([0, 1], len(df) // 2)
df["T"] = rng.permutation(assign)

# ---------------- 2. Outcome simulation (ground truth) ----------------
v_k = df["total_velocity"] / 1000.0          # velocity in k-points/month
tier = df["max_tier"]
cross = df["cross_program"]
inc_k = df["income"] / 100_000.0

logit_p0 = -3.5 + 0.08 * v_k + 0.40 * tier + 0.25 * cross + 0.25 * inc_k \
           - 0.06 * df["recency_months"]
p0 = 1 / (1 + np.exp(-logit_p0))

tau = np.full(len(df), np.nan)
seg = np.full(len(df), "mid", dtype=object)

m = v_k < 2.0
tau[m], seg[m] = 0.005, "lost_cause"
m = (tier == 3) & (v_k < 6)
tau[m], seg[m] = -0.02, "do_not_disturb"
m = (tier == 3) & (v_k >= 6)
tau[m], seg[m] = 0.01, "sure_thing"
m = tier.isin([1, 2]) & (v_k >= 6)
tau[m] = 0.04 + 0.010 * np.minimum(v_k[m], 12) + 0.03 * cross[m]
seg[m] = "persuadable"
m = np.isnan(tau)
tau[m] = 0.015 + 0.004 * np.minimum(v_k[m], 12)

p1 = np.clip(p0 + tau, 0.001, 0.97)
p = np.where(df["T"] == 1, p1, p0)
df["converted"] = (rng.random(len(df)) < p).astype(int)
df["true_tau"] = tau          # kept ONLY for simulation sanity checks
df["true_segment"] = seg
df.to_csv(f"{DATA_DIR}/experiment.csv", index=False)

# ---------------- 3. A/B readout ----------------
n1, n0 = (df["T"] == 1).sum(), (df["T"] == 0).sum()
c1 = df.loc[df["T"] == 1, "converted"].mean()
c0 = df.loc[df["T"] == 0, "converted"].mean()
ate = c1 - c0

# SRM check
chi2, srm_p = stats.chisquare([n1, n0])

# two-proportion z-test
x1, x0 = df.loc[df["T"] == 1, "converted"].sum(), df.loc[df["T"] == 0, "converted"].sum()
p_pool = (x1 + x0) / (n1 + n0)
se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n0))
z = ate / se_pool
pval = 2 * (1 - stats.norm.cdf(abs(z)))
se_unpooled = np.sqrt(c1 * (1 - c1) / n1 + c0 * (1 - c0) / n0)
ci = (ate - 1.96 * se_unpooled, ate + 1.96 * se_unpooled)

# placebo: permute T, effect should vanish
T_perm = rng.permutation(df["T"].to_numpy())
ate_placebo = df.loc[T_perm == 1, "converted"].mean() - df.loc[T_perm == 0, "converted"].mean()

print("=" * 60)
print("A/B TEST READOUT  (status-matched vs standard offer)")
print("=" * 60)
print(f"n_treatment={n1:,}  n_control={n0:,}   SRM p-value={srm_p:.3f} (OK if > .05)")
print(f"conv_treatment={c1:.4f}   conv_control={c0:.4f}")
print(f"ATE = {ate:+.4f}  (95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}])   z={z:.2f}  p={pval:.2e}")
print(f"Placebo (permuted T) ATE = {ate_placebo:+.4f}  -> ~0 as expected")
print("\nTrue segment mix (simulation ground truth):")
print(df["true_segment"].value_counts(normalize=True).round(3).to_string())
