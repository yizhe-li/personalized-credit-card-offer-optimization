"""
04_cate_estimation.py
Resume bullet 3: heterogeneous treatment effects on the experimental data.

Estimator: CausalForestDML (econml) -- Double Machine Learning with a
causal-forest final stage (Athey & Wager generalized random forest).
On RCT data the propensity is known (~0.5), so the treatment model is a
trivial nuisance; DML cross-fitting still partials out baseline outcome
variation, which sharpens the CATE estimates.

Validation (the two things interviewers ask):
  1. Uplift ranking quality on a holdout: Qini curve / AUUC vs random.
  2. Decile calibration: predicted CATE vs ACTUAL experimental uplift
     within each predicted-CATE decile.
  3. (Simulation-only sanity check) correlation with true tau.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from econml.dml import CausalForestDML
from config import SEED, DATA_DIR, HOLDOUT_FRAC

df = pd.read_csv(f"{DATA_DIR}/experiment.csv")

FEATURES = ["age", "income", "tenure_months", "delta_velocity", "aa_velocity",
            "total_velocity", "delta_tier", "aa_tier", "max_tier",
            "cross_program", "total_active_months", "mom_growth_total",
            "recency_months"]
X = pd.get_dummies(df[FEATURES + ["region"]], columns=["region"], drop_first=True)
y = df["converted"].to_numpy()
T = df["T"].to_numpy()

strata = T * 2 + y  # keep T x Y mix balanced across the split
X_tr, X_te, y_tr, y_te, T_tr, T_te, idx_tr, idx_te = train_test_split(
    X, y, T, df.index, test_size=HOLDOUT_FRAC, random_state=SEED, stratify=strata)

# ---------------- Fit DML causal forest ----------------
est = CausalForestDML(
    model_y=HistGradientBoostingRegressor(max_depth=3, random_state=SEED),
    model_t=make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    discrete_treatment=True,
    n_estimators=500,
    min_samples_leaf=40,
    cv=3,                                  # DML cross-fitting folds
    random_state=SEED,
)
est.fit(y_tr, T_tr, X=X_tr)
tau_hat = est.effect(X_te)

ate_forest = est.ate(X_te)
print(f"Forest ATE on holdout: {ate_forest:+.4f}")

# ---------------- Validation 1: Qini / AUUC ----------------
def qini_curve(y, t, score):
    order = np.argsort(-score)
    y, t = y[order], t[order]
    n_t = np.cumsum(t)
    n_c = np.cumsum(1 - t)
    y_t = np.cumsum(y * t)
    y_c = np.cumsum(y * (1 - t))
    with np.errstate(divide="ignore", invalid="ignore"):
        qini = y_t - y_c * np.where(n_c > 0, n_t / n_c, 0)  # incremental conversions
    frac = np.arange(1, len(y) + 1) / len(y)
    return frac, qini

frac, qini = qini_curve(y_te, T_te, tau_hat)
qini_random = frac * qini[-1]                       # straight-line baseline
auuc_model = np.trapezoid(qini, frac)
auuc_random = np.trapezoid(qini_random, frac)
qini_coef = auuc_model - auuc_random
print(f"AUUC(model) = {auuc_model:.1f} | AUUC(random) = {auuc_random:.1f} "
      f"| Qini coefficient = {qini_coef:.1f}  (>0 means real targeting signal)")

# ---------------- Validation 2: decile calibration ----------------
hold = pd.DataFrame({"tau_hat": tau_hat, "y": y_te, "T": T_te}).reset_index(drop=True)
hold["decile"] = pd.qcut(hold["tau_hat"], 10, labels=False, duplicates="drop")
rows = []
for d, g in hold.groupby("decile"):
    actual = g.loc[g["T"] == 1, "y"].mean() - g.loc[g["T"] == 0, "y"].mean()
    rows.append({"decile": int(d) + 1, "n": len(g),
                 "predicted_cate": g["tau_hat"].mean(), "actual_uplift": actual})
calib = pd.DataFrame(rows)
print("\nDecile calibration (1 = lowest predicted CATE, 10 = highest):")
print(calib.round(4).to_string(index=False))
calib.to_csv(f"{DATA_DIR}/decile_calibration.csv", index=False)

# ---------------- Validation 3: simulation-only sanity check ----------------
true_tau_te = df.loc[idx_te, "true_tau"].to_numpy()
corr = np.corrcoef(tau_hat, true_tau_te)[0, 1]
print(f"\n[sim-only] corr(tau_hat, true_tau) = {corr:.3f}")

# ---------------- Save scored holdout ----------------
scored = df.loc[idx_te].copy()
scored["tau_hat"] = tau_hat
scored.to_csv(f"{DATA_DIR}/scored_holdout.csv", index=False)
qini_df = pd.DataFrame({"frac_targeted": frac, "qini_model": qini,
                        "qini_random": qini_random})
qini_df.iloc[::25].to_csv(f"{DATA_DIR}/qini_curve.csv", index=False)
print("\nSaved scored_holdout.csv, decile_calibration.csv, qini_curve.csv")
