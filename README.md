# Personalized Credit Card Offer Optimization

Uplift modeling on randomized experiment data: run an A/B test of credit-card
sign-up offers, estimate **heterogeneous treatment effects (CATE)** with a
**DML causal forest**, and reallocate marketing spend toward "persuadable"
customer segments.

> **Data disclosure:** all data here is **synthetic**, generated to replicate
> the structure of the original course project (no real Delta/American
> Airlines customer data). The ground-truth treatment effect is known by
> construction, which lets us verify that the estimation pipeline recovers it.

## Pipeline (maps 1:1 to the resume bullets)

| Step | Script | Resume bullet |
|---|---|---|
| Raw data | `01_generate_data.py` | (synthetic Delta + AA transactions) |
| Feature engineering | `02_sql_features.py` + `sql/feature_engineering.sql` | SQL window functions -> Points Velocity, loyalty tiers |
| Randomized experiment | `03_ab_test.py` | A/B test: standard vs status-matched offer, ATE readout |
| CATE estimation | `04_cate_estimation.py` | DML causal forest on experimental data, Qini/calibration validation |
| Spend reallocation | `05_roi_simulation.py` | targeting policy -> projected ROI gain |
| Dashboard data | `06_tableau_export.py` | CSVs for the Tableau dashboard |

## Run it

```bash
pip install -r requirements.txt
python 01_generate_data.py
python 02_sql_features.py
python 03_ab_test.py
python 04_cate_estimation.py
python 05_roi_simulation.py
python 06_tableau_export.py
```

Total runtime ~2 minutes. Everything is seeded (`config.py: SEED=42`), so
results are reproducible.

## Reference results (seed 42)

- **A/B test:** control conv 10.4%, treatment conv 16.0%,
  **ATE = +5.5pp** (95% CI [+4.6, +6.5], p < 1e-15). SRM clean, placebo ~ 0.
- **CATE validation:** Qini coefficient = 28.5 (> 0 -> real targeting
  signal); decile calibration monotone at the top
  (decile 10: predicted 0.118 vs actual 0.134);
  corr(tau_hat, true tau) = 0.77 (simulation-only check).
- **Policy:** with LTV = $600 and $40 cost per status-matched offer,
  sending to **everyone loses money (ROI -17%)**; targeting the **top 20%
  by CATE yields ROI +104%**. The targeted group is 87% true persuadables.

## The two interview questions this repo answers

**"How did you compute the projected ROI gain?"**
See `05_roi_simulation.py`. The model only *selects* the target group
(top-k% by predicted CATE on a holdout). The lift inside that group is then
measured from the holdout's *actual experimental outcomes* — so the
projection is experimentally grounded, not the model grading its own
homework. ROI = (LTV x incremental conversions - offer cost) / offer cost,
compared against the untargeted send-to-all policy. The economics (LTV,
cost) live in `config.py`; the quoted number should match your run.

**"How did you validate the CATE estimates?"**
Three ways (`04_cate_estimation.py`): (1) Qini/AUUC on a holdout vs a
random-targeting baseline; (2) decile calibration — within each predicted-
CATE decile, compare predicted effect to the decile's actual
treatment-control gap; (3) because the data is synthetic, correlation with
the known true tau (0.77).

## Why DML on RCT data?

Randomization identifies the *average* effect, but deciding **who** should
get the expensive offer requires *individual-level* effects. CausalForestDML
(econml) estimates CATE; on RCT data the propensity nuisance is trivially
~0.5, and DML's cross-fitted outcome model partials out baseline conversion
variation, sharpening the heterogeneity estimates. (The CausalML library's
`UpliftRandomForestClassifier` / R-learner is an equivalent route; note
causalml currently requires Python <= 3.11, while this repo uses econml.)
