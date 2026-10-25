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
  
