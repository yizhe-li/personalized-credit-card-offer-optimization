"""Shared configuration for the Personalized Credit Card Offer Optimization project."""

SEED = 42
N_USERS = 20_000
N_MONTHS = 12          # months of pre-experiment transaction history

# ---- Experiment economics (tune to your own assumptions) ----
LTV_PER_CONVERSION = 600.0   # incremental lifetime value of one acquired cardholder ($)
COST_PER_TREATED = 40.0      # expected extra cost of status-matched offer vs standard, per user sent ($)

# ---- Paths ----
DATA_DIR = "data"
SQL_DIR = "sql"
TABLEAU_DIR = "tableau"

HOLDOUT_FRAC = 0.25
