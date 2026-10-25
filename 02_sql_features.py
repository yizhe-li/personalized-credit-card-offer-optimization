"""
02_sql_features.py
Executes sql/feature_engineering.sql in DuckDB against the raw CSVs and
writes one feature row per user to data/features.csv  (resume bullet 2).
"""
import duckdb
import os
from config import DATA_DIR, SQL_DIR

con = duckdb.connect()
con.execute(f"CREATE VIEW users AS SELECT * FROM read_csv_auto('{DATA_DIR}/users.csv')")
con.execute(f"CREATE VIEW delta_transactions AS SELECT * FROM read_csv_auto('{DATA_DIR}/delta_transactions.csv')")
con.execute(f"CREATE VIEW aa_transactions AS SELECT * FROM read_csv_auto('{DATA_DIR}/aa_transactions.csv')")

with open(os.path.join(SQL_DIR, "feature_engineering.sql")) as f:
    query = f.read()

features = con.execute(query).fetchdf()
features.to_csv(f"{DATA_DIR}/features.csv", index=False)

print(f"features.csv written: {features.shape[0]:,} rows x {features.shape[1]} cols")
print(features[["total_velocity", "max_tier", "cross_program", "recency_months"]]
      .describe().round(1).to_string())
print("\nTier distribution (max_tier):")
print(features["max_tier"].value_counts().sort_index().to_string())
