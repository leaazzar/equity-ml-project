"""Point-in-time data integration: merges CRSP, CCM, Compustat, and
Fama-French into one master security-month panel.

Consumes the typed/deduplicated Parquet files in `data/interim/` produced by
`data_validation` (run that pipeline first). Every merge/lag decision is
documented in `MERGE_REPORT.md` at the repository root. This package does
not perform feature engineering or modeling — see `PLAN.md`.
"""

__version__ = "0.1.0"
