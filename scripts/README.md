# scripts/

Standalone, runnable entry points (as opposed to importable library code,
which lives in `src/equity_ml/`). Run in this order, each depending on the
previous step's output — see the root `README.md` for full usage:

1. `run_data_validation.py` — profiles and validates the raw WRDS extracts
   in `data/raw/`, writes `data/interim/`.
2. `run_data_processing.py` — merges `data/interim/` into the point-in-time
   master panel at `data/processed/master_panel.parquet`.
3. `run_feature_engineering.py` — builds the two feature panels
   (`features_raw.parquet`, `features_model_ready.parquet`).
4. `run_modeling.py` — walk-forward trains benchmark and ML models, writes
   `reports/modeling/`.
5. `run_backtest.py` — backtests `run_modeling.py`'s predictions, writes
   `reports/backtest/`.

`download_wrds_data.py` — **not yet implemented.** Placeholder entry point
for pulling raw extracts from WRDS into `data/raw/` once access is
provisioned; see `configs/data_sources.yaml` and
`src/equity_ml/data/wrds_loader.py`. This project's own raw extracts were
supplied directly, not downloaded through it.

## Conventions

- Each script should be runnable as `python scripts/<name>.py ...` from the
  project root, with `src/` importable (achieved via `pip install -e .`).
- Scripts should import logic from `equity_ml.*` rather than containing
  business logic themselves — keep them thin CLI wrappers.
- Scripts must not require network/WRDS credentials to *import* successfully;
  they should only fail at the point of actually connecting.
