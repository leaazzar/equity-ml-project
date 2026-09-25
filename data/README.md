# data/

**No data is committed to this repository, and never will be.** CRSP,
Compustat, and CCM data are licensed WRDS products this project has no
right to redistribute — see the root `.gitignore`. Every
subdirectory here is gitignored except for a `.gitkeep` placeholder; clone
this repo and all three will be empty until you populate them yourself with
your own WRDS access.

- `raw/` — untouched WRDS extracts, exactly as pulled. Place the six CSV
  files described in `DATA_DICTIONARY.md` here (`crsp_monthly_stock.csv`,
  `crsp_delisting.csv`, `crsp_names.csv`, `ccm_link_table.csv`,
  `compustat_fundamentals_annual.csv`,
  `fama_french_5f_momentum_monthly.csv`). `scripts/download_wrds_data.py`
  remains a documented placeholder for automating this step in the future
  (see `src/equity_ml/data/wrds_loader.py`); this project's own extracts
  were provided directly, not downloaded through it.
- `interim/` — typed, deduplicated Parquet copies, written by
  `scripts/run_data_validation.py`. Never edited by hand.
- `processed/` — the point-in-time master panel and feature panels, written
  by `scripts/run_data_processing.py` and `scripts/run_feature_engineering.py`.

See the root `README.md`'s pipeline sections for the full command sequence
that turns `raw/` into everything downstream, through to
`reports/modeling/` and `reports/backtest/`.
