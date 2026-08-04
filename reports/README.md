# reports/

Generated research outputs (figures, tables, write-ups). Nothing here is
source code.

- `figures/` — generated plots (gitignored; regenerate from code).
- `tables/` — generated tables (gitignored; regenerate from code).
- `data_validation/` — output of `scripts/run_data_validation.py` (gitignored
  — it embeds literal sample values from the licensed raw WRDS extract, so
  it is never committed; regenerate locally any time). See
  `DATA_DICTIONARY.md` for the durable, git-tracked summary of its findings.

- `feature_engineering/` — output of `scripts/run_feature_engineering.py`
  (gitignored, except `feature_registry.json` — pure metadata, no licensed
  data). See `FEATURE_DICTIONARY.md` for the durable, git-tracked summary.
- `modeling/` — output of `scripts/run_modeling.py`: the out-of-sample
  prediction panel, per-model Information Coefficient series/summary, fold
  boundaries, and validation results (gitignored — regenerate locally). See
  `MODEL_REPORT.md` for the durable, git-tracked summary.
- `backtest/` — output of `scripts/run_backtest.py`: portfolio returns,
  performance metrics, cost sensitivity, and factor-exposure regression
  (gitignored — regenerate locally). See `MODEL_REPORT.md`.
