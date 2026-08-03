# reports/

Generated research outputs (figures, tables, write-ups). Nothing here is
source code.

- `figures/` — generated plots (gitignored; regenerate from code).
- `tables/` — generated tables (gitignored; regenerate from code).
- `data_validation/` — output of `scripts/run_data_validation.py` (gitignored
  — it embeds literal sample values from the licensed raw WRDS extract, so
  it is never committed; regenerate locally any time). See
  `DATA_DICTIONARY.md` for the durable, git-tracked summary of its findings.

No feature/model reports exist yet — feature engineering has not started.
