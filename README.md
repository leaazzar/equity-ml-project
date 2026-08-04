# equity-ml

A quantitative equity research pipeline built end to end on WRDS data (CRSP
monthly stock/delisting files, the CRSP/Compustat Merged link table,
Compustat annual fundamentals, and Fama-French factors — see
`DATA_DICTIONARY.md` for the exact datasets): point-in-time data
integration, 51 firm-characteristic features, purged/embargoed walk-forward
model training, and long/short backtesting, each phase's decisions and
findings documented as they were made.

## Status

**Modeling and backtesting phase — first pass complete.** Raw WRDS extracts
have been validated (`src/data_validation/`, see `DATA_DICTIONARY.md`) and
merged into a point-in-time-correct security-month panel
(`src/data_processing/`, see `MERGE_REPORT.md`). 51 point-in-time firm
characteristics have been built from that panel (`src/feature_engineering/`,
see `FEATURE_DICTIONARY.md`), in two layers:
`data/processed/features_raw.parquet` (interpretable units) and
`data/processed/features_model_ready.parquet` (winsorized + cross-
sectionally rank-normalized). Phase 4/5 (`src/equity_ml/models/`,
`src/equity_ml/backtest/`) now has a first working, real-data run — purged/
embargoed expanding-window walk-forward training across benchmark and ML
models, decile long/short backtesting, performance evaluation, and
explainability — see `MODEL_DESIGN.md` for the design and `MODEL_REPORT.md`
for what the first real run found. This is explicitly a **first pass**, not
a finished research result; see `PLAN.md` for what's next.

## Repository layout

```
.
├── configs/                # YAML configuration (project, logging, data sources)
├── data/
│   ├── raw/                # untouched WRDS extracts (gitignored; never modified)
│   ├── interim/            # typed, deduplicated Parquet copies (gitignored; generated)
│   └── processed/          # master panel + feature panels (gitignored; generated)
├── notebooks/              # exploratory notebooks (none yet)
├── reports/
│   ├── data_validation/    # generated data-quality reports (gitignored; regenerate anytime)
│   ├── data_processing/    # generated merge diagnostics (gitignored; regenerate anytime)
│   ├── feature_engineering/ # generated feature diagnostics (gitignored; feature_registry.json/.csv excepted)
│   ├── modeling/           # generated walk-forward/IC diagnostics (gitignored; regenerate anytime)
│   └── backtest/           # generated portfolio/performance diagnostics (gitignored; regenerate anytime)
├── scripts/                # runnable entry points (thin CLI wrappers)
├── src/
│   ├── equity_ml/          # project infrastructure (config, logging) + modeling/backtest pipelines
│   │   ├── data/           # WRDS connection + loading (placeholder — future live pulls only)
│   │   ├── features/       # (superseded by src/feature_engineering/)
│   │   ├── models/         # walk-forward model training (this phase — see MODEL_DESIGN.md)
│   │   ├── backtest/       # portfolio construction + backtesting (this phase)
│   │   └── utils/          # generic helpers
│   ├── data_validation/    # schema/quality validation pipeline
│   ├── data_processing/    # point-in-time merge pipeline
│   └── feature_engineering/ # point-in-time feature engineering
└── tests/                  # pytest suite (synthetic fixtures only, never real data)
```

## Setup

Requires Python >= 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
make install       # pip install -e ".[dev]" + pre-commit install
cp .env.example .env  # then fill in WRDS_USERNAME once you have WRDS access
```

## Development workflow

```bash
make lint          # ruff check
make format        # ruff format
make typecheck      # mypy
make test           # pytest
make check           # lint + format-check + typecheck + test (what CI runs)
make precommit      # run all pre-commit hooks against the whole repo
```

## Configuration

- `configs/config.yaml` — project-level settings (paths, random seed). Its
  `research:` block (date range, universe, rebalance frequency) is left as
  `null` placeholders deliberately: those decisions were made per-phase
  instead, as hardcoded, documented defaults in each phase's own frozen
  dataclass config (`FeatureConfig`, `MergeConfig`, `ModelConfig`,
  `BacktestConfig`) — see `MODEL_DESIGN.md`'s "Confirmed decisions" for
  where the modeling-relevant ones (horizon, universe) actually live.
- `configs/logging.yaml` — standard-library `logging.dictConfig` setup;
  console + rotating file handler writing to `logs/` (gitignored).
- `configs/data_sources.yaml` — WRDS library/table placeholders. **No table
  or field names are confirmed yet** — see `DATA_DICTIONARY.md`.
- `.env` (not committed; copy from `.env.example`) — local secrets, i.e.
  `WRDS_USERNAME`. WRDS authentication itself goes through a local
  `.pgpass` entry or interactive prompt, never a password in this repo.

## Data ingestion & validation pipeline

Raw CSV extracts live in `data/raw/` (gitignored — never committed, never
modified by any code in this repo). Run the validation pipeline with:

```bash
python scripts/run_data_validation.py
# equivalently: python -m data_validation
```

This will, for each of the six raw datasets:

1. Load the raw CSV and auto-profile it: schema, dtypes, null counts,
   duplicate-key detection (against a documented primary key, or
   auto-detected when none is declared — see `crsp_names`), date-range and
   invalid-date detection.
2. Run cross-dataset referential-integrity checks (CRSP ↔ CCM ↔ Compustat ↔
   Fama-French) — missing/duplicate identifiers, CCM link-period overlaps,
   missing delisting data, date-coverage mismatches, identifier-naming
   inconsistencies.
3. Write a typed, deduplicated copy to `data/interim/<name>.parquet` (exact
   full-row duplicates dropped; date columns parsed; nothing else changed).
4. Write reports to `reports/data_validation/`: `summary.md` (human-readable,
   everything above in one place) plus one JSON file per dataset profile,
   `integrity_checks.json`, and `cleaning_summary.json`.

Useful flags: `--raw-dir`, `--interim-dir`, `--reports-dir` (all default to
`data/raw`, `data/interim`, `reports/data_validation`).

See `DATA_DICTIONARY.md` for the resulting schema documentation and a
summary of the data-quality issues found, and `src/data_validation/` for the
implementation (`datasets.py` for the per-dataset specs, `profiling.py` for
generic schema/duplicate detection, `integrity.py` for the cross-dataset
checks, `cleaning.py` for the interim-output writer, `reporting.py` for
report generation, `pipeline.py` for orchestration). `tests/data_validation/`
exercises all of it against small synthetic fixtures — not the real raw
data — so `make test` and CI never need `data/raw/` to be populated.

## Point-in-time data integration pipeline

Once `data/interim/` exists (see above), build the master panel with:

```bash
python scripts/run_data_processing.py
# equivalently: python -m data_processing
```

This merges CRSP monthly stock, CRSP delisting, the CCM link table,
Compustat annual fundamentals, and Fama-French factors into one
security-month panel at `data/processed/master_panel.parquet`, preserving
point-in-time integrity throughout:

1. Resolves `crsp_monthly_stock`'s remaining conflicting-duplicate keys.
2. Resolves a point-in-time `gvkey` for every PERMNO-month via the CCM link
   table's validity windows.
3. Attaches Compustat fundamentals with a 6-month reporting lag and a
   12-month shelf life — the core look-ahead-bias guard.
4. Applies delisting-return adjustments (compounded onto each security's
   final observation).
5. Merges Fama-French factors by month.

Every security-month from `crsp_monthly_stock` is kept (a left join
throughout), including delisted and Compustat-unmatched securities, which is
what keeps the panel survivorship-bias-free. Diagnostics (coverage,
duplicate checks, missing-variable rates, firm coverage over time) are
written to `reports/data_processing/`.

**See `MERGE_REPORT.md` for the full, git-tracked explanation of every merge
decision and lag assumption** — reporting-lag rationale, the delisting-return
methodology and why missing values aren't imputed, survivorship-bias
handling, and known limitations. `src/data_processing/config.py` documents
every tunable assumption (`MergeConfig`). `tests/data_processing/` covers the
point-in-time logic against synthetic fixtures, including a regression test
for a real bug caught during development (CCM's open-ended-link sentinel not
surviving a round trip through the already-parsed interim Parquet).

## Point-in-time feature engineering pipeline

Once `data/processed/master_panel.parquet` exists (see above), build the
feature panels with:

```bash
python scripts/run_feature_engineering.py
# equivalently: python -m feature_engineering
```

This produces 51 point-in-time firm characteristics across nine categories
(size, value, momentum, reversal, quality/profitability, investment/growth,
leverage, liquidity, volatility/risk) in two layers:

- **`data/processed/features_raw.parquet`** — interpretable units, never
  winsorized or normalized, every master-panel row preserved.
- **`data/processed/features_model_ready.parquet`** — the same
  characteristics after one uniform, documented transform: month-by-month
  winsorization (configurable percentiles) followed by cross-sectional
  rank-scaling to `[-1, 1]`, both computed within that month's investable
  universe only.

An `is_investable` flag is a **coarse, approximate** research-universe proxy
built from the fields actually available (this extract has no CRSP
`SHRCD`/`EXCHCD`) — it must not be described as a verified common-share
classification or as membership in any named index (e.g. Russell 1000). The
master panel itself is never filtered by it. Every feature's exact formula
(with units stated explicitly for every monetary quantity), economic
rationale, point-in-time treatment, and missing-value/outlier handling is
documented in **`FEATURE_DICTIONARY.md`**, generated from the machine-
readable registry (`src/feature_engineering/registry.py`, also written to
`reports/feature_engineering/feature_registry.json`/`.csv`, which — unlike
the rest of that gitignored reports directory — is git-tracked, since it's
pure metadata with no licensed data in it).

Validation (`src/feature_engineering/validation.py`) runs automatically on
every build: duplicate-key/calendar-gap/infinite-value checks, a registry-
vs-panel consistency check, denominator-handling checks, and — most
importantly — an automated **no-look-ahead check** that rebuilds every
feature from data truncated at several sample dates and confirms the
results are bit-for-bit identical to the full-sample build, plus automated
cross-sectional-isolation and reproducibility checks. `tests/feature_engineering/`
covers all of this against synthetic fixtures spanning missing fields, bad
denominators, sparse histories, delisted securities, duplicated rows, the
6-month lag boundary, extreme outliers, and universe entry/exit.

**See `UNIT_AUDIT_REPORT.md`** for a dedicated unit-integrity audit
performed before modeling began: it empirically verifies every CRSP/
Compustat field's actual unit (rather than assuming "standard convention"),
and found — and fixed — two real bugs this uncovered: every VALUE ratio
combining CRSP `MthCap` ($ thousands) directly with a Compustat field ($
millions) was off by ~1000x, and `liquidity_share_turnover` mixed `MthVol`
(actual shares) with raw `ShrOut` (thousands of shares) the same way. Both
are fixed, with regression tests in `tests/feature_engineering/test_units.py`.

## Modeling and backtesting pipeline

Once `data/processed/features_model_ready.parquet` exists (see above), train
walk-forward models with:

```bash
python scripts/run_modeling.py
# equivalently: python -m equity_ml.models
```

This builds a forward 1-month-return target from `ret_adj`, restricts to the
`is_investable` universe, and runs purged/embargoed expanding-window walk-
forward training (21 folds over 2005-2025 against the real panel) across
three benchmarks (equal-weight, momentum-sort, Fama-MacBeth) and, by
default, five scikit-learn estimators (Ridge, Lasso, ElasticNet,
RandomForestRegressor, HistGradientBoostingRegressor), each
hyperparameter-tuned per fold by validation-fold Information Coefficient.
`MODEL_REPORT.md`'s first real run used a custom, smaller model set
(excluding `RandomForestRegressor`) for compute-budget reasons — see its
"Compute-scoping decisions" for why and what a full-default run costs.
Writes `reports/modeling/` (out-of-sample predictions, IC series/summary,
fold boundaries, validation results).

Then backtest the resulting predictions with:

```bash
python scripts/run_backtest.py
# equivalently: python -m equity_ml.backtest
```

This builds decile long/short, dollar-neutral portfolios from the
out-of-sample predictions, applies the delisting-adjusted return
(`ret_adj`), and computes turnover, transaction-cost-sensitivity, Sharpe/
Sortino/drawdown, and a Fama-French-5-plus-momentum factor-exposure
regression (alpha). Writes `reports/backtest/`.

**See `MODEL_DESIGN.md`** for the full technical design (target
construction, walk-forward/purge-embargo methodology, benchmark and ML
models, hyperparameter tuning, portfolio construction, backtesting,
performance evaluation, explainability) and the project owner's confirmed
decisions on prediction horizon, universe, and how Phase 1-3's residual
data-quality follow-ups are treated. **See `MODEL_REPORT.md`** for the
durable record of the first real run's results, two real bugs found and
fixed running against real (not synthetic-fixture) data, and this pass's
explicit compute-scoping decisions — this is a first pass, not a finished
research result. **See `MODEL_CARD.md`** for a compact, at-a-glance summary
of the whole modeling system (objective, target, universe, methodology,
leakage safeguards, limitations) if you don't need the full design/results
narrative.

## WRDS raw-data notes

- `configs/data_sources.yaml` documents the datasets currently on hand; it is
  not yet a complete WRDS library/table registry.
- `src/equity_ml/data/wrds_loader.py` and `scripts/download_wrds_data.py`
  remain placeholders (`TODO(WRDS)`) for automating *future* WRDS pulls —
  the current raw files were provided directly, not downloaded by this repo.

## Authorship

See `CLAUDE.md` for this repository's authorship and git conventions.
