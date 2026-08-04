# Research Plan

**Status:** Data ingestion, validation, point-in-time integration, and
point-in-time feature engineering complete. Phase 4 (modeling) and Phase 5
(backtesting) have a first working pass — see `MODEL_REPORT.md` for what
was actually run and found; it is explicitly a first pass, not a finished
research result or a model-selection decision. Nothing in this document
should be read as a final research conclusion — it is a plan of phases, and
every phase's findings so far are pipeline-correctness/first-look findings,
not a validated trading strategy.

## Phase 0 — Infrastructure (done)

- [x] Repository scaffolding: package layout, `pyproject.toml`, tests,
      linting/formatting/type-checking, pre-commit, CI, logging, Makefile.

## Phase 1 — Data ingestion & validation (done)

Raw extracts were provided directly (not pulled via the WRDS API) and placed
in `data/raw/`. Completed:

- [x] Built `src/data_validation/` to auto-profile schema/dtypes/nulls/
      duplicates/date-ranges and cross-validate identifiers across CRSP,
      CCM, Compustat, and Fama-French (see `DATA_DICTIONARY.md`).
- [x] Populated `DATA_DICTIONARY.md` from the inspected schema (not assumed).
- [x] Wrote typed, exact-duplicate-free interim copies to `data/interim/`.

Still open:

- [ ] Decide and document the sample period and universe filters in
      `configs/config.yaml` (currently `null` placeholders).
- [ ] Resolve or explicitly accept the residual data-quality issues found
      (conflicting duplicate keys, `crsp_names`' missing validity periods,
      unexplained CCM coverage gaps) before building on top of them.
- [ ] `equity_ml.data.wrds_loader` remains unimplemented — it targets
      *future* live WRDS pulls, which weren't needed for this data drop.

## Phase 2 — Point-in-time data integration (done)

- [x] Built `src/data_processing/` to merge CRSP monthly stock, CRSP
      delisting, the CCM link table, Compustat annual fundamentals, and
      Fama-French factors into `data/processed/master_panel.parquet`, with
      point-in-time integrity enforced throughout (see `MERGE_REPORT.md`
      for every decision: reporting lag, staleness cutoff, delisting-return
      compounding, survivorship-bias handling).
- [x] Resolved `crsp_monthly_stock`'s conflicting-duplicate-key rows (a
      documented, deterministic rule — see `MERGE_REPORT.md`).

Still open:

- [ ] Decide and document the sample period and universe filters in
      `configs/config.yaml` (currently `null` placeholders) — the master
      panel currently includes every security as provided, unfiltered.
- [ ] Resolve the residual issues carried forward from Phase 1
      (`crsp_names`' missing validity periods, unexplained CCM coverage
      gaps, unconfirmed WRDS code definitions) before relying on them.
- [ ] Decide on quarterly Compustat data, `DelPERMNO` return-chaining, and
      any `DelReasonType`-dependent delisting-return proxy — all
      explicitly deferred (see `MERGE_REPORT.md`'s "Known limitations").

## Phase 3 — Point-in-time feature engineering (done)

- [x] Built `src/feature_engineering/` to compute 51 point-in-time firm
      characteristics across 9 categories (size, value, momentum, reversal,
      quality, growth, leverage, liquidity, volatility) from the master
      panel — see `FEATURE_DICTIONARY.md` for every formula, economic
      rationale, and point-in-time treatment.
- [x] Two output layers: `data/processed/features_raw.parquet`
      (interpretable units) and `data/processed/features_model_ready.parquet`
      (month-by-month, universe-only winsorize + rank-scale).
- [x] Documented `is_investable` universe approximation (this extract has
      no `SHRCD`/`EXCHCD`) — the raw panel is never filtered by it.
- [x] Automated validation on every build, including a no-look-ahead check
      (rebuild from truncated data, compare to full-sample) that passed on
      the full 2.5M-row real panel.
- [x] **Unit integrity audit** (see `UNIT_AUDIT_REPORT.md`): empirically
      verified every CRSP/Compustat field's actual unit and fixed two real
      bugs this uncovered — VALUE ratios mixing `MthCap` ($ thousands)
      directly with Compustat fields ($ millions), and
      `liquidity_share_turnover` mixing `MthVol` (actual shares) with raw
      `ShrOut` (thousands). `reversal_1m` flagged as perfectly collinear
      with `mom_1m`; `is_investable` documentation strengthened to rule out
      "verified common-share"/named-index framing.

Still open (see `FEATURE_DICTIONARY.md`'s "Requested features that
could not be created" for detail on the first item):

- [ ] Zero-return-frequency/LOT liquidity, dividend yield, ROIC, and the
      balance-sheet accrual method were not implemented — no daily data,
      no dividend field, no clean tax-expense field, and scope control,
      respectively.
- [ ] The `is_investable` flag is a price/size/trading-activity
      approximation, not a precise share-code-based common-equity
      classification (the underlying fields don't exist in this extract).

## Phase 4 — Modeling (first pass done)

- [x] Technical design document written (`MODEL_DESIGN.md`): target
      construction, train/val/test methodology, walk-forward evaluation,
      benchmark models, hyperparameter tuning, portfolio construction,
      backtesting, performance evaluation, explainability, reporting.
- [x] Blocking design decisions resolved by the project owner
      (2026-08-03, see `MODEL_DESIGN.md`'s "Confirmed decisions"):
      1-month prediction horizon with monthly rebalance; `is_investable`
      accepted as the modeling universe with no additional filter; residual
      Phase 1-3 data-quality follow-ups explicitly deferred.
- [x] Implemented `src/equity_ml/models/`: target/label construction
      (`targets.py`), purged/embargoed expanding-window walk-forward splits
      (`splits.py`), benchmarks (`baselines.py`) + ML estimators
      (`estimators.py`), hyperparameter tuning by validation-fold IC
      (`tuning.py`), walk-forward training orchestration (`training.py`),
      leakage/consistency validation (`validation.py`), IC/permutation-
      importance diagnostics (`diagnostics.py`), reporting, pipeline, CLI.
      253 tests total in the repo (65 new this phase, all against synthetic
      fixtures) — see `tests/models/`.
- [x] First real run against the full panel: 21 walk-forward folds,
      2005-2025 out-of-sample, 7 models (3 benchmarks + 4 ML estimators) —
      see `MODEL_REPORT.md` for full results, two real bugs found and fixed
      running against real data, and this pass's explicit compute-scoping
      decisions (e.g. `random_forest` excluded from this run — too slow at
      this panel's size with its default hyperparameters, though fully
      implemented and tested).

## Phase 5 — Backtesting & evaluation (first pass done)

- [x] Backtest methodology defined and implemented in
      `src/equity_ml/backtest/`: decile long/short portfolio construction
      (`portfolio.py`), turnover/transaction-cost engine using the
      delisting-adjusted return (`engine.py`), Sharpe/Sortino/drawdown/
      IC/factor-exposure performance evaluation (`performance.py`),
      reporting, pipeline, CLI — see `MODEL_DESIGN.md` Sections 6-8 for the
      design and `MODEL_REPORT.md` for the first real results (net-of-cost
      Sharpe, cost sensitivity, alpha vs. Fama-French 5 + momentum).
- [ ] Sector-neutral and score-weighted portfolio variants (both supported
      by `BacktestConfig` flags) were not run in the first pass.

## Phase 6 — Reporting (first pass done)

- [x] `MODEL_REPORT.md` — the durable, git-tracked record of Phase 4/5's
      first real run, generated from code-produced diagnostics
      (`reports/modeling/`, `reports/backtest/`), matching
      `MERGE_REPORT.md`/`UNIT_AUDIT_REPORT.md`'s convention.
- [ ] Figures (cumulative return curves, IC time series, importance charts)
      are not yet generated as images — only CSV/Markdown tables so far.

## Open questions (resolved before Phase 4 implementation began)

- **Target universe:** `is_investable` as currently defined, no additional
  restriction — confirmed by the project owner 2026-08-03. It remains only
  an approximation (see `FEATURE_DICTIONARY.md`), accepted as such.
- **Prediction horizon and rebalance frequency:** 1 month, monthly rebalance
  — confirmed 2026-08-03.
- **Residual data-quality/integration issues from Phases 1-2, and
  not-yet-implemented features from Phase 3:** explicitly deferred, carried
  forward with their existing documented handling — confirmed 2026-08-03.

See `MODEL_DESIGN.md`'s "Confirmed decisions" section for the full
reasoning behind each.
