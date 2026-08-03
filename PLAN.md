# Research Plan

**Status:** Data ingestion, validation, point-in-time integration, and
point-in-time feature engineering complete; modeling has not started.
Nothing in this document should be read as a research result — it is a plan
of phases, and Phases 1-3's findings are data *quality*/*integration*/
*feature-construction* findings, not research findings.

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

Still open (see `TASKS.md` "Follow-ups identified by validation"):

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

Still open (see `TASKS.md`'s follow-ups):

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

Still open (see `TASKS.md`'s follow-ups and `FEATURE_DICTIONARY.md`'s
"Requested features that could not be created"):

- [ ] Zero-return-frequency/LOT liquidity, dividend yield, ROIC, and the
      balance-sheet accrual method were not implemented — no daily data,
      no dividend field, no clean tax-expense field, and scope control,
      respectively.
- [ ] The `is_investable` flag is a price/size/trading-activity
      approximation, not a precise share-code-based common-equity
      classification (the underlying fields don't exist in this extract).

## Phase 4 — Modeling (TODO)

- [ ] Define target/label construction, using
      `data/processed/features_model_ready.parquet` (or
      `features_raw.parquet`, if a different transform is wanted).
- [ ] Define train/validation/test split methodology appropriate for panel
      financial data (e.g. purged/embargoed walk-forward).
- [ ] Implement in `src/equity_ml/models/`.

## Phase 5 — Backtesting & evaluation (TODO)

- [ ] Define backtest methodology (transaction costs, turnover constraints,
      rebalance frequency).
- [ ] Implement in `src/equity_ml/backtest/`.

## Phase 6 — Reporting (TODO)

- [ ] Generate figures/tables into `reports/` from code, not by hand.

## Open questions (to resolve before Phase 4)

- What is the target universe (e.g. all US common equities, an index
  subset)? Neither the master panel nor the feature panels are filtered to
  a particular universe (share code, exchange, etc.) — `is_investable` is
  only an approximation (see `FEATURE_DICTIONARY.md`).
- What is the intended prediction horizon and rebalance frequency?
- How should the residual data-quality/integration issues from Phases 1-2,
  and the not-yet-implemented features from Phase 3, be resolved (see
  `TASKS.md`)?

These are intentionally unanswered — do not guess at them.
