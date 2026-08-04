# Model Card

A compact reference summary of the modeling system built in this repository,
following the general spirit of Mitchell et al. (2019) "Model Cards for
Model Reporting," adapted for a cross-sectional equity return-prediction
system rather than a single classifier. This is a summary — `MODEL_DESIGN.md`
has the full design rationale for every decision below, and `MODEL_REPORT.md`
has the full results, real-data findings, and known limitations of the first
run. Where this card and either of those documents seem to disagree, the
longer document is authoritative; this card is kept in sync with them.

**Status:** first pass, not a finished research result or a production
trading system. See "Limitations" below.

## Project objective

Predict which US-listed equities will outperform their peers over the next
month, using only point-in-time-available firm characteristics, and
evaluate whether that prediction translates into a risk-adjusted return
after realistic implementation frictions (turnover, transaction costs) —
not just a statistically significant signal in isolation.

## Prediction target

Forward 1-month total return (`ret_adj`, which includes CRSP's
delisting-return adjustment — see `MERGE_REPORT.md`), computed with the
same log-compounding convention the codebase's own trailing-momentum
features use, applied forward instead of backward. Two variants are built:

- `target_raw` — the literal forward return, used to compute realized
  portfolio P&L in the backtest.
- `target_demeaned` — `target_raw` minus that month's cross-sectional mean
  among investable-universe stocks, used to *train* every model. A
  dollar-neutral long-short portfolio (see "Portfolio construction" below)
  doesn't monetize the market-wide return component, so training on it
  would be optimizing for the wrong quantity.

A row's target is `NaN` (never imputed) whenever fewer than the horizon's
worth of future calendar-contiguous return observations exist for that
security — e.g. the sample's last month, or a security within one month of
its final observation.

## Investment universe

`is_investable`: non-missing price ≥ $1 (configurable), non-missing
positive market capitalization, and an actual trade that month. This is a
**coarse, approximate research-universe proxy**, not a verified common-share
classification or membership in any named index (this data extract has no
CRSP `SHRCD`/`EXCHCD` fields) — see `FEATURE_DICTIONARY.md`'s "The
investable universe" section for the full caveat. Used as-is, with no
additional filter, per the project owner's confirmed decision
(`MODEL_DESIGN.md`).

## Data sources

Six WRDS extracts, documented field-by-field in `DATA_DICTIONARY.md`:
CRSP monthly stock file, CRSP delisting events, CRSP historical
names/identifiers (not used downstream — no point-in-time validity
columns), the CRSP/Compustat Merged (CCM) link table, Compustat annual
fundamentals, and the Fama-French 5-factor + momentum monthly file.
Coverage: 1999-01 to 2025-12, ~2.5 million security-months. No licensed
data is redistributed in this repository — see "Reproducibility notes."

## Feature summary

51 point-in-time firm characteristics across 9 categories (size, value,
momentum, reversal, quality/profitability, investment/growth, leverage,
liquidity, volatility/risk), built in `src/feature_engineering/` from the
point-in-time master panel — full formulas, economic rationale, and
point-in-time treatment for every feature in `FEATURE_DICTIONARY.md`. Two
output layers: raw (interpretable units, $ millions for every monetary
quantity) and model-ready (month-by-month winsorized + cross-sectionally
rank-scaled to `[-1, 1]`, computed within the investable universe only).
Models train on `feature_engineering.registry.model_feature_names()` — 50
of the 51 (excludes `reversal_1m`, perfectly collinear with `mom_1m`).

## Modeling methodology

Three benchmarks and, by default, five scikit-learn estimators
(`src/equity_ml/models/`):

| Model | Type | Notes |
| --- | --- | --- |
| `equal_weight` | Baseline | Zero-skill floor — identical score every row. |
| `momentum_sort` | Baseline | Single-factor sort on `mom_12_1` alone. |
| `fama_macbeth` | Baseline | Monthly cross-sectional OLS, coefficients averaged over each fold's training window. |
| `ridge` | ML | L2-regularized linear regression. |
| `lasso` | ML | L1-regularized linear regression. |
| `elastic_net` | ML | L1+L2-regularized linear regression. |
| `random_forest` | ML | Shallow trees (`max_depth` ≤ 8), 300 estimators. |
| `hist_gradient_boosting` | ML | Histogram-based gradient boosting; natively tolerant of missing features. |

Every model is scored through the identical walk-forward harness — no
benchmark gets a more favorable evaluation protocol than an ML model.
Hyperparameters are grid-searched per fold, selected by validation-fold
Information Coefficient (Spearman rank correlation of predicted score vs.
realized return), not MSE — IC directly measures the cross-sectional
ranking quality a long-short portfolio actually monetizes.

Training uses **complete cases only**: a row is used if and only if all 50
features and the target are non-missing, matching this codebase's
never-impute convention throughout. This is strict — only ~13% of
investable-universe rows qualify (see `MODEL_REPORT.md`), concentrated from
2001 onward.

## Walk-forward validation protocol

Purged/embargoed expanding-window walk-forward (`src/equity_ml/models/splits.py`):

- **Expanding, not rolling** — every fold's training window starts at the
  sample's first month and grows; never truncated from the left.
- **Minimum initial training window:** 72 months (6 years), long enough for
  the longest-lookback feature (36-month trailing skewness) and to move past
  the panel's early Compustat-coverage ramp-up.
- **Annual refit** — one refit per calendar year of test data.
- **Purge + embargo = the prediction horizon** (1 month) at every fold
  boundary: the last `horizon` months are dropped from training, and the
  first `horizon` months after that are excluded from the following test
  window — so no forward-looking label computed inside training can reach
  into a scored test month.
- **Nested inner split** for hyperparameter tuning: the trailing ~20% of
  each fold's own training window, with the identical purge/embargo rule
  applied at that inner boundary — hyperparameters are never selected by
  looking at a fold's own test data.

First real run: 21 folds, out-of-sample test period 2005-02 to 2025-12.

## Portfolio construction methodology

`src/equity_ml/backtest/portfolio.py`: each rebalance month, investable
stocks with a valid prediction are ranked by score and split into deciles.
Long the top decile, short the bottom decile, equal-weighted within each
leg by default (score-weighting also supported via `BacktestConfig`),
**dollar-neutral** (100% long / 100% short, 0% net exposure) — this is why
training uses the demeaned target: it's optimizing for exactly the quantity
this construction monetizes. Rebalanced monthly, matching the 1-month
prediction horizon. Sector-neutral construction is supported by config but
was not exercised in the first run.

## Transaction cost assumptions

**10 basis points one-way** (20 bps round-trip) per unit of turnover — a
**documented placeholder**, not a measured figure: this project has no
bid-ask-spread or commission data from any of its six source extracts.
Turnover is computed explicitly each rebalance (half the sum of absolute
weight changes vs. the prior rebalance, including the cost of building the
initial portfolio from cash). Every result is reported net of this
assumption alongside a 0/5/10/20/50 bps sensitivity table
(`reports/backtest/cost_sensitivity.csv`), so the reader can judge
cost-sensitivity rather than trust one hardcoded number.

## Evaluation metrics

- **Signal quality:** monthly cross-sectional Information Coefficient
  (Spearman rank correlation, predicted score vs. realized forward return),
  reported as pooled mean/std/t-statistic across all test months.
- **Portfolio performance** (gross and net of transaction cost):
  annualized return, annualized volatility, Sharpe ratio, Sortino ratio,
  maximum drawdown, hit rate, average turnover.
- **Factor-exposure regression:** long-short return series regressed on the
  Fama-French 5 + momentum factors (already point-in-time merged into the
  master panel), with Newey-West (HAC) standard errors — reports alpha (the
  return unexplained by known factors) and factor loadings, so a result
  that's just a repackaged momentum or value tilt is visible rather than
  mistaken for new signal.
- **Explainability:** permutation feature importance per fold, rolled up to
  the 9 feature categories, computed for at least one linear and one
  tree-ensemble model so the two families' attributions can be cross-checked
  against each other.

## Leakage safeguards

- **Accounting data:** 6-month reporting lag + 12-month shelf life after
  each fiscal period-end (`MERGE_REPORT.md`) — a fiscal year's fundamentals
  are only usable starting 6 months after period-end, and expire (nulled,
  not carried forward indefinitely) 12 months after that.
- **Return-based features:** backward-looking, calendar-contiguous windows
  only; a window is either fully populated or `NaN`, never partial.
- **Cross-sectional operations** (winsorization, rank-scaling, universe
  membership): computed independently within each month, using only that
  month's own investable universe.
- **No-look-ahead check:** automated, on every feature-engineering build —
  rebuilds every feature from data truncated at several sample cutoff dates
  and asserts the results are bit-for-bit identical to the full-sample
  build (`feature_engineering/validation.py`).
- **Walk-forward purge/embargo:** see "Walk-forward validation protocol"
  above — prevents a forward-looking label from crossing a fold's own
  train/test boundary.
- **Automated post-hoc checks on the modeling output itself**
  (`equity_ml/models/validation.py`): every fold expands from the same
  start date, every fold's test window starts strictly after
  `train_end + horizon`, no duplicate `(security, date, model)` predictions,
  and every prediction's date falls inside its own fold's test window —
  checked against the actual output, not just trusted by construction.
- **No feature or target is ever imputed anywhere in this codebase.**
  Missing source data always produces a missing value, never a filled-in
  guess (two narrow, explicitly documented accounting-identity exceptions
  are noted in `FEATURE_DICTIONARY.md`).

## Limitations

- **First pass, not a finished result.** Exists to validate the pipeline
  works end to end and get a first read on relative model performance, not
  to declare a winning model or a production strategy.
- **Single prediction horizon** (1 month) run; 3-month/12-month robustness
  checks are supported by the same target-construction code but not yet run.
- **`random_forest` not exercised against real data** in the first pass — a
  single fit at its default hyperparameters took 165.8 seconds on the
  ~313K-row complete-case sample, making a full grid search across 21 folds
  infeasible within that pass's compute budget. Implemented and tested;
  `hist_gradient_boosting` served as the tree-ensemble representative
  instead.
- **`is_investable` is an approximation**, not a verified common-share/
  index-membership filter (no `SHRCD`/`EXCHCD` in this data extract).
- **Transaction cost is a documented placeholder**, not measured
  bid-ask-spread or commission data.
- **Several Phase 1-3 data-quality items remain open** (unexplained CCM
  coverage gaps, missing delisting returns left unadjusted rather than
  proxied, no quarterly Compustat, no `DelPERMNO` return-chaining) —
  explicitly deferred by the project owner rather than fixed, carried
  forward with their existing documented handling. See `MERGE_REPORT.md`
  and `PLAN.md`.
- **One random seed.** All stochastic estimators use `random_state=42`
  throughout; conclusions from a single seed shouldn't be over-read.
- **Feature importance computed for two models only** (`ridge`,
  `hist_gradient_boosting`), not the full seven-model comparison table.
- **Sector-neutral and score-weighted portfolio variants**, though
  supported by `BacktestConfig`, were not exercised in the first run.

## Reproducibility notes

- **No data is redistributed.** CRSP, Compustat, and CCM data are licensed
  WRDS products; `data/raw/`, `data/interim/`, `data/processed/`, and every
  gitignored `reports/*` subdirectory are excluded from version control
  (see the root `.gitignore`). Reproducing this project's exact numbers
  requires your own WRDS access to the same six datasets (see
  `DATA_DICTIONARY.md` for exact table/field expectations).
- **Deterministic given the same input data:** `project.random_seed` (42,
  `configs/config.yaml`) seeds every stochastic estimator; the walk-forward
  splits, feature formulas, and portfolio construction are otherwise
  non-random.
- **Full pipeline, in order** (see the root `README.md` for full detail on
  each step):
  ```bash
  python scripts/run_data_validation.py
  python scripts/run_data_processing.py
  python scripts/run_feature_engineering.py
  python scripts/run_modeling.py
  python scripts/run_backtest.py
  ```
- **Tests never require real data.** The full test suite
  (`make test` / `pytest`) runs entirely against synthetic fixtures, so
  `make check` and CI never need WRDS access.
