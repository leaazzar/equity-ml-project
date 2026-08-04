# Phase 4 Design: Model Development

**Status: design finalized, decisions confirmed by the project owner,
first-pass implementation complete.** This is the plan for Phase 4 (see
`PLAN.md`; results from running it are in `MODEL_REPORT.md`), written after
reading the full Phase 1-3 codebase and documentation (`README.md`,
`PLAN.md`, this project's private task tracker, `DATA_DICTIONARY.md`,
`FEATURE_DICTIONARY.md`, `MERGE_REPORT.md`, `UNIT_AUDIT_REPORT.md`, all of
`src/`, all of `tests/`). The three decisions originally flagged here as
**DECISION NEEDED** — explicitly called out in `PLAN.md` ("Open questions ...
do not guess at them") and in that private task tracker ("Do not start
modeling until the follow-ups above are resolved or explicitly deferred by
the project owner") as requiring the project owner's explicit answer before
any code is written — were put to the project owner and resolved on
2026-08-03 (see "Confirmed decisions" below). Implementation then proceeded
component by component per the module layout below, in the order given at
the end of this document.

## Inputs this phase builds on

- `data/processed/features_model_ready.parquet` — winsorized, cross-sectionally
  rank-scaled `[-1, 1]` features, `NaN` for non-`is_investable` rows. Primary
  input for model training (already leakage-checked, already on a bounded
  scale suitable for regularized linear models and tree models alike).
- `data/processed/features_raw.parquet` — interpretable-unit counterpart,
  useful for diagnostics/explainability plots in real units, not for training.
- `data/processed/master_panel.parquet` — source of `ret_adj` (the
  delisting-adjusted return, needed for target construction — it is **not**
  in either feature panel) and of `SICCD`/`is_investable`/`MthCalDt`/`PERMNO`
  for universe filtering, joining, and sector bucketing.
- `src/feature_engineering/registry.py`'s `model_feature_names()` — the
  50-feature recommended list (51 features minus `reversal_1m`, perfectly
  collinear with `mom_1m`). Phase 4 should import this rather than
  hand-listing feature columns, so a future registry change (new feature
  added/retired) propagates automatically.

---

## 1. Target construction

**Definition:** for each `(PERMNO, MthCalDt=t)` row with an investable-universe
feature vector, the label is the forward cumulative return over the
prediction horizon `h`, built from `ret_adj` (not raw `MthRet` — `ret_adj`
already carries the delisting-return correction from `data_processing`, so a
delisted stock's true realized return, including its final compounded
delisting leg, is what gets predicted, not an artificially-truncated return):

```
y(PERMNO, t) = exp( sum_{i=1..h} log(1 + ret_adj(PERMNO, t+i)) ) - 1
```

i.e. the same log-compounding convention `time_series_features.py` already
uses for `mom_3m`/`mom_6m`/etc., applied forward instead of backward, for
consistency with the rest of the codebase.

**Point-in-time safety:** the label is attached by a **forward** shift
(`groupby("PERMNO")["ret_adj"].shift(-i)`, summed in log space over `i=1..h`),
computed once per `PERMNO` on the full return series and then merged onto the
feature panel by `(PERMNO, t)` — mirroring `accounting.py`'s
groupby-then-shift pattern, just shifted the opposite direction. A row's
label is `NaN` whenever fewer than `h` future calendar-contiguous
observations exist for that `PERMNO` (i.e. within `h` months of the sample's
end, or within `h` months of a delisting event with no further rows) — those
rows are dropped from training/evaluation, never imputed, matching every
other missing-value rule already established in this codebase
(`FEATURE_DICTIONARY.md`: "no feature is ever imputed").

**Calendar-gap handling:** `validation.check_no_calendar_gaps` already
guarantees zero gaps in `ret_adj`'s underlying series, so a plain forward
shift is safe — no risk of silently bridging across a missing month the way
an unguarded `.shift()` on a gapped series could.

**Why a raw/excess-return label, not a rank/classification label:** a
continuous label preserves the magnitude information a regression model (and
later, a portfolio-construction step that wants a *score*, not just a
sign) can use, and lets Section 8's IC/regression-based evaluation work in
the usual asset-pricing sense. A **cross-sectionally demeaned** version,
`y - mean(y | t, is_investable)`, is also worth carrying alongside the raw
label — it removes the market-wide return common to every stock that month
(that component is neither forecastable from firm characteristics nor
useful for a long-short portfolio, which is dollar-neutral by construction —
see Section 6) and empirically stabilizes the effective sample size point.
Both should be materialized as label columns so a model can be trained on
either; the long-short backtest in Section 6-7 only needs relative ranking
within a month, so training on the demeaned label is expected to be the
primary target, with the raw label kept for reporting realized (not
excess) portfolio returns.

**Universe (CONFIRMED — `is_investable` as-is, no additional filter):** only
rows with `is_investable == True` at time `t` are used for training and
evaluation, with no further restriction beyond that flag's existing
definition. The project owner confirmed this on 2026-08-03 (the recommended
option). This matches how the model-ready features were already scaled
(winsorization/ranking references the investable universe only —
`transforms.py`), and keeps the model's job "rank stocks that are actually
part of the reference cross-section" rather than also having to implicitly
learn to distinguish investable from non-investable rows, which
`is_investable` already does deterministically. As always in this codebase,
`is_investable` is accepted with its documented limitations — a coarse
price/size/trading-activity proxy, not a verified common-share or
index-membership filter (`FEATURE_DICTIONARY.md`) — rather than layering on
an unverified additional restriction (e.g. an ad hoc `SICCD` exclusion) that
this project has no more evidentiary basis for than `is_investable` itself.

**CONFIRMED — prediction horizon `h=1` month, monthly rebalance.** `PLAN.md`
asked *"What is the intended prediction horizon and rebalance frequency?"*;
the project owner confirmed **`h=1` month, monthly rebalance** on 2026-08-03
(the recommended option below), as the primary horizon for Phase 4. This
choice determines the label (`h` above), the walk-forward purge/embargo width
(Section 3, one month), and the backtest's rebalance cadence (Section 6).
Rationale (given the panel is monthly-only — no daily data,
`FEATURE_DICTIONARY.md`'s "Requested features that could not be created"):
`h=1` maximizes usable observations per walk-forward fold, which matters
given the modest 27-year (324-month) sample, and every existing return-based
feature (`mom_*`, `vol_12m`, `beta_24m`, etc.) is already built and validated
at monthly granularity, so a 1-month target keeps train and feature timing
aligned. The trade-off accepted: highest turnover/transaction-cost drag and
the noisiest single-month label of the three candidates considered (`h=3`
quarterly, `h=12` annual — both remain available as secondary/robustness
targets later, since the target module is horizon-parameterized regardless,
but are not part of the first implementation pass).

---

## 2. Train/validation/test methodology

Panel financial data with monthly, cross-sectionally-correlated observations
violates the IID assumption behind an ordinary random train/test split in two
ways this design addresses explicitly:

1. **Temporal leakage.** A random row-level split would let a model trained
   on, say, March 2015 data "see" January 2015 in training and be tested on
   February 2015 — future information about the same market regime leaking
   backward. **Fix:** all splits are chronological — every training fold's
   dates precede every corresponding test fold's dates. No row is ever
   assigned to train and test by anything other than its `MthCalDt`.
2. **Cross-sectional + serial correlation inflating apparent test
   performance.** Even with a chronological split, if the test period starts
   the month immediately after training ends, a 1-month-horizon label's
   information (return over `t+1`) can overlap with a training-set label's
   window if `h > 1` (e.g. a `h=3` label at the last training month uses
   returns through `t+3`, which can extend into the first "test" months).
   **Fix: purging + embargo** (Lopez de Prado's terminology, already
   conceptually present in this codebase's own reporting-lag/shelf-life
   design for accounting data — Phase 4 applies the same discipline to the
   *label*, not just the features): drop the last `h` months of every
   training fold (purge), and additionally drop the first `h` months of
   every test fold from *scoring* against a model that could have been
   influenced by adjacent training labels (embargo). With `h=1`, purge/embargo
   is a single month and the effect is small; it becomes material at `h=3`
   or `h=12`, which is one more reason `h`'s choice (Decision #1) matters
   structurally, not just economically.

**Three-way split, chronological:**

- **Train:** used to fit model parameters.
- **Validation:** used only for hyperparameter selection (Section 5) and
  model-family comparison (Section 4) — never seen by the final fit.
- **Test:** touched exactly once per walk-forward fold, only to record
  out-of-sample predictions for the backtest (Sections 6-8). No hyperparameter
  or model-family decision is ever made by looking at test-fold performance.

Because the full history is only ~27 years (324 months) and Phase 4 is asked
for a walk-forward *expanding-window* protocol (not a single static
train/val/test cut), the three-way split above is realized **inside each
walk-forward step** (Section 3), not as one global partition of the whole
sample — a single static split would waste most of the sample's later years
on "test only" and never let the model retrain on regime changes, which is
exactly what expanding-window walk-forward is for.

**Initial training window:** needs enough history for the longest-lookback
feature (`skew_36m`, 36 months trailing, requiring 24 non-missing) and the
longest-shelf-life accounting cycle to have stabilized past the panel's early
ramp-up (`MERGE_REPORT.md`: Compustat coverage in 1999 is a fraction of later
years — 579 vs. ~5,000-7,800 gvkey-matched PERMNOs). A minimum initial
training window of **5 years** (60 months) is proposed, starting the first
walk-forward test fold no earlier than **2005** — leaving 1999-2004 as
warm-up history that's available to *features* (they're already point-in-time
correct and usable from month 1) but excluded from being a *test* period,
since early-sample Compustat coverage is thin enough to make performance
estimates from that window unreliable regardless of model quality. This is a
recommendation to confirm alongside Decision #1, not an irreversible
structural choice — later.

---

## 3. Walk-forward expanding-window evaluation

```
Fold 1: train [1999-01 .. 2004-12]                        -> test [2005-01+purge .. 2005-12]
Fold 2: train [1999-01 .. 2005-12]                        -> test [2006-01+purge .. 2006-12]
Fold 3: train [1999-01 .. 2006-12]                        -> test [2007-01+purge .. 2007-12]
...
Fold N: train [1999-01 .. 2024-12]                        -> test [2025-01+purge .. 2025-12]
```

- **Expanding, not rolling:** each fold's training window starts at the panel
  start and grows; it is never truncated from the left. This is the
  conventional choice for this data volume (rolling windows would discard
  early history that's still informative and would only be preferable if
  there were a specific non-stationarity argument for it, e.g. a suspected
  structural break — none is established here).
- **Refit cadence:** annual (one refit per calendar year of test data) is
  proposed over monthly refitting, primarily for computational tractability
  given nested nested hyperparameter search (Section 5) — refit granularity
  is a tunable, not a hard architectural constraint, and finer refit
  cadences can be evaluated as a robustness check once the pipeline exists.
- **Purge/embargo width = `h`** (the prediction horizon from Decision #1),
  applied at every fold boundary as described in Section 2.
- **Within each fold's training window**, the train/validation split used
  for hyperparameter tuning (Section 5) is itself a smaller
  purged/embargoed walk-forward split (nested CV) — e.g. the trailing 20% of
  each training window, chronologically, held out as validation, with the
  same purge/embargo rule applied at that inner boundary too.
- **Output:** one out-of-sample prediction per `(PERMNO, t)` test row across
  all folds, concatenated into a single panel of predictions spanning
  2005-2025 (or whatever the confirmed start year is) — this is what
  Sections 6-8 consume. Critically, every prediction in this panel was
  produced by a model that never had access to that row's own label or to
  any label within `h` months of it, so the backtest is a genuine
  out-of-sample simulation, not an in-sample fit dressed up as one.

---

## 4. Benchmark models

Ordered from weakest/simplest to strongest, each included so later models'
value-add is measurable against something, not just against "no model":

1. **Zero-skill / equal-weight benchmark.** Every investable stock gets an
   identical score each month — realized long-short portfolio return should
   be ~0 in expectation. Establishes the floor.
2. **Single-factor momentum sort.** Rank by `mom_12_1` alone (the
   already-computed, skip-month 12-month momentum feature — the most
   canonical, best-documented cross-sectional predictor in the equity
   literature). A model that can't beat this on a risk-adjusted basis isn't
   adding value over a one-line factor sort.
3. **Fama-MacBeth cross-sectional regression.** Month-by-month OLS of
   forward return on the full feature set, coefficients averaged over time
   (Newey-West or similar for the standard errors) — the standard academic
   linear benchmark, and a natural fit given `statsmodels` is already a
   project dependency (`pyproject.toml`).
4. **Regularized linear (Ridge / Lasso / ElasticNet, scikit-learn).**
   Same feature set as the ML models below, but linear — isolates "does
   nonlinearity/interaction-modeling actually help" as a testable question
   rather than an assumption.
5. **Tree ensembles (Random Forest, Gradient Boosting).** `scikit-learn`'s
   `RandomForestRegressor` and `HistGradientBoostingRegressor` are already
   available with zero new dependencies (`pyproject.toml` already has
   `scikit-learn>=1.4`). A more specialized gradient-boosting library
   (LightGBM/XGBoost/CatBoost) is **not** currently a project dependency —
   proposed as a fast-follow once the sklearn-only pipeline is validated
   end-to-end, flagged here as a small, explicit dependency-addition
   decision for later rather than assumed now.
6. **(Optional, later) A small MLP** (`sklearn.neural_network.MLPRegressor`,
   also a zero-new-dependency option) as a nonlinear-but-not-tree-based
   comparison point, lower priority than 1-5.

Every model in this list is evaluated through the *same* walk-forward
harness (Section 3) and the *same* metrics (Section 8) — no benchmark gets a
different, more favorable evaluation protocol than the "real" models.

---

## 5. Hyperparameter tuning

- **Search space per model family**, kept intentionally small given the
  computational cost of nested walk-forward CV (each hyperparameter
  candidate must be refit across every fold's inner train/validation split):
  - Ridge/Lasso/ElasticNet: regularization strength `alpha` (log-spaced
    grid), ElasticNet also `l1_ratio`.
  - Random Forest: `max_depth`, `min_samples_leaf`, `max_features` — depth
    and leaf-size matter far more than tree count for panel data with this
    much cross-sectional noise; tree count itself is set high and fixed
    (diminishing returns, not worth tuning).
  - HistGradientBoostingRegressor: `max_depth` or `max_leaf_nodes`,
    `learning_rate`, `l2_regularization`, with early stopping on the inner
    validation fold standing in for `n_estimators` tuning.
- **Search method:** a small explicit grid (not random/Bayesian search) for
  the first implementation pass — the search space above is deliberately
  narrow enough that grid search is tractable and fully reproducible (no
  extra randomness beyond the models' own `random_state`, which is fixed
  from `configs/config.yaml`'s existing `project.random_seed: 42`).
- **Selection metric:** validation-fold **Information Coefficient** (Spearman
  rank correlation between predicted score and realized forward return,
  averaged across validation months) rather than MSE — MSE rewards getting
  the *magnitude* of a noisy return right, which is not what a long-short,
  rank-based portfolio (Section 6) actually needs; IC directly measures
  cross-sectional ranking quality, which is what gets monetized. MSE is
  still logged as a secondary diagnostic.
- **Refit cadence for tuning:** tied to the annual refit cadence in Section
  3 — hyperparameters are re-selected once per fold, not once globally and
  not every month, balancing adaptivity against overfitting the
  hyperparameters themselves to a shrinking effective validation sample.

---

## 6. Portfolio construction

- **Signal → weights:** each month `t`, every investable stock with a
  non-missing model prediction is ranked by predicted score.
- **Long/short buckets:** long the top decile, short the bottom decile
  (10/10), the standard convention in this literature and a reasonable
  starting granularity given typical monthly investable-universe size in
  this panel (thousands of names in most months per
  `reports/feature_engineering/universe_coverage.csv`) — decile buckets keep
  each leg's stock count in the hundreds, not so few that idiosyncratic
  single-name risk dominates, not so many that the signal is diluted toward
  the median. Quintile (5/5) is a natural robustness check, not a
  structural alternative requiring a different pipeline.
- **Weighting within a leg:** equal-weighted as the default (simplest,
  least assumption-laden, standard first-pass convention), with
  score-weighted (predicted-score-proportional, then normalized) as a
  documented alternative the same portfolio-construction module supports via
  a config flag — not a second code path.
- **Dollar-neutrality:** long leg and short leg sized to equal dollar
  exposure each month (100% long / 100% short, 0% net), removing
  market-directional exposure so the portfolio's return is attributable to
  the cross-sectional ranking signal, not to beta — this is also why the
  demeaned label from Section 1 is the natural training target: it's
  optimizing for exactly the quantity this portfolio construction monetizes.
- **Sector neutrality (optional, config flag):** `SICCD` is present in
  `master_panel.parquet` (0% null per `DATA_DICTIONARY.md`) and can be
  joined onto the prediction panel by `(PERMNO, t)` to demean predicted
  scores within `SICCD` groups before ranking, or to construct long/short
  legs within-sector rather than universe-wide — proposed as an optional
  robustness variant, not the default, since `SICCD`'s point-in-time
  correctness for this purpose hasn't been separately audited the way the
  feature-relevant fields have (it's a straightforward monthly field, same
  provenance as `MthPrc`/`MthCap`, so risk is low, but this should be stated
  rather than silently assumed).
- **Rebalance frequency:** tied directly to Decision #1's horizon `h` — a
  portfolio built from an `h`-month-ahead prediction rebalances every `h`
  months, holding each position for the horizon the prediction actually
  targets (rebalancing monthly against a 3-month-ahead signal would mean
  holding overlapping, staggered tranches — a legitimate but materially more
  complex construction, deferred as a later enhancement rather than the
  first implementation).

---

## 7. Backtesting

- **Return realization:** `ret_adj` (the same delisting-adjusted return used
  for the label) applied to each fold's realized portfolio weights —
  ensures the backtest isn't silently benefiting from ignoring delisting
  losses the way a naive `MthRet`-only backtest would (a well-known source
  of overstated backtest performance in the literature this repo's own
  `MERGE_REPORT.md` already cites, Shumway 1997).
- **Transaction costs:** this project has no bid-ask-spread or commission
  data (not in any of the six raw extracts — see `DATA_DICTIONARY.md`). A
  flat, explicitly-labeled placeholder assumption is proposed — e.g. **10
  basis points one-way** (20 bps round-trip) per unit of turnover, applied
  to the fraction of the portfolio actually turned over each rebalance
  (names entering/exiting the long or short leg, or re-weighted) — a
  commonly cited order-of-magnitude for liquid US equities, stated plainly
  as an assumption rather than a fact, with the net-of-cost Sharpe reported
  alongside a gross-of-cost figure and a sensitivity table (0 / 5 / 10 / 20 /
  50 bps) so the reader can see how cost-sensitive the strategy is rather
  than trusting one hardcoded number.
- **Turnover accounting:** computed explicitly each rebalance (sum of
  absolute weight changes / 2), reported as its own diagnostic — needed
  both for the cost calculation above and as a standalone strategy-capacity
  signal.
- **No look-ahead in the backtest itself:** every weight at month `t` derives
  only from a prediction whose model was trained on data through
  `t - purge_window` (Section 3); every realized return used to score that
  weight is `ret_adj` over `[t, t+h]`, i.e. strictly after the weight was
  knowable. This is the same discipline already enforced for feature
  construction (`validation.check_no_leakage_via_truncation`) — Phase 4
  should add an analogous automated check for the *backtest* (assert every
  test-fold prediction's model-fit cutoff date is chronologically before
  that prediction's own feature date, and that no realized-return window
  used for scoring starts before the prediction date), not just trust the
  walk-forward loop's bookkeeping by construction.

---

## 8. Performance evaluation

- **Return-based metrics** (on both gross- and net-of-cost long-short
  return series): annualized return, annualized volatility, Sharpe ratio,
  Sortino ratio, maximum drawdown, hit rate (fraction of positive months).
- **Signal-quality metrics** (independent of any portfolio-construction
  choice, so they isolate model quality from portfolio-construction
  choices): monthly cross-sectional **Information Coefficient** (Spearman
  rank correlation, predicted score vs. realized forward return), its
  time-series mean/std/t-stat, and IC decay (does predictive power persist
  if measured against `t+2h`, `t+3h`, not just `t+h`).
- **Factor-exposure regression:** long-short return series regressed on
  the Fama-French 5 + momentum factors already merged into the panel
  (`ff_mktrf`, `ff_smb`, `ff_hml`, `ff_rmw`, `ff_cma`, `ff_umd`) — reports
  **alpha** (the return unexplained by known factors — the actual quantity
  of interest for a claimed "new" signal) and factor loadings, so a result
  that's "just momentum in disguise" or "just a value tilt" is visible
  rather than mistaken for genuine new signal. This is a natural fit given
  `ff_*` columns are already point-in-time merged onto the panel — no new
  data source needed.
- **Per-fold and pooled reporting:** every metric above reported both
  per walk-forward fold (to see stability/regime-dependence over
  2005-2025) and pooled across the full out-of-sample period (the headline
  number).
- **Benchmark comparison table:** every metric computed identically for
  every model in Section 4's list, in one comparison table — the actual
  deliverable answering "did the ML model beat momentum / Fama-MacBeth /
  equal-weight."

---

## 9. Explainability

- **Global feature importance:** permutation importance (model-agnostic,
  works identically across linear/tree/ensemble models, no new dependency —
  `sklearn.inspection.permutation_importance`) computed on each fold's
  validation set, aggregated across folds to see which features are
  *consistently* important versus important in only one regime.
- **Linear-model coefficients** reported directly (already interpretable)
  for the Ridge/Lasso/Fama-MacBeth benchmarks — a natural cross-check
  against the tree models' permutation importances (do both model families
  agree on which characteristics matter?).
- **Per-category attribution:** using the registry's category labels (size,
  value, momentum, reversal, quality, growth, leverage, liquidity,
  volatility — the same nine categories already in `FEATURE_DICTIONARY.md`)
  to roll individual feature importances up into a category-level summary —
  directly answers "is this a value model, a momentum model, a quality
  model," which is a more economically legible question than a 50-feature
  importance list.
- **Stability over time:** feature/category importance plotted per
  walk-forward fold, not just pooled — a feature that mattered in
  2008-2009 and nowhere else is a different finding than one that's
  consistently important, and both are worth surfacing rather than
  averaging away.
- **SHAP** (more detailed local explanations, interaction effects) is
  **not** proposed for the first pass — it would be a new dependency
  (`shap` is not in `pyproject.toml`) and permutation importance plus
  per-category attribution already answers the primary explainability
  questions this project needs; flagged as a possible fast-follow, same
  status as LightGBM in Section 4, not assumed.

---

## 10. Final report generation

Following the existing repo convention exactly (`MERGE_REPORT.md`,
`UNIT_AUDIT_REPORT.md`, `FEATURE_DICTIONARY.md` — durable, git-tracked
Markdown explaining *decisions*, generated diagnostics in gitignored
`reports/`):

- **`MODEL_REPORT.md`** (git-tracked, root-level, alongside the existing
  three phase reports) — the durable record of what was actually built:
  target definition used, walk-forward configuration, model families and
  their tuned hyperparameters per fold, and a pointer to the generated
  diagnostics. Written by hand/generated once modeling is real, analogous to
  how `MERGE_REPORT.md` documents `data_processing`'s actual decisions
  rather than being auto-dumped.
- **`reports/modeling/`** and **`reports/backtest/`** (gitignored,
  regeneratable — `reports/figures/` and `reports/tables/` already exist as
  empty placeholders in the repo for exactly this, per `reports/README.md`):
  - `summary.md` — the per-model comparison table from Section 8, formatted
    like `feature_engineering`'s `summary.md` (validation-style pass/fail +
    metric tables).
  - `predictions.parquet` — the full out-of-sample prediction panel from
    Section 3.
  - `metrics.json` / `metrics.csv` — every metric from Section 8, per fold
    and pooled, machine-readable (mirroring
    `feature_registry.json`/`.csv`'s dual JSON+CSV convention).
  - Figures: cumulative return curves (gross/net), IC time series, feature/
    category importance bar charts, factor-exposure regression table
    rendered as an image or Markdown table.
- **Generation, not hand-authoring:** every number and figure in these
  reports comes from pipeline code (`reporting.py` in the new
  `src/equity_ml/models/` and `src/equity_ml/backtest/` packages,
  mirroring `feature_engineering/reporting.py`'s pattern precisely — CSV/JSON
  writers plus an f-string-assembled `summary.md`), never hand-edited
  numbers, matching this repo's established practice.

---

## Proposed module layout

Mirrors the existing three-phase pattern (`config.py` / pipeline
orchestration / CLI / registry-or-validation / reporting /
`tests/<phase>/conftest.py` + synthetic fixtures) exactly, split across the
two placeholder packages that already exist for this purpose
(`src/equity_ml/models/`, `src/equity_ml/backtest/` — currently
`TODO(WRDS)` stubs):

```
src/equity_ml/models/
  config.py        # ModelConfig: horizon h, universe filter, walk-forward
                    # window sizes, refit cadence, purge/embargo, model list,
                    # hyperparameter grids, random_seed (from config.yaml)
  targets.py        # forward-return label construction (Section 1)
  splits.py          # expanding-window + purge/embargo split generator (Sections 2-3)
  baselines.py       # equal-weight, momentum-sort, Fama-MacBeth (Section 4)
  estimators.py       # sklearn model wrappers/factory (Ridge/Lasso/RF/HGB) (Section 4)
  tuning.py           # grid search over each fold's inner validation split (Section 5)
  training.py          # orchestrates fit-per-fold -> out-of-sample predictions
  validation.py         # leakage/consistency checks specific to modeling (fold-boundary
                        # purge verification, no-train-label-in-test-window checks) —
                        # same "return ValidationResult, never raise" pattern as
                        # feature_engineering.validation
  diagnostics.py         # importances, IC series, per-fold metric tables (Section 9)
  reporting.py            # writes reports/modeling/ (Section 10)
  pipeline.py              # run_pipeline(...) end-to-end entry point
  cli.py / __main__.py      # argparse CLI, mirrors feature_engineering.cli exactly

src/equity_ml/backtest/
  config.py        # BacktestConfig: decile/quintile choice, weighting scheme,
                    # dollar-neutrality, sector-neutrality flag, transaction-cost bps
  portfolio.py       # predicted scores -> long/short weights (Section 6)
  engine.py            # applies weights to ret_adj, computes turnover + costs (Section 7)
  performance.py         # Sharpe/Sortino/drawdown/IC/factor-regression (Section 8)
  reporting.py             # writes reports/backtest/ (Section 10)
  pipeline.py               # run_pipeline(...) end-to-end entry point
  cli.py / __main__.py       # argparse CLI

scripts/run_modeling.py       # thin wrapper, mirrors scripts/run_feature_engineering.py
scripts/run_backtest.py        # thin wrapper

tests/models/         # synthetic fixtures + tests, mirrors tests/feature_engineering/
tests/backtest/         # synthetic fixtures + tests
```

Every module above follows conventions already established in
`src/feature_engineering/` and `src/data_processing/` (confirmed by
re-reading those packages before writing this document): frozen
`@dataclass` configs with documented defaults, `logging.getLogger(__name__)`
(not the `equity_ml.logging_utils.get_logger` wrapper, which this codebase
uses only in `wrds_loader.py`), validation functions that return a result
object rather than raising, dual JSON+CSV machine-readable outputs plus a
hand-composed `summary.md`, and `tests/<phase>/conftest.py` synthetic
fixtures — never real data — covering named edge cases the way
`tests/feature_engineering/conftest.py` does.

---

## Confirmed decisions (project owner, 2026-08-03)

`PLAN.md`'s explicit blockers, resolved:

1. **Prediction horizon `h` and rebalance frequency:** `h=1` month, monthly
   rebalance (Section 1's recommended option). Affects the label, the
   purge/embargo width, and the backtest cadence throughout.
2. **Target universe:** `is_investable` as currently defined, no additional
   restriction (Section 1's recommended option).
3. **Residual data-quality/scope follow-ups from Phases 1-3**
   (`MERGE_REPORT.md`'s "Known limitations": unexplained CCM coverage gaps,
   missing `DelRet` left unadjusted rather than proxied, no `DelPERMNO`
   return-chaining, no quarterly Compustat, unresolved WRDS code
   definitions): **explicitly deferred**, carried forward into Phase 4 with
   their existing documented handling (flag-and-leave-null, never silently
   guessed or fixed). This formally closes the "do not start modeling until
   resolved or explicitly deferred" gate this project's task tracker held
   over Phase 4.

Implementation proceeds component by component in the order presented above
(targets -> splits -> baselines -> estimators -> tuning -> training ->
portfolio -> backtest engine -> performance -> explainability -> reporting),
each with its own tests against synthetic fixtures before moving to the
next, matching how Phases 1-3 were each built and tested incrementally.
