# Feature Dictionary

Documents every one of the 51 point-in-time firm characteristics produced by
`src/feature_engineering/` from `data/processed/master_panel.parquet`. This
file is **generated from, and must stay in sync with, the machine-readable
registry** (`src/feature_engineering/registry.py`,
`reports/feature_engineering/feature_registry.json`/`.csv`) — `pipeline.py`'s
`registry_source_columns_exist` validation check fails loudly if a computed
feature's declared source columns don't actually exist, which is the
guardrail against this document silently drifting from the code.

**Status:** point-in-time feature engineering complete, including a
dedicated **unit-integrity audit** (see `UNIT_AUDIT_REPORT.md`) that found
and fixed two real bugs in the original implementation — every formula
below already reflects the fix. No labels, targets, or models are built
here — see `PLAN.md`.

## How to read this document

Every feature entry lists:

- **Formula** — the exact calculation, **with units stated explicitly** for
  every monetary quantity (see "Units" below — this was added by the audit;
  earlier versions of this document did not state units and that omission
  is exactly what let a real ~1000x scaling bug go unnoticed).
- **Economic interpretation** — why the characteristic is included.
- **Source columns** — every input column it reads (validated against the
  actual panel at pipeline run time).
- **Lookback window** — how far back the calculation looks.
- **Minimum observations** — how much history is required before the
  feature is populated (rather than left `NaN`).
- **Point-in-time notes** — the specific mechanism that prevents look-ahead
  bias for this feature.
- **Missing-value treatment** — exactly when and why a value is `NaN`.
  **No feature is ever imputed** — see "Missing values" below.
- **Outlier treatment** — always the same, described once here: raw values
  are never altered; the model-ready layer winsorizes at configurable
  percentiles (default 1%/99%) computed **within that month's investable
  universe only**, then rank-scales to `[-1, 1]`.
- **Collinearity**, where a feature is flagged as redundant with another.
- **Known limitations**, where applicable.

## Units — read this before using any monetary feature

**Every unit stated below was empirically verified against the actual raw
extracts in this project — none was assumed from generic "standard
convention" folklore.** See `UNIT_AUDIT_REPORT.md` for the full methodology
(three independent cross-checks) and findings. Summary:

| Field | Source | Verified unit |
| --- | --- | --- |
| `MthCap` | CRSP | **thousands** of USD |
| `ShrOut` | CRSP | **thousands** of shares |
| `MthPrc` | CRSP | USD per share (unscaled) |
| `MthVol` | CRSP | **actual shares** (not thousands) |
| `cst_csho` | Compustat | **millions** of shares |
| `cst_prcc_f` | Compustat | USD per share (unscaled) |
| All other `cst_*` dollar fields | Compustat | **millions** of USD |

**Canonical unit adopted throughout this package: millions of USD.**
`units.py` converts every CRSP monetary/share quantity into this unit
(`mktcap_millions`, `dollar_volume_millions`, `shares_outstanding_actual`)
*before* any other module touches it. Every monetary raw feature in
`features_raw.parquet` — not only the ones combining CRSP and Compustat —
is in millions of USD, so the whole panel has one consistent, documented
unit throughout.

This matters because CRSP and Compustat use *different* native units:
directly combining `MthCap` ($ thousands) with a `cst_*` dollar field ($
millions) — by ratio or by sum — silently produces a value off by ~1000x.
This exact bug existed in `value_bm`, `value_earnings_yield`,
`value_cf_yield`, `value_sales_to_price`, and `enterprise_value` (and hence
`value_ebitda_to_ev`/`value_cf_to_ev`) in an earlier version of this code,
and a related bug (mixing `MthVol`, actual shares, with raw `ShrOut`,
thousands of shares) existed in `liquidity_share_turnover`. Both are fixed;
see `UNIT_AUDIT_REPORT.md` for the before/after numbers.

## Inputs and the two output layers

- **Input:** `data/processed/master_panel.parquet` (from `data_processing`)
  plus `data/interim/compustat_fundamentals_annual.parquet` (for
  year-over-year growth rates — see "Growth rates," below).
- **`data/processed/features_raw.parquet`** — every characteristic in its
  natural, interpretable unit (monetary features in millions of USD — see
  "Units" above). Nothing is winsorized, ranked, or standardized. Every row
  from the master panel is preserved, whether or not it's in the investable
  universe.
- **`data/processed/features_model_ready.parquet`** — the same
  characteristics after the one, uniform, documented transform: winsorize
  (month-by-month, investable-universe-only thresholds) then cross-sectional
  percentile-rank scaled to `[-1, 1]` (also month-by-month,
  investable-universe-only). Non-investable rows are `NaN` here — a
  cross-sectional rank relative to a universe a security isn't part of
  isn't well-defined. See `transforms.py`.
- **Recommended model feature list:** `registry.model_feature_names()` —
  every feature *except* `reversal_1m`, which is perfectly collinear with
  `mom_1m` (identical values; see "Collinearity" below). Both output
  layers still contain all 51 features; this function only defines which
  columns a model should actually be trained on.

## The investable universe (`is_investable`)

**This extract has no CRSP `SHRCD`/`EXCHCD` fields** — the standard, precise
way to identify U.S. common stock and major-exchange listings (see
`DATA_DICTIONARY.md`). `is_investable` is a **coarse, approximate research-
universe proxy**: non-missing price ≥ $1 (configurable), non-missing
positive market cap, and an actual trade that month (return flag ≠ `"NT"`).

**`is_investable` MUST NOT be described or relied upon as:**
- a verified common-share classification (no `SHRCD` field exists here);
- membership in any named index or benchmark — e.g. **not** "Russell 1000",
  **not** "S&P 500", **not** "NYSE/AMEX/NASDAQ common stock" — no such
  membership is checked or implied;
- a substitute for the precise share-code/exchange-code-based investability
  filters used in published academic asset-pricing research.

It is exactly what it is: price/size/trading-activity evidence, nothing
more. The master panel (and `features_raw.parquet`) is never filtered by
this flag — every row is kept; `is_investable` just marks which rows are
used as the cross-sectional reference for winsorization/ranking. See
`universe.py`.

## Point-in-time safety, in one place

- **Accounting data** (`cst_*` fields and everything derived from them —
  value, quality, growth, leverage) uses the exact same 6-month reporting
  lag and 12-month shelf-life expiry as `data_processing`'s master panel
  (see `MERGE_REPORT.md`). Growth rates get this treatment too: they're
  computed on the gvkey/fiscal-year sequence first (`groupby` + `shift`,
  never by shifting the monthly panel), then merged in with the identical
  lag/expiry logic — see `accounting.py`.
- **Return-based features** (momentum, reversal, volatility, liquidity) use
  only backward-looking, calendar-contiguous windows (verified: zero
  calendar gaps across every PERMNO — see `validation.check_no_calendar_gaps`),
  with `min_periods` always equal to the declared minimum observation
  count — a window is either fully populated or `NaN`, never partial.
- **Cross-sectional operations** (relative size, winsorization, ranking) are
  computed **within each month independently**, using only that month's own
  investable-universe cross-section — verified automatically by
  `validation.check_cross_sectional_isolation`.
- **No look-ahead**: verified automatically by
  `validation.check_no_leakage_via_truncation`, which rebuilds every feature
  from data truncated at several sample cutoff dates and confirms the
  cutoff month's values are bit-for-bit identical to building from the full
  sample — if any feature used future data, truncating the input would
  change its value, and it doesn't.

## Missing values, outliers, and derived accounting quantities

- **No feature is silently imputed anywhere in this package.** Missing
  source data always produces a missing feature value.
- Two exceptions are explicitly documented (not silent): `book_equity`
  treats missing `PSTK`/`TXDITC` as 0 (most firms genuinely have none), and
  `quality_operating_profitability` treats missing `COGS`/`XSGA`/`XINT` as 0
  (the Fama-French 2015 construction's own convention). Both are stated
  again on the specific features they apply to.
- **`at_proxy` (total-assets proxy, $ millions):** this extract has no `AT`
  field. `at_proxy = cst_lt + cst_seq` (total liabilities + total
  stockholders' equity, both $ millions), via the accounting identity
  Assets = Liabilities + Equity. Used by every asset-denominated ratio (ROA,
  asset growth, asset turnover, gross/operating profitability, leverage
  ratios) — flagged individually below wherever it's used.
- **`book_equity` ($ millions):** `cst_seq - cst_pstk(fillna 0) +
  cst_txditc(fillna 0)` — a simplified Fama-French book-equity construction
  using the fields this extract actually has.
- **`mktcap_millions` ($ millions):** `MthCap / 1,000` — see "Units" above.
- **`enterprise_value` / `total_debt` / `net_debt` (all $ millions):**
  `total_debt = cst_dlc + cst_dltt`; `enterprise_value = mktcap_millions +
  total_debt + cst_pstk(fillna 0) - cst_che` (minority interest omitted —
  not available in this extract; **uses `mktcap_millions`, not raw
  `MthCap`** — mixing the two was a real bug found during the unit audit);
  `net_debt = total_debt - cst_che`.
- `has_fundamentals` (on the raw panel): whether a row has usable,
  non-expired Compustat data at all — a single flag explaining missingness
  across every accounting-derived feature at once, rather than one flag per
  ratio. `link_matched` (from `data_processing`) similarly flags whether a
  CCM link was active that month.

## Collinearity

- **`reversal_1m` is perfectly collinear with `mom_1m`** (identical values —
  verified: correlation = 1.0 across the full 2.5M-row panel). Both remain
  in the data (raw and model-ready layers) for `reversal_1m`'s distinct,
  documented economic-interpretation label, but `reversal_1m` is excluded
  from `registry.model_feature_names()` — the recommended feature list for
  actual model training. Anyone building a custom feature list by hand
  should include at most one of the two.

## Requested features that could not be created (and why)

- **Zero-return frequency / LOT liquidity measure** — needs daily trading
  data (to count zero-return days within the month); this extract has only
  monthly CRSP observations.
- **Dividend yield / payout ratio** — no dividend field (`DVC` or
  equivalent) exists in this extract.
- **Return on invested capital (ROIC)** — would need a clean effective tax
  rate (income tax expense / pretax income), which isn't cleanly available
  here (`TXDI` is not the same as tax expense); a shaky proxy was avoided
  rather than fabricated.
- **Balance-sheet-method accruals** (Sloan 1996's original ΔWC-based
  construction) — the simpler cash-flow-statement method (`NI - OANCF`) was
  used instead (see `quality_accruals`), to control scope; both are
  legitimate, but implementing both wasn't judged worth the added
  complexity for a ~50-60 feature target.
- **Precise common-equity/exchange classification** — see "The investable
  universe" above; `SHRCD`/`EXCHCD` don't exist in this extract, and
  `crsp_names` (which has some related fields) has no validity-period
  columns, so it can't be used point-in-time without risking look-ahead
  bias (excluded entirely from `data_processing`, for the same reason).

## Feature reference

Total: **51 features across 9 categories.**

## SIZE

### `size_mktcap`

- **Formula:** mktcap_millions = MthCap ($ thousands) / 1,000, masked to NaN if <= 0. Units: $ millions.
- **Economic interpretation:** Firm size in market-value terms; the classic size (SMB) characteristic.
- **Source columns:** MthCap
- **Lookback window:** contemporaneous (current month)
- **Minimum observations:** n/a (single-period)
- **Point-in-time notes:** Uses only the current month's own market cap.
- **Missing-value treatment:** Missing source data -> missing feature (NaN). Never imputed.
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)
- **Known limitations:** MthCap's native unit ($ thousands) was empirically verified, not assumed — see UNIT_AUDIT_REPORT.md. This feature reports the converted $-millions value, matching the unit used by every other monetary feature in this package.

### `size_log_mktcap`

- **Formula:** log(mktcap_millions). Units: log($ millions).
- **Economic interpretation:** Log-scale size; reduces the influence of extreme mega-caps versus the raw level.
- **Source columns:** MthCap
- **Lookback window:** contemporaneous (current month)
- **Minimum observations:** n/a (single-period)
- **Point-in-time notes:** Uses only the current month's own market cap.
- **Missing-value treatment:** Missing source data -> missing feature (NaN). Never imputed.
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)

### `size_relative_mktcap`

- **Formula:** mktcap_millions / median(mktcap_millions among that month's investable universe). Dimensionless (a same-unit ratio; numerically identical whether computed from mktcap_millions or raw MthCap).
- **Economic interpretation:** Size relative to the typical investable firm that month (e.g. 2.3x the median).
- **Source columns:** MthCap, is_investable, MthCalDt
- **Lookback window:** contemporaneous (current month's cross-section only)
- **Minimum observations:** n/a (single-period)
- **Point-in-time notes:** Cross-sectional reference (the median) is computed using only that month's investable-universe rows — never other months, never the full sample.
- **Missing-value treatment:** Missing source data -> missing feature (NaN). Never imputed.
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)


## VALUE

### `value_bm`

- **Formula:** book_equity ($M) / mktcap_millions ($M), where book_equity = cst_seq - cst_pstk(fillna 0) + cst_txditc(fillna 0), all $ millions (Compustat's native unit); NaN if book_equity <= 0. Dimensionless.
- **Economic interpretation:** Book-to-market; classic HML value characteristic.
- **Source columns:** cst_seq, cst_pstk, cst_txditc, MthCap
- **Lookback window:** most recent point-in-time-available fiscal year
- **Minimum observations:** n/a (single-period)
- **Point-in-time notes:** book_equity uses the same 6-month-lagged, 12-month-shelf-life Compustat merge as every cst_* field; mktcap_millions is contemporaneous. Updates monthly with the current price, not annually-only as in the classic Fama-French June-updating convention — a deliberate simplification for this monthly panel.
- **Missing-value treatment:** NaN if book_equity <= 0 (negative book equity makes BM sign-flipped/uninterpretable) or if mktcap_millions missing/non-positive.
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)
- **Known limitations:** Numerator and denominator were confirmed to be in different native units (Compustat $ millions vs. CRSP $ thousands) during a dedicated unit audit — see UNIT_AUDIT_REPORT.md. mktcap_millions (MthCap / 1,000), not raw MthCap, is used here; this was a real bug in an earlier version of this feature.

### `value_earnings_yield`

- **Formula:** cst_ni ($M) / mktcap_millions ($M). Dimensionless.
- **Economic interpretation:** Earnings yield (E/P); inverse of the P/E ratio.
- **Source columns:** cst_ni, MthCap
- **Lookback window:** most recent point-in-time-available fiscal year
- **Minimum observations:** n/a (single-period)
- **Point-in-time notes:** cst_ni is lag/expiry-safe; mktcap_millions is contemporaneous.
- **Missing-value treatment:** NaN if mktcap_millions missing/non-positive.
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)
- **Known limitations:** Uses net income (NI); income before extraordinary items (IB) is an available alternative not computed separately, to avoid a near-duplicate feature. Denominator is market capitalization (mktcap_millions = MthCap / 1,000), never the raw per-share price alone — see UNIT_AUDIT_REPORT.md.

### `value_cf_yield`

- **Formula:** cst_oancf ($M) / mktcap_millions ($M). Dimensionless.
- **Economic interpretation:** Operating cash-flow yield; a value measure less sensitive to accrual manipulation than earnings yield.
- **Source columns:** cst_oancf, MthCap
- **Lookback window:** most recent point-in-time-available fiscal year
- **Minimum observations:** n/a (single-period)
- **Point-in-time notes:** cst_oancf is lag/expiry-safe; mktcap_millions is contemporaneous.
- **Missing-value treatment:** NaN if mktcap_millions missing/non-positive.
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)
- **Known limitations:** Denominator is market capitalization (mktcap_millions = MthCap / 1,000), never the raw per-share price alone.

### `value_sales_to_price`

- **Formula:** cst_revt ($M) / mktcap_millions ($M). Dimensionless.
- **Economic interpretation:** Sales-to-price; a value measure robust to earnings-management concerns.
- **Source columns:** cst_revt, MthCap
- **Lookback window:** most recent point-in-time-available fiscal year
- **Minimum observations:** n/a (single-period)
- **Point-in-time notes:** cst_revt is lag/expiry-safe; mktcap_millions is contemporaneous.
- **Missing-value treatment:** NaN if mktcap_millions missing/non-positive.
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)
- **Known limitations:** Denominator is market capitalization (mktcap_millions = MthCap / 1,000), never the raw per-share price alone.

### `value_ebitda_to_ev`

- **Formula:** cst_ebitda ($M) / enterprise_value ($M), where enterprise_value = mktcap_millions + total_debt + cst_pstk(fillna 0) - cst_che, all $ millions. Dimensionless.
- **Economic interpretation:** EBITDA yield on enterprise value; a capital-structure-neutral value measure.
- **Source columns:** cst_ebitda, MthCap, cst_dlc, cst_dltt, cst_pstk, cst_che
- **Lookback window:** most recent point-in-time-available fiscal year
- **Minimum observations:** n/a (single-period)
- **Point-in-time notes:** All cst_* inputs are lag/expiry-safe; mktcap_millions is contemporaneous.
- **Missing-value treatment:** NaN if enterprise_value missing/non-positive.
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)
- **Known limitations:** Enterprise value omits minority interest (not available in this extract). Uses mktcap_millions (MthCap / 1,000), not raw MthCap — mixing $-thousands MthCap directly with $-millions Compustat fields in enterprise_value was a real bug found during a dedicated unit audit; see UNIT_AUDIT_REPORT.md.

### `value_cf_to_ev`

- **Formula:** cst_oancf ($M) / enterprise_value ($M). Dimensionless.
- **Economic interpretation:** Operating cash flow yield on enterprise value.
- **Source columns:** cst_oancf, MthCap, cst_dlc, cst_dltt, cst_pstk, cst_che
- **Lookback window:** most recent point-in-time-available fiscal year
- **Minimum observations:** n/a (single-period)
- **Point-in-time notes:** All cst_* inputs are lag/expiry-safe; mktcap_millions is contemporaneous.
- **Missing-value treatment:** NaN if enterprise_value missing/non-positive.
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)
- **Known limitations:** Enterprise value omits minority interest (not available in this extract). Uses mktcap_millions, not raw MthCap — see UNIT_AUDIT_REPORT.md.


## MOMENTUM

### `mom_1m`

- **Formula:** ret_adj (current month)
- **Economic interpretation:** Most recent month's return.
- **Source columns:** ret_adj
- **Lookback window:** 1 month (contemporaneous)
- **Minimum observations:** 1
- **Point-in-time notes:** Uses only the current month's realized return, known as of that month's end.
- **Missing-value treatment:** Missing source data -> missing feature (NaN). Never imputed.
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)

### `mom_3m`

- **Formula:** exp(sum(log(1+ret_adj) over trailing 3 months)) - 1
- **Economic interpretation:** Cumulative return over the trailing 3 months (inclusive of the current month).
- **Source columns:** ret_adj, PERMNO
- **Lookback window:** 3 months, backward-looking
- **Minimum observations:** 3
- **Point-in-time notes:** Rolling window strictly requires 3 full prior (calendar-contiguous) monthly observations; no partial windows.
- **Missing-value treatment:** NaN unless all 3 months in the window are non-missing.
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)

### `mom_6m`

- **Formula:** exp(sum(log(1+ret_adj) over trailing 6 months)) - 1
- **Economic interpretation:** Cumulative return over the trailing 6 months (inclusive of the current month).
- **Source columns:** ret_adj, PERMNO
- **Lookback window:** 6 months, backward-looking
- **Minimum observations:** 6
- **Point-in-time notes:** Rolling window strictly requires 6 full prior (calendar-contiguous) monthly observations; no partial windows.
- **Missing-value treatment:** NaN unless all 6 months in the window are non-missing.
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)

### `mom_9m`

- **Formula:** exp(sum(log(1+ret_adj) over trailing 9 months)) - 1
- **Economic interpretation:** Cumulative return over the trailing 9 months (inclusive of the current month).
- **Source columns:** ret_adj, PERMNO
- **Lookback window:** 9 months, backward-looking
- **Minimum observations:** 9
- **Point-in-time notes:** Rolling window strictly requires 9 full prior (calendar-contiguous) monthly observations; no partial windows.
- **Missing-value treatment:** NaN unless all 9 months in the window are non-missing.
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)

### `mom_12m`

- **Formula:** exp(sum(log(1+ret_adj) over trailing 12 months)) - 1
- **Economic interpretation:** Cumulative return over the trailing 12 months (inclusive of the current month).
- **Source columns:** ret_adj, PERMNO
- **Lookback window:** 12 months, backward-looking
- **Minimum observations:** 12
- **Point-in-time notes:** Rolling window strictly requires 12 full prior (calendar-contiguous) monthly observations; no partial windows.
- **Missing-value treatment:** NaN unless all 12 months in the window are non-missing.
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)

### `mom_12_1`

- **Formula:** exp(sum(log(1+ret_adj) over months t-12..t-1)) - 1
- **Economic interpretation:** Classic 12-month momentum skipping the most recent month (Jegadeesh-Titman / Fama-French UMD convention).
- **Source columns:** ret_adj, PERMNO
- **Lookback window:** 12 months ending one month ago
- **Minimum observations:** 12
- **Point-in-time notes:** Return series shifted forward one row before the rolling sum, so month t itself is excluded.
- **Missing-value treatment:** NaN unless all 12 months in the window are non-missing.
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)

### `mom_6_1`

- **Formula:** exp(sum(log(1+ret_adj) over months t-6..t-1)) - 1
- **Economic interpretation:** 6-month momentum skipping the most recent month.
- **Source columns:** ret_adj, PERMNO
- **Lookback window:** 6 months ending one month ago
- **Minimum observations:** 6
- **Point-in-time notes:** Return series shifted forward one row before the rolling sum, so month t itself is excluded.
- **Missing-value treatment:** NaN unless all 6 months in the window are non-missing.
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)


## REVERSAL

### `reversal_1m`

- **Formula:** ret_adj (current month)
- **Economic interpretation:** Short-term reversal (STR): the most recent month's return, conventionally treated in the literature as having a negative relationship with next-period returns — a distinct economic interpretation from mom_1m even though the computed value is identical.
- **Source columns:** ret_adj
- **Lookback window:** 1 month (contemporaneous)
- **Minimum observations:** 1
- **Point-in-time notes:** Uses only the current month's realized return.
- **Missing-value treatment:** Missing source data -> missing feature (NaN). Never imputed.
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)
- **Collinearity:** perfectly collinear with `mom_1m` — excluded from the recommended model feature list (`registry.model_feature_names()`)
- **Known limitations:** Perfectly collinear with mom_1m (identical values in both output layers). Kept in the data for its documented, distinct economic-interpretation label, but excluded from the recommended model feature list — see model_feature_names().


## QUALITY / PROFITABILITY

### `quality_roe`

- **Formula:** cst_ni / cst_seq, NaN if cst_seq <= 0
- **Economic interpretation:** Return on equity.
- **Source columns:** cst_ni, cst_seq
- **Lookback window:** most recent point-in-time-available fiscal year
- **Minimum observations:** n/a (single-period)
- **Point-in-time notes:** Lag/expiry-safe Compustat inputs.
- **Missing-value treatment:** NaN if cst_seq missing or <= 0.
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)

### `quality_roa`

- **Formula:** cst_ni / at_proxy, where at_proxy = cst_lt + cst_seq; NaN if at_proxy <= 0
- **Economic interpretation:** Return on assets.
- **Source columns:** cst_ni, cst_lt, cst_seq
- **Lookback window:** most recent point-in-time-available fiscal year
- **Minimum observations:** n/a (single-period)
- **Point-in-time notes:** Lag/expiry-safe Compustat inputs.
- **Missing-value treatment:** NaN if at_proxy missing or <= 0.
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)
- **Known limitations:** at_proxy is a derived total-assets proxy (this extract has no AT field) — see accounting.py / DATA_DICTIONARY.md.

### `quality_gross_profitability`

- **Formula:** cst_gp / at_proxy, NaN if at_proxy <= 0
- **Economic interpretation:** Gross profitability (Novy-Marx 2013): a robust quality/profitability characteristic.
- **Source columns:** cst_gp, cst_lt, cst_seq
- **Lookback window:** most recent point-in-time-available fiscal year
- **Minimum observations:** n/a (single-period)
- **Point-in-time notes:** Lag/expiry-safe Compustat inputs.
- **Missing-value treatment:** NaN if at_proxy missing or <= 0.
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)
- **Known limitations:** Uses the at_proxy total-assets proxy.

### `quality_operating_profitability`

- **Formula:** (cst_revt - cst_cogs.fillna(0) - cst_xsga.fillna(0) - cst_xint.fillna(0)) / book_equity, NaN if book_equity <= 0
- **Economic interpretation:** Operating profitability (Fama-French 2015 RMW definition).
- **Source columns:** cst_revt, cst_cogs, cst_xsga, cst_xint, cst_seq, cst_pstk, cst_txditc
- **Lookback window:** most recent point-in-time-available fiscal year
- **Minimum observations:** n/a (single-period)
- **Point-in-time notes:** Lag/expiry-safe Compustat inputs.
- **Missing-value treatment:** COGS/XSGA/XINT missing treated as 0 (Fama-French's own construction — most firms genuinely have none, not an unknown value); NaN if REVT missing or book_equity <= 0.
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)

### `quality_net_profit_margin`

- **Formula:** cst_ni / cst_revt, NaN if cst_revt <= 0
- **Economic interpretation:** Net profit margin.
- **Source columns:** cst_ni, cst_revt
- **Lookback window:** most recent point-in-time-available fiscal year
- **Minimum observations:** n/a (single-period)
- **Point-in-time notes:** Lag/expiry-safe Compustat inputs.
- **Missing-value treatment:** NaN if cst_revt missing or <= 0.
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)

### `quality_gross_margin`

- **Formula:** cst_gp / cst_revt, NaN if cst_revt <= 0
- **Economic interpretation:** Gross margin.
- **Source columns:** cst_gp, cst_revt
- **Lookback window:** most recent point-in-time-available fiscal year
- **Minimum observations:** n/a (single-period)
- **Point-in-time notes:** Lag/expiry-safe Compustat inputs.
- **Missing-value treatment:** NaN if cst_revt missing or <= 0.
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)

### `quality_asset_turnover`

- **Formula:** cst_revt / at_proxy, NaN if at_proxy <= 0
- **Economic interpretation:** Asset turnover; sales generated per dollar of assets.
- **Source columns:** cst_revt, cst_lt, cst_seq
- **Lookback window:** most recent point-in-time-available fiscal year
- **Minimum observations:** n/a (single-period)
- **Point-in-time notes:** Lag/expiry-safe Compustat inputs.
- **Missing-value treatment:** NaN if at_proxy missing or <= 0.
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)
- **Known limitations:** Uses the at_proxy total-assets proxy.

### `quality_cf_profitability`

- **Formula:** cst_oancf / at_proxy, NaN if at_proxy <= 0
- **Economic interpretation:** Cash-flow-based profitability (assets-scaled), analogous to ROA but cash- rather than earnings-based.
- **Source columns:** cst_oancf, cst_lt, cst_seq
- **Lookback window:** most recent point-in-time-available fiscal year
- **Minimum observations:** n/a (single-period)
- **Point-in-time notes:** Lag/expiry-safe Compustat inputs.
- **Missing-value treatment:** NaN if at_proxy missing or <= 0.
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)
- **Known limitations:** Uses the at_proxy total-assets proxy.

### `quality_cf_margin`

- **Formula:** cst_oancf / cst_revt, NaN if cst_revt <= 0
- **Economic interpretation:** Cash-flow margin, analogous to net profit margin but cash-based.
- **Source columns:** cst_oancf, cst_revt
- **Lookback window:** most recent point-in-time-available fiscal year
- **Minimum observations:** n/a (single-period)
- **Point-in-time notes:** Lag/expiry-safe Compustat inputs.
- **Missing-value treatment:** NaN if cst_revt missing or <= 0.
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)

### `quality_accruals`

- **Formula:** (cst_ni - cst_oancf) / at_proxy, NaN if at_proxy <= 0
- **Economic interpretation:** Total accruals (cash-flow-statement / Sloan 1996 method), scaled by assets; high accruals are associated with lower subsequent returns/earnings quality.
- **Source columns:** cst_ni, cst_oancf, cst_lt, cst_seq
- **Lookback window:** most recent point-in-time-available fiscal year
- **Minimum observations:** n/a (single-period)
- **Point-in-time notes:** Lag/expiry-safe Compustat inputs.
- **Missing-value treatment:** NaN if at_proxy missing or <= 0.
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)
- **Known limitations:** Uses the simpler cash-flow-statement accrual method (NI - OANCF); the balance-sheet method (change in non-cash working capital) was not implemented, to control scope.


## INVESTMENT / GROWTH

### `growth_asset`

- **Formula:** (at_proxy_t - at_proxy_(t-1)) / abs(at_proxy_(t-1)), NaN if at_proxy_(t-1) <= 0
- **Economic interpretation:** Year-over-year growth in asset.
- **Source columns:** cst_at_proxy_growth
- **Lookback window:** 1 fiscal year (current vs. immediately prior fiscal year)
- **Minimum observations:** 2
- **Point-in-time notes:** Computed on the gvkey/fiscal-year sequence via groupby-shift (never by shifting the monthly panel), then merged in with the same 6-month lag / 12-month shelf life as the level data — see accounting.py.
- **Missing-value treatment:** NaN if the prior fiscal year is missing or its value is <= 0 (a non-positive base makes percentage growth uninterpretable).
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)

### `growth_sales`

- **Formula:** (revt_t - revt_(t-1)) / abs(revt_(t-1)), NaN if revt_(t-1) <= 0
- **Economic interpretation:** Year-over-year growth in sales.
- **Source columns:** cst_revt_growth
- **Lookback window:** 1 fiscal year (current vs. immediately prior fiscal year)
- **Minimum observations:** 2
- **Point-in-time notes:** Computed on the gvkey/fiscal-year sequence via groupby-shift (never by shifting the monthly panel), then merged in with the same 6-month lag / 12-month shelf life as the level data — see accounting.py.
- **Missing-value treatment:** NaN if the prior fiscal year is missing or its value is <= 0 (a non-positive base makes percentage growth uninterpretable).
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)

### `growth_capx`

- **Formula:** (capx_t - capx_(t-1)) / abs(capx_(t-1)), NaN if capx_(t-1) <= 0
- **Economic interpretation:** Year-over-year growth in capx.
- **Source columns:** cst_capx_growth
- **Lookback window:** 1 fiscal year (current vs. immediately prior fiscal year)
- **Minimum observations:** 2
- **Point-in-time notes:** Computed on the gvkey/fiscal-year sequence via groupby-shift (never by shifting the monthly panel), then merged in with the same 6-month lag / 12-month shelf life as the level data — see accounting.py.
- **Missing-value treatment:** NaN if the prior fiscal year is missing or its value is <= 0 (a non-positive base makes percentage growth uninterpretable).
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)

### `growth_inventory`

- **Formula:** (invt_t - invt_(t-1)) / abs(invt_(t-1)), NaN if invt_(t-1) <= 0
- **Economic interpretation:** Year-over-year growth in inventory.
- **Source columns:** cst_invt_growth
- **Lookback window:** 1 fiscal year (current vs. immediately prior fiscal year)
- **Minimum observations:** 2
- **Point-in-time notes:** Computed on the gvkey/fiscal-year sequence via groupby-shift (never by shifting the monthly panel), then merged in with the same 6-month lag / 12-month shelf life as the level data — see accounting.py.
- **Missing-value treatment:** NaN if the prior fiscal year is missing or its value is <= 0 (a non-positive base makes percentage growth uninterpretable).
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)

### `growth_receivables`

- **Formula:** (rect_t - rect_(t-1)) / abs(rect_(t-1)), NaN if rect_(t-1) <= 0
- **Economic interpretation:** Year-over-year growth in receivables.
- **Source columns:** cst_rect_growth
- **Lookback window:** 1 fiscal year (current vs. immediately prior fiscal year)
- **Minimum observations:** 2
- **Point-in-time notes:** Computed on the gvkey/fiscal-year sequence via groupby-shift (never by shifting the monthly panel), then merged in with the same 6-month lag / 12-month shelf life as the level data — see accounting.py.
- **Missing-value treatment:** NaN if the prior fiscal year is missing or its value is <= 0 (a non-positive base makes percentage growth uninterpretable).
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)

### `growth_ppent`

- **Formula:** (ppent_t - ppent_(t-1)) / abs(ppent_(t-1)), NaN if ppent_(t-1) <= 0
- **Economic interpretation:** Year-over-year growth in ppent.
- **Source columns:** cst_ppent_growth
- **Lookback window:** 1 fiscal year (current vs. immediately prior fiscal year)
- **Minimum observations:** 2
- **Point-in-time notes:** Computed on the gvkey/fiscal-year sequence via groupby-shift (never by shifting the monthly panel), then merged in with the same 6-month lag / 12-month shelf life as the level data — see accounting.py.
- **Missing-value treatment:** NaN if the prior fiscal year is missing or its value is <= 0 (a non-positive base makes percentage growth uninterpretable).
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)

### `growth_equity`

- **Formula:** (seq_t - seq_(t-1)) / abs(seq_(t-1)), NaN if seq_(t-1) <= 0
- **Economic interpretation:** Year-over-year growth in equity.
- **Source columns:** cst_seq_growth
- **Lookback window:** 1 fiscal year (current vs. immediately prior fiscal year)
- **Minimum observations:** 2
- **Point-in-time notes:** Computed on the gvkey/fiscal-year sequence via groupby-shift (never by shifting the monthly panel), then merged in with the same 6-month lag / 12-month shelf life as the level data — see accounting.py.
- **Missing-value treatment:** NaN if the prior fiscal year is missing or its value is <= 0 (a non-positive base makes percentage growth uninterpretable).
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)


## LEVERAGE / FINANCIAL HEALTH

### `leverage_debt_to_assets`

- **Formula:** total_debt / at_proxy, where total_debt = cst_dlc + cst_dltt; NaN if at_proxy <= 0
- **Economic interpretation:** Total debt relative to assets.
- **Source columns:** cst_dlc, cst_dltt, cst_lt, cst_seq
- **Lookback window:** most recent point-in-time-available fiscal year
- **Minimum observations:** n/a (single-period)
- **Point-in-time notes:** Lag/expiry-safe Compustat inputs.
- **Missing-value treatment:** NaN if at_proxy missing or <= 0.
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)
- **Known limitations:** Uses the at_proxy total-assets proxy.

### `leverage_debt_to_equity`

- **Formula:** total_debt / cst_seq, NaN if cst_seq <= 0
- **Economic interpretation:** Total debt relative to book equity.
- **Source columns:** cst_dlc, cst_dltt, cst_seq
- **Lookback window:** most recent point-in-time-available fiscal year
- **Minimum observations:** n/a (single-period)
- **Point-in-time notes:** Lag/expiry-safe Compustat inputs.
- **Missing-value treatment:** NaN if cst_seq missing or <= 0.
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)

### `leverage_lt_debt_ratio`

- **Formula:** cst_dltt / at_proxy, NaN if at_proxy <= 0
- **Economic interpretation:** Long-term debt relative to assets.
- **Source columns:** cst_dltt, cst_lt, cst_seq
- **Lookback window:** most recent point-in-time-available fiscal year
- **Minimum observations:** n/a (single-period)
- **Point-in-time notes:** Lag/expiry-safe Compustat inputs.
- **Missing-value treatment:** NaN if at_proxy missing or <= 0.
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)
- **Known limitations:** Uses the at_proxy total-assets proxy.

### `leverage_current_ratio`

- **Formula:** cst_act / cst_lct, NaN if cst_lct <= 0
- **Economic interpretation:** Current assets relative to current liabilities; short-term liquidity/solvency.
- **Source columns:** cst_act, cst_lct
- **Lookback window:** most recent point-in-time-available fiscal year
- **Minimum observations:** n/a (single-period)
- **Point-in-time notes:** Lag/expiry-safe Compustat inputs.
- **Missing-value treatment:** NaN if cst_lct missing or <= 0.
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)

### `leverage_cash_ratio`

- **Formula:** cst_che / cst_lct, NaN if cst_lct <= 0
- **Economic interpretation:** Cash relative to current liabilities; the strictest short-term solvency measure.
- **Source columns:** cst_che, cst_lct
- **Lookback window:** most recent point-in-time-available fiscal year
- **Minimum observations:** n/a (single-period)
- **Point-in-time notes:** Lag/expiry-safe Compustat inputs.
- **Missing-value treatment:** NaN if cst_lct missing or <= 0.
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)

### `leverage_net_debt_to_assets`

- **Formula:** net_debt / at_proxy, where net_debt = total_debt - cst_che; NaN if at_proxy <= 0
- **Economic interpretation:** Debt net of cash holdings, relative to assets.
- **Source columns:** cst_dlc, cst_dltt, cst_che, cst_lt, cst_seq
- **Lookback window:** most recent point-in-time-available fiscal year
- **Minimum observations:** n/a (single-period)
- **Point-in-time notes:** Lag/expiry-safe Compustat inputs.
- **Missing-value treatment:** NaN if at_proxy missing or <= 0 (net_debt itself may legitimately be negative — a net cash position).
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)
- **Known limitations:** Uses the at_proxy total-assets proxy.

### `leverage_interest_coverage`

- **Formula:** cst_ebit / cst_xint, NaN if cst_xint <= 0
- **Economic interpretation:** Ability to cover interest expense from operating earnings.
- **Source columns:** cst_ebit, cst_xint
- **Lookback window:** most recent point-in-time-available fiscal year
- **Minimum observations:** n/a (single-period)
- **Point-in-time notes:** Lag/expiry-safe Compustat inputs.
- **Missing-value treatment:** NaN if cst_xint missing, zero, or negative (a zero-interest-expense firm has an undefined/infinite coverage ratio, not a real signal).
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)


## LIQUIDITY

### `liquidity_dollar_volume`

- **Formula:** dollar_volume_millions = (MthPrc [$/share] * MthVol [actual shares]) / 1,000,000. Units: $ millions.
- **Economic interpretation:** Dollar trading volume; a basic liquidity/size-of-trading measure.
- **Source columns:** MthPrc, MthVol
- **Lookback window:** contemporaneous (current month)
- **Minimum observations:** n/a (single-period)
- **Point-in-time notes:** Uses only the current month's own price and volume.
- **Missing-value treatment:** Missing source data -> missing feature (NaN). Never imputed.
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)
- **Known limitations:** MthPrc*MthVol is a genuine dollar amount either way (this was never a unit-mismatch bug, unlike value_bm etc.) — expressed in $ millions here purely for consistency with every other monetary feature in this package.

### `liquidity_share_turnover`

- **Formula:** MthVol [actual shares] / shares_outstanding_actual [actual shares, = ShrOut * 1,000], NaN if shares_outstanding_actual <= 0. Dimensionless.
- **Economic interpretation:** Shares traded as a fraction of shares outstanding.
- **Source columns:** MthVol, ShrOut
- **Lookback window:** contemporaneous (current month)
- **Minimum observations:** n/a (single-period)
- **Point-in-time notes:** Uses only the current month's own volume and shares outstanding.
- **Missing-value treatment:** NaN if shares_outstanding_actual missing or <= 0.
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)
- **Known limitations:** MthVol is in actual shares while ShrOut is in thousands of shares — dividing MthVol directly by raw ShrOut overstated turnover by ~1000x in an earlier version of this feature; a dedicated unit audit caught it (median implied turnover was ~104x/month, which is impossible). See UNIT_AUDIT_REPORT.md.

### `liquidity_amihud_illiq`

- **Formula:** abs(ret_adj) / dollar_volume_millions, NaN if dollar_volume_millions <= 0. Units: 1 / $ millions.
- **Economic interpretation:** Amihud (2002) illiquidity: price impact per dollar traded.
- **Source columns:** ret_adj, MthPrc, MthVol
- **Lookback window:** contemporaneous (current month)
- **Minimum observations:** n/a (single-period)
- **Point-in-time notes:** Uses only the current month's own return and dollar volume.
- **Missing-value treatment:** NaN if dollar_volume_millions missing or <= 0.
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)
- **Known limitations:** This extract has no daily data, so this is a monthly-frequency single-observation proxy, not the standard measure averaged over daily observations within the month — materially noisier. Its scale depends on the $-millions convention for dollar_volume_millions (changing that convention rescales this feature proportionally).


## VOLATILITY / RISK

### `vol_12m`

- **Formula:** std(ret_adj over trailing 12 months)
- **Economic interpretation:** Total return volatility.
- **Source columns:** ret_adj, PERMNO
- **Lookback window:** 12 months, backward-looking
- **Minimum observations:** 6
- **Point-in-time notes:** Rolling std over the trailing 12 calendar-contiguous months, requiring at least 6 non-missing observations.
- **Missing-value treatment:** NaN unless at least 6 of the trailing 12 months are non-missing.
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)
- **Known limitations:** Monthly-frequency volatility, not the more common daily-based measure (no daily data available).

### `downside_vol_12m`

- **Formula:** std(ret_adj where ret_adj < 0, over trailing 12 months)
- **Economic interpretation:** Downside (semi-deviation) volatility; risk from negative returns only.
- **Source columns:** ret_adj, PERMNO
- **Lookback window:** 12 months, backward-looking
- **Minimum observations:** 6
- **Point-in-time notes:** Rolling std of the negative-return subset within the trailing 12 calendar-contiguous months.
- **Missing-value treatment:** NaN unless at least 6 negative-return months are present in the trailing 12-month window.
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)
- **Known limitations:** Monthly-frequency; requiring 6 of 12 months to be negative is a meaningfully strict bar, so this feature is missing more often than vol_12m by construction.

### `beta_24m`

- **Formula:** Cov(ff_mktrf, ret_adj - ff_rf) / Var(ff_mktrf), over trailing 24 months
- **Economic interpretation:** CAPM market beta.
- **Source columns:** ret_adj, ff_mktrf, ff_rf, PERMNO
- **Lookback window:** 24 months, backward-looking
- **Minimum observations:** 12
- **Point-in-time notes:** Rolling covariance/variance over the trailing 24 calendar-contiguous months, requiring at least 12 non-missing observations.
- **Missing-value treatment:** NaN unless at least 12 of the trailing 24 months are non-missing, or if the market factor's variance is zero over that window.
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)
- **Known limitations:** Monthly-frequency beta, computed from population (not small-sample-adjusted) rolling moments — a standard, documented simplification.

### `idio_vol_24m`

- **Formula:** sqrt(Var(ret_adj - ff_rf) - beta_24m^2 * Var(ff_mktrf)), over trailing 24 months
- **Economic interpretation:** Idiosyncratic (firm-specific) volatility from the same rolling market-model regression as beta_24m.
- **Source columns:** ret_adj, ff_mktrf, ff_rf, PERMNO
- **Lookback window:** 24 months, backward-looking
- **Minimum observations:** 12
- **Point-in-time notes:** Same rolling window and minimum-observation rule as beta_24m; computed from the identical closed-form variance decomposition.
- **Missing-value treatment:** NaN under the same conditions as beta_24m.
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)
- **Known limitations:** Same population-moment simplification as beta_24m; negative variance from numerical noise is clipped to zero before the square root.

### `max_ret_12m`

- **Formula:** max(ret_adj over trailing 12 months)
- **Economic interpretation:** Maximum monthly return over the trailing year (a monthly analogue of the Bali-Cakici-Whitelaw MAX factor, which is normally built from daily returns).
- **Source columns:** ret_adj, PERMNO
- **Lookback window:** 12 months, backward-looking
- **Minimum observations:** 6
- **Point-in-time notes:** Rolling max over the trailing 12 calendar-contiguous months.
- **Missing-value treatment:** NaN unless at least 6 of the trailing 12 months are non-missing.
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)
- **Known limitations:** Monthly-frequency proxy for what the literature usually computes from daily returns.

### `min_ret_12m`

- **Formula:** min(ret_adj over trailing 12 months)
- **Economic interpretation:** Minimum monthly return over the trailing year — crash/tail-risk exposure.
- **Source columns:** ret_adj, PERMNO
- **Lookback window:** 12 months, backward-looking
- **Minimum observations:** 6
- **Point-in-time notes:** Rolling min over the trailing 12 calendar-contiguous months.
- **Missing-value treatment:** NaN unless at least 6 of the trailing 12 months are non-missing.
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)
- **Known limitations:** Monthly-frequency proxy.

### `skew_36m`

- **Formula:** skewness(ret_adj over trailing 36 months)
- **Economic interpretation:** Return skewness; lottery-like (positively skewed) versus crash-prone (negatively skewed) return profiles.
- **Source columns:** ret_adj, PERMNO
- **Lookback window:** 36 months, backward-looking
- **Minimum observations:** 24
- **Point-in-time notes:** Rolling skewness over the trailing 36 calendar-contiguous months, requiring at least 24 non-missing observations.
- **Missing-value treatment:** NaN unless at least 24 of the trailing 36 months are non-missing.
- **Outlier treatment:** No imputation. In the model-ready layer, winsorized at configurable percentiles (default 1%/99%) computed within that month's investable universe only (see transforms.py); the raw layer is never winsorized.
- **Layer:** raw (model-ready counterpart: yes)
- **Known limitations:** Skewness estimated from ~24-36 monthly points is statistically noisy; the relatively high min_observations bar is a deliberate attempt to keep this 'meaningful,' per the task's own caveat, but does not eliminate estimation noise.

## Reproducing this

```bash
python scripts/run_data_validation.py    # if data/interim/ doesn't exist yet
python scripts/run_data_processing.py    # if data/processed/master_panel.parquet doesn't exist yet
python scripts/run_feature_engineering.py
```

Diagnostics (missingness, coverage by year/month, distributions, pairwise
correlations, category correlation heatmaps, outlier rates, feature
stability, universe coverage) are written to `reports/feature_engineering/`
(gitignored — derived from the licensed WRDS extract). The feature registry
itself (`feature_registry.json`/`.csv`, the machine-readable source this
document is generated from) is git-tracked, since it contains no licensed
data — only formulas and column names.
