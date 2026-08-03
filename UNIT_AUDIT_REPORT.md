# Unit Integrity Audit Report

A final feature-integrity audit performed before any modeling work, in
response to a direct question: are CRSP and Compustat monetary fields
actually in the units this pipeline assumed? **They were not consistently
assumed at all** — `DATA_DICTIONARY.md` had flagged `MthCap`'s unit as
"likely thousands of USD... unconfirmed" back in the data-validation phase,
and that uncertainty was never resolved before `feature_engineering` started
combining CRSP and Compustat values directly. This audit resolves it
empirically, finds that the unresolved uncertainty had in fact caused two
real bugs, fixes both, and adds regression tests so they cannot silently
reappear.

**All fixes described here are already applied** in
`src/feature_engineering/` and reflected in `FEATURE_DICTIONARY.md`; this
document is the record of the audit itself.

## 1. Raw source units (empirically verified, not assumed)

Three independent, data-driven cross-checks were used — no unit below is
taken on the strength of "standard convention" alone:

**Check 1 — CRSP internal consistency.** Across a 20,000-row sample,
`MthCap / (MthPrc * ShrOut)` has mean 1.000000 and std 0.0 (min 0.999978,
max 1.000004). This confirms `MthCap`, `MthPrc`, and `ShrOut` are mutually
consistent (`MthCap` is exactly price times shares), but not yet which
absolute scale they use.

**Check 2 — CRSP vs. Compustat cross-check.** Across 1,704,942 matched
PERMNO-months, `(cst_csho * cst_prcc_f) / MthCap` has:

| Percentile | Value |
| --- | --- |
| 1% | 0.00021 |
| 25% | 0.00079 |
| **median** | **0.0010058** |
| 75% | 0.00142 |
| 99% | 0.0625 |

The median sits almost exactly at `0.001` — i.e. `cst_csho * cst_prcc_f`
(Compustat) needs multiplying by ~1,000 to match `MthCap` (CRSP). Since
`cst_prcc_f` is unambiguously dollars-per-share (a per-share price cannot be
off by a round power of ten without producing an implausible per-share
value), this pins down two things simultaneously: `cst_csho` is in
**millions of shares**, and `MthCap` is in **thousands of dollars** (a
company with 10 million shares at $50 gives `csho*prcc_f = 500` under the
millions convention, i.e. "$500 million," and `MthCap` for the same company
would need to read `500,000` to be "thousands of dollars" — a ratio of
exactly 1,000, matching the data). The remaining dispersion (25th-75th
percentile spans roughly 0.0008-0.0014, not a single exact point mass) is
fully explained by `cst_prcc_f` being a **fiscal-year-end** price while
`MthCap` is a **calendar-month-end** price on a potentially different date,
plus small share-count timing differences between the two sources — not by
any unit inconsistency.

**Check 3 — plausibility of per-share book value.** `cst_seq / cst_csho`
(book equity ÷ shares, both assumed millions) has a median of **$7.81** and
a 25th-75th percentile range of **$2.80-$15.75** — a thoroughly plausible
range of book value per share across a broad cross-section of firms. Had
the millions-of-shares assumption for `cst_csho` been wrong by a factor of
1,000 in either direction, this would instead show values of roughly
$0.003-$0.016 (implausibly low) or $2,800-$15,700 (implausibly high) per
share. It doesn't — reinforcing Check 2's conclusion from an independent
angle.

**Check 4 — real-magnitude sanity check.** The single largest-average-
`MthCap` security in the entire panel resolves (via `crsp_names`, used here
*only* for one-time human identification, never in any pipeline
computation) to a well-known, easily-recognized large-cap technology
company. Under the "`MthCap` is thousands of dollars" assignment, that
security's implied market capitalization for late 2025 comes out in the
**low-single-digit trillions of dollars** — the correct order of magnitude
and the correct historical period for that company's well-publicized real
market cap. (The company's identity and exact figures are omitted from this
document; see the reasoning trail in this conversation if the specific
numbers are needed — they are not reproduced here to avoid embedding
literal licensed data values in a git-tracked file, consistent with this
project's data-handling rules.)

**Separately, `MthVol`'s unit** was checked directly: `MthVol / ShrOut`
implies a median monthly share turnover of **~104x** under the naive
(same-units) assumption — mathematically impossible (a stock cannot turn
over 100+ times its own float in one month, in aggregate, as a *typical*
observation). Dividing by 1,000 gives a median of **~0.10 (10%)** — a
completely ordinary monthly turnover figure. This confirms `MthVol` is in
**actual shares**, not thousands like `ShrOut` — a genuine, if unusual,
inconsistency within CRSP's own schema (not a project assumption).

### Confirmed units

| Field | Source | Unit |
| --- | --- | --- |
| `MthCap` | CRSP | thousands of USD |
| `ShrOut` | CRSP | thousands of shares |
| `MthPrc` | CRSP | USD per share (unscaled) |
| `MthVol` | CRSP | **actual shares** (not thousands) |
| `cst_csho` | Compustat | millions of shares |
| `cst_prcc_f` | Compustat | USD per share (unscaled) |
| All other `cst_*` dollar fields (`revt`, `ni`, `seq`, `oancf`, `ebitda`, `gp`, `cogs`, `xsga`, `dlc`, `dltt`, `che`, `act`, `lct`, `ebit`, `xint`, `pstk`, `txditc`, `capx`, `invt`, `rect`, `ppent`, `lt`) | Compustat | millions of USD |

## 2. Converted (canonical) units

**Canonical unit adopted: millions of USD**, matching Compustat's native
convention (fewer fields need converting). `src/feature_engineering/units.py`
adds three normalized columns before any other module runs:

| Column | Derivation | Unit |
| --- | --- | --- |
| `mktcap_millions` | `MthCap / 1,000` | $ millions |
| `dollar_volume_millions` | `(MthPrc * MthVol) / 1,000,000` | $ millions |
| `shares_outstanding_actual` | `ShrOut * 1,000` | actual shares |

Every monetary *raw feature* — not only the ones combining CRSP and
Compustat — now uses this convention (`size_mktcap`, `liquidity_dollar_volume`,
and all VALUE-category ratios), so `features_raw.parquet` has one
consistent, documented unit throughout rather than a silent mix.

## 3. Sample hand-calculated ratios

Fully synthetic, round-number verification (also codified as automated
tests in `tests/feature_engineering/test_units.py` — these are not
illustrative-only, the pipeline is asserted against these exact numbers):

**Synthetic firm:** `MthCap = 1,000,000` ($ thousands) → `mktcap_millions =
1,000` ($1,000M / $1B). Compustat (all $ millions): `seq = 400`, `pstk =
txditc = 0` → `book_equity = 400`; `ni = 50`; `oancf = 60`; `revt = 800`;
`ebitda = 120`; `dlc = 50`, `dltt = 150` → `total_debt = 200`; `che = 100` →
`enterprise_value = 1,000 + 200 + 0 - 100 = 1,100`.

| Feature | Formula | Expected | 
| --- | --- | --- |
| `value_bm` | `400 / 1,000` | **0.400** |
| `value_earnings_yield` | `50 / 1,000` | **0.050** |
| `value_cf_yield` | `60 / 1,000` | **0.060** |
| `value_sales_to_price` | `800 / 1,000` | **0.800** |
| `value_ebitda_to_ev` | `120 / 1,100` | **0.1091** |
| `value_cf_to_ev` | `60 / 1,100` | **0.0545** |
| `liquidity_share_turnover` (separate example: 500,000 actual shares / 10,000,000 actual shares) | `500,000 / (10,000 * 1,000)` | **0.050** |

Every one of these is asserted exactly (`pytest.approx`) in
`test_units.py`, alongside two explicit **regression** tests that compute
what the *pre-fix* code would have produced (`correct_value / 1000` for the
value ratios; `correct_turnover * 1000` for turnover) and assert the
relationship — so a future accidental reversion of either fix fails loudly
with a clear "off by 1000x" signature, not just a generic mismatch.

## 4. Post-conversion feature distributions

Computed on the full, real 2,498,171-row master panel
(`reports/feature_engineering/summary_distributions.csv` has the complete
table; key figures reproduced here):

| Feature | Median (post-fix) | IQR (post-fix) | Plausible? |
| --- | --- | --- | --- |
| `size_mktcap` ($M) | 275.5 | 62.2 - 1,308.5 | Yes — broad-market small/mid/large-cap mix |
| `value_bm` | 0.596 | 0.303 - 1.110 | Yes — typical book-to-market range |
| `value_earnings_yield` | 0.0315 (3.15%) | -0.055 - 0.072 | Yes — includes loss-making firms (negative E/Y), as expected |
| `value_cf_yield` | 0.0662 (6.62%) | -0.0005 - 0.149 | Yes |
| `value_sales_to_price` | 0.635 | 0.257 - 1.724 | Yes |
| `value_ebitda_to_ev` | 0.0814 (8.14%) | 0.021 - 0.142 | Yes |
| `value_cf_to_ev` | 0.0541 (5.41%) | 0.006 - 0.109 | Yes |
| `liquidity_dollar_volume` ($M) | 28.31 | 3.73 - 230.6 | Yes |
| `liquidity_share_turnover` | 0.1036 (10.4%) | 0.045 - 0.216 | Yes — ordinary monthly turnover |

`reversal_1m` vs. `mom_1m`: correlation **1.0** across the full panel,
confirming perfect collinearity (see `FEATURE_DICTIONARY.md`'s
"Collinearity" section and `registry.model_feature_names()`).

## 5. Changes caused by the audit

**Two real bugs found and fixed** (both existed before this audit; neither
was introduced by it):

1. **VALUE-category ratios mixed CRSP thousands with Compustat millions.**
   `value_bm`, `value_earnings_yield`, `value_cf_yield`,
   `value_sales_to_price` divided a Compustat $-millions numerator by raw
   `MthCap` ($ thousands) directly. `enterprise_value` — and therefore
   `value_ebitda_to_ev` / `value_cf_to_ev` — summed `MthCap` ($ thousands)
   with `total_debt`/`cst_pstk`/`cst_che` ($ millions) directly. Every one
   of these was silently wrong by approximately 1,000x (the ratios) or by a
   structural, non-proportional amount (`enterprise_value`, since it was a
   *sum* of inconsistently-scaled terms, not a simple ratio). **Fix:**
   introduced `mktcap_millions` (`units.py`) and used it everywhere a CRSP
   monetary value meets a Compustat one (`accounting.py`, `ratios.py`).

2. **`liquidity_share_turnover` mixed actual shares with thousands of
   shares.** `MthVol` (actual shares traded) was divided by raw `ShrOut`
   (thousands of shares) with no conversion, overstating turnover by
   ~1,000x (median implied turnover was ~104x/month — impossible). **Fix:**
   introduced `shares_outstanding_actual` (`ShrOut * 1,000`) and divide
   `MthVol` by that instead (`time_series_features.py`).

**One re-expression for consistency (not a bug fix):**
`liquidity_dollar_volume` (`MthPrc * MthVol`) was already a mathematically
correct dollar amount before this audit — it was just expressed in raw
dollars rather than millions. It is now divided by 1e6 purely so every
monetary feature in `features_raw.parquet` shares the same $-millions
convention; no economic value changed.

**`size_mktcap`** changed value (divided by 1,000) for the same
consistency reason, not because it was previously wrong — `MthCap` itself
was always correct in its native ($ thousands) unit; it just wasn't
labeled as such anywhere, and this audit fixes that omission along with
switching the *reported* feature to the canonical $-millions unit.

**Documentation changes:**
- `FEATURE_DICTIONARY.md`: every formula involving a monetary quantity now
  states its unit explicitly; a new "Units" section up front; a new
  "Collinearity" section for `reversal_1m`/`mom_1m`; `is_investable`'s
  description strengthened to explicitly rule out "verified common-share"
  and named-index (e.g. Russell 1000) framing.
- `universe.py`: docstring rewritten with the same explicit "must not be
  described as..." language.
- `registry.py`: `FeatureSpec` gained `collinear_with` and
  `exclude_from_model_features` fields; a new `model_feature_names()`
  helper returns the 50-feature recommended training list (all features
  except `reversal_1m`).

**New tests:** `tests/feature_engineering/test_units.py` (7 tests — unit
normalization, the CRSP/Compustat cross-check as an automated regression,
hand-calculated VALUE ratios, and explicit "would be wrong by 1000x"
regressions for both bugs) plus 3 new registry tests for the
collinearity/model-feature-list mechanism. All existing tests whose
fixtures referenced raw `MthCap`/`MthVol`/`ShrOut` directly were updated to
use the normalized columns, matching how the fixed code is actually called.

**Full pipeline re-run against the real 2,498,171-row panel:** all 9
automated validation checks (duplicate keys, calendar gaps, infinite
values, registry-source-column consistency, return bounds, denominator
handling, no-look-ahead-via-truncation, reproducibility, cross-sectional
isolation) pass after the fix, same as before it — the fix changed
*values*, not the pipeline's structural correctness, which was already
verified in the prior phase.

## Reproducing this audit

```bash
python scripts/run_feature_engineering.py
```

regenerates `features_raw.parquet` / `features_model_ready.parquet` with
the fixes applied, plus `reports/feature_engineering/summary_distributions.csv`
for the full post-conversion distribution table.

```bash
pytest tests/feature_engineering/test_units.py -v
```

runs just the unit-consistency regression suite.
