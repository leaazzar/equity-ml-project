# Data Dictionary

Describes the six raw WRDS extracts in `data/raw/`, as empirically determined
by running the validation pipeline in `src/data_validation/` against the
actual files (not from general WRDS documentation or memory). Column
descriptions below are limited to what can be confirmed from the data itself
or from unambiguous public conventions (e.g. Fama-French factor names);
anything whose exact definition depends on WRDS/Compustat documentation this
project doesn't have on hand is marked **TODO: confirm against vendor docs**
rather than guessed.

This is a live document: regenerate the underlying statistics at any time
with `python scripts/run_data_validation.py`, which writes full per-column
detail to `reports/data_validation/` (JSON) and `reports/data_validation/summary.md`
(human-readable). The numbers quoted here are a snapshot from the most recent
run; treat the generated report as the source of truth if they ever drift.

## How this file is organized

Each dataset section lists: row/column counts, the primary key used for
duplicate-checking, a column table (dtype as loaded, % null, short
description), and dataset-specific quality notes. A final section covers
cross-dataset identifier mappings and a summary of validation findings.

---

## `crsp_monthly_stock.csv` — CRSP Monthly Stock File

- **Rows:** 2,528,440 &nbsp;&nbsp; **Columns:** 14
- **Coverage:** 1999-01-29 to 2025-12-31 (month-end observations)
- **Primary key:** `PERMNO` + `MthCalDt` (declared; see quality notes below —
  this key is *not* fully unique in the raw file)

| Column | Type | Null % | Description |
| --- | --- | --- | --- |
| PERMNO | int | 0% | CRSP permanent security identifier. Stable for the life of the security. |
| HdrCUSIP | string | 0% | "Header" CUSIP for the security. TODO: confirm exact WRDS semantics vs. `CUSIP`. |
| CUSIP | string | 0.7% | CUSIP as of that month; can change over time via corporate actions. |
| Ticker | string | 1.9% | Ticker symbol as of that month. |
| PERMCO | int | 0% | CRSP permanent *company* identifier; groups PERMNOs belonging to the same company (e.g. share classes). |
| SICCD | int | 0% | Standard Industrial Classification code as of that month. `0` is observed and likely means unclassified/unavailable. |
| MthCalDt | date | 0% | Month-end calendar date of the observation. |
| MthPrc | float | 1.2% | Month-end price. No negative values observed in this extract. TODO: confirm units/sign convention against CRSP docs. |
| MthCap | float | 1.2% | Market capitalization at month end. TODO: confirm units (likely thousands of USD). |
| MthRet | float | 1.3% | Monthly holding-period return (return including distributions). Null exactly when `MthRetFlg == "NT"`. |
| MthRetx | float | 1.3% | Monthly return excluding distributions. |
| MthRetFlg | string | 0% | Return-calculation flag. Observed values: `CR`, `DE`, `GP`, `IP`, `MP`, `NS`, `NT`. TODO: confirm exact definitions against CRSP documentation. |
| MthVol | float | 1.2% | Monthly trading volume (shares). Zero observed for some illiquid names. |
| ShrOut | int | 0% | Shares outstanding. TODO: confirm units (likely thousands). |

**Quality notes:**
- 53,323 rows share a `PERMNO`+`MthCalDt` key with another row. Of these,
  50,079 are exact full-row duplicates (safe to drop — done in the interim
  output) and 3,244 share a key but differ in other columns (e.g. one copy
  has a populated `CUSIP`/`Ticker`, the other has both null) — these are
  **not** auto-resolved; see the validation report.
- `MthRetFlg` fully explains missingness in `MthRet`: every null `MthRet` row
  has flag `NT`, and no other flag value ever has a null `MthRet`.

---

## `crsp_delisting.csv` — CRSP Delisting Events

- **Rows:** 15,665 &nbsp;&nbsp; **Columns:** 6
- **Coverage:** 1999-01-04 to 2025-12-30
- **Primary key:** `PERMNO` (verified unique — no duplicates)

| Column | Type | Null % | Description |
| --- | --- | --- | --- |
| PERMNO | int | 0% | Security identifier. FK to `crsp_monthly_stock.PERMNO`. |
| DelistingDt | date | 0% | Date of the delisting event. |
| DelActionType | string | 0% | Delisting action code. Observed: `MER`, `GDR`, `GLI`, `GEX`. TODO: confirm definitions. |
| DelReasonType | string | 0% | Delisting reason code. Observed: `UNAV`, `FING`, `LP`, `CORQ`, `BKPY`, `INSC`, `MVOT`, `DELQ`, `INSF`, `PUBI`, `EQRQ`, `DERE`, `SHLD`, `FDCV`, `VIO`, `MTMK`, `SERQ`, `OFFRE`, `DEEX`, `FARG`, `MVTO`. TODO: confirm definitions. |
| DelPERMNO | int | 0% | Successor security's PERMNO, if the delisting was into another CRSP-covered security. `0` is a sentinel for "no successor." |
| DelRet | float | 4.6% | Delisting return. Null for 726 of 15,665 rows — literal "missing delisting information." |

**Quality notes:**
- 2 of 15,665 `PERMNO` values are not found in `crsp_monthly_stock` at all.
- 11 of 3,840 non-zero `DelPERMNO` values are not found in `crsp_monthly_stock`.
- A heuristic check (securities with no observation in the last 6 months of
  the sample and no delisting record) found **zero** unexplained cases in
  this extract — every apparent "stopped trading" security has a matching
  delisting record.

---

## `crsp_names.csv` — CRSP Historical Name/Identifier Records

- **Rows:** 191,048 &nbsp;&nbsp; **Columns:** 9
- **Primary key:** none — see below.

| Column | Type | Null % | Description |
| --- | --- | --- | --- |
| permno | int | 0% | Security identifier. |
| permco | int | 0% | Company identifier. |
| cusip | string | 18.5% | CUSIP as of this name record. |
| ticker | string | 18.2% | Ticker as of this name record. |
| issuernm | string | 0% | Issuer/company name. |
| securitytype | string | 15.6% | Observed: `EQTY`, `FUND`, `DERV`. |
| sharetype | string | 15.6% | Observed: `NS`, `AD`, `SB`, `UG`, `CE`. |
| siccd | int | 0% | SIC code as of this name record. |
| primaryexch | string | 0% | Observed: `Q`, `N`, `X`, `A`, `R`, `B`, `I`. Likely exchange codes; not confirmed here. |

**Quality notes — this file has real structural limitations:**
- **No `NAMEDT`/`NAMEENDDT` validity-period columns exist in this extract.**
  Rows cannot be date-bounded, so there is no way to determine from this file
  alone which period a given name/CUSIP/ticker was valid for.
- Because of that, there is **no reliable row-level primary key**: 113,353
  rows (out of 191,048) share a fully-duplicated row with at least one other
  row, and no combination of up to 3 columns (searched automatically —
  see `profiling.suggest_primary_key`) uniquely identifies a row.
- 27,672 of 40,518 unique `permno` values have a synthetic `"... (Last
  Known)"` row in `issuernm`, with `cusip`/`ticker`/`securitytype`/`sharetype`
  all null — an apparent "last known identity" placeholder record.

---

## `ccm_link_table.csv` — CRSP/Compustat Merged (CCM) Link Table

- **Rows:** 32,814 &nbsp;&nbsp; **Columns:** 9
- **Coverage:** `LINKDT` 1949-07-01 to 2025-12-18
- **Primary key:** `gvkey` + `LIID` + `LINKDT` (declared; verified unique — no duplicates)

| Column | Type | Null % | Description |
| --- | --- | --- | --- |
| gvkey | int | 0% | Compustat firm identifier. |
| tic | string | 0.01% | Compustat ticker. |
| LINKPRIM | string | 0% | Observed: `P`, `C`, `J`, `N` (`P` is ~74% of rows). TODO: confirm exact definitions. |
| LIID | string | 0% | Compustat issue/security ID within the firm. |
| LINKTYPE | string | 0% | Observed: `LU`, `LC`. TODO: confirm exact definitions. |
| LPERMNO | int | 0% | Linked CRSP PERMNO. |
| LPERMCO | int | 0% | Linked CRSP PERMCO. |
| LINKDT | date | 0% | Start of the link's validity window. |
| LINKENDDT | date or `"E"` | 0% | End of the link's validity window. `"E"` (6,109 rows) is a sentinel meaning the link is still open/active — not a parse failure. |

**Quality notes:**
- No overlapping validity windows were found within any `(gvkey, LIID)`
  group, nor within primary (`LINKPRIM == "P"`) links per `gvkey`.
- 10,403 unique `LPERMNO` values don't appear in `crsp_monthly_stock`; 10,382
  of those have their entire link window ending before `crsp_monthly_stock`'s
  earliest date (1999-01-29) — an expected consequence of CCM's longer
  history, not a data error. 21 are unexplained by that alone.
- 11,572 unique `gvkey` values don't appear in `compustat_fundamentals_annual`;
  10,500 are explained the same way (link ends before Compustat's window
  starts). 1,072 are unexplained by that alone.
- Every `gvkey` in `compustat_fundamentals_annual` has at least one row here
  (full coverage in that direction).

---

## `compustat_fundamentals_annual.csv` — Compustat Annual Fundamentals

- **Rows:** 160,320 &nbsp;&nbsp; **Columns:** 42
- **Coverage:** `datadate` 1999-06-30 to 2026-05-31
- **Primary key:** `GVKEY` + `datadate` (declared)

| Column | Type | Null % | Description |
| --- | --- | --- | --- |
| GVKEY | int | 0% | Compustat firm identifier. |
| datadate | date | 0% | Fiscal period end date. |
| fyear | int | 0% | Fiscal year label. |
| indfmt | string | 0% | Constant `INDL` in this extract. |
| consol | string | 0% | Constant `C` in this extract. |
| popsrc | string | 0% | Constant `D` in this extract. |
| datafmt | string | 0% | Constant `STD` in this extract. |
| curcd | string | 0% | Constant `USD` in this extract. |
| apdedate | date | 26.2% | "As of" date. TODO: confirm exact meaning against Compustat docs. |
| act, capx, ceq, che, cogs, csho, dlc, dltt, dp, ebit, ebitda, gp, ib, invt, lct, lt, ni, oancf, ppent, pstk, pstkr, rect, revt, sale, seq, sret, txdi, txditc, xint, xrd, xsga | float | 0.03%–97.1% (varies; see report) | Standard Compustat annual line-item codes. TODO: confirm exact definitions against the Compustat data manual before using downstream — do not assume from the abbreviation alone. `sret` is 97.1% null (extremely sparse). |
| costat | string | 0% | Observed: `A`, `I`. Commonly "Active"/"Inactive" company status in Compustat convention — TODO confirm. |
| prcc_f | float | 0.03% | Fiscal-year-end price. |

**Quality notes:**
- `indfmt`/`consol`/`popsrc`/`datafmt`/`curcd` are each single-valued in this
  extract — this pull is already filtered to one Compustat data-format
  combination, so no filtering on those columns is needed downstream.
- 3,320 rows are exact full-row duplicates of another row (dropped in the
  interim output; 1,689 excess rows removed).
- Every `GVKEY` here has at least one row in `ccm_link_table` (full coverage).
- 2 rows have `csho <= 0`; no negative values found in any numeric column.

---

## `fama_french_5f_momentum_monthly.csv` — Fama-French 5 Factors + Momentum

- **Rows:** 324 &nbsp;&nbsp; **Columns:** 8
- **Coverage:** 1999-01-29 to 2025-12-31, one row per month
- **Primary key:** `dateff` (verified unique)

| Column | Type | Null % | Description |
| --- | --- | --- | --- |
| dateff | date | 0% | Month-end date. |
| mktrf | float | 0% | Market excess return (market return minus risk-free rate). |
| smb | float | 0% | Size factor (small minus big). |
| hml | float | 0% | Value factor (high minus low book-to-market). |
| rmw | float | 0% | Profitability factor (robust minus weak). |
| cma | float | 0% | Investment factor (conservative minus aggressive). |
| rf | float | 0% | Risk-free rate. |
| umd | float | 0% | Momentum factor (up minus down). |

**Quality notes:**
- No missing values anywhere in this file.
- Its 324 month-end dates match `crsp_monthly_stock`'s month-ends exactly
  (verified 1:1 in both directions — no gaps either way).

---

## Cross-dataset identifiers

The same real-world identifier is spelled differently across files — joins
must map these explicitly rather than assuming a shared column name:

| Identifier | Spellings observed |
| --- | --- |
| PERMNO (security) | `PERMNO` (crsp_monthly_stock, crsp_delisting), `permno` (crsp_names), `LPERMNO` (ccm_link_table) |
| PERMCO (company) | `PERMCO` (crsp_monthly_stock), `permco` (crsp_names), `LPERMCO` (ccm_link_table) |
| GVKEY (Compustat firm) | `GVKEY` (compustat_fundamentals_annual), `gvkey` (ccm_link_table) |

**Join path:** `compustat_fundamentals_annual.GVKEY` → `ccm_link_table.gvkey`
(filtering to the row whose `[LINKDT, LINKENDDT]` window covers the relevant
`datadate`) → `ccm_link_table.LPERMNO` → `crsp_monthly_stock.PERMNO` /
`crsp_delisting.PERMNO` / `crsp_names.permno`. `fama_french_5f_momentum_monthly`
has no identifiers at all — it joins purely on month-end date.

## Known data-quality issues (summary)

Full detail (exact counts, samples, per-column stats) is generated by the
pipeline into `reports/data_validation/`. As of the last run:

1. **Duplicate rows** in `crsp_monthly_stock` (50,079 exact + 3,244
   conflicting), `crsp_names` (113,353 exact), and
   `compustat_fundamentals_annual` (3,320 exact). Exact duplicates are
   dropped when writing `data/interim/`; conflicting duplicates and
   `crsp_names`' lack of a primary key are left for a future cleaning phase.
2. **Missing delisting information:** 726 of 15,665 `crsp_delisting` rows
   have a null `DelRet`.
3. **Coverage gaps** between `ccm_link_table` and `crsp_monthly_stock` /
   `compustat_fundamentals_annual` — mostly explained by CCM's link history
   extending before both extracts' 1999 start, with a small unexplained
   residual (21 PERMNOs, 1,072 GVKEYs).
4. **Referential integrity gaps:** 2 delisting PERMNOs and 11 successor
   PERMNOs not found in `crsp_monthly_stock`.
5. **Identifier naming inconsistency** across files (see table above).
6. **`crsp_names` has no validity-period columns**, so it cannot support
   point-in-time name/CUSIP lookups on its own.

None of the above have been silently resolved by this pipeline — see
`reports/data_validation/summary.md` for the full findings and `TASKS.md`
for what's deferred to later phases.

## Derived / engineered fields

None yet. This project has not started feature engineering — see `PLAN.md`.
