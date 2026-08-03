"""Writes merge diagnostics to reports/data_processing/ (JSON + Markdown).

Like data_validation's reports, this directory is gitignored: the firm
coverage table and missing-variable breakdowns are aggregate statistics, but
regenerating them is cheap and keeping the durable, git-tracked narrative in
`MERGE_REPORT.md` (hand-written, at the repository root) avoids committing
any pipeline-run artifacts derived from the licensed WRDS extract.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from data_processing.dedup import DedupSummary
from data_processing.diagnostics import MergeDiagnostics

logger = logging.getLogger(__name__)


def _write_json(obj: object, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


def _firm_coverage_table(diagnostics: MergeDiagnostics) -> str:
    header = "| Year | # PERMNO | # with gvkey | # with Compustat |\n"
    header += "| --- | --- | --- | --- |\n"
    rows = "\n".join(
        f"| {r['year']} | {r['n_permno']:,} | {r['n_permno_with_gvkey']:,} | "
        f"{r['n_permno_with_compustat']:,} |"
        for r in diagnostics.firm_coverage_by_year
    )
    return header + rows


def _missing_accounting_table(diagnostics: MergeDiagnostics) -> str:
    header = "| Column | % missing in final panel |\n| --- | --- |\n"
    rows = "\n".join(
        f"| {col} | {pct:.2f}% |"
        for col, pct in sorted(diagnostics.accounting_var_missing_pct.items())
    )
    return header + rows


def write_reports(
    diagnostics: MergeDiagnostics,
    dedup_summary: DedupSummary,
    reports_dir: Path | str,
) -> Path:
    """Write diagnostics.json, dedup_summary.json, and diagnostics.md.

    Returns the path to diagnostics.md.
    """
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    _write_json(asdict(diagnostics), reports_dir / "diagnostics.json")
    _write_json(asdict(dedup_summary), reports_dir / "dedup_summary.json")

    generated_at = datetime.now(UTC).isoformat()
    body = f"""# Merge Diagnostics

Generated: {generated_at}

## Coverage

- Rows: {diagnostics.n_rows:,} (unique PERMNO-months: {diagnostics.n_unique_permno_months:,}, \
duplicates: {diagnostics.n_duplicate_permno_months:,})
- PERMNO-months matched to a gvkey: {diagnostics.pct_gvkey_matched:.2f}%
- PERMNO-months with usable (non-expired) Compustat fundamentals: \
{diagnostics.pct_compustat_usable:.2f}%
- PERMNO-months matched to Compustat but subsequently expired: \
{diagnostics.pct_compustat_expired:.2f}%
- PERMNO-months matched to Fama-French factors: {diagnostics.pct_ff_matched:.2f}%
- Delisted PERMNO-months: {diagnostics.n_delisted_rows:,} \
(missing DelRet: {diagnostics.n_delisted_missing_return:,})

## Duplicate keys before dedup (crsp_monthly_stock)

- Rows in: {dedup_summary.rows_in:,}
- Rows out: {dedup_summary.rows_out:,}
- Duplicate-key groups resolved: {dedup_summary.n_duplicate_groups:,}
- Rows dropped: {dedup_summary.n_rows_dropped:,}

## Missing accounting variables (% null in final panel)

{_missing_accounting_table(diagnostics)}

## Firm coverage over time

{_firm_coverage_table(diagnostics)}
"""
    diagnostics_path = reports_dir / "diagnostics.md"
    diagnostics_path.write_text(body)
    logger.info("Wrote merge diagnostics to %s", diagnostics_path)
    return diagnostics_path
