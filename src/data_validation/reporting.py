"""Renders profiling/integrity/cleaning results to reports/data_validation/.

Produces one machine-readable JSON file per dataset profile, one JSON file
for the cross-dataset integrity findings, one for the cleaning summary, and
a single human-readable `summary.md` tying everything together.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from data_validation.cleaning import CleaningSummary
from data_validation.integrity import IntegrityFinding
from data_validation.profiling import DatasetProfile

logger = logging.getLogger(__name__)


def _write_json(obj: object, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


def _column_table(profile: DatasetProfile) -> str:
    header = "| Column | Dtype | Null % | # Unique | Sample values |\n"
    header += "| --- | --- | --- | --- | --- |\n"
    rows = [
        f"| {c.name} | {c.dtype} | {c.null_pct:.2f}% | {c.n_unique} | "
        f"{', '.join(c.sample_values) or '—'} |"
        for c in profile.columns
    ]
    return header + "\n".join(rows)


def _date_table(profile: DatasetProfile) -> str:
    if not profile.date_columns:
        return "_No declared date columns._"
    header = "| Column | Min | Max | Valid | Sentinel | Invalid | Invalid samples |\n"
    header += "| --- | --- | --- | --- | --- | --- | --- |\n"
    rows = [
        f"| {d.column} | {d.min_date} | {d.max_date} | {d.n_valid} | {d.n_sentinel} | "
        f"{d.n_invalid} | {', '.join(d.invalid_samples) or '—'} |"
        for d in profile.date_columns
    ]
    return header + "\n".join(rows)


def _dataset_section(profile: DatasetProfile) -> str:
    dup = profile.duplicates
    return f"""## {profile.name}

- Row count: **{profile.row_count:,}**
- Column count: **{profile.column_count}**
- Primary key ({dup.key_source}): `{", ".join(dup.key_columns)}`
- Rows sharing a duplicate key: **{dup.n_duplicate_key_rows:,}** \
({dup.n_exact_full_row_duplicates:,} exact full-row duplicates, \
{dup.n_conflicting_duplicates:,} same-key rows with differing values)
- Excess duplicate rows (droppable to make the key unique): **{dup.n_excess_duplicate_rows:,}**

### Columns

{_column_table(profile)}

### Date columns

{_date_table(profile)}
"""


def _integrity_table(findings: list[IntegrityFinding]) -> str:
    header = "| Check | Status | Affected / Total | Datasets | Description |\n"
    header += "| --- | --- | --- | --- | --- |\n"
    rows = [
        f"| {f.check_name} | {f.status.upper()} | {f.n_affected:,} / {f.n_total:,} | "
        f"{', '.join(f.datasets)} | {f.description} |"
        for f in findings
    ]
    return header + "\n".join(rows)


def _display_path(path_str: str) -> str:
    """Render a path relative to the current working directory when possible."""
    path = Path(path_str)
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _cleaning_table(summaries: list[CleaningSummary]) -> str:
    header = "| Dataset | Rows in | Rows out | Exact duplicates dropped | Output |\n"
    header += "| --- | --- | --- | --- | --- |\n"
    rows = [
        f"| {s.name} | {s.rows_in:,} | {s.rows_out:,} | {s.n_exact_duplicates_dropped:,} | "
        f"`{_display_path(s.output_path)}` |"
        for s in summaries
    ]
    return header + "\n".join(rows)


def write_reports(
    profiles: dict[str, DatasetProfile],
    findings: list[IntegrityFinding],
    cleaning_summaries: list[CleaningSummary],
    reports_dir: Path | str,
) -> Path:
    """Write per-dataset JSON, integrity JSON, cleaning JSON, and a summary.md.

    Returns the path to `summary.md`.
    """
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    for name, profile in profiles.items():
        _write_json(asdict(profile), reports_dir / f"{name}_profile.json")

    _write_json([asdict(f) for f in findings], reports_dir / "integrity_checks.json")
    _write_json([asdict(s) for s in cleaning_summaries], reports_dir / "cleaning_summary.json")

    generated_at = datetime.now(UTC).isoformat()
    n_warnings = sum(1 for f in findings if f.status == "warning")

    overview_header = "| Dataset | Rows | Columns | Duplicate-key rows |\n"
    overview_header += "| --- | --- | --- | --- |\n"
    overview_rows = "\n".join(
        f"| {p.name} | {p.row_count:,} | {p.column_count} | {p.duplicates.n_duplicate_key_rows:,} |"
        for p in profiles.values()
    )
    dataset_sections = "\n".join(_dataset_section(p) for p in profiles.values())

    body = f"""# Data Validation Report

Generated: {generated_at}

Integrity checks flagged as **WARNING**: {n_warnings} / {len(findings)}

## Overview

{overview_header}{overview_rows}

## Cross-dataset integrity checks

{_integrity_table(findings)}

## Cleaning summary (data/interim/ outputs)

{_cleaning_table(cleaning_summaries)}

{dataset_sections}
"""
    summary_path = reports_dir / "summary.md"
    summary_path.write_text(body)
    logger.info("Wrote validation report to %s", summary_path)
    return summary_path
