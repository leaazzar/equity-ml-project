"""Writes Phase 4 diagnostics under reports/modeling/ (gitignored, derived
from the licensed WRDS extract — same convention as
`feature_engineering.reporting`) — the out-of-sample prediction panel,
IC series/summary, validation results, and a human-readable summary.md.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from equity_ml.models.splits import WalkForwardFold
from equity_ml.models.validation import ValidationResult

logger = logging.getLogger(__name__)


def _write_json(obj: object, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


def write_reports(
    predictions: pd.DataFrame,
    ic_series: pd.DataFrame,
    ic_summary: pd.DataFrame,
    validation_results: list[ValidationResult],
    folds: list[WalkForwardFold],
    reports_dir: Path | str,
) -> tuple[Path, Path]:
    """Write all diagnostics and summary.md. Returns (predictions_path, ic_summary_path)."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = reports_dir / "predictions.parquet"
    predictions.to_parquet(predictions_path, index=False)

    ic_series.to_csv(reports_dir / "ic_series.csv", index=False)
    ic_summary_path = reports_dir / "ic_summary.csv"
    ic_summary.to_csv(ic_summary_path, index=False)

    folds_records = [asdict(fold) for fold in folds]
    _write_json(folds_records, reports_dir / "folds.json")
    _write_json([asdict(r) for r in validation_results], reports_dir / "validation_results.json")

    generated_at = datetime.now(UTC).isoformat()
    n_failed = sum(1 for r in validation_results if not r.passed)
    validation_table = "\n".join(
        f"| {r.check_name} | {'PASS' if r.passed else 'FAIL'} | {r.n_affected} | {r.detail} |"
        for r in validation_results
    )
    ic_table = "\n".join(
        f"| {row.model} | {row.ic_mean:.4f} | {row.ic_std:.4f} | "
        f"{row.ic_t_stat:.2f} | {row.n_months} |"
        for row in ic_summary.itertuples()
    )

    body = f"""# Modeling Diagnostics

Generated: {generated_at}

## Walk-forward configuration

{len(folds)} folds, spanning {folds[0].test_start.date() if folds else "n/a"} to \
{folds[-1].test_end.date() if folds else "n/a"}.

## Validation

{n_failed} / {len(validation_results)} checks failed.

| Check | Status | # Affected | Detail |
| --- | --- | --- | --- |
{validation_table}

## Information Coefficient summary (pooled across all folds)

| Model | IC mean | IC std | IC t-stat | # months |
| --- | --- | --- | --- | --- |
{ic_table}

## Files in this directory

- `predictions.parquet` — full out-of-sample prediction panel
  (permno, date, model, fold_id, y_true, y_pred).
- `ic_series.csv` — monthly IC per model.
- `ic_summary.csv` — pooled IC summary per model (this file, as a table above).
- `folds.json` — every walk-forward fold's date boundaries.
- `validation_results.json` — this file's validation table, machine-readable.
"""
    summary_path = reports_dir / "summary.md"
    summary_path.write_text(body)
    logger.info("Wrote modeling diagnostics to %s", reports_dir)
    return predictions_path, ic_summary_path
