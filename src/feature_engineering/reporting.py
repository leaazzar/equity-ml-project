"""Writes the feature registry (durable, git-tracked metadata — no licensed
data) and all diagnostics (gitignored — derived from the licensed WRDS
extract) under reports/feature_engineering/.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from feature_engineering.config import FeatureConfig
from feature_engineering.diagnostics import (
    coverage_by_month,
    coverage_by_year,
    feature_stability_through_time,
    missingness_report,
    outlier_diagnostics,
    pairwise_correlations,
    save_correlation_heatmaps,
    save_universe_coverage_plot,
    summary_distributions,
    universe_coverage_through_time,
)
from feature_engineering.registry import feature_names, registry_as_records
from feature_engineering.validation import ValidationResult

logger = logging.getLogger(__name__)


def write_registry(reports_dir: Path | str) -> Path:
    """Write the feature registry as JSON and CSV — durable, git-tracked metadata."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    records = registry_as_records()
    json_path = reports_dir / "feature_registry.json"
    json_path.write_text(json.dumps(records, indent=2, default=str))

    csv_path = reports_dir / "feature_registry.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)

    logger.info("Wrote feature registry to %s and %s", json_path, csv_path)
    return json_path


def write_reports(
    raw_df: pd.DataFrame,
    model_ready_df: pd.DataFrame,
    validation_results: list[ValidationResult],
    config: FeatureConfig,
    reports_dir: Path | str,
) -> Path:
    """Write all diagnostics (gitignored) and a summary.md. Returns summary.md's path."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    names = feature_names()

    missingness_report(raw_df, names).to_csv(reports_dir / "missingness_report.csv", index=False)
    coverage_by_year(raw_df, names).to_csv(reports_dir / "coverage_by_year.csv")
    coverage_by_month(raw_df, names).to_csv(reports_dir / "coverage_by_month.csv")
    summary_distributions(raw_df, names).to_csv(
        reports_dir / "summary_distributions.csv", index=False
    )
    outlier_diagnostics(raw_df, names).to_csv(reports_dir / "outlier_diagnostics.csv", index=False)
    feature_stability_through_time(raw_df, names).to_csv(reports_dir / "feature_stability.csv")

    universe_coverage = universe_coverage_through_time(raw_df)
    universe_coverage.to_csv(reports_dir / "universe_coverage.csv")
    save_universe_coverage_plot(universe_coverage, reports_dir / "universe_coverage.png")

    full_corr = pairwise_correlations(model_ready_df, names)
    full_corr.to_csv(reports_dir / "correlations_all.csv")
    save_correlation_heatmaps(model_ready_df, reports_dir)

    _write_json([asdict(r) for r in validation_results], reports_dir / "validation_results.json")

    generated_at = datetime.now(UTC).isoformat()
    n_failed = sum(1 for r in validation_results if not r.passed)
    validation_table = "\n".join(
        f"| {r.check_name} | {'PASS' if r.passed else 'FAIL'} | {r.n_affected} | {r.detail} |"
        for r in validation_results
    )

    body = f"""# Feature Engineering Diagnostics

Generated: {generated_at}

## Validation

{n_failed} / {len(validation_results)} checks failed.

| Check | Status | # Affected | Detail |
| --- | --- | --- | --- |
{validation_table}

## Files in this directory

- `missingness_report.csv`, `coverage_by_year.csv`, `coverage_by_month.csv`
- `summary_distributions.csv`, `outlier_diagnostics.csv`
- `feature_stability.csv` (monthly cross-sectional median per feature)
- `universe_coverage.csv` / `.png`
- `correlations_all.csv`, `correlation_all.png`, `correlation_<category>.png`
- `feature_registry.json` / `.csv` (also git-tracked — see FEATURE_DICTIONARY.md)
"""
    summary_path = reports_dir / "summary.md"
    summary_path.write_text(body)
    logger.info("Wrote feature engineering diagnostics to %s", reports_dir)
    return summary_path


def _write_json(obj: object, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))
