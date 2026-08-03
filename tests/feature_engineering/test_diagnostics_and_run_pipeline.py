"""Tests for diagnostics computation, report writing, and the full
run_pipeline (disk-to-disk) entry point — using the synthetic multi-scenario
fixtures, never real data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from feature_engineering.diagnostics import (
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
from feature_engineering.pipeline import (
    build_model_ready_features,
    build_raw_features,
    run_pipeline,
)
from feature_engineering.registry import feature_names
from feature_engineering.reporting import write_registry, write_reports


def test_missingness_report_shape(synthetic_master_panel, synthetic_compustat_interim) -> None:
    raw = build_raw_features(synthetic_master_panel, synthetic_compustat_interim)
    report = missingness_report(raw, feature_names())
    assert set(report.columns) == {"feature", "n_non_null", "pct_missing"}
    assert len(report) == len(feature_names())


def test_coverage_by_year_and_summary_distributions_run(
    synthetic_master_panel, synthetic_compustat_interim
) -> None:
    raw = build_raw_features(synthetic_master_panel, synthetic_compustat_interim)
    coverage = coverage_by_year(raw, feature_names())
    assert set(coverage.index) == {2019, 2020}

    stats = summary_distributions(raw, feature_names())
    assert "mean" in stats.columns
    assert "50%" in stats.columns


def test_outlier_diagnostics_and_stability_run(
    synthetic_master_panel, synthetic_compustat_interim
) -> None:
    raw = build_raw_features(synthetic_master_panel, synthetic_compustat_interim)
    outliers = outlier_diagnostics(raw, feature_names())
    assert "pct_beyond_3std" in outliers.columns

    stability = feature_stability_through_time(raw, feature_names())
    assert len(stability) > 0


def test_universe_coverage_and_plot(
    synthetic_master_panel, synthetic_compustat_interim, tmp_path: Path
) -> None:
    raw = build_raw_features(synthetic_master_panel, synthetic_compustat_interim)
    coverage = universe_coverage_through_time(raw)
    assert {"n_total", "n_investable", "pct_investable"}.issubset(coverage.columns)

    plot_path = tmp_path / "universe.png"
    save_universe_coverage_plot(coverage, plot_path)
    assert plot_path.exists()


def test_pairwise_correlations_and_heatmaps(
    synthetic_master_panel, synthetic_compustat_interim, tmp_path: Path
) -> None:
    raw = build_raw_features(synthetic_master_panel, synthetic_compustat_interim)
    model_ready = build_model_ready_features(raw)

    corr = pairwise_correlations(model_ready, feature_names())
    assert corr.shape == (len(feature_names()), len(feature_names()))

    save_correlation_heatmaps(model_ready, tmp_path)
    assert (tmp_path / "correlation_all.png").exists()
    assert (tmp_path / "correlation_size.png").exists()


def test_write_registry_creates_json_and_csv(tmp_path: Path) -> None:
    path = write_registry(tmp_path)
    assert path.exists()
    assert (tmp_path / "feature_registry.csv").exists()


def test_write_reports_creates_summary(
    synthetic_master_panel, synthetic_compustat_interim, tmp_path: Path
) -> None:
    from feature_engineering.config import FeatureConfig
    from feature_engineering.validation import ValidationResult

    raw = build_raw_features(synthetic_master_panel, synthetic_compustat_interim)
    model_ready = build_model_ready_features(raw)
    results = [ValidationResult("dummy_check", True, "ok", 0)]

    summary_path = write_reports(raw, model_ready, results, FeatureConfig(), tmp_path)
    assert summary_path.exists()
    assert "dummy_check" in summary_path.read_text()


def test_run_pipeline_end_to_end(
    synthetic_master_panel, synthetic_compustat_interim, tmp_path: Path
) -> None:
    processed_dir = tmp_path / "processed"
    interim_dir = tmp_path / "interim"
    reports_dir = tmp_path / "reports"
    processed_dir.mkdir()
    interim_dir.mkdir()

    synthetic_master_panel.to_parquet(processed_dir / "master_panel.parquet", index=False)
    synthetic_compustat_interim.to_parquet(
        interim_dir / "compustat_fundamentals_annual.parquet", index=False
    )

    result = run_pipeline(processed_dir, interim_dir, reports_dir)

    assert result.raw_path.exists()
    assert result.model_ready_path.exists()
    assert result.registry_path.exists()
    assert len(result.validation_results) > 0

    raw = pd.read_parquet(result.raw_path)
    model_ready = pd.read_parquet(result.model_ready_path)
    assert len(raw) == len(synthetic_master_panel)
    assert len(model_ready) == len(raw)
    assert (reports_dir / "summary.md").exists()
