"""Orchestrates the full feature-engineering build: derived accounting
quantities, growth rates, the investable universe, every raw feature, and
the model-ready transform — plus validation and reporting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data_processing.config import MergeConfig
from feature_engineering.accounting import add_derived_accounting_columns, merge_growth_features
from feature_engineering.config import FeatureConfig
from feature_engineering.io import load_compustat_interim, load_master_panel
from feature_engineering.missing import add_missingness_flags
from feature_engineering.ratios import (
    compute_investment_growth_features,
    compute_leverage_features,
    compute_quality_features,
    compute_value_features,
)
from feature_engineering.registry import feature_names
from feature_engineering.reporting import write_registry, write_reports
from feature_engineering.size_features import compute_size_features
from feature_engineering.time_series_features import (
    compute_liquidity_features,
    compute_momentum_reversal_features,
    compute_volatility_features,
)
from feature_engineering.transforms import build_model_ready_column
from feature_engineering.units import add_unit_normalized_columns
from feature_engineering.universe import build_investable_universe
from feature_engineering.validation import (
    ValidationResult,
    check_cross_sectional_isolation,
    check_denominator_handling,
    check_no_calendar_gaps,
    check_no_duplicate_keys,
    check_no_infinite_values,
    check_no_leakage_via_truncation,
    check_registry_source_columns_exist,
    check_reproducibility,
    check_return_based_feature_bounds,
)

logger = logging.getLogger(__name__)

_KEY_COLUMNS = ["PERMNO", "PERMCO", "MthCalDt", "gvkey", "is_investable", "has_fundamentals"]


def build_raw_features(
    master_panel: pd.DataFrame,
    compustat_interim: pd.DataFrame,
    feature_config: FeatureConfig | None = None,
    merge_config: MergeConfig | None = None,
) -> pd.DataFrame:
    """Build the raw (interpretable-unit) feature panel from a master panel.

    Every intermediate/derived accounting column (at_proxy, book_equity,
    etc.) is retained on the returned frame alongside the requested
    characteristics, so downstream validation can confirm every registry
    source column actually exists.
    """
    feature_config = feature_config or FeatureConfig()
    merge_config = merge_config or MergeConfig()

    df = master_panel.sort_values(["PERMNO", "MthCalDt"]).reset_index(drop=True)
    df = add_unit_normalized_columns(df)
    df = add_derived_accounting_columns(df)
    df = merge_growth_features(df, compustat_interim, feature_config, merge_config)
    df = add_missingness_flags(df)
    df["is_investable"] = build_investable_universe(df, feature_config)

    features: dict[str, pd.Series] = {}
    features.update(compute_size_features(df, df["is_investable"]))
    features.update(compute_value_features(df))
    features.update(compute_momentum_reversal_features(df, feature_config))
    features.update(compute_quality_features(df))
    features.update(compute_investment_growth_features(df))
    features.update(compute_leverage_features(df))
    features.update(compute_liquidity_features(df))
    features.update(compute_volatility_features(df, feature_config))

    for name, series in features.items():
        df[name] = series

    return df


def build_model_ready_features(
    raw_df: pd.DataFrame, feature_config: FeatureConfig | None = None
) -> pd.DataFrame:
    """Winsorize + rank-scale every registry feature, month-by-month, universe-only."""
    feature_config = feature_config or FeatureConfig()
    month = raw_df["MthCalDt"].dt.to_period("M")
    is_investable = raw_df["is_investable"]

    out = raw_df[_KEY_COLUMNS].copy()
    for name in feature_names():
        out[name] = build_model_ready_column(raw_df[name], month, is_investable, feature_config)
    return out


@dataclass
class FeaturePipelineResult:
    raw_path: Path
    model_ready_path: Path
    registry_path: Path
    validation_results: list[ValidationResult]


def _sample_validation_inputs(
    raw_master_panel: pd.DataFrame, n_permnos: int = 40, n_dates: int = 3
) -> tuple[list[int], list[pd.Timestamp]]:
    rng = pd.Series(raw_master_panel["PERMNO"].unique())
    sample_permnos = rng.sample(n=min(n_permnos, len(rng)), random_state=0).tolist()
    all_dates = sorted(raw_master_panel["MthCalDt"].unique())
    step = max(len(all_dates) // (n_dates + 1), 1)
    sample_dates = [pd.Timestamp(all_dates[i]) for i in range(step, len(all_dates), step)][:n_dates]
    return sample_permnos, sample_dates


def run_validations(
    raw_df: pd.DataFrame,
    model_ready_df: pd.DataFrame,
    raw_master_panel: pd.DataFrame,
    compustat_interim: pd.DataFrame,
    feature_config: FeatureConfig,
    merge_config: MergeConfig,
) -> list[ValidationResult]:
    """Run the full validation suite (structural checks on the full data,
    behavioral checks — leakage, isolation, reproducibility — on a sample)."""
    names = feature_names()
    results = [
        check_no_duplicate_keys(raw_df),
        check_no_calendar_gaps(raw_df),
        check_no_infinite_values(raw_df, names),
        check_registry_source_columns_exist(set(raw_df.columns)),
        check_return_based_feature_bounds(raw_df),
        check_denominator_handling(raw_df),
    ]

    sample_permnos, sample_dates = _sample_validation_inputs(raw_master_panel)
    sample_panel = raw_master_panel[raw_master_panel["PERMNO"].isin(sample_permnos)]

    def build_fn(panel: pd.DataFrame) -> pd.DataFrame:
        return build_raw_features(panel, compustat_interim, feature_config, merge_config)

    results.append(
        check_no_leakage_via_truncation(
            build_fn, sample_panel, sample_dates, names, sample_permnos=sample_permnos
        )
    )
    results.append(check_reproducibility(build_fn, sample_panel, names))

    sample_raw = build_fn(sample_panel)
    month = sample_raw["MthCalDt"].dt.to_period("M")
    sample_months = sorted(month.unique())[:3]
    if names:
        probe_feature = names[0]

        def transform_fn(subset: pd.DataFrame) -> pd.Series:
            sub_month = subset["MthCalDt"].dt.to_period("M")
            return build_model_ready_column(
                subset[probe_feature], sub_month, subset["is_investable"], feature_config
            )

        results.append(
            check_cross_sectional_isolation(transform_fn, sample_raw, month, sample_months)
        )

    return results


def run_pipeline(
    processed_dir: Path | str,
    interim_dir: Path | str,
    reports_dir: Path | str,
    feature_config: FeatureConfig | None = None,
    merge_config: MergeConfig | None = None,
) -> FeaturePipelineResult:
    """Load the master panel, build both feature layers, validate, and write outputs."""
    feature_config = feature_config or FeatureConfig()
    merge_config = merge_config or MergeConfig()
    processed_dir = Path(processed_dir)

    logger.info("Loading master panel and Compustat interim data")
    master_panel = load_master_panel(processed_dir)
    compustat_interim = load_compustat_interim(interim_dir)

    logger.info("Building raw feature panel")
    raw_df = build_raw_features(master_panel, compustat_interim, feature_config, merge_config)

    logger.info("Building model-ready feature panel")
    model_ready_df = build_model_ready_features(raw_df, feature_config)

    logger.info("Running validation suite")
    validation_results = run_validations(
        raw_df, model_ready_df, master_panel, compustat_interim, feature_config, merge_config
    )
    n_failed = sum(1 for r in validation_results if not r.passed)
    if n_failed:
        logger.error("%d / %d validation checks FAILED", n_failed, len(validation_results))
    else:
        logger.info("All %d validation checks passed", len(validation_results))

    output_columns = _KEY_COLUMNS + feature_names()
    raw_output = raw_df[[c for c in output_columns if c in raw_df.columns]]

    raw_path = processed_dir / "features_raw.parquet"
    model_ready_path = processed_dir / "features_model_ready.parquet"
    raw_output.to_parquet(raw_path, index=False)
    model_ready_df.to_parquet(model_ready_path, index=False)
    logger.info("Wrote raw features (%d rows) to %s", len(raw_output), raw_path)
    logger.info("Wrote model-ready features (%d rows) to %s", len(model_ready_df), model_ready_path)

    reports_dir = Path(reports_dir)
    registry_path = write_registry(reports_dir)
    write_reports(raw_df, model_ready_df, validation_results, feature_config, reports_dir)

    return FeaturePipelineResult(
        raw_path=raw_path,
        model_ready_path=model_ready_path,
        registry_path=registry_path,
        validation_results=validation_results,
    )
