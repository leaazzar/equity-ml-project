"""Orchestrates the full Phase 4 modeling build (MODEL_DESIGN.md): forward-
return target construction, joining `ret_adj` from the master panel onto the
model-ready feature panel, purged/embargoed expanding-window walk-forward
training across benchmark and ML models, out-of-sample prediction
collection, validation, and diagnostics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from equity_ml.models.baselines import DEFAULT_BASELINE_RUNNERS
from equity_ml.models.config import ModelConfig
from equity_ml.models.diagnostics import compute_ic_series, summarize_ic
from equity_ml.models.estimators import DEFAULT_ESTIMATORS, EstimatorSpec
from equity_ml.models.io import load_master_panel, load_model_ready_features
from equity_ml.models.reporting import write_reports
from equity_ml.models.splits import WalkForwardFold, build_walk_forward_folds
from equity_ml.models.targets import build_targets
from equity_ml.models.training import ModelRunner, make_ml_model_runner, run_walk_forward_training
from equity_ml.models.validation import ValidationResult, run_all_checks
from feature_engineering.registry import model_feature_names

logger = logging.getLogger(__name__)


def build_training_panel(
    master_panel: pd.DataFrame,
    model_ready_features: pd.DataFrame,
    config: ModelConfig | None = None,
) -> pd.DataFrame:
    """Join model-ready features with `ret_adj` (needed for target
    construction — not present in either feature panel) and attach the
    forward-return targets (`target_raw`, `target_demeaned`,
    `target_valid` — see targets.py)."""
    config = config or ModelConfig()
    joined = model_ready_features.merge(
        master_panel[[config.permno_column, config.date_column, config.return_column]],
        on=[config.permno_column, config.date_column],
        how="left",
    )
    targets = build_targets(joined, config)
    return joined.merge(
        targets[
            [
                config.permno_column,
                config.date_column,
                "target_raw",
                "target_demeaned",
                "target_valid",
            ]
        ],
        on=[config.permno_column, config.date_column],
        how="left",
    )


def default_model_runners(
    estimators: tuple[EstimatorSpec, ...] = DEFAULT_ESTIMATORS,
) -> dict[str, ModelRunner]:
    """Every benchmark (MODEL_DESIGN.md Section 4) plus every ML estimator
    (Section 4-5), wrapped into the common model-runner signature
    `training.py`'s walk-forward loop expects."""
    runners: dict[str, ModelRunner] = dict(DEFAULT_BASELINE_RUNNERS)
    for spec in estimators:
        runners[spec.name] = make_ml_model_runner(spec)
    return runners


@dataclass
class ModelPipelineResult:
    predictions_path: Path
    ic_summary_path: Path
    folds: list[WalkForwardFold]
    validation_results: list[ValidationResult]


def run_pipeline(
    processed_dir: Path | str,
    reports_dir: Path | str,
    config: ModelConfig | None = None,
    target_column: str = "target_demeaned",
    model_runners: dict[str, ModelRunner] | None = None,
) -> ModelPipelineResult:
    """Load the master panel and model-ready features, build targets,
    restrict to the confirmed modeling universe (`is_investable`, as-is —
    MODEL_DESIGN.md's confirmed decisions), run walk-forward training across
    every model, validate, and write reports.
    """
    config = config or ModelConfig()
    processed_dir = Path(processed_dir)

    logger.info("Loading master panel and model-ready features")
    master_panel = load_master_panel(processed_dir)
    model_ready_features = load_model_ready_features(processed_dir)

    logger.info("Building training panel (features + forward-return targets)")
    panel = build_training_panel(master_panel, model_ready_features, config)
    panel = panel[panel[config.universe_column].fillna(False)].reset_index(drop=True)
    logger.info("Modeling universe (is_investable): %d rows", len(panel))

    feature_columns = model_feature_names()

    folds = build_walk_forward_folds(panel[config.date_column], config)
    runners = model_runners or default_model_runners()
    logger.info("Running walk-forward training: %d models x %d folds", len(runners), len(folds))
    predictions = run_walk_forward_training(
        panel, feature_columns, target_column, folds, runners, config
    )

    logger.info("Running validation suite")
    validation_results = run_all_checks(predictions, folds, config)
    n_failed = sum(1 for r in validation_results if not r.passed)
    if n_failed:
        logger.error("%d / %d validation checks FAILED", n_failed, len(validation_results))
    else:
        logger.info("All %d validation checks passed", len(validation_results))

    ic_series = compute_ic_series(predictions, config)
    ic_summary = summarize_ic(ic_series, config)

    predictions_path, ic_summary_path = write_reports(
        predictions, ic_series, ic_summary, validation_results, folds, reports_dir
    )

    return ModelPipelineResult(
        predictions_path=predictions_path,
        ic_summary_path=ic_summary_path,
        folds=folds,
        validation_results=validation_results,
    )
