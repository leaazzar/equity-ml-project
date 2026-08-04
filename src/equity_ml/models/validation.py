"""Automated leakage/consistency checks for Phase 4 (MODEL_DESIGN.md Section
7's "the backtest should add an analogous automated check ... not just trust
the walk-forward loop's bookkeeping by construction"). Mirrors
`feature_engineering.validation`'s pattern exactly: every check is a plain
function that returns a `ValidationResult` rather than raising, so a
pipeline run can report every check's outcome in one summary instead of
stopping at the first failure.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from equity_ml.models.config import ModelConfig
from equity_ml.models.splits import WalkForwardFold


@dataclass(frozen=True)
class ValidationResult:
    check_name: str
    passed: bool
    detail: str
    n_affected: int = 0


def check_fold_purge_embargo(folds: list[WalkForwardFold], config: ModelConfig) -> ValidationResult:
    """Every fold's test window must start strictly after `train_end +
    horizon_months` — the minimum separation that guarantees a forward
    label computed at the last training month cannot reach into the test
    window (MODEL_DESIGN.md Section 2)."""
    violations = 0
    for fold in folds:
        min_gap_months = (fold.test_start.year - fold.train_end.year) * 12 + (
            fold.test_start.month - fold.train_end.month
        )
        if min_gap_months <= config.horizon_months:
            violations += 1
    passed = violations == 0
    detail = (
        "every fold's test window starts more than horizon_months after train_end"
        if passed
        else f"{violations} fold(s) have test_start within horizon_months of train_end"
    )
    return ValidationResult("fold_purge_embargo", passed, detail, violations)


def check_folds_expanding(folds: list[WalkForwardFold]) -> ValidationResult:
    """Every fold's training window must start at the same (sample-start)
    date and train_end must be strictly increasing — an expanding, not
    rolling, window (MODEL_DESIGN.md Section 3)."""
    if not folds:
        return ValidationResult("folds_expanding", True, "no folds to check", 0)
    violations = 0
    first_start = folds[0].train_start
    prev_train_end = None
    for fold in folds:
        if fold.train_start != first_start:
            violations += 1
        if prev_train_end is not None and fold.train_end <= prev_train_end:
            violations += 1
        prev_train_end = fold.train_end
    passed = violations == 0
    detail = "all folds expand from the same start date" if passed else f"{violations} violation(s)"
    return ValidationResult("folds_expanding", passed, detail, violations)


def check_no_duplicate_predictions(
    predictions: pd.DataFrame, config: ModelConfig
) -> ValidationResult:
    """No (permno, date, model) combination should appear more than once —
    would indicate a row scored by two overlapping folds."""
    key = [config.permno_column, config.date_column, "model"]
    n_dupes = int(predictions.duplicated(subset=key).sum())
    passed = n_dupes == 0
    detail = (
        "no duplicate (permno, date, model) predictions" if passed else f"{n_dupes} duplicate rows"
    )
    return ValidationResult("no_duplicate_predictions", passed, detail, n_dupes)


def check_predictions_within_fold_test_window(
    predictions: pd.DataFrame, folds: list[WalkForwardFold], config: ModelConfig
) -> ValidationResult:
    """Every prediction's date must fall inside its own fold_id's [test_start,
    test_end] window — catches a bookkeeping bug that mislabeled a fold_id or
    sliced the wrong date range."""
    bounds = {fold.fold_id: (fold.test_start, fold.test_end) for fold in folds}
    violations = 0
    for fold_id_raw, group in predictions.groupby("fold_id"):
        fold_id = int(fold_id_raw)  # type: ignore[arg-type]
        if fold_id not in bounds:
            violations += len(group)
            continue
        start, end = bounds[fold_id]
        out_of_range = ~group[config.date_column].between(start, end)
        violations += int(out_of_range.sum())
    passed = violations == 0
    detail = (
        "every prediction falls within its fold's test window"
        if passed
        else f"{violations} out-of-range rows"
    )
    return ValidationResult("predictions_within_fold_test_window", passed, detail, violations)


def run_all_checks(
    predictions: pd.DataFrame, folds: list[WalkForwardFold], config: ModelConfig
) -> list[ValidationResult]:
    return [
        check_folds_expanding(folds),
        check_fold_purge_embargo(folds, config),
        check_no_duplicate_predictions(predictions, config),
        check_predictions_within_fold_test_window(predictions, folds, config),
    ]
