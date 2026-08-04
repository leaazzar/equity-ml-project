"""Purged/embargoed expanding-window walk-forward splits (MODEL_DESIGN.md
Sections 2-3).

Two temporal-leakage risks a plain chronological split doesn't handle:

1. A forward-looking label's window can extend past a fold's train/test
   boundary (e.g. a 3-month-horizon label computed at the last training
   month uses returns through month+3, reaching into the test period).
   Fixed by *purging* the last `purge_embargo_months` months from a training
   window and *embargoing* the first `purge_embargo_months` months of the
   following test window.
2. Hyperparameter selection must never see the fold's own test data. Fixed
   by carving each fold's *inner* train/validation split the same way,
   entirely inside that fold's outer training window.

Folds are built by position in the panel's own sorted, deduplicated month
grid (not by calendar-date arithmetic), so this works correctly for
whatever the actual observed month-end dates are without assuming a fixed
day-of-month or pulling in a calendar-arithmetic dependency. Every function
here returns date *boundaries*, never touches the label or feature values
themselves — callers slice a panel by `config.date_column` against the
boundaries returned.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from equity_ml.models.config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WalkForwardFold:
    """One expanding-window fold's date boundaries (all inclusive)."""

    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    """Last month included in training, after purging."""
    test_start: pd.Timestamp
    """First month scored, after embargo."""
    test_end: pd.Timestamp

    inner_train_end: pd.Timestamp
    """Last month of the inner (hyperparameter-tuning) training split, after
    purging — a subset of [train_start, train_end]."""
    inner_val_start: pd.Timestamp
    """First month of the inner validation split, after embargo."""
    inner_val_end: pd.Timestamp
    """Equal to train_end — the inner validation split is the trailing slice
    of the same outer training window."""


def _purge_embargo_months(config: ModelConfig) -> int:
    return (
        config.purge_embargo_months
        if config.purge_embargo_months is not None
        else config.horizon_months
    )


def build_walk_forward_folds(
    dates: pd.Series, config: ModelConfig | None = None
) -> list[WalkForwardFold]:
    """Build expanding-window, purged/embargoed walk-forward folds spanning
    the full range of `dates` (e.g. `panel[config.date_column]`).

    - Expanding: every fold's training window starts at the sample's first
      month and grows; it is never truncated from the left (MODEL_DESIGN.md
      Section 3).
    - The first fold's training window is at least
      `config.min_initial_train_months` months long.
    - Subsequent folds advance the test window by
      `config.refit_frequency_months`.
    - Each fold purges the last `purge_embargo_months` months of training and
      embargoes the first `purge_embargo_months` months after the purge
      boundary from the test window, so no label window computed inside
      training can reach into a scored test month.
    - Each fold additionally carries an inner train/validation split (the
      trailing `config.inner_validation_fraction` of the outer training
      window, purge/embargo applied identically) for hyperparameter tuning
      (MODEL_DESIGN.md Section 5) — entirely within [train_start, train_end],
      never touching the fold's own test window.
    """
    config = config or ModelConfig()
    months = pd.Series(pd.to_datetime(dates.unique())).sort_values().reset_index(drop=True)
    n_months = len(months)
    if n_months == 0:
        return []

    purge = _purge_embargo_months(config)
    inner_val_months = max(
        1, round(config.min_initial_train_months * config.inner_validation_fraction)
    )

    if config.min_initial_train_months > n_months:
        raise ValueError(
            f"min_initial_train_months ({config.min_initial_train_months}) exceeds the "
            f"available sample length ({n_months} distinct months)"
        )
    min_window_for_inner_split = inner_val_months + 2 * purge + 1
    if config.min_initial_train_months < min_window_for_inner_split:
        raise ValueError(
            "min_initial_train_months is too small to carve out an inner "
            f"validation split of {inner_val_months} months with {purge}-month "
            f"purge/embargo on each side (need at least {min_window_for_inner_split})"
        )

    folds: list[WalkForwardFold] = []
    fold_id = 0
    train_end_idx = config.min_initial_train_months - 1
    while True:
        purged_train_end_idx = train_end_idx - purge
        test_start_idx = train_end_idx + 1 + purge
        test_end_idx = min(train_end_idx + config.refit_frequency_months, n_months - 1)
        if test_start_idx > n_months - 1:
            break

        inner_val_start_unpurged_idx = purged_train_end_idx - inner_val_months + 1
        inner_train_end_idx = inner_val_start_unpurged_idx - 1 - purge
        inner_val_start_idx = inner_val_start_unpurged_idx + purge

        folds.append(
            WalkForwardFold(
                fold_id=fold_id,
                train_start=months.iloc[0],
                train_end=months.iloc[purged_train_end_idx],
                test_start=months.iloc[test_start_idx],
                test_end=months.iloc[test_end_idx],
                inner_train_end=months.iloc[inner_train_end_idx],
                inner_val_start=months.iloc[inner_val_start_idx],
                inner_val_end=months.iloc[purged_train_end_idx],
            )
        )
        fold_id += 1
        if test_end_idx >= n_months - 1:
            break
        train_end_idx = test_end_idx

    logger.info(
        "Built %d walk-forward folds (purge/embargo=%d months, refit every %d months)",
        len(folds),
        purge,
        config.refit_frequency_months,
    )
    return folds


def slice_by_dates(
    panel: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, config: ModelConfig | None = None
) -> pd.DataFrame:
    """Rows of `panel` with `config.date_column` in `[start, end]` inclusive."""
    config = config or ModelConfig()
    mask = (panel[config.date_column] >= start) & (panel[config.date_column] <= end)
    return panel.loc[mask]
