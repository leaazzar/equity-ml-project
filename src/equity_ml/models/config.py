"""Configuration for Phase 4 modeling — every tunable assumption lives here,
with its rationale, rather than being hardcoded inside model logic. See
MODEL_DESIGN.md for the full design and the project owner's confirmed
decisions (prediction horizon, universe, deferred data-quality follow-ups).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Tunable assumptions for target construction, walk-forward splitting,
    and model fitting. Defaults are documented per-parameter below and match
    MODEL_DESIGN.md's confirmed decisions.
    """

    # --- Column names (source: data/processed/master_panel.parquet and
    # data/processed/features_model_ready.parquet) ---
    permno_column: str = "PERMNO"
    date_column: str = "MthCalDt"
    return_column: str = "ret_adj"
    universe_column: str = "is_investable"

    # --- Target construction (MODEL_DESIGN.md Section 1) ---
    horizon_months: int = 1
    """Forward-return prediction horizon in months. Confirmed by the project
    owner 2026-08-03 as 1 month, monthly rebalance (MODEL_DESIGN.md)."""

    # --- Train/validation/test methodology (MODEL_DESIGN.md Sections 2-3) ---
    purge_embargo_months: int | None = None
    """Months purged from the end of a training window / embargoed from the
    start of a test window at every walk-forward fold boundary, to prevent a
    forward-looking label from overlapping a fold's own train/test split.
    None (default) means "use horizon_months" — the minimum width that
    guarantees no label window spans the boundary."""

    min_initial_train_months: int = 72
    """Minimum length of the first walk-forward training window (6 years).
    Long enough to cover the longest-lookback feature (skew_36m, 36-month
    window / 24 min. observations) and to let Compustat coverage move past
    the panel's early ramp-up (MERGE_REPORT.md: 1999 has a small fraction of
    later years' gvkey-matched coverage) before the first fold is scored."""

    refit_frequency_months: int = 12
    """How often the expanding-window model is refit (annual, matching the
    Compustat data's own annual update cadence). Test folds between refits
    use the most recently fit model."""

    inner_validation_fraction: float = 0.2
    """Fraction of each training window's most recent months held out
    (chronologically, with the same purge/embargo rule applied at that inner
    boundary) as a validation split for hyperparameter tuning (Section 5)."""

    random_seed: int = 42
    """Matches configs/config.yaml's project.random_seed."""

    momentum_benchmark_feature: str = "mom_12_1"
    """Feature used for the single-factor momentum-sort benchmark (Section 4)
    — the skip-month 12-month momentum characteristic, the most canonical
    cross-sectional predictor in the literature."""
