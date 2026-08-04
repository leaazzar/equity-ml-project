"""Configuration for Phase 4 backtesting (MODEL_DESIGN.md Sections 6-7) —
every tunable assumption lives here, with its rationale, rather than being
hardcoded inside portfolio-construction or backtest logic.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BacktestConfig:
    """Tunable assumptions for portfolio construction, cost modeling, and
    performance evaluation. Defaults match MODEL_DESIGN.md Section 6-7's
    recommended (not yet separately confirmed by the project owner, since
    Section 6 flags these as "documented alternative" choices, not blocking
    decisions the way target/universe/horizon were)."""

    permno_column: str = "PERMNO"
    date_column: str = "MthCalDt"
    model_column: str = "model"
    score_column: str = "y_pred"
    realized_return_column: str = "target_raw"

    # --- Portfolio construction (MODEL_DESIGN.md Section 6) ---
    decile_count: int = 10
    """Number of score-ranked buckets each month. Long the top bucket, short
    the bottom bucket (10/10 decile convention — the standard granularity
    in this literature, keeping each leg's stock count in the hundreds given
    this panel's typical monthly investable-universe size)."""

    weighting_scheme: str = "equal"
    """"equal" (default — simplest, least assumption-laden) or "score"
    (score-proportional within each leg, normalized to sum to 1). Both live
    behind this one flag, not two code paths."""

    # --- Costs (MODEL_DESIGN.md Section 7) ---
    transaction_cost_bps: float = 10.0
    """One-way transaction cost in basis points per unit of turnover — a
    documented placeholder assumption (this project has no bid-ask-spread
    or commission data). Reported alongside a 0/5/10/20/50 bps sensitivity
    table, not trusted as a precise figure on its own."""

    cost_sensitivity_bps: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0, 50.0)

    # --- Performance evaluation (MODEL_DESIGN.md Section 8) ---
    periods_per_year: int = 12
    """Monthly rebalancing (MODEL_DESIGN.md's confirmed 1-month horizon) ->
    12 periods/year for annualization."""

    factor_columns: tuple[str, ...] = ("ff_mktrf", "ff_smb", "ff_hml", "ff_rmw", "ff_cma", "ff_umd")
    """Fama-French 5 + momentum factor columns already merged into the
    master panel by data_processing — used for the alpha/factor-loading
    regression (MODEL_DESIGN.md Section 8)."""
