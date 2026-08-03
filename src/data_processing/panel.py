"""Orchestrates the full point-in-time merge into one master security-month panel.

Pipeline order (each stage's rationale is in its own module's docstring and
in MERGE_REPORT.md):

1. Resolve `crsp_monthly_stock`'s conflicting duplicate PERMNO+month keys
   (`dedup.resolve_duplicate_keys`) — a master panel needs exactly one row
   per PERMNO-month.
2. Resolve a point-in-time `gvkey` for every PERMNO-month via the CCM link
   table (`ccm_linking.resolve_gvkey_for_panel`).
3. Attach Compustat fundamentals with a reporting lag and staleness cutoff
   (`compustat_merge.merge_compustat_point_in_time`) — the core
   look-ahead-bias guard.
4. Apply delisting-return adjustments (`delisting.apply_delisting_returns`).
5. Merge Fama-French factors by month (`fama_french.merge_fama_french`).

The panel is built as a left join from `crsp_monthly_stock` throughout —
every CRSP security-month is kept even without a CCM/Compustat match, which
is what keeps the panel survivorship-bias-free (delisted, unmatched, and
never-covered-by-Compustat securities all remain in the output; only their
`cst_*` columns are null).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from data_processing.ccm_linking import resolve_gvkey_for_panel
from data_processing.compustat_merge import merge_compustat_point_in_time
from data_processing.config import MergeConfig
from data_processing.dedup import DedupSummary, resolve_duplicate_keys
from data_processing.delisting import apply_delisting_returns
from data_processing.fama_french import merge_fama_french

logger = logging.getLogger(__name__)

DEDUP_KEY_COLUMNS = ["PERMNO", "MthCalDt"]


def build_master_panel(
    frames: dict[str, pd.DataFrame], config: MergeConfig | None = None
) -> tuple[pd.DataFrame, DedupSummary]:
    """Build the master panel from the six loaded interim datasets.

    `frames` must have keys: crsp_monthly_stock, crsp_delisting,
    ccm_link_table, compustat_fundamentals_annual,
    fama_french_5f_momentum_monthly (see `data_processing.io.load_all_interim`).
    """
    config = config or MergeConfig()

    crsp = frames["crsp_monthly_stock"]
    logger.info("Resolving conflicting duplicate PERMNO+month keys in crsp_monthly_stock")
    crsp_dedup, dedup_summary = resolve_duplicate_keys(crsp, DEDUP_KEY_COLUMNS)

    logger.info("Resolving point-in-time gvkey for each PERMNO-month")
    linked = resolve_gvkey_for_panel(crsp_dedup, frames["ccm_link_table"], config)
    panel = crsp_dedup.reset_index(drop=True).copy()
    panel["gvkey"] = linked["gvkey"].to_numpy()
    panel["link_matched"] = linked["link_matched"].to_numpy()

    logger.info("Merging Compustat fundamentals point-in-time")
    panel = merge_compustat_point_in_time(panel, frames["compustat_fundamentals_annual"], config)

    logger.info("Applying delisting return adjustments")
    panel = apply_delisting_returns(panel, frames["crsp_delisting"])

    logger.info("Merging Fama-French factors")
    panel = merge_fama_french(panel, frames["fama_french_5f_momentum_monthly"])

    logger.info("Master panel built: %d rows, %d columns", len(panel), panel.shape[1])
    return panel, dedup_summary


def write_panel(
    panel: pd.DataFrame, processed_dir: Path | str, filename: str = "master_panel.parquet"
) -> Path:
    """Write the master panel to data/processed/ as Parquet."""
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / filename
    panel.to_parquet(output_path, index=False)
    logger.info("Wrote master panel (%d rows) to %s", len(panel), output_path)
    return output_path
