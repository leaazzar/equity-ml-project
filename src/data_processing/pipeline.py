"""End-to-end orchestration: load interim data, build the master panel,
write it to data/processed/, compute diagnostics, and write reports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from data_processing.config import MergeConfig
from data_processing.dedup import DedupSummary
from data_processing.diagnostics import MergeDiagnostics, compute_diagnostics
from data_processing.io import load_all_interim
from data_processing.panel import build_master_panel, write_panel
from data_processing.reporting import write_reports

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Everything the pipeline produced, for programmatic inspection or testing."""

    output_path: Path
    report_path: Path
    diagnostics: MergeDiagnostics
    dedup_summary: DedupSummary


def run_pipeline(
    interim_dir: Path | str,
    processed_dir: Path | str,
    reports_dir: Path | str,
    config: MergeConfig | None = None,
) -> PipelineResult:
    """Run the full point-in-time merge pipeline and return its results."""
    config = config or MergeConfig()

    logger.info("Loading interim datasets from %s", interim_dir)
    frames = load_all_interim(interim_dir)

    panel, dedup_summary = build_master_panel(frames, config)
    output_path = write_panel(panel, processed_dir)

    logger.info("Computing merge diagnostics")
    diagnostics = compute_diagnostics(panel)

    report_path = write_reports(diagnostics, dedup_summary, reports_dir)

    return PipelineResult(
        output_path=output_path,
        report_path=report_path,
        diagnostics=diagnostics,
        dedup_summary=dedup_summary,
    )
