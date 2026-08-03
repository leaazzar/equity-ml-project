"""End-to-end orchestration: load raw data, profile it, check cross-dataset
integrity, write typed/deduplicated interim copies, and generate reports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data_validation.cleaning import CleaningSummary, clean_and_write
from data_validation.datasets import DATASETS, DatasetSpec
from data_validation.integrity import IntegrityFinding, run_all_integrity_checks
from data_validation.io import load_raw
from data_validation.profiling import DatasetProfile, profile_dataset
from data_validation.reporting import write_reports

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Everything the pipeline produced, for programmatic inspection or testing."""

    profiles: dict[str, DatasetProfile]
    findings: list[IntegrityFinding]
    cleaning_summaries: list[CleaningSummary]
    report_path: Path


def load_all_raw(
    raw_dir: Path | str, specs: dict[str, DatasetSpec] = DATASETS
) -> dict[str, pd.DataFrame]:
    """Load every registered raw dataset into memory, keyed by dataset name."""
    frames: dict[str, pd.DataFrame] = {}
    for name, spec in specs.items():
        logger.info("Loading %s from %s", name, spec.filename)
        frames[name] = load_raw(spec, raw_dir)
        logger.info("%s: %d rows, %d columns", name, len(frames[name]), frames[name].shape[1])
    return frames


def run_pipeline(
    raw_dir: Path | str,
    interim_dir: Path | str,
    reports_dir: Path | str,
    specs: dict[str, DatasetSpec] = DATASETS,
) -> PipelineResult:
    """Run the full ingestion/validation pipeline and return its results.

    Steps: load raw CSVs -> profile each dataset -> run cross-dataset integrity
    checks -> write deduplicated/typed interim Parquet copies -> write reports.
    """
    frames = load_all_raw(raw_dir, specs)

    logger.info("Profiling %d datasets", len(frames))
    profiles = {name: profile_dataset(df, specs[name]) for name, df in frames.items()}

    logger.info("Running cross-dataset integrity checks")
    findings = run_all_integrity_checks(frames)
    n_warnings = sum(1 for f in findings if f.status == "warning")
    logger.info("Integrity checks complete: %d warning(s) of %d", n_warnings, len(findings))

    logger.info("Writing interim outputs to %s", interim_dir)
    cleaning_summaries = []
    for name, df in frames.items():
        _, summary = clean_and_write(df, specs[name], interim_dir)
        cleaning_summaries.append(summary)

    report_path = write_reports(profiles, findings, cleaning_summaries, reports_dir)

    return PipelineResult(
        profiles=profiles,
        findings=findings,
        cleaning_summaries=cleaning_summaries,
        report_path=report_path,
    )
