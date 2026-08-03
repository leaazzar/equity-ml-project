"""Writes typed, deduplicated intermediate copies of the raw data.

"Cleaning" here is intentionally narrow: parse declared date columns and
drop rows that are *exact, full-row duplicates* (a deterministic, lossless
operation). No imputation, business-rule deduplication, or schema changes
happen here — that is deferred to the feature-engineering phase in
`PLAN.md`. Raw files under `data/raw/` are only ever read, never modified.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data_validation.datasets import DatasetSpec
from data_validation.io import parse_date_column

logger = logging.getLogger(__name__)


@dataclass
class CleaningSummary:
    """Record of what happened when writing one dataset's interim output."""

    name: str
    rows_in: int
    rows_out: int
    n_exact_duplicates_dropped: int
    date_columns_cast: list[str]
    output_path: str


def clean_and_write(
    df: pd.DataFrame, spec: DatasetSpec, interim_dir: Path | str
) -> tuple[pd.DataFrame, CleaningSummary]:
    """Drop exact duplicate rows, parse date columns, and write to Parquet.

    Returns the cleaned in-memory DataFrame alongside a summary of what changed,
    so callers can log or report on it without re-reading the output file.
    """
    rows_in = len(df)
    deduped = df.drop_duplicates(keep="first").reset_index(drop=True)
    n_dropped = rows_in - len(deduped)
    if n_dropped:
        logger.info(
            "%s: dropped %d exact full-row duplicate(s) of %d input rows",
            spec.name,
            n_dropped,
            rows_in,
        )

    cleaned = deduped.copy()
    for col in spec.date_columns:
        cleaned[col] = parse_date_column(cleaned[col], col, spec).parsed

    interim_dir = Path(interim_dir)
    interim_dir.mkdir(parents=True, exist_ok=True)
    output_path = interim_dir / f"{spec.name}.parquet"
    cleaned.to_parquet(output_path, index=False)
    logger.info("%s: wrote %d rows to %s", spec.name, len(cleaned), output_path)

    summary = CleaningSummary(
        name=spec.name,
        rows_in=rows_in,
        rows_out=len(cleaned),
        n_exact_duplicates_dropped=n_dropped,
        date_columns_cast=list(spec.date_columns),
        output_path=str(output_path),
    )
    return cleaned, summary
