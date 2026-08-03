"""Raw CSV loading, with explicit, auditable date parsing.

Date columns are parsed separately from `pandas.read_csv` (rather than via
`parse_dates=`) so that unparseable values can be counted and sampled instead
of silently becoming `NaT`. Known non-date sentinels (e.g. CCM's `LINKENDDT`
`"E"` value) are recognized and excluded from the "invalid" count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from data_validation.datasets import DatasetSpec

MAX_SAMPLES = 5


@dataclass
class DateParseResult:
    """Outcome of parsing one date column: the parsed series plus any anomalies."""

    column: str
    parsed: pd.Series
    n_valid: int
    n_sentinel: int
    n_invalid: int
    invalid_samples: list[str] = field(default_factory=list)


def parse_date_column(raw: pd.Series, column: str, spec: DatasetSpec) -> DateParseResult:
    """Parse one raw (string) column to datetime, tracking sentinels and failures."""
    sentinel_map = spec.date_sentinels.get(column, {})
    is_sentinel = raw.isin(sentinel_map.keys())

    parsed = pd.to_datetime(raw.where(~is_sentinel), format="mixed", errors="coerce")
    is_blank = raw.isna() | (raw.astype("string").str.strip() == "")
    is_invalid = parsed.isna() & ~is_sentinel & ~is_blank

    invalid_samples = list(raw.loc[is_invalid].astype("string").unique()[:MAX_SAMPLES])

    return DateParseResult(
        column=column,
        parsed=parsed,
        n_valid=int((~parsed.isna() & ~is_sentinel).sum()),
        n_sentinel=int(is_sentinel.sum()),
        n_invalid=int(is_invalid.sum()),
        invalid_samples=invalid_samples,
    )


def load_raw(spec: DatasetSpec, raw_dir: Path | str) -> pd.DataFrame:
    """Load a raw CSV as-is (strings/numbers only; dates are left as raw strings).

    Use `parse_date_column` on the returned frame for each of `spec.date_columns`
    to get auditable, sentinel-aware parsing rather than pandas' silent coercion.
    """
    path = Path(raw_dir) / spec.filename
    if not path.exists():
        raise FileNotFoundError(
            f"Raw file not found for dataset {spec.name!r}: {path}. "
            "WRDS raw data must be placed under data/raw/ before running validation."
        )
    return pd.read_csv(path, low_memory=False, dtype_backend="numpy_nullable")
