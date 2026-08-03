"""Data ingestion and validation pipeline for raw WRDS extracts.

This package profiles, cross-validates, and produces typed intermediate
copies of the raw CSV extracts in `data/raw/`. It does not perform feature
engineering, cleaning beyond exact-duplicate removal, or modeling — see
`PLAN.md` for those later phases.
"""

__version__ = "0.1.0"
