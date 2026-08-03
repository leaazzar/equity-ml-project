"""Point-in-time feature engineering on the merged master panel.

Consumes `data/processed/master_panel.parquet` (produced by
`data_processing`) and produces two outputs:

- a raw characteristics panel (`data/processed/features_raw.parquet`) —
  interpretable-unit economic characteristics, no cross-sectional
  normalization;
- a model-ready panel (`data/processed/features_model_ready.parquet`) —
  the same characteristics after documented, month-by-month
  winsorization + cross-sectional rank normalization.

Every feature is documented in `FEATURE_DICTIONARY.md` and in the
machine-readable registry (`registry.py`). No labels, targets, or models are
built here — see `PLAN.md`.
"""

__version__ = "0.1.0"
