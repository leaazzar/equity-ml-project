# scripts/

Standalone, runnable entry points (as opposed to importable library code,
which lives in `src/equity_ml/`).

- `download_wrds_data.py` — **not yet implemented.** Placeholder entry point
  for pulling raw extracts from WRDS into `data/raw/` once access is
  provisioned. See `TASKS.md` and `configs/data_sources.yaml`.

## Conventions

- Each script should be runnable as `python scripts/<name>.py ...` from the
  project root, with `src/` importable (achieved via `pip install -e .`).
- Scripts should import logic from `equity_ml.*` rather than containing
  business logic themselves — keep them thin CLI wrappers.
- Scripts must not require network/WRDS credentials to *import* successfully;
  they should only fail at the point of actually connecting.
