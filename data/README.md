# data/

**No data is committed to this repository.** WRDS data has not been
downloaded yet. This directory only holds the folder structure and
`.gitignore` rules that will govern data once it exists.

- `raw/` — untouched WRDS extracts, exactly as pulled (TODO: not yet
  populated; see `scripts/download_wrds_data.py`).
- `interim/` — intermediate, partially-cleaned data.
- `processed/` — final, analysis-ready datasets.

All three subdirectories are gitignored except for `.gitkeep` placeholders.
Do not commit raw data, licensed datasets, or anything derived from them —
see the authorship/data rules in `CLAUDE.md`.
