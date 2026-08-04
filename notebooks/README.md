# notebooks/

Exploratory Jupyter notebooks. Nothing here is production code — anything
that becomes reusable should be promoted into `src/equity_ml/`.

## Conventions

- Prefix notebooks with a two-digit stage number and a short description,
  e.g. `01_wrds_data_exploration.ipynb`, `02_feature_exploration.ipynb`.
- Clear cell outputs before committing where practical — notebooks should
  not be a backdoor for committing data extracts (see `.gitignore`).
- No notebooks exist yet; every phase so far (data validation through
  modeling/backtesting) has been driven by the pipelines in `src/` and
  `scripts/` instead. This directory stays available for ad hoc exploration
  should that become useful.
