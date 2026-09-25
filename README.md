# equity-ml

A cross-sectional equity return prediction project built on WRDS data (CRSP,
Compustat, the CCM link table and Fama-French factors). It covers the whole
path from raw extracts to a backtested long/short strategy:

1. Validating and cleaning the raw WRDS files
2. Merging them into a point-in-time monthly panel (no look-ahead, no
   survivorship bias)
3. Building 51 firm characteristics (size, value, momentum, quality,
   investment, leverage, liquidity, volatility...)
4. Training models with purged, embargoed walk-forward validation
5. Backtesting decile long/short portfolios, net of transaction costs

I wrote up each stage's decisions as I went, so the markdown files in the
root are worth reading if you want the reasoning and not just the code.

## Results

Out-of-sample, 2005–2025 (21 annual walk-forward folds), decile long/short,
net of 10 bps one-way costs:

| Model | Mean IC | IC t-stat | Ann. return | Sharpe | Max drawdown |
| --- | --- | --- | --- | --- | --- |
| Fama-MacBeth | 0.023 | 4.50 | 15.5% | 1.01 | -46% |
| Ridge | 0.020 | 3.97 | 12.5% | 0.89 | -37% |
| Gradient boosting | 0.030 | 5.53 | 15.6% | 0.69 | -72% |
| Elastic net | 0.026 | 4.56 | 10.6% | 0.63 | -54% |
| Lasso | 0.025 | 4.40 | 10.1% | 0.60 | -54% |
| Momentum sort | 0.043 | 4.82 | 10.2% | 0.48 | -74% |

The most interesting result to me: gradient boosting had the most consistent
signal (highest IC t-stat), but the simple linear models turned their weaker
signal into better risk-adjusted returns because their portfolios were much
less volatile. Signal quality and portfolio performance don't always agree.

This is a first pass, not a finished strategy. Random forest was left out for
compute reasons, costs are a flat placeholder, and only ~13% of investable
rows have every feature available. [MODEL_REPORT.md](MODEL_REPORT.md) has the
full numbers, the factor regressions and the list of known limitations.

## Documentation

| File | What's in it |
| --- | --- |
| [DATA_DICTIONARY.md](DATA_DICTIONARY.md) | Raw datasets, schemas, data-quality issues found |
| [MERGE_REPORT.md](MERGE_REPORT.md) | How the panel is merged: reporting lags, CCM linking, delisting returns |
| [FEATURE_DICTIONARY.md](FEATURE_DICTIONARY.md) | Formula, rationale and point-in-time handling for every feature |
| [UNIT_AUDIT_REPORT.md](UNIT_AUDIT_REPORT.md) | A units audit that caught a 1000x bug in the value ratios |
| [MODEL_DESIGN.md](MODEL_DESIGN.md) | Target, walk-forward setup, models, tuning, backtest design |
| [MODEL_REPORT.md](MODEL_REPORT.md) | Results from the first real run |
| [MODEL_CARD.md](MODEL_CARD.md) | One-page summary of the modeling system |
| [PLAN.md](PLAN.md) | What's next |

## A few design choices worth knowing about

- **Point-in-time fundamentals.** Compustat data is only attached 6 months
  after fiscal year end and expires after 12 months, so a model never sees
  numbers that weren't public yet.
- **No survivorship bias.** Every CRSP security-month is kept, including
  delisted firms, and delisting returns are compounded into the final month.
- **Automated look-ahead test.** The feature pipeline rebuilds every feature
  from data truncated at several dates and checks the results match the
  full-sample build exactly.
- **Purge and embargo.** A one-month gap sits at every train/test boundary so
  the forward-return target can't leak across folds.
- **Units checked, not assumed.** CRSP market cap is in $ thousands and
  Compustat is in $ millions. Mixing them silently broke the value features
  until the unit audit caught it.

## Repository layout

```
configs/              YAML config (project settings, logging, data sources)
data/                 raw / interim / processed (all gitignored, see data/README.md)
reports/              generated diagnostics (gitignored, regenerate anytime)
scripts/              entry points for each pipeline stage
src/
  data_validation/    stage 1: raw data profiling and integrity checks
  data_processing/    stage 2: point-in-time merge
  feature_engineering/ stage 3: features + validation
  equity_ml/models/   stage 4: walk-forward training
  equity_ml/backtest/ stage 5: portfolios and performance
tests/                pytest suite (synthetic fixtures only)
```

## Setup

Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
make install              # installs the package + dev tools + pre-commit
cp .env.example .env      # add your WRDS_USERNAME
```

## Running it

You need your own WRDS access. The data is licensed and isn't included here.
Put the six raw CSVs listed in [DATA_DICTIONARY.md](DATA_DICTIONARY.md) into
`data/raw/`, then run the stages in order:

```bash
python scripts/run_data_validation.py     # -> data/interim/
python scripts/run_data_processing.py     # -> data/processed/master_panel.parquet
python scripts/run_feature_engineering.py # -> data/processed/features_*.parquet
python scripts/run_modeling.py            # -> reports/modeling/
python scripts/run_backtest.py            # -> reports/backtest/
```

Each stage writes its diagnostics to the matching folder under `reports/`.

## Development

```bash
make lint        # ruff
make typecheck   # mypy
make test        # pytest
make check       # everything CI runs
```

The tests use small synthetic datasets, so they run without any WRDS data.

## License

MIT for the code (see [LICENSE](LICENSE)). This doesn't cover CRSP,
Compustat or CCM data, which are WRDS products and are never stored in this
repo.
