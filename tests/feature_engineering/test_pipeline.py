"""Integration tests: the full raw + model-ready build across all required
test scenarios (missing fields, bad denominators, sparse history, delisted
securities, duplicated rows, lag boundaries, extreme outliers, and universe
entry/exit), run together to prove the pipeline wiring — not to re-derive
formulas already proven correct in the per-module unit tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_engineering.pipeline import build_model_ready_features, build_raw_features
from feature_engineering.registry import feature_names
from feature_engineering.validation import check_no_duplicate_keys


@pytest.fixture
def raw_features(synthetic_master_panel, synthetic_compustat_interim) -> pd.DataFrame:
    return build_raw_features(synthetic_master_panel, synthetic_compustat_interim)


@pytest.fixture
def model_ready_features(raw_features) -> pd.DataFrame:
    return build_model_ready_features(raw_features)


def test_build_raw_features_produces_every_registered_feature(raw_features) -> None:
    for name in feature_names():
        assert name in raw_features.columns


def test_build_raw_features_preserves_full_row_count(raw_features, synthetic_master_panel) -> None:
    assert len(raw_features) == len(synthetic_master_panel)


def test_missing_accounting_fields_propagate_not_imputed(raw_features) -> None:
    permno_2 = raw_features[raw_features["PERMNO"] == 2].sort_values("MthCalDt")
    # Rows where cst_ni was null must have null quality_roe/roa, never a
    # silently-imputed value.
    missing_ni_rows = permno_2[permno_2["cst_ni"].isna()]
    assert len(missing_ni_rows) > 0
    assert missing_ni_rows["quality_roe"].isna().all()


def test_bad_denominators_produce_nan_not_inf_or_garbage(raw_features) -> None:
    permno_3 = raw_features[raw_features["PERMNO"] == 3]
    # Restrict to rows where fundamentals are actually available (post-lag) —
    # earlier rows are correctly NaN for the unrelated reason of not having
    # matched any fiscal year yet.
    post_lag = permno_3[permno_3["MthCalDt"] >= "2019-06-30"]

    assert post_lag["quality_roe"].isna().all()  # negative seq
    assert post_lag["value_sales_to_price"].notna().all()  # revt=0 numerator is valid (=0)
    assert post_lag["leverage_current_ratio"].isna().all()  # lct == 0
    assert post_lag["leverage_interest_coverage"].isna().all()  # xint == 0
    numeric_cols = [c for c in feature_names() if c in permno_3.columns]
    assert not np.isinf(permno_3[numeric_cols].astype("float64")).any().any()


def test_sparse_return_history_yields_missing_momentum(raw_features) -> None:
    permno_4 = raw_features[raw_features["PERMNO"] == 4].sort_values("MthCalDt")
    # With only every-other-month trading, a full 12-consecutive-month
    # window is never available -> mom_12m must stay NaN throughout.
    assert permno_4["mom_12m"].isna().all()


def test_delisted_security_uses_ret_adj_not_raw_return(raw_features) -> None:
    permno_5 = raw_features[raw_features["PERMNO"] == 5].sort_values("MthCalDt")
    last_row = permno_5.iloc[-1]
    assert last_row["is_delisted"]
    assert last_row["mom_1m"] == pytest.approx(-0.40)
    assert last_row["reversal_1m"] == pytest.approx(-0.40)


def test_duplicate_rows_are_flagged_by_validation(
    duplicated_master_panel, synthetic_compustat_interim
) -> None:
    raw = build_raw_features(duplicated_master_panel, synthetic_compustat_interim)
    result = check_no_duplicate_keys(raw)
    assert not result.passed
    assert result.n_affected == 1  # the one excess (droppable) copy


def test_lag_boundary_matches_data_processing_convention(raw_features) -> None:
    permno_7 = raw_features[raw_features["PERMNO"] == 7].sort_values("MthCalDt")
    before = permno_7[permno_7["MthCalDt"] == "2019-05-31"]
    at_boundary = permno_7[permno_7["MthCalDt"] == "2019-06-30"]
    assert before["quality_roe"].isna().all()
    assert at_boundary["quality_roe"].notna().all()


def test_extreme_outlier_preserved_in_raw_but_bounded_in_model_ready(
    raw_features, model_ready_features
) -> None:
    permno_8_raw = raw_features[raw_features["PERMNO"] == 8].sort_values("MthCalDt")
    extreme_row = permno_8_raw.iloc[15]
    assert extreme_row["quality_roe"] > 1000  # the raw layer keeps the true extreme value

    permno_8_model = model_ready_features[model_ready_features["PERMNO"] == 8].sort_values(
        "MthCalDt"
    )
    extreme_model_value = permno_8_model.iloc[15]["quality_roe"]
    assert pd.isna(extreme_model_value) or abs(extreme_model_value) <= 1.0


def test_universe_entry_and_exit_preserves_raw_but_nulls_model_ready(
    raw_features, model_ready_features
) -> None:
    permno_9_raw = (
        raw_features[raw_features["PERMNO"] == 9].sort_values("MthCalDt").reset_index(drop=True)
    )
    # Rows 8-12 (0-indexed) are the sub-$1 stretch -> not investable, but the
    # raw price/size feature must still be computed (panel never filtered).
    below_threshold = permno_9_raw.iloc[8:13]
    assert not below_threshold["is_investable"].any()
    assert below_threshold["size_mktcap"].notna().all()

    permno_9_model = (
        model_ready_features[model_ready_features["PERMNO"] == 9]
        .sort_values("MthCalDt")
        .reset_index(drop=True)
    )
    assert permno_9_model.iloc[8:13]["size_mktcap"].isna().all()
    # Recovery afterward should be investable again.
    assert permno_9_raw.iloc[15]["is_investable"]
