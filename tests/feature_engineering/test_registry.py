"""Sanity checks for the feature registry itself."""

from __future__ import annotations

from feature_engineering.registry import (
    FEATURE_REGISTRY,
    collinear_feature_pairs,
    feature_names,
    feature_names_by_category,
    model_feature_names,
)

EXPECTED_CATEGORIES = {
    "size",
    "value",
    "momentum",
    "reversal",
    "quality",
    "growth",
    "leverage",
    "liquidity",
    "volatility",
}


def test_registry_has_no_duplicate_names() -> None:
    names = feature_names()
    assert len(names) == len(set(names))


def test_registry_covers_all_expected_categories() -> None:
    assert set(feature_names_by_category()) == EXPECTED_CATEGORIES


def test_registry_feature_count_is_in_target_range() -> None:
    # Task target: ~50-60 robust characteristics.
    assert 45 <= len(FEATURE_REGISTRY) <= 65


def test_every_spec_has_non_empty_documentation() -> None:
    for spec in FEATURE_REGISTRY:
        assert spec.formula.strip()
        assert spec.economic_intuition.strip()
        assert spec.source_columns
        assert spec.missing_value_treatment.strip()
        assert spec.outlier_treatment.strip()


def test_reversal_1m_flagged_collinear_with_mom_1m() -> None:
    by_name = {spec.name: spec for spec in FEATURE_REGISTRY}
    assert by_name["reversal_1m"].collinear_with == "mom_1m"
    assert by_name["reversal_1m"].exclude_from_model_features is True
    assert by_name["mom_1m"].exclude_from_model_features is False


def test_model_feature_names_excludes_reversal_1m_but_keeps_mom_1m() -> None:
    names = model_feature_names()
    assert "reversal_1m" not in names
    assert "mom_1m" in names
    # Every other feature should still be included — only the flagged
    # collinear one is dropped.
    assert len(names) == len(feature_names()) - 1


def test_collinear_feature_pairs_reports_reversal_1m() -> None:
    pairs = collinear_feature_pairs()
    assert ("reversal_1m", "mom_1m") in pairs
