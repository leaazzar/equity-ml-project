"""Integration tests for the full master-panel build (all five merge stages
composed together)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_processing.panel import build_master_panel, write_panel


def test_build_master_panel_has_no_duplicate_keys(interim_frames, crsp_monthly_df) -> None:
    panel, dedup_summary = build_master_panel(interim_frames)
    assert panel.duplicated(subset=["PERMNO", "MthCalDt"]).sum() == 0
    assert len(panel) == len(crsp_monthly_df) - dedup_summary.n_rows_dropped
    assert dedup_summary.n_rows_dropped == 1


def test_build_master_panel_resolves_conflicting_duplicate(interim_frames) -> None:
    panel, _ = build_master_panel(interim_frames)
    permno_6 = panel[panel["PERMNO"] == 6]
    assert len(permno_6) == 1
    assert permno_6.iloc[0]["Ticker"] == "FFF"


def test_build_master_panel_carries_gvkey_and_compustat(interim_frames) -> None:
    panel, _ = build_master_panel(interim_frames)
    row = panel[(panel["PERMNO"] == 1) & (panel["MthCalDt"] == "2020-06-30")].iloc[0]
    assert row["gvkey"] == 100
    assert row["cst_revt"] == 100.0


def test_build_master_panel_unlinked_permno_has_null_compustat(interim_frames) -> None:
    panel, _ = build_master_panel(interim_frames)
    row = panel[panel["PERMNO"] == 3].iloc[0]
    assert pd.isna(row["gvkey"])
    assert pd.isna(row["cst_revt"])


def test_build_master_panel_applies_delisting_adjustment(interim_frames) -> None:
    panel, _ = build_master_panel(interim_frames)
    row = panel[panel["PERMNO"] == 7].iloc[0]
    assert row["is_delisted"]
    assert row["ret_adj"] != row["MthRet"]


def test_build_master_panel_merges_fama_french(interim_frames) -> None:
    panel, _ = build_master_panel(interim_frames)
    assert panel["ff_mktrf"].notna().all()


def test_write_panel_writes_parquet(interim_frames, tmp_path: Path) -> None:
    panel, _ = build_master_panel(interim_frames)
    output_path = write_panel(panel, tmp_path / "processed")
    assert output_path.exists()
    roundtrip = pd.read_parquet(output_path)
    assert len(roundtrip) == len(panel)
