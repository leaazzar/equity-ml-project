"""Point-in-time resolution of CRSP PERMNO -> Compustat gvkey via the CCM
link table.

For each CRSP PERMNO-month, finds the `ccm_link_table` row whose validity
window `[LINKDT, LINKENDDT]` covers that month (`LINKENDDT == "E"` means the
link is still open). Implemented as a backward `merge_asof` on `LINKDT`
(grouped by PERMNO) followed by a filter on `LINKENDDT`, which is the
standard efficient way to do a point-in-time interval join in pandas.

As of the last validation run, no PERMNO ever has two *simultaneously*
valid links (verified empirically — see DATA_DICTIONARY.md /
reports/data_validation), so the `LINKPRIM` tie-break below is a defensive
measure, not something exercised by the current data.
"""

from __future__ import annotations

import logging

import pandas as pd

from data_processing.config import MergeConfig

logger = logging.getLogger(__name__)

OPEN_ENDED_SENTINEL_DATE = pd.Timestamp("2099-12-31")


def _prepare_ccm(ccm: pd.DataFrame, config: MergeConfig) -> pd.DataFrame:
    priority = {code: rank for rank, code in enumerate(config.linkprim_priority)}
    out = ccm.copy()
    out["LPERMNO"] = out["LPERMNO"].astype("int64")
    out["LINKDT"] = pd.to_datetime(out["LINKDT"])
    # data_validation's cleaning already parses the raw "E" sentinel (meaning
    # "link still open") to NaT, and validated there are zero genuinely
    # invalid LINKENDDT values — so every NaT here is an open link, never a
    # parse failure. `errors="coerce"` keeps this robust if ever given the
    # raw, unparsed string column instead (e.g. in a test fixture).
    linkenddt = pd.to_datetime(out["LINKENDDT"], errors="coerce")
    out["_linkenddt"] = linkenddt.fillna(OPEN_ENDED_SENTINEL_DATE)
    # Rows not in the configured priority list rank last (least preferred).
    out["_priority_rank"] = out["LINKPRIM"].map(priority).fillna(len(priority)).astype(int)
    # merge_asof requires the right frame sorted by the `on` column (LINKDT)
    # globally — grouping by `by` (LPERMNO) is handled internally and must
    # NOT be used as the primary sort key here. Within an exact LINKDT tie,
    # sort the most preferred LINKPRIM last, since a backward asof match
    # picks the last row on a tie.
    return out.sort_values(["LINKDT", "_priority_rank"], ascending=[True, False])


def resolve_gvkey_for_panel(
    crsp_panel: pd.DataFrame, ccm: pd.DataFrame, config: MergeConfig | None = None
) -> pd.DataFrame:
    """Attach a point-in-time `gvkey` (+ `link_matched` flag) to a CRSP PERMNO-month panel.

    `crsp_panel` must have `PERMNO` and `MthCalDt` columns. Returns a frame
    with those two columns plus `gvkey` (nullable) and `link_matched` (bool),
    in the same row order as the input.
    """
    config = config or MergeConfig()

    left = crsp_panel[["PERMNO", "MthCalDt"]].copy()
    left["PERMNO"] = left["PERMNO"].astype("int64")
    left["MthCalDt"] = pd.to_datetime(left["MthCalDt"])
    left["_orig_order"] = range(len(left))
    left = left.sort_values("MthCalDt")

    right = _prepare_ccm(ccm, config)

    merged = pd.merge_asof(
        left,
        right[["LPERMNO", "gvkey", "LINKDT", "_linkenddt"]],
        left_on="MthCalDt",
        right_on="LINKDT",
        left_by="PERMNO",
        right_by="LPERMNO",
        direction="backward",
    )

    valid = merged["MthCalDt"] <= merged["_linkenddt"]
    merged["link_matched"] = valid.fillna(False)
    merged["gvkey"] = merged["gvkey"].where(merged["link_matched"])

    n_unmatched = int((~merged["link_matched"]).sum())
    logger.info(
        "CCM link resolution: %d / %d PERMNO-months matched to a gvkey (%d unmatched)",
        len(merged) - n_unmatched,
        len(merged),
        n_unmatched,
    )

    merged = merged.sort_values("_orig_order").reset_index(drop=True)
    return merged[["PERMNO", "MthCalDt", "gvkey", "link_matched"]]
