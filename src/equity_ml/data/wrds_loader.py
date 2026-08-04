"""WRDS connection and data-loading scaffolding.

STATUS: this module's live-connection code path has never been exercised.
This project's raw extracts (see `data/README.md`) were supplied directly
as CSV files, not pulled through here. The functions below are placeholders
defining the intended interface for a future live WRDS pull — do not assume
any table/field names or schemas beyond what's confirmed in
`DATA_DICTIONARY.md` until real WRDS access is provisioned and the actual
schema has been inspected against it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from equity_ml.config import wrds_username
from equity_ml.logging_utils import get_logger

if TYPE_CHECKING:
    import pandas as pd
    import wrds as wrds_module

logger = get_logger(__name__)


def get_wrds_connection() -> wrds_module.Connection:
    """Open a WRDS connection.

    TODO(WRDS): Not yet exercised against real WRDS infrastructure. Requires
    the `wrds` package and valid credentials (a local .pgpass entry, or an
    interactive prompt on first connect).
    """
    try:
        import wrds
    except ImportError as exc:  # pragma: no cover - exercised only if dep missing
        raise ImportError(
            "The 'wrds' package is required to connect to WRDS. "
            "Install project dependencies with `pip install -e .`."
        ) from exc

    username = wrds_username()
    if username is None:
        logger.warning(
            "WRDS_USERNAME not set in environment; wrds.Connection() will prompt interactively."
        )
    return wrds.Connection(wrds_username=username)


def load_crsp_monthly(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
    """TODO(WRDS): Implement once CRSP monthly access and schema are confirmed."""
    raise NotImplementedError(
        "Live WRDS pulls are not implemented in this module; raw extracts were "
        "supplied directly as CSV files instead. See data/README.md."
    )


def load_compustat_annual(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
    """TODO(WRDS): Implement once Compustat annual access and schema are confirmed."""
    raise NotImplementedError(
        "Live WRDS pulls are not implemented in this module; raw extracts were "
        "supplied directly as CSV files instead. See data/README.md."
    )
