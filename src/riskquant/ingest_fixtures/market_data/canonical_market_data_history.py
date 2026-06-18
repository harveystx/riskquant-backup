# src/riskquant/ingest_fixtures/market_data/canonical_market_data_history.py
"""
Canonical historical market data fixture.

Builds a synthetic `(T, N)` historical risk-factor matrix whose columns are
generated dynamically from `CANONICAL_RISK_STATIC_CONFIG_DATA` — guaranteeing
referential integrity with the static metadata.

Math
----
- Equity / FX / commodity spot-like factors follow geometric Brownian motion
  with drift `-0.5 * sigma**2 * dt` so simulated levels remain unbiased in
  log-space.
- Rate factors follow arithmetic Brownian motion (random walk).

Memory layout
-------------
The output matrix is allocated F-contiguous so a downstream caller doing
column-major time-series access reads cache-aligned. `np.cumsum(..., out=...)`
and `np.exp(..., out=...)` mutate the buffer in place; no intermediates.

Output dtype is `float32` to match `build_risk_factor_matrix_from_df`'s
default — generating in float64 would just be downcast immediately.

TODO: split this file into `_parts/{nodes,curves,surfaces}.py` once the
universe count grows past comfort. The dynamic-build pattern makes the split
mechanical.
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from riskquant.ingest_fixtures.as_of import CANONICAL_AS_OF_DATE
from riskquant.ingest_fixtures.config.risk_static_config.canonical_risk_static_config import (
    CANONICAL_RISK_STATIC_CONFIG_DATA,
)
from riskquant.ingest_fixtures.constants import (
    BUSINESS_DAYS_PER_YEAR,
    CANONICAL_HISTORY_BUSINESS_DAYS,
    EQUITY_DEFAULTS,
    FX_DEFAULTS,
    GAS_COMMODITY_DEFAULTS,
    HISTORY_DTYPE,
    OIL_COMMODITY_DEFAULTS,
    RATE_DEFAULTS,
    VOL_OF_VOL_DEFAULTS,
    _MarketRiskFactorParams,
)
from riskquant.ingest_fixtures.rng import get_canonical_rng

_logger: logging.Logger = logging.getLogger(__name__)


# ==============================================================================
# Time series setup
# ==============================================================================

HISTORICAL_DATES: Final[npt.NDArray[np.datetime64]] = np.sort(
    np.busday_offset(
        CANONICAL_AS_OF_DATE,
        -np.arange(CANONICAL_HISTORY_BUSINESS_DAYS - 1, -1, -1),
    )
)


# ==============================================================================
# Universe build (dynamic — derived from the static config, not parallel-listed)
# ==============================================================================


def _build_canonical_universe() -> list[tuple[str, _MarketRiskFactorParams]]:
    """
    Derive the full historical universe from `CANONICAL_RISK_STATIC_CONFIG_DATA`.
    Returns `[(factor_name, params), ...]` in node->curve_points->surface_points
    order.
    """
    specs: list[tuple[str, _MarketRiskFactorParams]] = []

    # ---- 1. Nodes ----
    node_meta = CANONICAL_RISK_STATIC_CONFIG_DATA["node_meta"]
    for node_name, asset_class in zip(
        node_meta["node_name"], node_meta["asset_class"], strict=True
    ):
        if asset_class == "EQUITY":
            specs.append((str(node_name), EQUITY_DEFAULTS))
        else:
            specs.append((str(node_name), FX_DEFAULTS))

    # ---- 2. Curve points ----
    curve_meta = CANONICAL_RISK_STATIC_CONFIG_DATA["curve_meta"]
    curve_info: dict[object, tuple[str, str]] = {
        cid: (str(name), str(utype))
        for cid, name, utype in zip(
            curve_meta["curve_id"],
            curve_meta["curve_name"],
            curve_meta["quote_type"],
            strict=True,
        )
    }

    curve_point_meta = CANONICAL_RISK_STATIC_CONFIG_DATA["curve_point_meta"]
    for cid, tenor in zip(
        curve_point_meta["curve_id"],
        curve_point_meta["tenor_token"],
        strict=True,
    ):
        curve_name, quote_type = curve_info[cid]
        factor_name = f"{curve_name}{tenor}"

        if quote_type == "RATE":
            specs.append((factor_name, RATE_DEFAULTS))
        elif "GAS" in curve_name:
            specs.append((factor_name, GAS_COMMODITY_DEFAULTS))
        else:
            specs.append((factor_name, OIL_COMMODITY_DEFAULTS))

    # ---- 3. Surface points ----
    surface_meta = CANONICAL_RISK_STATIC_CONFIG_DATA["surface_meta"]
    surface_info: dict[object, tuple[str, str]] = {
        sid: (str(name), str(y_kind))
        for sid, name, y_kind in zip(
            surface_meta["surface_id"],
            surface_meta["surface_name"],
            surface_meta["y_axis_kind"],
            strict=True,
        )
    }

    surface_point_meta = CANONICAL_RISK_STATIC_CONFIG_DATA["surface_point_meta"]
    for sid, x_tenor, y_label in zip(
        surface_point_meta["surface_id"],
        surface_point_meta["x_tenor_token"],
        surface_point_meta["y_label"],
        strict=True,
    ):
        surface_name, y_axis_kind = surface_info[sid]
        factor_name = f"{surface_name}{x_tenor}[{y_axis_kind}={y_label}]"
        specs.append((factor_name, VOL_OF_VOL_DEFAULTS))

    return specs


_FACTOR_SPECS: Final[list[tuple[str, _MarketRiskFactorParams]]] = (
    _build_canonical_universe()
)

CANONICAL_HISTORICAL_FACTOR_NAMES: Final[list[str]] = [
    name for name, _ in _FACTOR_SPECS
]
CANONICAL_HISTORICAL_FACTOR_COUNT: Final[int] = len(CANONICAL_HISTORICAL_FACTOR_NAMES)


# ==============================================================================
# Path generation (in-place, F-contiguous, batched RNG draw)
# ==============================================================================

_rng: np.random.Generator = get_canonical_rng()
_dt: float = 1.0 / BUSINESS_DAYS_PER_YEAR
_sqrt_dt: float = float(np.sqrt(_dt))

# Pre-allocate F-contiguous so per-column traversal is contiguous in memory.
_historical_values: npt.NDArray[np.floating] = np.empty(
    (CANONICAL_HISTORY_BUSINESS_DAYS, CANONICAL_HISTORICAL_FACTOR_COUNT),
    dtype=HISTORY_DTYPE,
    order="F",
)

# Single batched draw — avoids per-column RNG state thrash.
_Z: npt.NDArray[np.floating] = _rng.standard_normal(
    (CANONICAL_HISTORY_BUSINESS_DAYS, CANONICAL_HISTORICAL_FACTOR_COUNT),
    dtype=HISTORY_DTYPE,
)

for _i, (_, _params) in enumerate(_FACTOR_SPECS):
    _col = _historical_values[:, _i]
    _z_col = _Z[:, _i]
    _sigma = _params.annualized_vol

    if _params.is_rate:
        # Arithmetic Brownian motion: level_t = s0 + cumsum(sigma * sqrt(dt) * Z)
        np.cumsum(_sigma * _sqrt_dt * _z_col, out=_col)
        _col += _params.s0
    else:
        # Geometric Brownian motion in log-space, then exponentiate in place.
        _drift = -0.5 * (_sigma * _sigma) * _dt
        _log_returns = _drift + (_sigma * _sqrt_dt * _z_col)
        np.cumsum(_log_returns, out=_col)
        np.exp(_col, out=_col)
        _col *= _params.s0

_logger.debug(
    "Generated %d business-day history for %d canonical factors (dtype=%s)",
    CANONICAL_HISTORY_BUSINESS_DAYS,
    CANONICAL_HISTORICAL_FACTOR_COUNT,
    HISTORY_DTYPE.__name__,
)


# ==============================================================================
# DataFrame view (edge-only — strip before crossing into the engine)
# ==============================================================================

CANONICAL_HISTORICAL_MARKET_DATA_DF: Final[pd.DataFrame] = pd.DataFrame(
    _historical_values,
    index=HISTORICAL_DATES,
    columns=CANONICAL_HISTORICAL_FACTOR_NAMES,
)


# ==============================================================================
# Diagnostics
# ==============================================================================


def _debug_print() -> None:
    from riskquant.utils.printer import (
        print_calling_file,
        print_df,
        print_success_message,
    )

    print_calling_file()

    print("=========================================")
    print("=== HISTORICAL RISK FACTOR MATRIX DF  ===")
    print("=========================================")
    print(
        f"Shape: {CANONICAL_HISTORICAL_MARKET_DATA_DF.shape} "
        f"(T={CANONICAL_HISTORY_BUSINESS_DAYS}, "
        f"N={CANONICAL_HISTORICAL_FACTOR_COUNT}, dtype={HISTORY_DTYPE.__name__})"
    )

    print_df(
        CANONICAL_HISTORICAL_MARKET_DATA_DF.iloc[:, :5].head(5),
        title="Head (First 5 Factors)",
        index=True,
        float_format="{:.4f}",
    )
    print_df(
        CANONICAL_HISTORICAL_MARKET_DATA_DF.iloc[:, :5].tail(5),
        title="Tail (First 5 Factors)",
        index=True,
        float_format="{:.4f}",
    )

    print_success_message(add_space_before=False)


if __name__ == "__main__":
    _debug_print()
