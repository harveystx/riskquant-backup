# src/riskquant/portfolio/ingest/fx_resolver.py
"""
Vectorized resolution of FX-spot risk-factor ids for currency conversion.

Strategy: two gathers into static_config.fx_spot_matrix + a np.where mask.
For every row whose base currency differs from its settlement currency:

    fx_matrix[base_id, quote_id]   -> direct hit
    fx_matrix[quote_id, base_id]   -> inverse hit (record invert=1)

No string concatenation, no name fallback (e.g. "[ON]"), no priority chain.
The static_config validator guarantees `fx_spot_matrix` is the single
source of truth for FX-pair resolution.

The output `fx_invert` array is int8 (project convention for flags), not
int32. The flag carries a per-row "1 / fx_rate vs fx_rate" decision that
the pricer can apply branchlessly via `r * (1 - invert) + (1/r) * invert`.

Pandas migration note
---------------------
The prior ``pd.Series(arr).map(ccy_name_to_id)`` calls are replaced with
numpy dict lookups using a -1 sentinel. Valid ccy_ids are non-negative.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from riskquant.config.static_config.static_config import RiskStaticConfig
from riskquant.constants import NO_FX_RF_REQUIRED
from riskquant.exceptions import PrettyKeyError

_MISS = np.int32(-1)


def _detect_unknown_ccys(
    raw: npt.NDArray[np.str_],
    unknown: npt.NDArray[np.bool_],
    *,
    ctx: str,
) -> None:
    """Raise if any currency string failed to map to a ccy_id."""
    if not unknown.any():
        return
    bad = sorted({str(v) for v in raw[unknown]})[:10]
    raise PrettyKeyError(
        f"{ctx}: {int(unknown.sum())} currency name(s) not in static_config "
        f"ccy_name_to_id.\nUnresolved currencies:\n{bad}"
    )


def resolve_fx_rf_ids_inplace(
    *,
    base_ccy: npt.NDArray[np.str_],
    settlement_ccy: npt.NDArray[np.str_],
    out_fx_rf_ids: npt.NDArray[np.int32],
    out_fx_invert: npt.NDArray[np.int8],
    static_config: RiskStaticConfig,
) -> None:
    """
    Resolve FX-spot risk-factor ids and invert flags in place.

    Parameters
    ----------
    base_ccy, settlement_ccy
        Per-row currency arrays (ISO 3-letter strings, uppercased upstream
        by canonicalize). Rows where they match are tagged NO_FX_RF_REQUIRED.
    out_fx_rf_ids
        int32 output buffer. Receives the FX-pair rf_id or NO_FX_RF_REQUIRED.
    out_fx_invert
        int8 output buffer. 1 where the inverse pair was used, else 0.
    static_config
        Compiled static config. Supplies `ccy_name_to_id` for the per-row
        currency -> id translation and `fx_spot_matrix` for the principled
        (base_id, quote_id) -> rf_id lookup.

    Raises
    ------
    PrettyKeyError
        If any row carries a currency name not in ccy_name_to_id, or if
        neither the direct nor the inverse pair has been configured.
    """
    if base_ccy.shape != settlement_ccy.shape:
        raise ValueError(
            f"base_ccy {base_ccy.shape} and settlement_ccy {settlement_ccy.shape} "
            "must have identical shapes."
        )
    if out_fx_rf_ids.shape != base_ccy.shape:
        raise ValueError(
            f"out_fx_rf_ids {out_fx_rf_ids.shape} must match input {base_ccy.shape}."
        )
    if out_fx_invert.shape != base_ccy.shape:
        raise ValueError(
            f"out_fx_invert {out_fx_invert.shape} must match input {base_ccy.shape}."
        )
    if out_fx_rf_ids.dtype != np.int32:
        raise TypeError(f"out_fx_rf_ids must be int32; got {out_fx_rf_ids.dtype}.")
    if out_fx_invert.dtype != np.int8:
        raise TypeError(f"out_fx_invert must be int8; got {out_fx_invert.dtype}.")

    out_fx_rf_ids.fill(NO_FX_RF_REQUIRED)
    out_fx_invert.fill(np.int8(0))

    # Only rows where base != settle need resolution.
    needs_fx = base_ccy != settlement_ccy
    if not needs_fx.any():
        return

    # ---- Currency NAME -> ID (numpy dict lookup; no pandas) ----
    # Sentinel -1 is safe: valid ccy_ids are non-negative.
    ccy_name_to_id = static_config.ccy_name_to_id

    base_sub = base_ccy[needs_fx]
    settle_sub = settlement_ccy[needs_fx]

    base_ids_raw = np.array(
        [ccy_name_to_id.get(n, _MISS) for n in base_sub], dtype=np.int32
    )
    settle_ids_raw = np.array(
        [ccy_name_to_id.get(n, _MISS) for n in settle_sub], dtype=np.int32
    )

    _detect_unknown_ccys(base_sub, base_ids_raw == _MISS, ctx="base_ccy")
    _detect_unknown_ccys(settle_sub, settle_ids_raw == _MISS, ctx="settlement_ccy")

    base_ids = base_ids_raw
    quote_ids = settle_ids_raw

    # ---- Two-gather lookup into the dense FX matrix ----
    fx_matrix = static_config.fx_spot_matrix
    std_hits = fx_matrix[base_ids, quote_ids]
    inv_hits = fx_matrix[quote_ids, base_ids]

    std_valid = std_hits != int(NO_FX_RF_REQUIRED)
    inv_valid = inv_hits != int(NO_FX_RF_REQUIRED)

    # ---- Unresolved rows are hard errors ----
    missing = ~(std_valid | inv_valid)
    if missing.any():
        i = int(np.flatnonzero(missing)[0])
        raise PrettyKeyError(
            "No FX spot rf configured for currency conversion:\n"
            f"  base_ccy={base_sub[i]!r} (id={int(base_ids[i])})\n"
            f"  quote_ccy={settle_sub[i]!r} (id={int(quote_ids[i])})\n"
            f"  matrix[base, quote] = {int(std_hits[i])}\n"
            f"  matrix[quote, base] = {int(inv_hits[i])}"
        )

    # ---- Scatter resolved ids + invert flags ----
    # Direct hit wins over inverse (canonical orientation). Where direct
    # missed but inverse is present, record invert=1.
    final_ids = np.where(std_valid, std_hits, inv_hits).astype(np.int32, copy=False)
    final_inv = (~std_valid).astype(np.int8, copy=False)

    out_fx_rf_ids[needs_fx] = final_ids
    out_fx_invert[needs_fx] = final_inv
