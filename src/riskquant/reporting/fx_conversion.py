# src/riskquant/reporting/fx_conversion.py
"""
Reporting-currency conversion and aggregation for scenario P&L.

The sensitivity/P&L engines are deliberately currency-agnostic: a trade's
scenario P&L is denominated in that trade's OWN settlement currency and
nothing downstream of `solutions.sensitivity_pnl` knows about currency. That
stays true. This module is the explicit layer placed AROUND scenario P&L
aggregation, not a currency concept threaded through the engines.

Pipeline position
-----------------
    settlement-currency sensitivities
        -> scenario shocks
        -> scenario settlement-currency P&L        (currency-agnostic engine)
        -> scenario FX state
        -> settlement -> reporting FX conversion   (THIS MODULE)
        -> reporting-currency scenario P&L
        -> portfolio aggregation                   (THIS MODULE)
        -> risk measure

Reporting P&L equation (ordinary mark-to-market; D081, superseding D014)
------------------------------------------------------------------------
Reporting-currency value is ``W = V * q`` with ``V`` the settlement-currency
value and ``q`` the settlement -> reporting rate. Reporting P&L is the exact
identity

    dW[i, s] = V_s * q_s - V_0 * q_0
             = dV[i, s] * q_s  +  V_0 * (q_s - q_0)

where ``dV`` is the settlement-currency scenario P&L, ``q_s`` the scenario
rate, ``q_0`` the base rate and ``V_0`` the position's base settlement value.
The second term translates the BASE value. Omitting it (the former D014
convention ``dV * q_s``) is not reporting-currency P&L: for a USD-priced,
EUR-settled position under USD reporting it manufactures an FX P&L on an asset
that has none and flips the sign of the true P&L once FX moves (see
`tests/reporting/test_fx_conversion.py::test_ordinary_mtm_brent_internal_fx_cancels`).
``V_0`` and ``q_0`` are REQUIRED inputs, so the D014 form is unreachable by
construction rather than by discipline.

Quotation convention
--------------------
The FX pair name comes from the FX authority, `portfolio.fx`:

    get_fx_pair_for_under_to_settle_conversion(underlying_ccy=X,
                                               settlement_ccy=Y)  ->  "XY"

so a factor named ``"USDEUR"`` multiplies USD -> EUR. Converting settlement
ccy S to reporting ccy R therefore looks for the pair ``"SR"``; when only the
opposite pair ``"RS"`` exists the reciprocal is used, recorded as
``invert=1`` (the same direct/inverse representation
`portfolio.ingest.fx_resolver` uses). The convention is never hard-coded
here; it is always derived from that authority.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from riskquant.constants import INVALID_UIX
from riskquant.portfolio.fx import (
    get_fx_pair_for_under_to_settle_conversion,
    validate_ccy,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True, slots=True, kw_only=True)
class FxConversionPlan:
    """
    Compiled, per-position settlement -> reporting FX routing.

    Attributes
    ----------
    fx_row_index
        ``(M,) int32``. Row of the scenario FX level matrix supplying each
        position's rate. ``INVALID_UIX`` marks an identity conversion
        (settlement currency already equals the reporting currency); no FX
        row is read for those positions.
    invert
        ``(M,) int8``. ``0`` uses the rate directly, ``1`` uses its
        reciprocal because only the opposite pair is quoted.
    settlement_ccys
        Per-position settlement currency, retained so the currency dimension
        stays visible and auditable rather than implicit in the indices.
    reporting_ccy
        The single reporting currency every position is converted into.
    """

    fx_row_index: npt.NDArray[np.int32]
    invert: npt.NDArray[np.int8]
    settlement_ccys: tuple[str, ...]
    reporting_ccy: str

    def __post_init__(self) -> None:
        num_positions = self.fx_row_index.shape[0]
        _require(self.fx_row_index.ndim == 1, "fx_row_index must be 1-D.")
        _require(self.invert.ndim == 1, "invert must be 1-D.")
        _require(
            self.invert.shape[0] == num_positions,
            "invert must align with fx_row_index.",
        )
        _require(
            len(self.settlement_ccys) == num_positions,
            "settlement_ccys must align with fx_row_index.",
        )
        _require(self.fx_row_index.dtype == np.int32, "fx_row_index must be int32.")
        _require(self.invert.dtype == np.int8, "invert must be int8.")
        _require(
            bool(np.all((self.invert == 0) | (self.invert == 1))),
            "invert entries must be 0 or 1.",
        )

    @property
    def num_positions(self) -> int:
        return int(self.fx_row_index.shape[0])

    @property
    def is_identity(self) -> npt.NDArray[np.bool_]:
        """Positions already denominated in the reporting currency."""
        return self.fx_row_index == INVALID_UIX


@dataclass(frozen=True, slots=True, kw_only=True)
class ReportingCurrencyPnl:
    """
    Scenario P&L expressed in one reporting currency.

    Attributes
    ----------
    converted_pnl
        ``(M, S) float64`` per-position scenario P&L in `reporting_ccy`.
    portfolio_pnl
        ``(S,) float64`` scenario-wise portfolio total. Summed ONLY after
        every position has been converted, so settlement currencies are
        never mixed.
    reporting_ccy
        The currency both arrays are denominated in.
    """

    converted_pnl: npt.NDArray[np.float64]
    portfolio_pnl: npt.NDArray[np.float64]
    reporting_ccy: str

    @property
    def num_positions(self) -> int:
        return int(self.converted_pnl.shape[0])

    @property
    def num_scenarios(self) -> int:
        return int(self.converted_pnl.shape[1])


def build_fx_conversion_plan(
    *,
    settlement_ccys: Sequence[str],
    reporting_ccy: str,
    fx_factor_names: Sequence[str],
) -> FxConversionPlan:
    """
    Resolve each position's settlement -> reporting FX route. Fails fast.

    Parameters
    ----------
    settlement_ccys
        Per-position settlement currency code.
    reporting_ccy
        Target reporting currency code.
    fx_factor_names
        Names of the rows of the scenario FX level matrix, in row order
        (e.g. ``("USDEUR",)``).

    Raises
    ------
    TypeError, ValueError
        On an invalid or missing currency code, or when neither the direct
        nor the inverse pair required by a cross-currency position is quoted.
        No rate is ever defaulted, filled, or assumed to be 1.0.
    """
    target = validate_ccy(reporting_ccy)

    _require(len(settlement_ccys) > 0, "settlement_ccys must not be empty.")

    name_to_row: dict[str, int] = {}
    for row, raw_name in enumerate(fx_factor_names):
        if not isinstance(raw_name, str) or not raw_name:
            raise ValueError(
                f"fx_factor_names[{row}] must be a non-empty string; got {raw_name!r}."
            )
        if raw_name in name_to_row:
            raise ValueError(
                f"Duplicate FX factor name {raw_name!r} at rows "
                f"{name_to_row[raw_name]} and {row}."
            )
        name_to_row[raw_name] = row

    num_positions = len(settlement_ccys)
    fx_row_index = np.empty(num_positions, dtype=np.int32)
    invert = np.zeros(num_positions, dtype=np.int8)
    resolved_ccys: list[str] = []

    for position, raw_ccy in enumerate(settlement_ccys):
        settlement = validate_ccy(raw_ccy)
        resolved_ccys.append(settlement)

        # "" means the two currencies match: an explicit identity conversion.
        direct_pair = get_fx_pair_for_under_to_settle_conversion(
            underlying_ccy=settlement, settlement_ccy=target
        )
        if direct_pair == "":
            fx_row_index[position] = INVALID_UIX
            continue

        if direct_pair in name_to_row:
            fx_row_index[position] = np.int32(name_to_row[direct_pair])
            continue

        inverse_pair = get_fx_pair_for_under_to_settle_conversion(
            underlying_ccy=target, settlement_ccy=settlement
        )
        if inverse_pair in name_to_row:
            fx_row_index[position] = np.int32(name_to_row[inverse_pair])
            invert[position] = np.int8(1)
            continue

        raise ValueError(
            f"No FX factor available to convert position {position} from "
            f"{settlement} to {target}: neither {direct_pair!r} nor "
            f"{inverse_pair!r} is quoted. Available: "
            f"{sorted(name_to_row)}."
        )

    return FxConversionPlan(
        fx_row_index=fx_row_index,
        invert=invert,
        settlement_ccys=tuple(resolved_ccys),
        reporting_ccy=target,
    )


def convert_scenario_pnl_to_reporting_currency(
    *,
    settlement_pnl: npt.NDArray[np.float64],
    fx_levels: npt.NDArray[np.float64],
    plan: FxConversionPlan,
    base_settlement_values: npt.NDArray[np.float64],
    base_fx_levels: npt.NDArray[np.float64],
) -> ReportingCurrencyPnl:
    """
    Convert settlement-currency scenario P&L to reporting currency (ordinary
    mark-to-market, D081) and aggregate the portfolio.

    ``dW[i, s] = dV[i, s] * rate_s[i, s] + V0[i] * (rate_s[i, s] - rate_0[i])``
    with ``rate_s = fx_levels[row, s]`` (or its reciprocal when ``invert`` is set,
    or exactly 1.0 for an identity conversion).

    Parameters
    ----------
    settlement_pnl
        ``(M, S) float64`` C-contiguous, finite. Each row is denominated in
        that position's own settlement currency.
    base_settlement_values
        ``(M,) float64`` finite: each position's BASE value in its own
        settlement currency (``V_0``). Required; no default exists and the
        former ``dV * q_s`` convention cannot be requested.
    base_fx_levels
        ``(K,) float64``: BASE-state FX levels row-aligned with `fx_levels`
        (``q_0``); finite and positive for the rates in use.
    fx_levels
        ``(K, S) float64`` C-contiguous, finite, strictly positive. Scenario
        FX levels, row-aligned with the `fx_factor_names` used to build the
        plan. May be an empty ``(0, S)`` array when every conversion is an
        identity.
    plan
        Compiled routing from `build_fx_conversion_plan`.

    Raises
    ------
    TypeError, ValueError
        On dtype, layout, dimension, scenario-count, or finiteness
        violations, or on a non-positive FX rate. Nothing is broadcast,
        transposed, or filled to make the inputs fit.
    """
    if settlement_pnl.dtype != np.dtype(np.float64):
        raise TypeError(f"settlement_pnl must be float64; got {settlement_pnl.dtype}.")
    if fx_levels.dtype != np.dtype(np.float64):
        raise TypeError(f"fx_levels must be float64; got {fx_levels.dtype}.")

    _require(settlement_pnl.ndim == 2, "settlement_pnl must be 2-D (M, S).")
    _require(fx_levels.ndim == 2, "fx_levels must be 2-D (K, S).")
    if base_settlement_values.dtype != np.dtype(np.float64):
        raise TypeError(
            "base_settlement_values must be float64; got "
            f"{base_settlement_values.dtype}."
        )
    if base_fx_levels.dtype != np.dtype(np.float64):
        raise TypeError(f"base_fx_levels must be float64; got {base_fx_levels.dtype}.")
    _require(base_settlement_values.ndim == 1, "base_settlement_values must be (M,).")
    _require(base_fx_levels.ndim == 1, "base_fx_levels must be (K,).")
    _require(
        base_settlement_values.shape[0] == settlement_pnl.shape[0],
        f"base_settlement_values has {base_settlement_values.shape[0]} value(s) "
        f"but settlement_pnl has {settlement_pnl.shape[0]} position(s).",
    )
    _require(
        base_fx_levels.shape[0] == fx_levels.shape[0],
        f"base_fx_levels has {base_fx_levels.shape[0]} row(s) but fx_levels has "
        f"{fx_levels.shape[0]}.",
    )
    _require(
        bool(np.isfinite(base_settlement_values).all()),
        "base_settlement_values must contain only finite values.",
    )
    _require(
        settlement_pnl.flags["C_CONTIGUOUS"], "settlement_pnl must be C-contiguous."
    )
    _require(fx_levels.flags["C_CONTIGUOUS"], "fx_levels must be C-contiguous.")

    num_positions, num_scenarios = settlement_pnl.shape
    _require(num_positions > 0, "settlement_pnl must contain at least one position.")
    _require(num_scenarios > 0, "settlement_pnl must contain at least one scenario.")
    _require(
        num_positions == plan.num_positions,
        f"settlement_pnl has {num_positions} position(s) but the plan covers "
        f"{plan.num_positions}.",
    )
    _require(
        fx_levels.shape[1] == num_scenarios,
        f"fx_levels has {fx_levels.shape[1]} scenario(s) but settlement_pnl "
        f"has {num_scenarios}.",
    )
    _require(
        bool(np.isfinite(settlement_pnl).all()),
        "settlement_pnl must contain only finite values.",
    )

    needs_fx = ~plan.is_identity
    if np.any(needs_fx):
        max_row = int(plan.fx_row_index[needs_fx].max())
        _require(
            max_row < fx_levels.shape[0],
            f"FX plan references row {max_row} but fx_levels has only "
            f"{fx_levels.shape[0]} row(s).",
        )
        used_rows = np.unique(plan.fx_row_index[needs_fx]).astype(np.intp)
        used = fx_levels[used_rows]
        _require(
            bool(np.isfinite(used).all()),
            "fx_levels must contain only finite values for the rates in use.",
        )
        _require(
            bool((used > 0.0).all()),
            "fx_levels must be strictly positive for the rates in use; a "
            "non-positive FX rate is not a usable quote.",
        )

    # Per-position rate matrix. Identity rows are exactly 1.0 -- an explicit
    # conversion, not a fallback for a missing rate.
    rates = np.ones((num_positions, num_scenarios), dtype=np.float64)
    base_rates = np.ones(num_positions, dtype=np.float64)
    if np.any(needs_fx):
        rows = plan.fx_row_index[needs_fx].astype(np.intp)
        used_base = base_fx_levels[rows]
        _require(
            bool(np.isfinite(used_base).all()) and bool((used_base > 0.0).all()),
            "base_fx_levels must be finite and strictly positive for the rates in use.",
        )
        gathered = fx_levels[rows]
        inverted = plan.invert[needs_fx].astype(bool)
        gathered = np.where(inverted[:, None], 1.0 / gathered, gathered)
        rates[needs_fx] = gathered
        base_rates[needs_fx] = np.where(inverted, 1.0 / used_base, used_base)
    # Ordinary MTM (D081): dW = dV*q_s + V0*(q_s - q0). The second term is the
    # translation of the base value; it vanishes only when q_s == q0 or V0 == 0.
    translation = base_settlement_values[:, None] * (rates - base_rates[:, None])
    converted = np.ascontiguousarray(
        settlement_pnl * rates + translation, dtype=np.float64
    )

    # Aggregate ONLY after every row is in the reporting currency.
    portfolio = np.ascontiguousarray(converted.sum(axis=0), dtype=np.float64)

    return ReportingCurrencyPnl(
        converted_pnl=converted,
        portfolio_pnl=portfolio,
        reporting_ccy=plan.reporting_ccy,
    )
