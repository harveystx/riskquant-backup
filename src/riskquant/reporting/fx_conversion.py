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

Conversion convention (deliberate, and load-bearing)
----------------------------------------------------
Reporting P&L is obtained by converting the SETTLEMENT-CURRENCY scenario P&L
at the SCENARIO FX rate:

    dV_report[i, s] = dV_settlement[i, s] * F[i, s]

where ``F[i, s]`` is the settlement -> reporting rate in scenario ``s``.

It is NOT computed as ``V_shocked*F_shocked - V_base*F_base``. For a trade
whose payoff already contains the FX factor (a USD-priced, EUR-settled
position, say), that alternative cancels the FX term algebraically and
silently DELETES the FX exposure the sensitivity book explicitly carries.
The two conventions disagree in sign once FX moves, so this is a correctness
choice, not a formatting one. See
`tests/reporting/test_fx_conversion.py::test_moving_fx_distinguishes_the_two_conventions`.

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
) -> ReportingCurrencyPnl:
    """
    Convert settlement-currency scenario P&L and aggregate the portfolio.

    ``dV_report[i, s] = dV_settlement[i, s] * rate[i, s]`` with
    ``rate = fx_levels[row, s]`` (or its reciprocal when ``invert`` is set,
    or exactly 1.0 for an identity conversion).

    Parameters
    ----------
    settlement_pnl
        ``(M, S) float64`` C-contiguous, finite. Each row is denominated in
        that position's own settlement currency.
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
    if np.any(needs_fx):
        rows = plan.fx_row_index[needs_fx].astype(np.intp)
        gathered = fx_levels[rows]
        inverted = plan.invert[needs_fx].astype(bool)
        gathered = np.where(inverted[:, None], 1.0 / gathered, gathered)
        rates[needs_fx] = gathered

    converted = np.ascontiguousarray(settlement_pnl * rates, dtype=np.float64)

    # Aggregate ONLY after every row is in the reporting currency.
    portfolio = np.ascontiguousarray(converted.sum(axis=0), dtype=np.float64)

    return ReportingCurrencyPnl(
        converted_pnl=converted,
        portfolio_pnl=portfolio,
        reporting_ccy=plan.reporting_ccy,
    )
