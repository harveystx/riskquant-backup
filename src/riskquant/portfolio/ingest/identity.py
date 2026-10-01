# src/riskquant/portfolio/ingest/identity.py
"""
Semantic identity gate for market linkages at the ingest edge.

Global ids are untyped integers. A curve container, a volatility node and a
price node are all admissible `int32` values, and every downstream layer
(universe, producer, fold, kernel) indexes them without knowing what they are.
This module is therefore the LAST layer that holds the semantic kind, the
currency and the contract period of each linkage, and it is the owner of the
following contracts:

    D063  source kind       the SCALAR QUOTE SEMANTICS each linkage admits.
                            SPOT underlying        -> PRICE/FX_RATE/INDEX rf
                            NODE forward/option    -> a point of a PRICE curve
                            NODE option volatility -> VOL rf
                            An exact CURVE POINT rf is read exactly like a NODE
                            rf: one universe row, no interpolation. The topology
                            condition here is therefore NOT about interpolation;
                            it is the current configuration model's only proxy
                            for "this scalar level is a dated forward quote",
                            which is what the forward valuation model consumes.
                            Whether a spot-quoted node may also serve as a
                            forward/option underlying is an open economic
                            question and is NOT settled here.
    D065  contract identity FORWARDS ONLY. The parent curve's x-axis kind
                            decides the identity class (D101):
                            RELATIVE  OFFSET_* axis: the name resolves at the
                                      run as-of through a roll method.
                                      delivery_date is required and must fall
                                      in the resolved contract period. Only
                                      CALENDAR_PERIOD contracts are supported;
                                      every other roll method fails closed.
                            EXACT     LABEL axis: the name does not depend on
                                      the as-of, so the rf IS the contract.
                                      delivery_date is optional and selects
                                      nothing; it only sets the asset-leg
                                      discount date (settlement when absent).
                            Options never require delivery (D077).
    D066  discount linkage  a linked curve is a discount curve AND its currency
                            equals the currency of the leg it discounts

Every check raises before any id reaches `build_uix_mapper`. Nothing here is
a hot path: loops run over DISTINCT ids, never over trades.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import numpy.typing as npt

from riskquant.config.enums.container_topology import ContainerTopologyType
from riskquant.config.enums.x_axis_kinds import XAxisKind
from riskquant.config.static_config.rf_resolution import (
    build_risk_factor_name_mapping_for_as_of_date,
)
from riskquant.config.static_config.static_config import RiskStaticConfig
from riskquant.constants import NO_DATE_ORD32, NO_ID
from riskquant.exceptions import PrettyKeyError
from riskquant.portfolio.instruments import (
    SPOT_ID,
    VANILLA_EUROPEAN_OPTION_NODE_ID,
    VANILLA_FORWARD_NODE_ID,
)

_SPOT_QUOTE_TYPES: frozenset[str] = frozenset({"PRICE", "FX_RATE", "INDEX"})
_FORWARD_QUOTE_TYPES: frozenset[str] = frozenset({"PRICE"})
_VOL_QUOTE_TYPES: frozenset[str] = frozenset({"VOL"})

# (x_axis_kind, roll_method) -> width of the resolved period label.
# CALENDAR_PERIOD labels are the contract period itself (YYYYMM / YYYY), so the
# delivery date identifies the contract without any further market convention.
# Roll methods whose label is a boundary date or a roll-adjusted year carry a
# period definition this module does not own; they fail closed (D065).
_SUPPORTED_PERIOD_LABELS: dict[tuple[str, str], int] = {
    ("OFFSET_M", "CALENDAR_PERIOD"): 6,
    ("OFFSET_Y", "CALENDAR_PERIOD"): 4,
}


def _uids_preview(
    uids: npt.NDArray[np.int64], mask: npt.NDArray[np.bool_]
) -> list[int]:
    return [int(u) for u in uids[mask][:10]]


def _rf_quote_type_and_parent(
    rf_id: int, static_config: RiskStaticConfig
) -> tuple[str, int, int] | None:
    """Return `(quote_type, parent_topology, parent_id)` or None if not an rf."""
    if rf_id < 0 or rf_id >= static_config.rf_id_to_parent_topology_type.shape[0]:
        return None
    topology = int(static_config.rf_id_to_parent_topology_type[rf_id])
    parent = int(static_config.rf_id_to_parent_id[rf_id])
    if topology == ContainerTopologyType.NODE:
        node = static_config.node_meta_dict.get(rf_id)
        return None if node is None else (node.quote_type, topology, parent)
    if topology == ContainerTopologyType.CURVE:
        curve = static_config.curve_meta_dict.get(parent)
        return None if curve is None else (curve.quote_type, topology, parent)
    if topology == ContainerTopologyType.SURFACE:
        surface = static_config.surface_meta_dict.get(parent)
        return None if surface is None else (surface.quote_type, topology, parent)
    return None


def _require_kind(
    *,
    ids: npt.NDArray[np.int32],
    rows: npt.NDArray[np.bool_],
    allowed_quote_types: frozenset[str],
    allowed_topologies: frozenset[int],
    uids: npt.NDArray[np.int64],
    ctx: str,
    static_config: RiskStaticConfig,
) -> None:
    if not bool(rows.any()):
        return
    bad_ids: list[int] = []
    for rf_id in np.unique(ids[rows]).tolist():
        info = _rf_quote_type_and_parent(int(rf_id), static_config)
        if (
            info is None
            or info[0] not in allowed_quote_types
            or info[1] not in allowed_topologies
        ):
            bad_ids.append(int(rf_id))
    if bad_ids:
        mask = rows & np.isin(ids, np.asarray(bad_ids, dtype=np.int32))
        raise PrettyKeyError(
            f"{ctx}: linkage has the wrong source kind (D063). Allowed quote "
            f"types {sorted(allowed_quote_types)} on topologies "
            f"{sorted(ContainerTopologyType(t).name for t in allowed_topologies)}. "
            f"Offending ids {bad_ids[:10]}; trade uids {_uids_preview(uids, mask)}."
        )


def _validate_discount_linkage(
    *,
    curve_ids: npt.NDArray[np.int32],
    leg_ccy: npt.NDArray[np.str_],
    rows: npt.NDArray[np.bool_],
    uids: npt.NDArray[np.int64],
    ctx: str,
    static_config: RiskStaticConfig,
) -> None:
    linked = rows & (curve_ids != NO_ID)
    if not bool(linked.any()):
        return
    discount_ids = frozenset(
        int(v) for v in static_config.discount_curve_lookup.values()
    )
    ccy_name_to_id = static_config.ccy_name_to_id
    for curve_id in np.unique(curve_ids[linked]).tolist():
        at = linked & (curve_ids == curve_id)
        if int(curve_id) not in discount_ids:
            raise PrettyKeyError(
                f"{ctx}: curve id {curve_id} is not a discount curve "
                f"(NOT_A_DISCOUNT_CURVE or unknown) (D066). Trade uids "
                f"{_uids_preview(uids, at)}."
            )
        curve_ccy = int(
            static_config.container_ccy_id[static_config.container_slot(int(curve_id))]
        )
        leg_ccy_ids = np.fromiter(
            (int(ccy_name_to_id.get(str(c), -1)) for c in leg_ccy[at]),
            dtype=np.int64,
            count=int(at.sum()),
        )
        wrong = leg_ccy_ids != curve_ccy
        if bool(wrong.any()):
            curve_ccy_name = static_config.ccy_id_to_name[curve_ccy]
            raise PrettyKeyError(
                f"{ctx}: discount curve {curve_id} is quoted in {curve_ccy_name} but "
                f"discounts a leg in {sorted(set(leg_ccy[at][wrong].tolist()))} "
                f"(D066: curve currency must equal leg cashflow currency). Trade uids "
                f"{[int(u) for u in uids[at][wrong][:10]]}."
            )


def _validate_contract_identity(
    *,
    underlying_ids: npt.NDArray[np.int32],
    delivery_ord: npt.NDArray[np.int32],
    rows: npt.NDArray[np.bool_],
    uids: npt.NDArray[np.int64],
    as_of_date: date,
    static_config: RiskStaticConfig,
) -> None:
    """
    Enforce D065 contract identity for NODE-forward rows (D101 classification).

    The identity class of a forward point is a property of its parent curve's
    x-axis kind, not of its roll method. `CurveMeta` and `rf_resolution` define
    a LABEL point as as-of independent (the bare label is the absolute name)
    and an OFFSET_* point as resolved at the run as-of through a roll method.

    EXACT (LABEL): no delivery check and no period check. Delivery never
    selects the rf; the compiler discounts the asset leg at settlement when
    delivery is absent.

    RELATIVE (OFFSET_*): the (x_axis_kind, roll_method) pair must have a
    period definition owned here (CALENDAR_PERIOD only), or the row fails
    closed. Delivery is then required and must fall in the resolved period.
    """
    if not bool(rows.any()):
        return

    relative = np.zeros_like(rows)
    for rf_id in np.unique(underlying_ids[rows]).tolist():
        parent = int(static_config.rf_id_to_parent_id[int(rf_id)])
        curve = static_config.curve_meta_dict[parent]
        if curve.x_axis_kind == XAxisKind.LABEL.name:
            continue
        at = rows & (underlying_ids == rf_id)
        if (curve.x_axis_kind, str(curve.roll_method)) not in _SUPPORTED_PERIOD_LABELS:
            raise PrettyKeyError(
                f"forward underlying rf {rf_id} on curve {curve.curve_name!r} uses "
                f"(x_axis_kind={curve.x_axis_kind}, roll_method={curve.roll_method}); "
                "delivery-period identity is only defined for CALENDAR_PERIOD "
                f"contracts (D065). Trade uids {_uids_preview(uids, at)}."
            )
        relative |= at & (delivery_ord != NO_DATE_ORD32)

    if not bool(relative.any()):
        return
    missing = relative & (delivery_ord == NO_DATE_ORD32)
    if bool(missing.any()):
        raise PrettyKeyError(
            "forward contract identity requires delivery_date for relative NODE "
            f"contracts (D065). Trade uids {_uids_preview(uids, missing)}."
        )
    mapping = build_risk_factor_name_mapping_for_as_of_date(
        risk_static_config=static_config, as_of_date=as_of_date
    )
    rf_to_abs = {int(v): k for k, v in mapping.abs_to_rf_id.items()}
    months = delivery_ord.astype("datetime64[D]").astype("datetime64[M]")
    for rf_id in np.unique(underlying_ids[relative]).tolist():
        at = relative & (underlying_ids == rf_id)
        parent = int(static_config.rf_id_to_parent_id[int(rf_id)])
        curve = static_config.curve_meta_dict[parent]
        width = _SUPPORTED_PERIOD_LABELS[(curve.x_axis_kind, str(curve.roll_method))]
        abs_name = rf_to_abs.get(int(rf_id))
        prefix = f"{curve.curve_name}_"
        if abs_name is None or not abs_name.startswith(prefix):
            raise PrettyKeyError(
                f"forward underlying rf {rf_id} did not resolve to an absolute "
                f"contract of curve {curve.curve_name!r} at as-of {as_of_date}."
            )
        label = abs_name[len(prefix) :]
        if width == 6:
            trade_labels = np.char.replace(months[at].astype(str), "-", "")
        else:
            trade_labels = months[at].astype("datetime64[Y]").astype(str)
        wrong = trade_labels != label
        if bool(wrong.any()):
            raise PrettyKeyError(
                f"forward underlying {abs_name!r} (as-of {as_of_date}) is the "
                f"{label} contract, but delivery dates fall in "
                f"{sorted(set(trade_labels[wrong].tolist()))} (D065). Trade uids "
                f"{[int(u) for u in uids[at][wrong][:10]]}."
            )


def validate_market_linkage_identity(
    *,
    instrument_ids: npt.NDArray[np.int32],
    uids: npt.NDArray[np.int64],
    underlying_ids: npt.NDArray[np.int32],
    volatility_ids: npt.NDArray[np.int32],
    under_ccy: npt.NDArray[np.str_],
    strike_ccy: npt.NDArray[np.str_],
    under_discount_curve_ids: npt.NDArray[np.int32],
    strike_discount_curve_ids: npt.NDArray[np.int32],
    delivery_ord: npt.NDArray[np.int32],
    as_of_date: date | None,
    static_config: RiskStaticConfig,
) -> None:
    """Enforce D063, D065 and D066 for every supported instrument row."""
    spot = instrument_ids == np.int32(SPOT_ID)
    fwd = instrument_ids == np.int32(VANILLA_FORWARD_NODE_ID)
    opt = instrument_ids == np.int32(VANILLA_EUROPEAN_OPTION_NODE_ID)
    dated = fwd | opt

    _require_kind(
        ids=underlying_ids,
        rows=spot,
        allowed_quote_types=_SPOT_QUOTE_TYPES,
        allowed_topologies=frozenset({
            int(ContainerTopologyType.NODE),
            int(ContainerTopologyType.CURVE),
        }),
        uids=uids,
        ctx="SPOT underlying",
        static_config=static_config,
    )
    _require_kind(
        ids=underlying_ids,
        rows=dated,
        allowed_quote_types=_FORWARD_QUOTE_TYPES,
        allowed_topologies=frozenset({int(ContainerTopologyType.CURVE)}),
        uids=uids,
        ctx="NODE forward/option underlying",
        static_config=static_config,
    )
    _require_kind(
        ids=volatility_ids,
        rows=opt,
        allowed_quote_types=_VOL_QUOTE_TYPES,
        allowed_topologies=frozenset({int(ContainerTopologyType.NODE)}),
        uids=uids,
        ctx="NODE option volatility",
        static_config=static_config,
    )
    _validate_discount_linkage(
        curve_ids=under_discount_curve_ids,
        leg_ccy=under_ccy,
        rows=dated,
        uids=uids,
        ctx="under_discount_curve",
        static_config=static_config,
    )
    _validate_discount_linkage(
        curve_ids=strike_discount_curve_ids,
        leg_ccy=strike_ccy,
        rows=dated,
        uids=uids,
        ctx="strike_discount_curve",
        static_config=static_config,
    )
    if bool(dated.any()):
        if as_of_date is None:
            raise ValueError(
                "forward/option rows require the run as-of date at ingest: relative "
                "forward names resolve to dated contracts only at an as-of (D065)."
            )
        # Delivery-period identity is a FORWARD contract check. An option's
        # value never reads delivery, so requiring one would be a requirement
        # inherited from the sibling instrument, not from the option contract.
        _validate_contract_identity(
            underlying_ids=underlying_ids,
            delivery_ord=delivery_ord,
            rows=fwd,
            uids=uids,
            as_of_date=as_of_date,
            static_config=static_config,
        )
