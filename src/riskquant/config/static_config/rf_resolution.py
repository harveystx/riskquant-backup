# src/riskquant/config/static_config/rf_resolution.py
"""
Date-sensitive resolution of relative risk-factor names to absolute names.

Examples for as-of=2026-05-23:
    BRENT[M+0]          ->   BRENT_202607        (NEXT_MONTH, CALENDAR_PERIOD)
    HYDRO[Y+0]          ->   HYDRO_2026          (ANNUAL_ROLL_APRIL_BGN, ROLL_ON_MMDD)
    ERCOT[HY+0]         ->   ERCOT_2026_BH       (HALF_YEAR_LABEL, HALF_YEAR_BY_LABEL)

The public entry point is `build_risk_factor_name_mapping_for_as_of_date`
which returns a frozen, bidirectionally indexed `RiskFactorAbsToRelNameIdMap`.

Dispatch is a registry keyed by (XAxisKind, RollMethod) — never a
string-equality ladder. The validator guarantees only allowed pairs
ever reach the resolver.

Nodes do not resolve: their absolute name equals their declared name
(node_name). Surface point names follow the same model as curve points
with an additional bracketed y label.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import overload

from riskquant.config.enums.roll_methods import RollMethod
from riskquant.config.enums.x_axis_kinds import XAxisKind
from riskquant.config.static_config.schema_objects import (
    SpotRollRule,
)
from riskquant.config.static_config.static_config import RiskStaticConfig
from riskquant.utils.tenors import (
    add_months_to_date_and_return_yyyymm01,
    fmt_date_as_str_yyyymm,
    get_spot_year,
    parse_bracket_token,
)

# ============================================================
# Resolver type
# ============================================================

ResolverFn = Callable[
    ["_ResolverContext"],
    str,  # absolute date suffix, e.g. "202607" / "2026" / "2026_FH" / "20260630"
]


@dataclass(frozen=True, slots=True)
class _ResolverContext:
    """Per-token resolution context. Holds everything the dispatch fn needs."""

    container_name: str
    relative_name: str
    rule: SpotRollRule
    token_offset: int
    as_of_date: date


# ============================================================
# Per-(XAxisKind, RollMethod) resolvers
# ============================================================


def _resolve_offset_m_calendar_period(ctx: _ResolverContext) -> str:
    """OFFSET_M + CALENDAR_PERIOD  ->  '<curve>_YYYYMM'."""
    spot = add_months_to_date_and_return_yyyymm01(
        d=date(ctx.as_of_date.year, ctx.as_of_date.month, 1),
        months_to_add=ctx.rule.offset_in_tenor_units,
    )
    target = add_months_to_date_and_return_yyyymm01(
        d=spot, months_to_add=ctx.token_offset
    )
    return fmt_date_as_str_yyyymm(target)


def _resolve_offset_m_half_year_by_date(ctx: _ResolverContext) -> str:
    """OFFSET_M + HALF_YEAR_BY_DATE  ->  '<curve>_YYYYMMDD' on the next FH/BH boundary."""
    fh_bh_mmdd = _require_fh_bh_mmdd(ctx)
    abs_dt = _half_year_boundary_by_date(
        as_of_date=ctx.as_of_date,
        fh_bh_mmdd=fh_bh_mmdd,
        token_offset=ctx.token_offset,
        relative_name=ctx.relative_name,
        container_name=ctx.container_name,
    )
    return f"{abs_dt.year:04d}{abs_dt.month:02d}{abs_dt.day:02d}"


def _resolve_offset_y_calendar_period(ctx: _ResolverContext) -> str:
    """OFFSET_Y + CALENDAR_PERIOD  ->  '<curve>_YYYY'."""
    target_year = (
        ctx.as_of_date.year + ctx.rule.offset_in_tenor_units + ctx.token_offset
    )
    return f"{target_year:04d}"


def _resolve_offset_y_roll_on_mmdd(ctx: _ResolverContext) -> str:
    """OFFSET_Y + ROLL_ON_MMDD  ->  '<curve>_YYYY' with year picked by roll boundary."""
    spot_year = get_spot_year(
        as_of_date=ctx.as_of_date,
        roll_month=int(ctx.rule.roll_month or 0),
        roll_day=int(ctx.rule.roll_day or 0),
        post_roll_shift=ctx.rule.post_roll_shift,
    )
    return f"{spot_year + ctx.token_offset:04d}"


def _resolve_offset_hy_half_year_by_date(ctx: _ResolverContext) -> str:
    """OFFSET_HY + HALF_YEAR_BY_DATE  ->  '<curve>_YYYYMMDD'."""
    fh_bh_mmdd = _require_fh_bh_mmdd(ctx)
    abs_dt = _half_year_boundary_by_date(
        as_of_date=ctx.as_of_date,
        fh_bh_mmdd=fh_bh_mmdd,
        token_offset=ctx.token_offset,
        relative_name=ctx.relative_name,
        container_name=ctx.container_name,
    )
    return f"{abs_dt.year:04d}{abs_dt.month:02d}{abs_dt.day:02d}"


def _resolve_offset_hy_half_year_by_label(ctx: _ResolverContext) -> str:
    """OFFSET_HY + HALF_YEAR_BY_LABEL  ->  '<curve>_YYYY_FH' / '<curve>_YYYY_BH'."""
    fh_bh_mmdd = _require_fh_bh_mmdd(ctx)
    return _half_year_label_suffix(
        as_of_date=ctx.as_of_date,
        fh_bh_mmdd=fh_bh_mmdd,
        token_offset=ctx.token_offset,
    )


# Dense registry: (XAxisKind, RollMethod) -> resolver. Branch-free dispatch.
_OFFSET_RESOLVERS: dict[tuple[XAxisKind, RollMethod], ResolverFn] = {
    (XAxisKind.OFFSET_M, RollMethod.CALENDAR_PERIOD): _resolve_offset_m_calendar_period,
    (
        XAxisKind.OFFSET_M,
        RollMethod.HALF_YEAR_BY_DATE,
    ): _resolve_offset_m_half_year_by_date,
    (XAxisKind.OFFSET_Y, RollMethod.CALENDAR_PERIOD): _resolve_offset_y_calendar_period,
    (XAxisKind.OFFSET_Y, RollMethod.ROLL_ON_MMDD): _resolve_offset_y_roll_on_mmdd,
    (
        XAxisKind.OFFSET_HY,
        RollMethod.HALF_YEAR_BY_DATE,
    ): _resolve_offset_hy_half_year_by_date,
    (
        XAxisKind.OFFSET_HY,
        RollMethod.HALF_YEAR_BY_LABEL,
    ): _resolve_offset_hy_half_year_by_label,
}


# ============================================================
# Half-year helpers
# ============================================================


def _require_fh_bh_mmdd(ctx: _ResolverContext) -> str:
    """Half-year roll methods require fh_bh_mmdd to be populated."""
    if not ctx.rule.fh_bh_mmdd:
        raise ValueError(
            f"Spot roll rule {ctx.rule.rule_name!r} requires fh_bh_mmdd "
            f"but it is absent.\n"
            f"relative={ctx.relative_name!r}, container={ctx.container_name!r}"
        )
    return ctx.rule.fh_bh_mmdd


def _parse_fh_bh_mmdd(fh_bh_mmdd: str) -> tuple[int, int, int, int]:
    """Parse 'MMDD|MMDD' into (fh_month, fh_day, bh_month, bh_day)."""
    parts = fh_bh_mmdd.split("|")
    if len(parts) != 2 or len(parts[0]) != 4 or len(parts[1]) != 4:
        raise ValueError(f"fh_bh_mmdd must be 'MMDD|MMDD', got {fh_bh_mmdd!r}")
    fh_mm, fh_dd = int(parts[0][:2]), int(parts[0][2:])
    bh_mm, bh_dd = int(parts[1][:2]), int(parts[1][2:])
    return fh_mm, fh_dd, bh_mm, bh_dd


def _half_year_boundary_by_date(
    *,
    as_of_date: date,
    fh_bh_mmdd: str,
    token_offset: int,
    relative_name: str,
    container_name: str,
) -> date:
    """
    Resolve a half-year boundary date.

    Builds a sorted list of FH/BH boundary dates spanning a window around
    as_of_date, then picks the boundary at (bisect_left index + token_offset).
    """
    fh_mm, fh_dd, bh_mm, bh_dd = _parse_fh_bh_mmdd(fh_bh_mmdd)
    as_of_year = as_of_date.year
    look = max(3, abs(token_offset) + 1)

    boundaries: list[date] = []
    for y in range(as_of_year - look, as_of_year + look + 1):
        boundaries.append(date(y, fh_mm, fh_dd))
        boundaries.append(date(y, bh_mm, bh_dd))
    if fh_mm > bh_mm or (fh_mm == bh_mm and fh_dd > bh_dd):
        boundaries.sort()

    idx_target = bisect_left(boundaries, as_of_date) + token_offset
    if idx_target < 0 or idx_target >= len(boundaries):
        raise ValueError(
            f"HALF_YEAR_BY_DATE offset {token_offset} out of range.\n"
            f"relative={relative_name!r}, container={container_name!r}"
        )
    return boundaries[idx_target]


def _half_year_label_suffix(
    *,
    as_of_date: date,
    fh_bh_mmdd: str,
    token_offset: int,
) -> str:
    """Resolve a half-year label suffix (YYYY_FH / YYYY_BH)."""
    fh_mm, fh_dd, _, _ = _parse_fh_bh_mmdd(fh_bh_mmdd)
    current_half_idx = 0 if as_of_date <= date(as_of_date.year, fh_mm, fh_dd) else 1
    total = (as_of_date.year * 2) + current_half_idx + token_offset
    target_year = total // 2
    target_label = "FH" if (total % 2 == 0) else "BH"
    return f"{target_year:04d}_{target_label}"


# ============================================================
# RiskFactorAbsToRelNameIdMap
# ============================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class RiskFactorAbsToRelNameIdMap:
    """
    Bidirectionally indexed mapping between absolute and relative
    risk-factor names for one as-of date.

    Conceptual model
    ----------------
    Relative names (from config):
        BRENT[M+0]                 -- curve point
        VOL_BRENT[M+0][STRIKE=50]  -- surface point
        USDEUR                     -- node (same as absolute)

    Absolute names (as-of-resolved):
        BRENT_202607
        VOL_BRENT_202607[STRIKE=50]
        USDEUR

    Lookup contract
    ---------------
    O(1) lookup for both absolute and relative forms via the unified
    `lookup` dict. Hot-path callers should use `get_rf_id(name)` and
    treat it as the only lookup function they need.
    """

    as_of_date: date
    abs_to_rf_id: dict[str, int]
    abs_to_rel_name: dict[str, str]

    _rel_to_abs: dict[str, str] = field(init=False, repr=False)
    _rel_to_id: dict[str, int] = field(init=False, repr=False)
    _lookup: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        abs_keys = set(self.abs_to_rf_id)
        rel_keys = set(self.abs_to_rel_name)
        if abs_keys != rel_keys:
            sym_diff = sorted(abs_keys ^ rel_keys)[:25]
            raise ValueError(
                f"abs_to_rf_id and abs_to_rel_name keys disagree: {sym_diff}"
            )

        rel_to_abs: dict[str, str] = {}
        rel_to_id: dict[str, int] = {}
        for abs_name, rel_name in self.abs_to_rel_name.items():
            rf_id = self.abs_to_rf_id[abs_name]
            if rel_name in rel_to_abs and rel_to_abs[rel_name] != abs_name:
                raise ValueError(
                    f"Duplicate relative name {rel_name!r} maps to both "
                    f"{rel_to_abs[rel_name]!r} and {abs_name!r}"
                )
            rel_to_abs[rel_name] = abs_name
            rel_to_id[rel_name] = rf_id

        # Defensive disjointness check.
        # The static_config validator already enforces (a) global name uniqueness
        # across all five namespaces and (b) that the only legitimate overlap is
        # nodes, where abs_name == rel_name == node_name AND abs_id == rel_id.
        # This loop only fires if those invariants are broken by a future change
        # to validate.py or schema_objects.py — at which point the loud error here
        # is exactly what we want.
        for name in abs_keys & set(rel_to_id):
            if self.abs_to_rf_id[name] != rel_to_id[name]:
                raise ValueError(
                    f"Name {name!r} appears as both absolute and relative "
                    f"with different rf_ids "
                    f"(abs={self.abs_to_rf_id[name]}, rel={rel_to_id[name]})"
                )

        lookup = {**self.abs_to_rf_id, **rel_to_id}

        object.__setattr__(self, "_rel_to_abs", rel_to_abs)
        object.__setattr__(self, "_rel_to_id", rel_to_id)
        object.__setattr__(self, "_lookup", lookup)

    # ----- Single hot-path lookup -----

    def get_rf_id(self, name: str) -> int:
        """Resolve either an absolute or relative name -> rf_id."""
        return self._lookup[name]

    # ----- Explicit lookups (clarity / validation paths) -----

    def get_rf_id_for_abs_name(self, abs_name: str) -> int:
        return self.abs_to_rf_id[abs_name]

    @overload
    def get_rf_id_for_rel_name(self, rel_name: str) -> int: ...
    @overload
    def get_rf_id_for_rel_name(self, rel_name: list[str]) -> list[int]: ...

    def get_rf_id_for_rel_name(self, rel_name: str | list[str]) -> int | list[int]:
        m = self._rel_to_id
        if isinstance(rel_name, str):
            return m[rel_name]
        return [m[r] for r in rel_name]

    def get_abs_name_for_rel_name(self, rel_name: str | list[str]) -> str | list[str]:
        m = self._rel_to_abs
        if isinstance(rel_name, str):
            return m[rel_name]
        return [m[r] for r in rel_name]

    def get_rel_name_for_abs_name(self, abs_name: str) -> str:
        return self.abs_to_rel_name[abs_name]

    # ----- Properties -----

    @property
    def num_risk_factors(self) -> int:
        return len(self.abs_to_rf_id)

    @property
    def rel_name_to_rf_id(self) -> dict[str, int]:
        return self._rel_to_id

    @property
    def rel_name_to_abs_name(self) -> dict[str, str]:
        return self._rel_to_abs

    @property
    def lookup(self) -> dict[str, int]:
        return self._lookup


# ============================================================
# Resolution engine
# ============================================================


def build_risk_factor_name_mapping_for_as_of_date(
    *,
    risk_static_config: RiskStaticConfig,
    as_of_date: date | str,
) -> RiskFactorAbsToRelNameIdMap:
    """
    Resolve every relative risk-factor name to its absolute name for one as_of_date.

    Three sources are merged:

      1. Nodes — abs_name == rel_name == node_name. No resolution.
      2. Curve points — abs_name = f"{curve_name}_{resolved_suffix}".
         LABEL curves: resolved_suffix is the bare label (e.g. "ON", "3M");
         OFFSET curves: resolved_suffix is computed via the registry.
      3. Surface points — abs_name = f"{surface_name}_{resolved_suffix}[{y_label}]".
         Same resolution rules as curves for the x axis.

    Parameters
    ----------
    risk_static_config
        Compiled static configuration.
    as_of_date
        Valuation date used to resolve OFFSET tokens. ISO string accepted.

    Returns
    -------
    RiskFactorAbsToRelNameIdMap
        Bidirectional abs<->rel<->rf_id mapping.
    """
    if as_of_date is None:
        raise ValueError("as_of_date is required.")
    if isinstance(as_of_date, str):
        as_of_date = date.fromisoformat(as_of_date)

    abs_to_rf_id: dict[str, int] = {}
    abs_to_rel_name: dict[str, str] = {}

    # ---- Nodes ----
    for node in risk_static_config.node_meta_dict.values():
        abs_to_rf_id[node.node_name] = int(node.node_id)
        abs_to_rel_name[node.node_name] = node.node_name

    # ---- Curve points ----
    for pt in risk_static_config.curve_point_meta_dict.values():
        cm = risk_static_config.curve_meta_dict[pt.curve_id]
        rel_name = f"{cm.curve_name}{pt.tenor_token}"
        suffix = _resolve_curve_or_surface_suffix(
            container_name=cm.curve_name,
            container_kind="curve",
            x_axis_kind_name=cm.x_axis_kind,
            spot_roll_rule_name=cm.spot_roll_rule_name,
            spot_roll_rule_dict=risk_static_config.spot_roll_rule_dict,
            tenor_token=pt.tenor_token,
            relative_name=rel_name,
            as_of_date=as_of_date,
        )
        abs_name = f"{cm.curve_name}_{suffix}"
        abs_to_rf_id[abs_name] = int(pt.rf_id)
        abs_to_rel_name[abs_name] = rel_name

    # ---- Surface points ----
    for pt in risk_static_config.surface_point_meta_dict.values():
        sm = risk_static_config.surface_meta_dict[pt.surface_id]
        rel_name = f"{sm.surface_name}{pt.x_tenor_token}[{pt.y_label}]"
        suffix = _resolve_curve_or_surface_suffix(
            container_name=sm.surface_name,
            container_kind="surface",
            x_axis_kind_name=sm.x_axis_kind,
            spot_roll_rule_name=sm.spot_roll_rule_name,
            spot_roll_rule_dict=risk_static_config.spot_roll_rule_dict,
            tenor_token=pt.x_tenor_token,
            relative_name=rel_name,
            as_of_date=as_of_date,
        )
        abs_name = f"{sm.surface_name}_{suffix}[{pt.y_label}]"
        abs_to_rf_id[abs_name] = int(pt.rf_id)
        abs_to_rel_name[abs_name] = rel_name

    return RiskFactorAbsToRelNameIdMap(
        as_of_date=as_of_date,
        abs_to_rf_id=abs_to_rf_id,
        abs_to_rel_name=abs_to_rel_name,
    )


def _resolve_curve_or_surface_suffix(
    *,
    container_name: str,
    container_kind: str,
    x_axis_kind_name: str,
    spot_roll_rule_name: str | None,
    spot_roll_rule_dict: dict[str, SpotRollRule],
    tenor_token: str,
    relative_name: str,
    as_of_date: date,
) -> str:
    """Resolve a single curve/surface point's absolute-name suffix."""
    x_kind = XAxisKind[x_axis_kind_name]

    if x_kind == XAxisKind.LABEL:
        label, signed_offset = parse_bracket_token(tenor_token)
        if signed_offset is not None:
            raise ValueError(
                f"LABEL {container_kind} {container_name!r} has offset token "
                f"{tenor_token!r}; LABEL containers accept label-only tokens."
            )
        return label

    if spot_roll_rule_name is None:
        raise ValueError(
            f"OFFSET {container_kind} {container_name!r} has no spot_roll_rule_name."
        )
    rule = spot_roll_rule_dict.get(spot_roll_rule_name)
    if rule is None:
        raise ValueError(
            f"Unknown spot_roll_rule_name={spot_roll_rule_name!r} for "
            f"{container_kind} {container_name!r}."
        )

    _, token_offset = parse_bracket_token(tenor_token)
    if token_offset is None:
        raise ValueError(
            f"OFFSET {container_kind} {container_name!r} requires a signed "
            f"integer offset token, got {tenor_token!r}."
        )

    method = RollMethod[rule.roll_method]
    resolver = _OFFSET_RESOLVERS.get((x_kind, method))
    if resolver is None:
        raise ValueError(
            f"Unsupported (x_axis_kind, roll_method) pair: "
            f"({x_kind.name}, {method.name})."
        )

    return resolver(
        _ResolverContext(
            container_name=container_name,
            relative_name=relative_name,
            rule=rule,
            token_offset=token_offset,
            as_of_date=as_of_date,
        )
    )
