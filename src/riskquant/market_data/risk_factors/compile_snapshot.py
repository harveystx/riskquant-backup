# src/riskquant/market_data/risk_factors/compile_snapshot.py
"""
Compile a flat `(rel_name, level)` levels dictionary into a typed
`RiskFactorLevelSnapshot`, with unit-scale conversion applied.

Public API mirror — analogous to `compile_risk_static_config_from_dict_of_dfs`:

  compile_risk_static_config_from_dict_of_dfs(dict_of_dfs) -> RiskStaticConfig
  compile_risk_factor_level_snapshot_from_levels_dict(
      *, as_of_date, levels_dict, static_config
  ) -> RiskFactorLevelSnapshot

Contract
--------
- Every key in `levels_dict` is a relative risk-factor name (e.g.
  "USD_SOFR_OIS[1Y]", "BRENT_CRUDE_VOL_SURF[M+0][50.00]") that
  `static_config.rel_name_to_rf_id` resolves.
- Each value is the level *in the unit_scale declared by the static
  config for that rf*. This compiler applies the per-rf
  `unit_scale_factor` so the emitted `factor_levels` are always in
  base DECIMAL — e.g. PERCENT inputs (5.30) become decimal (0.0530),
  and BP inputs (50.0) become decimal (0.0050).
- The resolved `factor_ids` array is sorted strictly ascending and
  C-contiguous; `RiskFactorLevelSnapshot.__post_init__` enforces this
  invariant downstream.

Single normalisation boundary (D069)
------------------------------------
This compiler is the only place a level is converted to base DECIMAL. The
`build_risk_factor_level_snapshot_from_{dict,df,xlsx}` adapters delegate here,
so every public route yields the same canonical float64 value. Name resolution
(rel_name -> rf_id), unit-scale promotion, finiteness and the FX_RATE / VOL
positivity domain (D070) all belong to the ingest perimeter, not to the
hot-path snapshot container.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date

import numpy as np
import numpy.typing as npt

from riskquant.config.enums.container_topology import ContainerTopologyType
from riskquant.config.static_config.static_config import RiskStaticConfig
from riskquant.market_data.risk_factors.snapshot import RiskFactorLevelSnapshot

# D070: quote types whose economic domain excludes zero and negatives. A PRICE,
# RATE, SPREAD or INDEX level may be negative or zero in a real market.
_STRICTLY_POSITIVE_QUOTE_TYPES: frozenset[str] = frozenset({"FX_RATE", "VOL"})


def _quote_type_of(rf_id: int, static_config: RiskStaticConfig) -> str:
    topology = int(static_config.rf_id_to_parent_topology_type[rf_id])
    parent = int(static_config.rf_id_to_parent_id[rf_id])
    if topology == ContainerTopologyType.NODE:
        return static_config.node_meta_dict[rf_id].quote_type
    if topology == ContainerTopologyType.CURVE:
        return static_config.curve_meta_dict[parent].quote_type
    return static_config.surface_meta_dict[parent].quote_type


def canonical_unit_scale(
    *,
    rf_ids: npt.NDArray[np.int32],
    static_config: RiskStaticConfig,
) -> npt.NDArray[np.float64]:
    """
    Per-rf multiplier from declared source units to canonical base units (D069).

    The one normalisation engine for level inputs: every source-unit route
    (level dicts, frames, workbooks, histories) multiplies by exactly this
    vector, once, at its edge. The factor comes from
    `static_config.rf_id_to_unit_scale_factor` (DECIMAL 1, PERCENT 0.01, BP
    1e-4, resolved at config compile). An rf id outside the array or on an
    un-occupied slot fails closed; no scale is ever assumed.
    """
    table = static_config.rf_id_to_unit_scale_factor
    ids = np.asarray(rf_ids, dtype=np.int64)
    in_range = (ids >= 0) & (ids < table.shape[0])
    if not bool(in_range.all()):
        raise KeyError(
            f"rf_ids {ids[~in_range][:10].tolist()} are not risk factors of the "
            "static config; no unit scale exists for them."
        )
    scale = np.ascontiguousarray(table[ids], dtype=np.float64)
    unoccupied = scale <= 0.0
    if bool(unoccupied.any()):
        raise KeyError(
            f"rf_ids {ids[unoccupied][:10].tolist()} map to an un-occupied / "
            "sentinel slot in rf_id_to_unit_scale_factor; they are not present in "
            "the current static config."
        )
    return scale


def strictly_positive_level_mask(
    *,
    rf_ids: npt.NDArray[np.int32],
    static_config: RiskStaticConfig,
) -> npt.NDArray[np.bool_]:
    """
    Per-rf flag: True where the factor's quote type requires a level > 0 (D070).

    Single source of the factor-domain rule, shared by the snapshot compiler and
    the market-data producer (AUD-05). The rule is keyed on `rf_id -> quote
    type`, never on the value. One Python step per distinct rf at the edge.
    """
    out = np.empty(rf_ids.shape[0], dtype=np.bool_)
    for i, rf_id in enumerate(rf_ids.tolist()):
        out[i] = (
            _quote_type_of(int(rf_id), static_config) in _STRICTLY_POSITIVE_QUOTE_TYPES
        )
    return out


def compile_risk_factor_level_snapshot_from_levels_dict(
    *,
    as_of_date: date,
    levels_dict: Mapping[str, float],
    static_config: RiskStaticConfig,
) -> RiskFactorLevelSnapshot:
    """
    Resolve relative-name -> rf_id, apply unit_scale, and return a snapshot.

    Parameters
    ----------
    as_of_date
        Calendar date the snapshot applies to. Pure metadata on the
        returned object — no arithmetic uses it.
    levels_dict
        Flat mapping of relative-name -> level. Levels are in each rf's
        declared `unit_scale` (PERCENT for IR, DECIMAL for prices /
        FX / vols, BP if anything declares it).
    static_config
        Compiled static config. Source of truth for the rel-name -> rf_id
        index and the per-rf unit_scale_factor.
    Canonical dtype
        `factor_levels` is always float64 (D069/MD-10). Scenario buffers narrow
        at their own perimeter; the base-level snapshot does not.

    Returns
    -------
    RiskFactorLevelSnapshot
        `factor_ids` int32 sorted-ascending unique; `factor_levels`
        in base DECIMAL, parallel to `factor_ids`.

    Raises
    ------
    ValueError
        Empty `levels_dict`.
    KeyError
        A relative name in `levels_dict` does not resolve to any rf_id
        in the static config. The error names the offending key.
    """
    if not levels_dict:
        raise ValueError(
            "levels_dict cannot be empty; the snapshot must reference at "
            "least one rf_id."
        )

    n = len(levels_dict)
    rel_to_rf_id = static_config.rel_name_to_rf_id

    # Single-pass resolution: name -> rf_id + per-rf scale -> base level.
    # Pre-allocated int32 / float64 buffers; no Python list intermediates
    # past the dict iteration itself.
    ids_buf = np.empty(n, dtype=np.int32)
    levels_buf = np.empty(n, dtype=np.float64)

    for i, (rel_name, raw_level) in enumerate(levels_dict.items()):
        rf_id = rel_to_rf_id.get(rel_name)
        if rf_id is None:
            raise KeyError(
                f"rel_name {rel_name!r} did not resolve through "
                f"static_config.rel_name_to_rf_id. The snapshot fixture is "
                f"out of sync with the static config."
            )
        ids_buf[i] = np.int32(rf_id)
        # D069: the one unit engine; fails closed on an un-occupied scale slot
        # (a stale rf_id must never multiply by a sentinel; RQ-AUD-012).
        _scale = float(
            canonical_unit_scale(
                rf_ids=ids_buf[i : i + 1], static_config=static_config
            )[0]
        )
        level = float(raw_level) * _scale
        if not np.isfinite(level):
            raise ValueError(
                f"rel_name {rel_name!r}: non-finite level {raw_level!r} (D070)."
            )
        quote_type = _quote_type_of(int(rf_id), static_config)
        if quote_type in _STRICTLY_POSITIVE_QUOTE_TYPES and level <= 0.0:
            raise ValueError(
                f"rel_name {rel_name!r}: {quote_type} level {raw_level!r} must be "
                "strictly positive (D070)."
            )
        levels_buf[i] = level

    # The snapshot constructor enforces sorted-unique + contiguous; pass
    # the buffers in directly. If callers fed us duplicate rel-names that
    # map to the same rf_id, the snapshot constructor will raise.
    return RiskFactorLevelSnapshot(
        as_of_date=as_of_date,
        factor_ids=ids_buf,
        factor_levels=levels_buf,
    )
