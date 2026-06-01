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

Why a separate compiler rather than reusing `build_*_from_dict`?
----------------------------------------------------------------
The existing `build_risk_factor_level_snapshot_from_dict` is layout-
agnostic and unit-agnostic — fine for already-canonical numeric inputs.
The compiler here adds two pieces the canonical fixture needs:
  1. Name resolution via the static config (rel_name -> rf_id).
  2. Unit-scale promotion to base DECIMAL.

Both belong to the ingest perimeter, not the hot-path snapshot
container, so they live here in a dedicated compile step.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date

import numpy as np

from riskquant.config.static_config.static_config import RiskStaticConfig
from riskquant.market_data.risk_factors.snapshot import RiskFactorLevelSnapshot


def compile_risk_factor_level_snapshot_from_levels_dict(
    *,
    as_of_date: date,
    levels_dict: Mapping[str, float],
    static_config: RiskStaticConfig,
    float_dtype: type[np.floating] = np.float64,
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
    float_dtype
        Output dtype for `factor_levels`. Default `float64` to preserve
        precision through the unit-scale multiplication; downstream
        scenario buffers narrow to `float32` at their own perimeter.

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
    rf_id_to_unit_scale = static_config.rf_id_to_unit_scale_factor

    # Single-pass resolution: name -> rf_id + per-rf scale -> base level.
    # Pre-allocated int32 / float64 buffers; no Python list intermediates
    # past the dict iteration itself.
    ids_buf = np.empty(n, dtype=np.int32)
    levels_buf = np.empty(n, dtype=float_dtype)

    for i, (rel_name, raw_level) in enumerate(levels_dict.items()):
        rf_id = rel_to_rf_id.get(rel_name)
        if rf_id is None:
            raise KeyError(
                f"rel_name {rel_name!r} did not resolve through "
                f"static_config.rel_name_to_rf_id. The snapshot fixture is "
                f"out of sync with the static config."
            )
        ids_buf[i] = np.int32(rf_id)
        # rf_id_to_unit_scale_factor is keyed by rf_id (numpy array indexed
        # by id). float()-wrap because the result will be cast to
        # float_dtype anyway and we want native multiplication.
        levels_buf[i] = float_dtype(raw_level) * float_dtype(
            rf_id_to_unit_scale[int(rf_id)]
        )

    # The snapshot constructor enforces sorted-unique + contiguous; pass
    # the buffers in directly. If callers fed us duplicate rel-names that
    # map to the same rf_id, the snapshot constructor will raise.
    return RiskFactorLevelSnapshot(
        as_of_date=as_of_date,
        factor_ids=ids_buf,
        factor_levels=levels_buf,
    )
