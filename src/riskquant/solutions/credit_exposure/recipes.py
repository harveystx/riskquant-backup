# src/riskquant/solutions/credit_exposure/recipes.py
"""
Exposure-model recipes: flat trade/config tables -> compiled exposure inputs.

The authoring layer of the credit-exposure solution, mirroring
``solutions.credit_var.recipes``: it turns the desk's flat tables plus a handful
of scalar knobs into the compiled netting-set view the engine needs, which the
author never sees. Replacement cost: marks + collateral only. CEM: the add-on
table becomes per-trade supervisory factors on the priced notional, with the
NGR-compression floor/coefficient as knobs. SA-CCR: the add-on table plus the
house hedging-set config (subclass -> hedging set, per-set rho) become the
hedging-set-aggregated add-on, with alpha and the multiplier floor as knobs.

This is solution-tier, not core: the core engine takes fully-compiled masters and
views; "construct them from flat tables" is authoring sugar that belongs with the
solution. Plain DataFrames + params in, engine objects out, fail-fast. Adding a
model is one assembler here plus a catalog entry.
"""

from __future__ import annotations

from typing import NotRequired, TypedDict

__all__ = [
    "SUPPORTED_EXPOSURE_RECIPES",
    "AssembledExposureInputs",
    "assemble_exposure_inputs",
]

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from riskquant.credit.exposure.add_on.compiled import CompiledAddOnFactors
from riskquant.credit.exposure.add_on.ingest import compile_add_on_factors
from riskquant.credit.exposure.add_on.resolve import (
    resolve_quantity_price_effective_notional,
    resolve_saccr_effective_notional,
    resolve_supervisory_factors,
)
from riskquant.credit.exposure.compile import (
    compile_exposure_cem,
    compile_exposure_rc,
    compile_exposure_saccr,
)
from riskquant.credit.exposure.compiled import CompiledExposure
from riskquant.credit.exposure.saccr.ingest import compile_saccr_config
from riskquant.credit.masters.netting_set.compiled import CompiledNettingSetMaster


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class AssembledExposureInputs:
    """Compiled engine inputs for one exposure run (what a recipe produces)."""

    compiled: CompiledExposure
    netting_set_master: CompiledNettingSetMaster


# --------------------------------------------------------------------------
# Shared helpers (each recipe is a thin composition of these)
# --------------------------------------------------------------------------


def _trade_uids(trades: pd.DataFrame) -> np.ndarray:
    return np.ascontiguousarray(trades["trade_uid"].to_numpy(dtype=np.int64))


def _trade_netting_set(trades: pd.DataFrame) -> np.ndarray:
    return np.ascontiguousarray(trades["netting_set_uid"].to_numpy(dtype=np.int64))


def _master(netting_sets: pd.DataFrame) -> CompiledNettingSetMaster:
    """Compile the netting-set master (ascending by uid, the master invariant)."""
    uid = netting_sets["netting_set_uid"].to_numpy(dtype=np.int64)
    order = np.argsort(uid, kind="stable")
    return CompiledNettingSetMaster(
        uid=np.ascontiguousarray(uid[order]),
        obligor_uid=np.ascontiguousarray(
            netting_sets["obligor_uid"].to_numpy(dtype=np.int64)[order]
        ),
        net_collateral=np.ascontiguousarray(
            netting_sets["net_collateral"].to_numpy(dtype=np.float64)[order]
        ),
    )


def _add_on(add_on_factors: pd.DataFrame | None, *, ctx: str) -> CompiledAddOnFactors:
    if add_on_factors is None:
        raise ValueError(f"{ctx}: 'add_on_factors' table is required.")
    return compile_add_on_factors(add_on_factors)


def _effective_notional(trades: pd.DataFrame) -> np.ndarray:
    return resolve_quantity_price_effective_notional(
        volume=trades["volume"].to_numpy(dtype=np.float64),
        current_price=trades["current_price"].to_numpy(dtype=np.float64),
    )


def _saccr_effective_notional(trades: pd.DataFrame) -> np.ndarray:
    """SA-CCR signed effective notional D_i = d_i * MF_i * delta_i (P2-11A/B).

    Direction string dies here: it is mapped to the canonical +-1 sign before the
    numeric resolver. Remaining maturity crosses as a plain float year. The
    request validator has already guaranteed both columns are present and typed
    for SA-CCR (fail-fast at the boundary); this reads them as canonical arrays.
    """
    direction = trades["direction"].to_numpy()
    # Explicit long/short mapping. `np.where(=="long", +1, -1)` would silently
    # turn any non-"long" value (typo, None-as-object, "SHORT") into a short;
    # map only the two legal literals and reject everything else. TradeRow's
    # Literal already enforces this upstream — this is defence in depth against a
    # frame that reached the recipe by another path.
    is_long = direction == "long"
    is_short = direction == "short"
    if not np.all(is_long | is_short):
        bad = sorted({str(d) for d in direction[~(is_long | is_short)]})
        raise ValueError(
            f"_saccr_effective_notional: direction must be 'long' or 'short' for "
            f"every trade; got unexpected value(s) {bad}."
        )
    direction_sign = np.where(is_long, 1.0, -1.0).astype(np.float64)
    return resolve_saccr_effective_notional(
        volume=trades["volume"].to_numpy(dtype=np.float64),
        current_price=trades["current_price"].to_numpy(dtype=np.float64),
        direction_sign=np.ascontiguousarray(direction_sign),
        remaining_maturity_years=trades["remaining_maturity_years"].to_numpy(
            dtype=np.float64
        ),
    )


def _subclass_uix(trades: pd.DataFrame) -> np.ndarray:
    return np.ascontiguousarray(trades["asset_subclass_uix"].to_numpy(dtype=np.int32))


def _overrides(params: Mapping[str, float], keys: tuple[str, ...]) -> dict[str, float]:
    """Pick the model's knobs from params; absent keys fall to engine defaults."""
    return {k: float(params[k]) for k in keys if params.get(k) is not None}


# --------------------------------------------------------------------------
# Per-model assemblers (uniform signature; unused inputs ignored)
# --------------------------------------------------------------------------


def _assemble_replacement_cost(
    trades: pd.DataFrame,
    netting_sets: pd.DataFrame,
    params: Mapping[str, float],
    add_on_factors: pd.DataFrame | None = None,
    subclass_hedging_sets: pd.DataFrame | None = None,
    hedging_set_correlations: pd.DataFrame | None = None,
) -> AssembledExposureInputs:
    master = _master(netting_sets)
    compiled = compile_exposure_rc(
        trade_uids=_trade_uids(trades),
        trade_netting_set_id=_trade_netting_set(trades),
        netting_set_master=master,
    )
    return AssembledExposureInputs(compiled=compiled, netting_set_master=master)


def _assemble_current_exposure_method(
    trades: pd.DataFrame,
    netting_sets: pd.DataFrame,
    params: Mapping[str, float],
    add_on_factors: pd.DataFrame | None = None,
    subclass_hedging_sets: pd.DataFrame | None = None,
    hedging_set_correlations: pd.DataFrame | None = None,
) -> AssembledExposureInputs:
    master = _master(netting_sets)
    add_on = _add_on(add_on_factors, ctx="current_exposure_method")
    compiled = compile_exposure_cem(
        trade_uids=_trade_uids(trades),
        trade_netting_set_id=_trade_netting_set(trades),
        trade_effective_notional=_effective_notional(trades),
        trade_supervisory_factor=resolve_supervisory_factors(
            _subclass_uix(trades), add_on
        ),
        netting_set_master=master,
        **_overrides(params, ("pfe_floor", "pfe_coefficient")),
    )
    return AssembledExposureInputs(compiled=compiled, netting_set_master=master)


class _SaccrConfigKwargs(TypedDict):
    alpha: NotRequired[float]
    multiplier_floor: NotRequired[float]


def _assemble_saccr(
    trades: pd.DataFrame,
    netting_sets: pd.DataFrame,
    params: Mapping[str, float],
    add_on_factors: pd.DataFrame | None = None,
    subclass_hedging_sets: pd.DataFrame | None = None,
    hedging_set_correlations: pd.DataFrame | None = None,
) -> AssembledExposureInputs:
    master = _master(netting_sets)
    add_on = _add_on(add_on_factors, ctx="saccr")
    if subclass_hedging_sets is None or hedging_set_correlations is None:
        raise ValueError(
            "saccr: 'subclass_hedging_sets' and 'hedging_set_correlations' tables "
            "are required."
        )
    knobs = _overrides(params, ("saccr_alpha", "saccr_multiplier_floor"))
    config_kwargs: _SaccrConfigKwargs = {}
    if "saccr_alpha" in knobs:
        config_kwargs["alpha"] = knobs["saccr_alpha"]
    if "saccr_multiplier_floor" in knobs:
        config_kwargs["multiplier_floor"] = knobs["saccr_multiplier_floor"]
    saccr_config = compile_saccr_config(
        subclass_hedging_sets,
        hedging_set_correlations,
        **config_kwargs,
    )
    compiled = compile_exposure_saccr(
        trade_uids=_trade_uids(trades),
        trade_netting_set_id=_trade_netting_set(trades),
        trade_effective_notional=_saccr_effective_notional(trades),
        trade_subclass_uix=_subclass_uix(trades),
        netting_set_master=master,
        add_on_factors=add_on,
        saccr_config=saccr_config,
    )
    return AssembledExposureInputs(compiled=compiled, netting_set_master=master)


SUPPORTED_EXPOSURE_RECIPES: dict[str, object] = {
    "replacement_cost": _assemble_replacement_cost,
    "current_exposure_method": _assemble_current_exposure_method,
    "saccr": _assemble_saccr,
}


def assemble_exposure_inputs(
    *,
    model_key: str,
    trades: pd.DataFrame,
    netting_sets: pd.DataFrame,
    params: Mapping[str, float] | None = None,
    add_on_factors: pd.DataFrame | None = None,
    subclass_hedging_sets: pd.DataFrame | None = None,
    hedging_set_correlations: pd.DataFrame | None = None,
) -> AssembledExposureInputs:
    """Build compiled engine inputs for `model_key` from flat tables + params.

    Raises ValueError on an unknown recipe; each assembler fail-fast validates its
    own inputs, and the tables flow through the master / add-on / config
    compilers' validation.
    """
    assembler = SUPPORTED_EXPOSURE_RECIPES.get(model_key)
    if assembler is None:
        raise ValueError(
            f"assemble_exposure_inputs: no recipe for model '{model_key}'. "
            f"Supported: {sorted(SUPPORTED_EXPOSURE_RECIPES)}."
        )
    return assembler(  # type: ignore[operator]
        trades,
        netting_sets,
        params or {},
        add_on_factors,
        subclass_hedging_sets,
        hedging_set_correlations,
    )
