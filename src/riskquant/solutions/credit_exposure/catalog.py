# src/riskquant/solutions/credit_exposure/catalog.py
"""
Exposure-model catalog: the front-end menu of exposure methods.

Mirrors ``solutions.credit_var.catalog``. Each entry maps a stable string key to
the engine's integer ``ExposureModel`` id and declares what inputs the method
needs, so the request layer can validate a payload against the chosen method the
same way credit-VaR validates against ``requires_loadings`` / ``requires_groups``.
"""

from __future__ import annotations

__all__ = [
    "EXPOSURE_MODEL_CATALOG",
    "ExposureModelEntry",
    "exposure_model_entry",
    "supported_exposure_models",
]

from dataclasses import dataclass

from riskquant.credit.exposure.models import (
    CURRENT_EXPOSURE_METHOD_ID,
    REPLACEMENT_COST_ID,
    SACCR_ID,
)
from riskquant.solutions.credit_exposure.recipes import SUPPORTED_EXPOSURE_RECIPES


@dataclass(frozen=True, slots=True, kw_only=True)
class ExposureModelEntry:
    """One exposure method: display label, engine id, and input requirements."""

    key: str
    display_name: str
    exposure_model_id: int
    #: Needs the supervisory add-on table + per-trade priced-notional inputs
    #: (volume, current price, asset subclass). False for a pure replacement-cost
    #: method, which needs only marks + collateral.
    requires_add_on: bool = False
    #: Needs the SA-CCR hedging-set config (subclass->hedging-set map + per-
    #: hedging-set correlation), analogous to credit-VaR's `requires_loadings`.
    requires_saccr_config: bool = False


EXPOSURE_MODEL_CATALOG: dict[str, ExposureModelEntry] = {
    "replacement_cost": ExposureModelEntry(
        key="replacement_cost",
        display_name="Replacement Cost",
        exposure_model_id=REPLACEMENT_COST_ID,
        requires_add_on=False,
    ),
    "current_exposure_method": ExposureModelEntry(
        key="current_exposure_method",
        display_name="Current Exposure Method",
        exposure_model_id=CURRENT_EXPOSURE_METHOD_ID,
        requires_add_on=True,
    ),
    "saccr": ExposureModelEntry(
        key="saccr",
        display_name="SA-CCR (house-calibrated)",
        exposure_model_id=SACCR_ID,
        requires_add_on=True,
        requires_saccr_config=True,
    ),
}


# Guard: every catalog model must have a recipe (mirrors credit_var's guard).
_missing = set(EXPOSURE_MODEL_CATALOG) - set(SUPPORTED_EXPOSURE_RECIPES)
if _missing:
    raise RuntimeError(
        f"exposure catalog/recipe mismatch: catalog model(s) {sorted(_missing)} "
        "have no recipe in SUPPORTED_EXPOSURE_RECIPES."
    )


def supported_exposure_models() -> tuple[str, ...]:
    return tuple(EXPOSURE_MODEL_CATALOG)


def exposure_model_entry(key: str) -> ExposureModelEntry:
    entry = EXPOSURE_MODEL_CATALOG.get(key)
    if entry is None:
        raise KeyError(
            f"unknown exposure model '{key}'. "
            f"Supported: {sorted(EXPOSURE_MODEL_CATALOG)}."
        )
    return entry
