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
    #: Human-readable statement of what the method computes and its supported
    #: domain — the public model contract a caller reads without opening source.
    #: Empty for methods whose display_name is self-explanatory.
    description: str = ""


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
        description=(
            "Standardised Approach for Counterparty Credit Risk (SA-CCR, CRE52). "
            "Supported population: UNMARGINED netting sets, LINEAR (non-option, "
            "non-CDO) trades, EQUITY and COMMODITY asset classes only. "
            "Trade-level effective notional is the full CRE52 structural product "
            "D = d * MF * delta: adjusted notional d = volume * current price "
            "(CRE52.36); supervisory delta = +1 long / -1 short (CRE52.39), "
            "required per trade via 'direction'; unmargined maturity factor "
            "MF = sqrt(clamp(M, 10/250, 1)) (CRE52.48-49), required per trade via "
            "'remaining_maturity_years' (10-business-day floor, one-year cap). "
            "delta and MF are STRUCTURAL CRE52 terms, not house parameters. "
            "'House-calibrated' means only the RiskQuant-owned CONFIGURATION: the "
            "asset-subclass definitions, the subclass->hedging-set mapping, the "
            "supervisory factors SF, the per-hedging-set correlation rho, alpha, "
            "and the multiplier floor. Hedging-set aggregation is the CRE52 "
            "single-factor form; signed effective notionals net fully within a "
            "subclass and partially (rho-weighted) across subclasses of one "
            "hedging set, with no offset across hedging sets. "
            "NOT supported: options, CDO tranches, interest-rate / FX / credit "
            "asset classes, and margined transactions. Not Basel-exact beyond "
            "this supported population and house-owned configuration."
        ),
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
