# src/riskquant/credit/exposure/models.py
"""
Exposure-model dispatch identifiers.

`ExposureModel` is the exposure-layer analogue of `portfolio.instruments.Instrument`:
the single integer the exposure engine dispatches on. It indexes directly into
`EXPOSURE_EVALUATORS`. The enum holds only IMPLEMENTED models — it grows by one
member each time a model is wired, and the import-time guard in `engine.py`
fails loudly if a member ships without an evaluator.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Final


class ExposureModel(IntEnum):
    """ExposureModel is the single integer dispatch key for the exposure engine."""

    # --- (0) Replacement Cost (RC) for netting sets with no margin agreements ---

    # RC(NS) = MAX(MTM(NS) - COLLATERAL(NS), 0)
    REPLACEMENT_COST = 0

    # --- (1) Current Exposure Method (CEM): RC + net PFE with NGR compression ---

    # EAD(NS) = NRC + Net_PFE
    #   NRC     = MAX(sum(MTM) - COLLATERAL, 0)
    #   GRC     = sum(MAX(MTM, 0))                      (collateral-free)
    #   NGR     = NRC / GRC
    #   Net_PFE = sum(gross_PFE) * (0.4 + 0.6 * NGR)
    CURRENT_EXPOSURE_METHOD = 1

    # --- (2) SA-CCR: hedging-set aggregation + supervisory multiplier ---
    # EAD(NS) = alpha * (RC + PFE);  PFE = multiplier * AddOn
    #   AddOn aggregated across house-defined hedging sets with a supervisory
    #   correlation rho; all taxonomy / rho / alpha / floor are config.
    SACCR = 2


# Hard-coded constants for the exposure engine dispatch table.

NUM_EXPOSURE_MODELS: Final[int] = len(ExposureModel)

REPLACEMENT_COST_ID: Final[int] = ExposureModel.REPLACEMENT_COST.value
CURRENT_EXPOSURE_METHOD_ID: Final[int] = ExposureModel.CURRENT_EXPOSURE_METHOD.value
SACCR_ID: Final[int] = ExposureModel.SACCR.value
