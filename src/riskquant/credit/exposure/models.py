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

    # --- (1) Replacement Cost with Margin (RCM) for netting sets with margin agreements ---


NUM_EXPOSURE_MODELS: Final[int] = len(ExposureModel)

REPLACEMENT_COST_ID: Final[int] = ExposureModel.REPLACEMENT_COST.value
