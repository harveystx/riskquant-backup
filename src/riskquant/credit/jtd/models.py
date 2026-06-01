# src/riskquant/credit/jtd/models.py
"""
JTD model dispatch identifiers.

`JtdModel` is the credit-layer analogue of `ExposureModel`: a compact integer
identifier used to dispatch into `JTD_EVALUATORS`.

Design principles
-----------------
- Values are contiguous in [0, N-1] → safe for direct indexing.
- Enum contains ONLY implemented models.
- Ordering reflects increasing model complexity:
    deterministic → single-factor → multi-factor
  (NOT production priority).
- The enum is the single source of truth for model IDs.

Any addition MUST:
    1. Append at the end (preserve ordering + contiguity)
    2. Provide a corresponding evaluator
"""

from __future__ import annotations

from enum import IntEnum
from typing import Final


class JtdModel(IntEnum):
    # ---------------------------------------------------------------------
    # Deterministic model
    # ---------------------------------------------------------------------
    DETERMINISTIC = 0  # scenario-independent worst-case JTD

    # ---------------------------------------------------------------------
    # Single-factor copula models
    # ---------------------------------------------------------------------
    COPULA_SF_GAUSSIAN = 1  # single-factor Gaussian copula
    COPULA_SF_T = 2  # single-factor Student-t copula

    # ---------------------------------------------------------------------
    # Multi-factor copula models
    # ---------------------------------------------------------------------
    # TODO


# -------------------------------------------------------------------------
# Derived constants (single source of truth = enum)
# -------------------------------------------------------------------------

NUM_JTD_MODELS: Final[int] = len(JtdModel)


# -------------------------------------------------------------------------
# Fast-path integer aliases (optional, performance-oriented)
# -------------------------------------------------------------------------
# These exist to avoid enum dereferencing in tight loops / NumPy code.
# Safe because they derive directly from the enum.

# Deterministic model
DETERMINISTIC_ID: Final[int] = int(JtdModel.DETERMINISTIC)

# Single-factor copula models
COPULA_SF_GAUSSIAN_ID: Final[int] = int(JtdModel.COPULA_SF_GAUSSIAN)
COPULA_SF_T_ID: Final[int] = int(JtdModel.COPULA_SF_T)

# Multi-factor copula models
# TODO
