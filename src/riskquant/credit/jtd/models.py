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

__all__ = [
    "JtdModel",
]

from enum import IntEnum
from typing import Final


class JtdModel(IntEnum):
    # ---------------------------------------------------------------------
    # Forced default model
    # ---------------------------------------------------------------------
    FORCED_DEFAULT = 0  # scenario-independent worst-case JTD

    # --------------------------------------------------------------------
    # INDEPENDENT DEFAULT MODELS
    # ---------------------------------------------------------------------
    INDEPENDENT_BERNOULLI = 1

    # ---------------------------------------------------------------------
    # BETA BINOMIAL MODEL
    # ---------------------------------------------------------------------

    BETA_BINOMIAL = 2

    # ---------------------------------------------------------------------
    # STRESS REGIME BERNOULLI MODEL
    # ---------------------------------------------------------------------
    STRESS_REGIME_BERNOULLI = 3

    # ---------------------------------------------------------------------
    # Single-factor copula models
    # ---------------------------------------------------------------------
    COPULA_SF_GAUSSIAN = 4  # single-factor Gaussian copula
    COPULA_SF_STUDENTS_T = 5  # single-factor Student-t copula

    # ---------------------------------------------------------------------
    # Multi-factor copula models
    # ---------------------------------------------------------------------
    COPULA_MF_GAUSSIAN = 6  # multi-factor Gaussian copula
    COPULA_MF_STUDENTS_T = 7  # multi-factor Student-t copula


# -------------------------------------------------------------------------
# Derived constants (single source of truth = enum)
# -------------------------------------------------------------------------

NUM_JTD_MODELS: Final[int] = len(JtdModel)

VALID_JTD_MODEL_NAMES: Final[set[str]] = set(item.name for item in JtdModel)
VALID_JTD_MODEL_IDS: Final[set[int]] = set(item.value for item in JtdModel)

# Multi-factor copula models consume the shared long-format loadings frame.
# Extend as further MF copulas land (e.g. COPULA_MF_T).
MF_COPULA_MODELS: Final[frozenset[JtdModel]] = frozenset(
    {JtdModel.COPULA_MF_GAUSSIAN, JtdModel.COPULA_MF_STUDENTS_T}
)
MF_COPULA_MODEL_NAMES: Final[frozenset[str]] = frozenset(
    m.name for m in MF_COPULA_MODELS
)

# -------------------------------------------------------------------------
# Fast-path integer aliases (optional, performance-oriented)
# -------------------------------------------------------------------------
# These exist to avoid enum dereferencing in tight loops / NumPy code.
# Safe because they derive directly from the enum.

# Forced default model
FORCED_DEFAULT_ID: Final[int] = int(JtdModel.FORCED_DEFAULT)

# Independent bernoulli model
INDEPENDENT_BERNOULLI_ID: Final[int] = int(JtdModel.INDEPENDENT_BERNOULLI)

# Beta binomial model
BETA_BINOMIAL_ID: Final[int] = int(JtdModel.BETA_BINOMIAL)

# Stress regime bernoulli model
STRESS_REGIME_BERNOULLI_ID: Final[int] = int(JtdModel.STRESS_REGIME_BERNOULLI)

# Single-factor copula models
COPULA_SF_GAUSSIAN_ID: Final[int] = int(JtdModel.COPULA_SF_GAUSSIAN)
COPULA_SF_T_ID: Final[int] = int(JtdModel.COPULA_SF_STUDENTS_T)

# Multi-factor copula models
COPULA_MF_GAUSSIAN_ID: Final[int] = int(JtdModel.COPULA_MF_GAUSSIAN)
COPULA_MF_T_ID: Final[int] = int(JtdModel.COPULA_MF_STUDENTS_T)
