# src/riskquant/simlab/scenarios/parametric_data.py
"""
Construction-data contract for the parametric scenario generator.

SimLab owns this contract: its shape is dictated by ``ParametricGenerator``'s
needs (the generator resolves Cholesky columns / vols / GMM multipliers by
``searchsorted`` on ``factor_ids``, which therefore MUST be sorted-ascending and
unique), and it is a typed-array data object of the same kind SimLab already
owns (``ScenarioSetBatch``). It is a plain frozen dataclass over NumPy — no
Pydantic, no strings — so it stays inside the NumPy/Numba core tier.

The object owns DATA invariants only (dtype, layout, dimensions, finiteness,
positivity, canonical ordering). Model semantics — total_scenarios, seed,
distribution selection and distribution parameters — remain owned by
``ParametricGenerator``. Inputs are validated, never silently cast: a caller
must supply canonical arrays or fail fast.

The data also declares its effective forecast period (D090): the period whose
law the supplied numbers represent. It is a producer assertion that no array
can prove, so it is validated for form only; its equality with the requested
period is a composition check in the solution tier. The period vocabulary is
defined once, here, as integer codes; the configuration edge maps its names
onto these codes and never defines a second mapping.

Each column also declares its driver chart τ (D091): ``chart_ids[i]`` is the
chart of ``factor_ids[i]``, one chart per factor, drawn from the existing
``ShiftType`` codes. The chart names the variable a column describes; nothing
here transforms a value by chart. ``cholesky_matrix`` is a dependence factor L
whose rows have unit norm, so that ``R = L Lᵀ`` has a unit diagonal; no
triangularity or definiteness is required, and L is accepted unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Final

import numpy as np

from riskquant.constants import NUMERICAL_EXCURSION_TOLERANCE, UNIT_DEPENDENCE_DIAGONAL
from riskquant.market_data.risk_factors.shift_types import (
    SHIFT_DIFFERENCE_ID,
    SHIFT_LOG_RETURN_ID,
    SHIFT_SIMPLE_RETURN_ID,
)
from riskquant.simlab.scenarios.base import require_positive_int


class ForecastPeriodUnit(IntEnum):
    """
    Forecast-period unit vocabulary (D090). The single definition of the
    mapping between unit names and their int8 codes.

    V1 admits exactly one unit. ``BUSINESS_DAY`` counts steps of business-day
    time, one step being the interval from one business-day close to the next;
    it names no holiday calendar or observation schedule. No unit converts into
    another, and a new unit requires a new decision.
    """

    BUSINESS_DAY = 1


#: int8 code of ``ForecastPeriodUnit.BUSINESS_DAY``, the form the data stores.
FORECAST_PERIOD_UNIT_BUSINESS_DAY_ID: Final[np.int8] = np.int8(
    ForecastPeriodUnit.BUSINESS_DAY
)

#: Largest admissible period count: the count is hashed as int64 (D090).
FORECAST_PERIOD_COUNT_MAX: Final[int] = int(np.iinfo(np.int64).max)

#: Driver charts a V1 parametric column may declare (D091): LOG, SIMPLE and
#: DIFFERENCE. NO_SHIFT and ABSOLUTE_BASE_RETURN are not parametric driver charts.
PARAMETRIC_DRIVER_CHART_IDS: Final[tuple[int, ...]] = (
    int(SHIFT_LOG_RETURN_ID),
    int(SHIFT_SIMPLE_RETURN_ID),
    int(SHIFT_DIFFERENCE_ID),
)


@dataclass(frozen=True, slots=True)
class ParametricRiskData:
    """
    Canonical numerical inputs to construct a ``ParametricGenerator``.

    Contract (all rejected, never coerced):
      - ``factor_ids``      : int32, 1-D, C-contiguous, non-empty, sorted
                              ascending and unique (length K).
      - ``cholesky_matrix`` : float64, 2-D, (K, K) square, C-contiguous, finite,
                              every row of unit norm: ``|sum_k L_ik^2 - 1|`` within
                              ``NUMERICAL_EXCURSION_TOLERANCE`` (D091). Accepted
                              unchanged; never clamped or renormalised.
      - ``volatilities``    : float64, 1-D, length K, C-contiguous, finite,
                              strictly positive.
      - ``effective_period_count`` : integer (int or NumPy integer, not bool),
                              1 <= count <= int64 max (D090).
      - ``effective_period_unit``  : ``np.int8`` code of a ``ForecastPeriodUnit``
                              (D090); V1 admits only ``BUSINESS_DAY``.
      - ``chart_ids``       : int8, 1-D, length K, C-contiguous; ``chart_ids[i]``
                              is the driver chart of ``factor_ids[i]``, one of
                              ``PARAMETRIC_DRIVER_CHART_IDS`` (D091).

    The effective period ``(effective_period_count, effective_period_unit)`` is
    the forecast period the supplied law represents. It is neither an
    observation count nor a lookback, and nothing here converts it.
    """

    factor_ids: np.ndarray
    cholesky_matrix: np.ndarray
    volatilities: np.ndarray
    effective_period_count: int
    effective_period_unit: np.int8
    chart_ids: np.ndarray

    def __post_init__(self) -> None:
        f = self.factor_ids
        if f.ndim != 1 or f.dtype != np.int32:
            raise TypeError(f"factor_ids must be 1-D int32, got {f.ndim}-D {f.dtype}.")
        if not f.flags.c_contiguous:
            raise ValueError("factor_ids must be C-contiguous.")
        if f.shape[0] == 0:
            raise ValueError("factor_ids must be non-empty.")
        if f.shape[0] > 1 and np.any(np.diff(f) <= 0):
            raise ValueError("factor_ids must be sorted ascending and unique.")
        k = f.shape[0]

        c = self.cholesky_matrix
        if c.ndim != 2 or c.dtype != np.float64:
            raise TypeError(
                f"cholesky_matrix must be 2-D float64, got {c.ndim}-D {c.dtype}."
            )
        if c.shape != (k, k):
            raise ValueError(
                f"cholesky_matrix must be square ({k}, {k}), got {c.shape}."
            )
        if not c.flags.c_contiguous:
            raise ValueError("cholesky_matrix must be C-contiguous.")
        with np.errstate(all="ignore"):
            if not np.isfinite(c).all():
                raise ValueError("cholesky_matrix must contain only finite values.")
        # D091: R = L Lᵀ must have a unit diagonal, i.e. every row of L has unit
        # norm. The row sums of squares are the diagonal of R without forming R.
        # The engine's excursion tolerance is used as an acceptance bound only:
        # L is input data and is never clamped or renormalised (D034).
        row_norm_sq = np.einsum("ij,ij->i", c, c)
        excursion = np.abs(row_norm_sq - UNIT_DEPENDENCE_DIAGONAL)
        if np.any(excursion > NUMERICAL_EXCURSION_TOLERANCE):
            worst = int(np.argmax(excursion))
            raise ValueError(
                "cholesky_matrix must be a correlation factor: every row needs unit "
                f"norm (diag(L Lᵀ) = 1) within {NUMERICAL_EXCURSION_TOLERANCE}; "
                f"row {worst} has sum of squares {float(row_norm_sq[worst])!r}."
            )

        v = self.volatilities
        if v.ndim != 1 or v.dtype != np.float64:
            raise TypeError(
                f"volatilities must be 1-D float64, got {v.ndim}-D {v.dtype}."
            )
        if v.shape[0] != k:
            raise ValueError(f"volatilities length must be {k}, got {v.shape[0]}.")
        if not v.flags.c_contiguous:
            raise ValueError("volatilities must be C-contiguous.")
        with np.errstate(all="ignore"):
            if not np.isfinite(v).all():
                raise ValueError("volatilities must contain only finite values.")
            if np.any(v <= 0.0):
                raise ValueError("volatilities must be strictly positive.")

        # D090 effective period: validated for form only, never converted.
        count = require_positive_int(
            self.effective_period_count, "effective_period_count"
        )
        if count > FORECAST_PERIOD_COUNT_MAX:
            raise ValueError(
                f"effective_period_count must be <= {FORECAST_PERIOD_COUNT_MAX} "
                f"(int64), got {count}."
            )
        unit = self.effective_period_unit
        if type(unit) is not np.int8:
            raise TypeError(
                "effective_period_unit must be an np.int8 ForecastPeriodUnit code, "
                f"got {type(unit).__name__}."
            )
        admissible = tuple(int(m) for m in ForecastPeriodUnit)
        if int(unit) not in admissible:
            raise ValueError(
                f"effective_period_unit {int(unit)} is not a ForecastPeriodUnit code; "
                f"admissible: {admissible}."
            )

        # D091 driver charts: one declared chart per factor, aligned by position.
        t = self.chart_ids
        if not isinstance(t, np.ndarray) or t.ndim != 1 or t.dtype != np.int8:
            raise TypeError(
                "chart_ids must be a 1-D int8 array, got "
                f"{getattr(t, 'ndim', '?')}-D {getattr(t, 'dtype', type(t).__name__)}."
            )
        if t.shape[0] != k:
            raise ValueError(f"chart_ids length must be {k}, got {t.shape[0]}.")
        if not t.flags.c_contiguous:
            raise ValueError("chart_ids must be C-contiguous.")
        bad = ~np.isin(t, np.array(PARAMETRIC_DRIVER_CHART_IDS, dtype=np.int8))
        if np.any(bad):
            raise ValueError(
                f"chart_ids must be driver charts {PARAMETRIC_DRIVER_CHART_IDS} "
                f"(LOG, SIMPLE, DIFFERENCE); got {int(t[np.argmax(bad)])} at "
                f"position {int(np.argmax(bad))}."
            )
