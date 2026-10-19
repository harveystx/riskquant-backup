# src/riskquant/factor_regression/diagnostics.py
"""
Durbin-Watson statistic and integer-coded fit diagnostics. PRIVATE.

The core reports diagnostics as an ``int8`` code array and a ``float64`` value
array (C.11). The facade maps each code to its name in ``labels.py``. The code
set is closed (``codes.DIAGNOSTIC_CODES``) and the rules are those of A.4:

* ``CONDITION_ABOVE_THRESHOLD``: the threshold applies and ``kappa`` is above
  it. Value: ``kappa``.
* ``LEVEL_TERM_PRESENT``: a term (response or regressor) uses LEVEL.
  Value: ``1.0``.
* ``R2_ABOVE_DURBIN_WATSON``: a LEVEL term is present and ``r_squared`` is
  above the Durbin-Watson statistic, the Granger and Newbold rule of thumb for
  a spurious regression. It is a heuristic, not a test. Value: ``r_squared``.
* ``OVERLAP_NOT_CORRECTED``: ``q > 0`` and the covariance is IID or HC0, or
  HAC_BARTLETT with a lag below ``q``. Value: ``float(q)``.

A diagnostic never blocks a fit, and the library issues no Python warning
(C.11). The core approves no assumption: a missing code means only that its
rule did not fire.

No threshold sentinel. The core takes no None, no NaN and no new sentinel
(D004): the condition threshold is a float plus a bool that says whether it
applies. When the bool is False the float is not read.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from riskquant.factor_regression import codes
from riskquant.factor_regression.errors import (
    CoreDegenerateFitError,
    CoreInputError,
    CoreParameterError,
)

_TRANSFORM_CODES: Final = np.array(codes.TRANSFORM_CODES, dtype=np.int8)
_COVARIANCE_CODES: Final = frozenset(int(code) for code in codes.COVARIANCE_CODES)


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class CoreDiagnostics:
    """
    The diagnostics of one fit, in ascending code order.

    Attributes
    ----------
    codes
        ``int8`` ``(n,)``: codes of ``codes.DIAGNOSTIC_CODES`` that fired,
        ascending, each at most once.
    values
        ``float64`` ``(n,)``: the value of each code, all finite (A.4).

    Notes
    -----
    Both arrays are new and read-only. A fixed order makes two diagnoses of
    the same inputs equal bit for bit. ``eq=False`` follows the D064
    precedent for array-bearing containers.
    """

    codes: NDArray[np.int8]
    values: NDArray[np.float64]

    @property
    def n_diagnostics(self) -> int:
        return int(self.codes.shape[0])


def _validate_vector(operand: NDArray[np.generic], dtype: type[np.generic]) -> None:
    # Exact type, not isinstance, before the first reduction. numpy.ma skips
    # masked elements in .all(): a NaN residual or an unknown transform code
    # under the mask passes the checks. The core converts nothing, so it
    # rejects every ndarray subclass.
    if type(operand) is not np.ndarray:
        raise CoreInputError(reason=codes.INPUT_NOT_NDARRAY, operand=0, positions=())
    if operand.dtype != dtype:
        raise CoreInputError(reason=codes.INPUT_WRONG_DTYPE, operand=0, positions=())
    if operand.ndim != 1:
        raise CoreInputError(reason=codes.INPUT_WRONG_NDIM, operand=0, positions=())
    if int(operand.shape[0]) == 0:
        raise CoreInputError(reason=codes.INPUT_EMPTY, operand=0, positions=())


def _declared_flag(value: bool) -> bool:
    # A flag is a bool or numpy.bool_; an int 0 or 1 is not read as a flag.
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError("a flag must be a bool")
    return bool(value)


def _declared_integer(value: int) -> int:
    # operator.index rejects a float or str with TypeError, so the core never
    # rounds. bool is an int subclass and operator.index reads True as 1; a
    # flag is not an order, a code or a lag.
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("an integer parameter must be an int, not a bool")
    return operator.index(value)


def _declared_real(value: float) -> float:
    # bool is an int subclass, so float(True) reads as 1.0. A str, None or
    # complex value raises: the core never parses.
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError("a real parameter must be an int or a float")
    return float(value)


def _finite_statistic(value: float) -> float:
    # A NaN statistic makes every comparison False, so a rule would not fire
    # and the miss would be silent. codes.py has no reason code for a scalar
    # statistic, so this is a ValueError, as align.py raises TypeError for a
    # bound of the wrong type.
    real = _declared_real(value)
    if not math.isfinite(real):
        raise ValueError(f"a statistic must be finite, got {real!r}")
    return real


def durbin_watson(resid: NDArray[np.float64]) -> float:
    """
    Durbin-Watson statistic of residuals in date order (A.4).

    ``sum_{t=1..R-1} (e_t - e_{t-1})^2 / sum_t e_t^2``.

    Parameters
    ----------
    resid
        ``float64`` ``(R,)``, ``R >= 1``, all finite, rows in date order.

    Returns
    -------
    float
        The statistic, in ``[0, 4]`` to rounding. ``0.0`` for ``R == 1``.

    Raises
    ------
    CoreInputError
        ``operand`` is ``0``. ``INPUT_NOT_NDARRAY`` (any ndarray subclass,
        ``numpy.ma`` included), ``INPUT_WRONG_DTYPE``, ``INPUT_WRONG_NDIM``,
        ``INPUT_EMPTY`` with empty ``positions``. ``INPUT_NON_FINITE``:
        ``positions`` holds the index of the first NaN or inf.
    CoreDegenerateFitError
        ``DEGENERATE_EXACT_FIT``: every residual is zero, so the statistic
        is ``0 / 0``. ``value`` and ``bound`` are ``0.0``.

    Notes
    -----
    The residuals are scaled by the power of two ``2**-e`` with ``e`` the
    binary exponent of their largest magnitude. The statistic is a ratio, so
    the scale cancels; the scaling is exact, so the result equals the
    unscaled formula bit for bit when that has no overflow or underflow. An
    unscaled sum of squares overflows for residuals near 1e160.
    """
    _validate_vector(resid, np.float64)
    finite = np.isfinite(resid)
    if not bool(finite.all()):
        first = int(np.flatnonzero(~finite)[0])
        raise CoreInputError(
            reason=codes.INPUT_NON_FINITE, operand=0, positions=(first,)
        )
    peak = float(np.max(np.abs(resid)))
    if peak == 0.0:
        raise CoreDegenerateFitError(
            reason=codes.DEGENERATE_EXACT_FIT, value=0.0, bound=0.0, columns=()
        )
    _, exponent = math.frexp(peak)
    unit = np.ldexp(resid, -exponent)
    step = np.diff(unit)
    return float(np.dot(step, step) / np.dot(unit, unit))


def diagnose(
    *,
    term_transforms: NDArray[np.int8],
    condition_number: float,
    condition_threshold: float,
    apply_condition_threshold: bool,
    r_squared: float,
    durbin_watson_value: float,
    overlap_order: int,
    covariance_kind: int,
    covariance_lag: int,
) -> CoreDiagnostics:
    """
    Evaluate the four A.4 rules and return the codes that fire.

    Parameters
    ----------
    term_transforms
        ``int8`` ``(n_terms,)``, ``n_terms >= 1``: the transform code of every
        term, the response included. The intercept is not a term.
    condition_number
        ``kappa`` of the scaled design (``CoreScaledSvd.condition_number``).
        Finite.
    condition_threshold
        The threshold of C.10. Read only when ``apply_condition_threshold``
        is True; then it must be above 1 (NaN is not).
    apply_condition_threshold
        ``True`` applies the threshold. ``False`` stands for the facade value
        ``None``: no flag, and ``condition_threshold`` is not read.
    r_squared
        ``r_squared`` of the fit. Finite.
    durbin_watson_value
        The Durbin-Watson statistic of the fit residuals. Finite.
    overlap_order
        ``q >= 0`` of the rows (``CoreRows.overlap_order``, A.1).
    covariance_kind
        A code of ``codes.COVARIANCE_CODES``.
    covariance_lag
        The HAC_BARTLETT lag ``L >= 0``. Read only when ``covariance_kind`` is
        ``HAC_BARTLETT``. The lag rules of C.13 are checked by the fit (S4).

    Returns
    -------
    CoreDiagnostics
        The codes that fire in ascending order, with their values.

    Raises
    ------
    TypeError
        A flag is not a ``bool`` or ``numpy.bool_``; an integer parameter is
        not an integer or is a ``bool``; a real parameter is not a real
        number or is a ``bool``.
    ValueError
        ``overlap_order`` is negative, or a statistic (``condition_number``,
        ``r_squared``, ``durbin_watson_value``) is not finite.
    CoreInputError
        ``term_transforms`` breaks its contract. ``operand`` is ``0``;
        ``positions`` is empty.
    CoreParameterError
        ``PARAM_UNKNOWN_TRANSFORM`` (``value`` is the code),
        ``PARAM_CONDITION_THRESHOLD_NOT_ABOVE_ONE`` (applied threshold not
        above 1), ``PARAM_UNKNOWN_COVARIANCE`` (``value`` is the code) or
        ``PARAM_NEGATIVE_LAG`` (HAC_BARTLETT with ``L < 0``).

    Notes
    -----
    The checks run in this order: array, parameters, statistics. Every check
    runs before the first rule.
    """
    _validate_vector(term_transforms, np.int8)
    known = np.isin(term_transforms, _TRANSFORM_CODES)
    if not bool(known.all()):
        first = int(np.flatnonzero(~known)[0])
        raise CoreParameterError(
            reason=codes.PARAM_UNKNOWN_TRANSFORM, value=float(term_transforms[first])
        )

    apply = _declared_flag(apply_condition_threshold)
    threshold = _declared_real(condition_threshold)
    kind = _declared_integer(covariance_kind)
    lag = _declared_integer(covariance_lag)
    q = _declared_integer(overlap_order)
    if apply and not threshold > 1.0:
        raise CoreParameterError(
            reason=codes.PARAM_CONDITION_THRESHOLD_NOT_ABOVE_ONE, value=threshold
        )
    if kind not in _COVARIANCE_CODES:
        raise CoreParameterError(
            reason=codes.PARAM_UNKNOWN_COVARIANCE, value=float(kind)
        )
    is_hac = kind == int(codes.HAC_BARTLETT)
    if is_hac and lag < 0:
        raise CoreParameterError(reason=codes.PARAM_NEGATIVE_LAG, value=float(lag))
    if q < 0:
        raise ValueError(f"overlap_order must be at least 0, got {q}")

    kappa = _finite_statistic(condition_number)
    r2 = _finite_statistic(r_squared)
    dw = _finite_statistic(durbin_watson_value)

    level_present = bool((term_transforms == codes.LEVEL).any())
    fired: list[tuple[np.int8, float]] = []
    if apply and kappa > threshold:
        fired.append((codes.CONDITION_ABOVE_THRESHOLD, kappa))
    if level_present:
        fired.append((codes.LEVEL_TERM_PRESENT, 1.0))
    if level_present and r2 > dw:
        fired.append((codes.R2_ABOVE_DURBIN_WATSON, r2))
    # A HAC lag of at least q covers every non-zero autocovariance of an
    # MA(q) error; IID and HC0 cover none (A.4).
    if q > 0 and (not is_hac or lag < q):
        fired.append((codes.OVERLAP_NOT_CORRECTED, float(q)))

    diagnostic_codes = np.array([code for code, _ in fired], dtype=np.int8)
    diagnostic_values = np.array([value for _, value in fired], dtype=np.float64)
    diagnostic_codes.setflags(write=False)
    diagnostic_values.setflags(write=False)
    return CoreDiagnostics(codes=diagnostic_codes, values=diagnostic_values)
