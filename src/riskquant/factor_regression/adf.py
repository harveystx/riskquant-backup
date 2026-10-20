# src/riskquant/factor_regression/adf.py
"""
Augmented Dickey-Fuller regression of a combination value (A.1, C.5 to C.7). PRIVATE.

Input: the combination value ``z`` on the aligned date axis (the output of
``combination.combine_core``), a declared lag ``p >= 0`` and a code of
``codes.DETERMINISTIC_CODES``. With ``m`` aligned dates (A.1)::

    dz_t = z_t - z_{t-1}                                   t = 1 .. m - 1
    dz_t = c [+ delta (t - p - 1)] + gamma z_{t-1}
           + sum_{q=1..p} phi_q dz_{t-q} + e_t             t = p + 1 .. m - 1

The regression has ``n = m - 1 - p`` rows and ``k = 1 + trend + 1 + p``
columns, in this order: ones; the trend ``0 .. n - 1`` (``DET_CONSTANT_TREND``
only); ``z_lag1`` (``z_{t-1}``); ``dz_lag1`` to ``dz_lag<p>``. The trend origin
does not change tau, because the ones column is present. Row ``r`` starts at
aligned index ``p + r`` and ends at aligned index ``p + r + 1``.

Solve (C.5). One call of ``fit.fit_core`` with an intercept, IID covariance
and covariance lag 0. ``tau = coef[j] / se[j]`` for the ``z_lag1`` column
``j`` (1 for ``DET_CONSTANT``, 2 for ``DET_CONSTANT_TREND``). ``se`` uses
``rss / (n - k)``: the DF convention of MacKinnon (2010) section 3 for N = 1.
The module adds no second solver. The Student t p-values of ``fit_core`` are
wrong for ``gamma``, so the result holds no p-value.

IID only. The lag augmentation absorbs serial correlation. HC0 or HAC changes
the statistic, and the MacKinnon tables do not apply to it.

Critical values (C.7). ``mackinnon.critical_values`` of the tau_c table
(``DET_CONSTANT``) or the tau_ct table (``DET_CONSTANT_TREND``) at the row
count ``n``. The tables start at ``n = 20`` (C.6), so ``n < 20`` raises before
any design is built and before ``fit_core`` runs.

Layout. ``z`` is read along its one contiguous axis. The differences ``dz``
are one ``(m - 1,)`` temporary. The design is allocated once as ``(n, k)``
C-order, one contiguous row per observation, as A.3 digests ``X``, and is
filled column by column from slices of ``z`` and ``dz``. The response is an
owned ``(n,)`` copy. No loop runs over the rows; the column loop runs ``p``
times. ``fit_core`` makes its own copies (design copy, scaled design, SVD).

Overflow. Every ``z_t`` is finite, but a difference ``z_t - z_{t-1}`` outside
the float64 range gives ``inf``. ``errstate`` keeps that silent, and
``fit_core`` rejects the non-finite design or response with
``CoreInputError`` ``INPUT_NON_FINITE`` (C.10 overflow case).

The core sees no name, unit or date (C.11). The result holds aligned indices;
the facade maps them to dates.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any, Final, TypeVar

import numpy as np
from numpy.typing import NDArray

from riskquant.factor_regression import codes, fit, mackinnon
from riskquant.factor_regression.errors import (
    CoreInputError,
    CoreParameterError,
    CoreSampleSizeError,
)

# Operand position of a CoreInputError raised by adf_core: z is the only
# array parameter.
OPERAND_Z: Final = 0

# C.6: the smallest sample size simulated by MacKinnon (2010).
_TABLE_MINIMUM_ROWS: Final = 20

_DETERMINISTIC_VALUES: Final = frozenset(
    int(code) for code in codes.DETERMINISTIC_CODES
)
_DET_CONSTANT_TREND: Final = int(codes.DET_CONSTANT_TREND)

_ScalarT = TypeVar("_ScalarT", bound=np.generic)


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class CoreAdf:
    """
    The result of one ADF regression (A.1, C.5 to C.7).

    Attributes
    ----------
    tau
        ``coef[j] / se[j]`` for the ``z_lag1`` column ``j``.
    gamma
        ``coef[j]``.
    coef, se
        ``float64`` ``(k,)``: coefficients and IID standard errors of every
        design column, in design order.
    critical_values
        ``float64`` ``(3,)``: MacKinnon (2010) ``c(n)`` at 0.01, 0.05, 0.10.
    condition_number
        ``kappa`` of the scaled design (``CoreScaledSvd.condition_number``).
    n_obs
        Row count ``n = m - 1 - p``, ``n >= 20``.
    n_columns
        Column count ``k``.
    lag
        The declared lag ``p``.
    deterministic
        The declared code of ``codes.DETERMINISTIC_CODES``.
    response
        ``float64`` ``(n,)``: the vector ``dz`` of the regression rows.
    design
        ``float64`` ``(n, k)`` C-order: the design before column scaling.
    row_start_index, row_end_index
        ``int32`` ``(n,)``: aligned index of the start date (``t - 1``) and
        of the end date (``t``) of each row.

    Notes
    -----
    All arrays are new and read-only and share no memory with the caller
    array (C.13). Every float is finite. The result holds no p-value and no
    verdict (D104.4). ``eq=False`` follows the D064 precedent for
    array-bearing containers.
    """

    tau: float
    gamma: float
    coef: NDArray[np.float64]
    se: NDArray[np.float64]
    critical_values: NDArray[np.float64]
    condition_number: float
    n_obs: int
    n_columns: int
    lag: int
    deterministic: int
    response: NDArray[np.float64]
    design: NDArray[np.float64]
    row_start_index: NDArray[np.int32]
    row_end_index: NDArray[np.int32]


def _read_only(array: NDArray[_ScalarT]) -> NDArray[_ScalarT]:
    array.setflags(write=False)
    return array


def _validate_z(z: NDArray[np.float64]) -> None:
    # Exact type, not isinstance, before the first reduction. numpy.ma skips
    # masked elements in .all(): a NaN under the mask passes the finite check.
    # The core converts nothing, so it rejects every ndarray subclass.
    if type(z) is not np.ndarray:
        raise CoreInputError(
            reason=codes.INPUT_NOT_NDARRAY, operand=OPERAND_Z, positions=()
        )
    if z.dtype != np.float64:
        raise CoreInputError(
            reason=codes.INPUT_WRONG_DTYPE, operand=OPERAND_Z, positions=()
        )
    if z.ndim != 1:
        raise CoreInputError(
            reason=codes.INPUT_WRONG_NDIM, operand=OPERAND_Z, positions=()
        )
    if int(z.shape[0]) == 0:
        raise CoreInputError(reason=codes.INPUT_EMPTY, operand=OPERAND_Z, positions=())
    # A non-finite z is the combination overflow case of C.10. The position is
    # the first NaN or inf.
    finite = np.isfinite(z)
    if not bool(finite.all()):
        first = int(np.flatnonzero(~finite)[0])
        raise CoreInputError(
            reason=codes.INPUT_NON_FINITE, operand=OPERAND_Z, positions=(first,)
        )


def _declared_lag(lag: int | np.integer[Any]) -> int:
    # bool is an int subclass, so operator.index reads True as 1; a flag is
    # not a lag. numpy.bool_ is rejected by the same check. operator.index
    # rejects a float or str with TypeError, so the core never rounds a lag.
    if isinstance(lag, (bool, np.bool_)):
        raise TypeError("lag must be an integer, not a bool")
    value = operator.index(lag)
    if value < 0:
        raise CoreParameterError(reason=codes.PARAM_NEGATIVE_LAG, value=float(value))
    return value


def _declared_deterministic(deterministic: int | np.integer[Any]) -> int:
    # Same type rule as the lag: no bool, no float, no str.
    if isinstance(deterministic, (bool, np.bool_)):
        raise TypeError("a deterministic code must be an integer, not a bool")
    code = operator.index(deterministic)
    if code not in _DETERMINISTIC_VALUES:
        # No reason code exists: the facade passes Deterministic member values
        # only, so this is an internal contract violation.
        raise ValueError(f"unknown deterministic code {code}")
    return code


def _build_regression(
    z: NDArray[np.float64], lag: int, has_trend: bool
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    The A.1 design ``(n, k)`` C-order and response ``(n,)``, both read-only.

    ``dz[i] = z[i + 1] - z[i]`` holds ``dz_{i+1}`` of A.1. Row ``r`` is the
    date ``t = p + 1 + r``, so the response is ``dz[p:]``, ``z_lag1`` is
    ``z[p : m - 1]`` and ``dz_lag q`` is ``dz[p - q : m - 1 - q]``.
    """
    m = int(z.shape[0])
    n_rows = m - 1 - lag
    column = 1 + int(has_trend)
    with np.errstate(over="ignore"):
        dz = z[1:] - z[:-1]
    response = np.array(dz[lag:], dtype=np.float64, copy=True)
    design = np.empty((n_rows, column + 1 + lag), dtype=np.float64)
    design[:, 0] = 1.0
    if has_trend:
        design[:, 1] = np.arange(n_rows, dtype=np.float64)
    design[:, column] = z[lag : m - 1]
    for q in range(1, lag + 1):
        design[:, column + q] = dz[lag - q : m - 1 - q]
    return _read_only(design), _read_only(response)


def adf_core(
    z: NDArray[np.float64],
    lag: int | np.integer[Any],
    deterministic: int | np.integer[Any],
    rank_rtol: float,
) -> CoreAdf:
    """
    Run the ADF regression of ``z`` and return tau and ``c(n)`` (A.1, C.5 to C.7).

    Parameters
    ----------
    z
        Exactly ``numpy.ndarray``, ``float64``, ``(m,)``, all finite: the
        combination value on the aligned date axis.
    lag
        The declared lag ``p >= 0``. An ``int`` or a numpy integer.
    deterministic
        ``codes.DET_CONSTANT`` or ``codes.DET_CONSTANT_TREND``. An ``int`` or
        a numpy integer.
    rank_rtol
        Relative rank tolerance, passed to ``fit.fit_core`` unchanged. Its
        checks are the checks of ``design.rank_gate``.

    Returns
    -------
    CoreAdf
        New read-only arrays. No caller array is changed or retained (C.13).

    Raises
    ------
    CoreInputError
        ``operand`` is ``OPERAND_Z``. ``INPUT_NOT_NDARRAY`` (any ndarray
        subclass, ``numpy.ma`` included), ``INPUT_WRONG_DTYPE``,
        ``INPUT_WRONG_NDIM`` and ``INPUT_EMPTY`` have empty ``positions``.
        ``INPUT_NON_FINITE``: ``positions`` holds the index of the first NaN
        or inf of ``z``. From ``fit_core``: ``INPUT_NON_FINITE`` on its
        design (operand 0) or response (operand 1) when a difference of two
        finite values of ``z`` overflows.
    TypeError
        ``lag`` or ``deterministic`` is a ``bool``, ``numpy.bool_``, or a type
        without an integer index (``float``, ``str``, ``None``). From
        ``fit_core``: ``rank_rtol`` is not a real number or is a ``bool``.
    CoreParameterError
        ``PARAM_NEGATIVE_LAG``: ``lag < 0``; ``value`` is the lag. From
        ``fit_core``: ``PARAM_RANK_RTOL_OUT_OF_RANGE``.
    ValueError
        ``deterministic`` is an integer that is not in
        ``codes.DETERMINISTIC_CODES``.
    CoreSampleSizeError
        ``SAMPLE_ROWS_BELOW_TABLE_MINIMUM``: ``n = m - 1 - p < 20``;
        ``n_available`` is ``n`` (zero or negative when ``p >= m - 1``),
        ``n_required`` is 20. From ``fit_core``:
        ``SAMPLE_ROWS_NOT_ABOVE_COLUMNS`` when ``n >= 20`` and ``n < k + 1``.
    CoreRankError, CoreDegenerateFitError
        From ``fit_core``, unchanged.

    Notes
    -----
    The checks run in this order:

    1. ``z``: type, dtype, ndim, empty, finite;
    2. ``lag``: type, then ``lag >= 0``;
    3. ``deterministic``: type, then membership;
    4. ``n >= 20``, with ``n`` computed with Python integers.

    Every check runs before the design is built and before ``fit_core``
    runs. When both ``n < 20`` and ``n < k + 1`` hold, rule 4 raises. Every
    error of ``fit_core`` propagates unchanged.
    """
    _validate_z(z)
    p = _declared_lag(lag)
    code = _declared_deterministic(deterministic)
    m = int(z.shape[0])
    n_rows = m - 1 - p
    if n_rows < _TABLE_MINIMUM_ROWS:
        raise CoreSampleSizeError(
            reason=codes.SAMPLE_ROWS_BELOW_TABLE_MINIMUM,
            n_available=n_rows,
            n_required=_TABLE_MINIMUM_ROWS,
        )

    has_trend = code == _DET_CONSTANT_TREND
    column = 1 + int(has_trend)
    design, response = _build_regression(z, p, has_trend)
    ols = fit.fit_core(
        design,
        response,
        intercept=True,
        rank_rtol=rank_rtol,
        covariance_kind=int(codes.IID),
        covariance_lag=0,
    )

    coef = _read_only(np.array(ols.coef, dtype=np.float64, copy=True))
    se = _read_only(np.array(ols.se, dtype=np.float64, copy=True))
    gamma = float(coef[column])
    table = mackinnon.TABLE_TAU_CT if has_trend else mackinnon.TABLE_TAU_C
    return CoreAdf(
        tau=gamma / float(se[column]),
        gamma=gamma,
        coef=coef,
        se=se,
        critical_values=mackinnon.critical_values(table, n_rows),
        condition_number=ols.gate.condition_number,
        n_obs=n_rows,
        n_columns=int(design.shape[1]),
        lag=p,
        deterministic=code,
        response=response,
        design=design,
        row_start_index=_read_only(np.arange(p, m - 1, dtype=np.int32)),
        row_end_index=_read_only(np.arange(p + 1, m, dtype=np.int32)),
    )
