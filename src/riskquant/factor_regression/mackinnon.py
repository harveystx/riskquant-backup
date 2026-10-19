# src/riskquant/factor_regression/mackinnon.py
"""
MacKinnon (2010) N = 1 critical values of the Dickey-Fuller test (A.2, C.7). PRIVATE.

Source: MacKinnon, J. G. (2010), "Critical Values for Cointegration Tests",
Queen's Economics Department Working Paper 1227, appendix Table 2 (tau_c,
printed page 13) and Table 3 (tau_ct, printed page 14), rows N = 1. N = 1
means one series: the test regression estimates no cointegrating vector, so
a combination with declared weights uses these rows (F-8).

The response surface is ``c(n) = b_inf + b1/n + b2/n**2 + b3/n**3``, with
``n`` the number of observations in the test regression (A.1, C.6). An empty
table cell is stored as ``0.0``. The 23 constants exist once, in
``_COEFFICIENTS`` (C.7).

Validity. The source simulated ``n`` from 20 to 1400; a value for a smaller
``n`` lies outside that range, and ``adf.py`` rejects ``n < 20`` (C.6). The
finite-n values are strictly valid for IID normal errors; with lagged
differences the use of ``n`` is a convention. tau_c assumes no drift under
the null (West 1988); tau_ct does not need that assumption.

The core sees no name (C.11). The table selector is an ``int`` code; slice S3
maps its deterministic codes to ``TABLE_TAU_C`` and ``TABLE_TAU_CT``.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

TABLE_TAU_C: Final = 0
TABLE_TAU_CT: Final = 1


def _read_only(array: NDArray[np.float64]) -> NDArray[np.float64]:
    array.setflags(write=False)
    return array


# Axis 0: table (TABLE_TAU_C, TABLE_TAU_CT). Axis 1: level 0.01, 0.05, 0.10.
# Axis 2: coefficient b_inf, b1, b2, b3. Typed from the PDF (A.2); each cell
# was compared with the page images of Tables 2 and 3 (S2 report).
_COEFFICIENTS: Final[NDArray[np.float64]] = _read_only(
    np.array(
        [
            [
                [-3.43035, -6.5393, -16.786, -79.433],
                [-2.86154, -2.8903, -4.234, -40.040],
                [-2.56677, -1.5384, -2.809, 0.0],
            ],
            [
                [-3.95877, -9.0531, -28.428, -134.155],
                [-3.41049, -4.3904, -9.036, -45.374],
                [-3.12705, -2.5856, -3.925, -22.380],
            ],
        ],
        dtype=np.float64,
    )
)


def _declared_integer(value: int) -> int:
    # bool is an int subclass and np.bool_ converts to 0 or 1: a flag is not a
    # table code or a row count. A float or str is never rounded or parsed.
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(
            f"an integer argument must be an int or a numpy integer, got "
            f"{type(value).__name__}"
        )
    return int(value)


def critical_values(table: int, n: int) -> NDArray[np.float64]:
    """
    Critical values ``c(n)`` at the levels 0.01, 0.05 and 0.10 (A.2, C.7).

    Parameters
    ----------
    table
        ``TABLE_TAU_C`` (constant only) or ``TABLE_TAU_CT`` (constant and
        linear trend). An ``int`` or a numpy integer.
    n
        Number of observations in the test regression, ``n >= 1``. An ``int``
        or a numpy integer. The source simulated ``n`` from 20 to 1400, so a
        value for a smaller ``n`` lies outside the simulated range; the rule
        ``n >= 20`` belongs to ``adf.py`` (C.6).

    Returns
    -------
    NDArray[np.float64]
        New read-only ``(3,)`` array: ``c(n)`` at 0.01, 0.05, 0.10, in this
        order. Each value is a lower-tail critical value of tau.

    Raises
    ------
    TypeError
        ``table`` or ``n`` is a ``bool``, ``np.bool_``, ``float``, ``str`` or
        another type that is not an ``int`` or a numpy integer.
    ValueError
        ``table`` is not ``TABLE_TAU_C`` or ``TABLE_TAU_CT``, or ``n < 1``.
    OverflowError
        ``n`` is above the largest finite float64.

    Notes
    -----
    Each power of ``1/n`` divides the coefficient by a float64 power of ``n``,
    as A.2 writes the formula. All terms of the six rows are negative, so the
    sum has no cancellation; S2.T5 bounds the relative error by 1e-15 against
    an exact rational evaluation.
    """
    table_code = _declared_integer(table)
    count = _declared_integer(n)
    if table_code not in (TABLE_TAU_C, TABLE_TAU_CT):
        raise ValueError(
            f"table must be TABLE_TAU_C ({TABLE_TAU_C}) or TABLE_TAU_CT "
            f"({TABLE_TAU_CT}), got {table_code}"
        )
    if count < 1:
        raise ValueError(f"n must be at least 1, got {count}")
    n_1 = float(count)
    n_2 = n_1 * n_1
    n_3 = n_2 * n_1
    rows = _COEFFICIENTS[table_code]
    out = rows[:, 0] + rows[:, 1] / n_1 + rows[:, 2] / n_2 + rows[:, 3] / n_3
    return _read_only(np.ascontiguousarray(out, dtype=np.float64))
