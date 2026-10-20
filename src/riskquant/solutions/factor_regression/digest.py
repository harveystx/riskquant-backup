# src/riskquant/solutions/factor_regression/digest.py
"""
Digest functions for the factor-regression library (A.6, C.16, C.21).

This is the only facade module that imports
``riskquant.solutions.run_identity.hashing`` (C.21).  No other module in
``riskquant.solutions.factor_regression`` may import it.

All four functions delegate to ``digest_named_arrays``, which folds in the
array name, dtype, shape and C-order bytes via SHA-256 (F-4).  The order and
names of the arrays in each call are specified in A.6 and are canonical: any
change of order or name changes the digest.

Canonical byte layout summary (A.6)
------------------------------------
``row_digest(start, end)``
    Arrays: ``row_start_date_ord`` int32, ``row_end_date_ord`` int32.

``sample_digest(start, end, response_name, intercept, regressor_names, y, x)``
    Arrays: ``row_start_date_ord`` int32, ``row_end_date_ord`` int32,
    ``column_names`` uint8 (see below), ``y`` float64, ``X`` float64 C-order.
    ``column_names`` text: lines ``Y\\x1f{response_name}``, ``C\\x1f`` (if
    intercept), then ``X\\x1f{name}`` per regressor, joined by ``\\x1e``.

``spec_digest(...)``
    Arrays in this order: ``schema_version`` int64[1] (constant 1),
    ``engine_version`` uint8 (UTF-8), ``terms`` uint8 (one line
    ``Y\\x1f{name}\\x1f{unit}\\x1f{TRANSFORM}`` for the response, then
    ``X\\x1f{name}\\x1f{unit}\\x1f{TRANSFORM}`` per regressor, joined by
    ``\\x1e``), ``intercept`` int8[0 or 1], ``step`` int64[h],
    ``stride`` int64[s], ``window`` int32[start_ord, end_ord],
    ``covariance_kind`` int8[0/1/2], ``covariance_lag`` int64[L or -1],
    ``rank_rtol`` float64[x], ``condition_threshold`` float64[x] or empty
    float64 when None.

``restriction_digest(restriction, rhs)``
    Arrays: ``restriction`` float64 C-order, ``rhs`` float64 C-order.

Stability
---------
Digests are stable across Python processes and platforms as long as
``digest_named_arrays`` is stable (SHA-256, native byte order normalised,
C-order bytes folded in) (C.16).

Input contract
--------------
No function casts, truncates or reshapes its input.  Each array argument must
be exactly ``numpy.ndarray`` with its A.6 dtype in native byte order, else
``TypeError``.  ``row_digest`` and ``sample_digest`` check the row shapes,
else ``ValueError``.  ``spec_digest`` raises ``ValueError`` when the regressor
name, unit and transform sequences differ in length.  ``restriction_digest``
checks type and dtype only: shape errors belong to the Wald core (C.18).
Valid inputs digest exactly as before these checks: the arrays pass to
``digest_named_arrays`` unchanged, and that function folds in the dtype, the
shape and the C-order bytes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
from numpy.typing import NDArray

from riskquant.factor_regression import codes
from riskquant.solutions.run_identity.hashing import digest_named_arrays

# Schema version for the spec bytes.  Increment only when the canonical byte
# layout of spec_digest changes.
_SCHEMA_VERSION: Final = np.array([1], dtype=np.int64)


def _require_array(name: str, arr: object, dtype: type[np.generic]) -> None:
    """
    Raise ``TypeError`` unless ``arr`` is exactly ``numpy.ndarray`` with dtype
    ``dtype`` in native byte order.

    The type check runs first: a subclass can redefine ``dtype`` access and
    reductions.  The dtype check replaces a cast, because a cast would digest
    values that the caller did not supply (R-017).
    """
    if type(arr) is not np.ndarray:
        raise TypeError(
            f"{name} must be exactly numpy.ndarray; got {type(arr).__name__}"
        )
    expected = np.dtype(dtype)
    if arr.dtype != expected or not arr.dtype.isnative:
        raise TypeError(
            f"{name} must have dtype {expected} in native byte order; "
            f"got {arr.dtype.str}"
        )


def _require_rows(
    row_start_date_ord: NDArray[np.int32], row_end_date_ord: NDArray[np.int32]
) -> int:
    """
    Return the row count ``R`` after a shape check of the row date arrays.

    Raises
    ------
    ValueError
        A row array is not 1-D, or the two lengths differ.
    """
    if row_start_date_ord.ndim != 1 or row_end_date_ord.ndim != 1:
        raise ValueError(
            f"row date arrays must be 1-D; got ndim "
            f"{row_start_date_ord.ndim} and {row_end_date_ord.ndim}"
        )
    n_rows = int(row_start_date_ord.shape[0])
    if int(row_end_date_ord.shape[0]) != n_rows:
        raise ValueError(
            f"row date arrays must have equal length; got {n_rows} and "
            f"{int(row_end_date_ord.shape[0])}"
        )
    return n_rows


def _text_uint8(text: str) -> NDArray[np.uint8]:
    """Encode text as UTF-8 into a uint8 ndarray."""
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).copy()


def row_digest(
    row_start_date_ord: NDArray[np.int32],
    row_end_date_ord: NDArray[np.int32],
) -> str:
    """
    SHA-256 digest of the row interval arrays only (A.6, C.16).

    Two fits on the same ``AlignedLevels`` with equal ``step`` and ``stride``
    share a ``row_digest`` (C.17).

    Parameters
    ----------
    row_start_date_ord, row_end_date_ord
        ``int32`` ``(R,)`` arrays of row start and end date ordinals.

    Returns
    -------
    str
        64-character lowercase hex digest.

    Raises
    ------
    TypeError
        A row array is not exactly ``numpy.ndarray`` with native ``int32``.
    ValueError
        A row array is not 1-D, or the lengths differ.
    """
    _require_array("row_start_date_ord", row_start_date_ord, np.int32)
    _require_array("row_end_date_ord", row_end_date_ord, np.int32)
    _require_rows(row_start_date_ord, row_end_date_ord)
    return digest_named_arrays(
        row_start_date_ord=row_start_date_ord,
        row_end_date_ord=row_end_date_ord,
    )


def sample_digest(
    *,
    row_start_date_ord: NDArray[np.int32],
    row_end_date_ord: NDArray[np.int32],
    response_name: str,
    intercept: bool,
    regressor_names: Sequence[str],
    y: NDArray[np.float64],
    x: NDArray[np.float64],
) -> str:
    """
    SHA-256 digest of the row intervals, column names, response and design (A.6).

    Changes when the row dates, any column name, ``y`` or ``X`` change.  Unit
    and covariance parameters are not included (those live in ``spec_digest``).

    The ``column_names`` bytes are: ``Y\\x1f{response_name}``, then ``C\\x1f``
    (when intercept is True), then ``X\\x1f{name}`` per regressor in declared
    order, all joined by ``\\x1e``.

    Parameters
    ----------
    row_start_date_ord, row_end_date_ord
        ``int32`` ``(R,)`` row interval dates.
    response_name
        Name of the response series.
    intercept
        True when the design includes an intercept column.
    regressor_names
        Names of the regressor columns in declared order.
    y
        ``float64`` ``(R,)`` response values.
    x
        ``float64`` ``(R, k)`` design matrix (C-order), intercept column first
        if declared.

    Returns
    -------
    str
        64-character lowercase hex digest.

    Raises
    ------
    TypeError
        An array is not exactly ``numpy.ndarray`` with its A.6 dtype in native
        byte order (``int32`` row dates, ``float64`` ``y`` and ``x``).
    ValueError
        Row arrays not 1-D of equal length ``R``; ``y`` not 1-D of length
        ``R``; ``x`` not 2-D with ``R`` rows.
    """
    _require_array("row_start_date_ord", row_start_date_ord, np.int32)
    _require_array("row_end_date_ord", row_end_date_ord, np.int32)
    _require_array("y", y, np.float64)
    _require_array("x", x, np.float64)
    n_rows = _require_rows(row_start_date_ord, row_end_date_ord)
    if y.ndim != 1 or int(y.shape[0]) != n_rows:
        raise ValueError(f"y must be 1-D of length {n_rows}; got shape {y.shape}")
    if x.ndim != 2 or int(x.shape[0]) != n_rows:
        raise ValueError(f"x must be 2-D with {n_rows} rows; got shape {x.shape}")

    parts = [f"Y\x1f{response_name}"]
    if intercept:
        parts.append("C\x1f")
    for name in regressor_names:
        parts.append(f"X\x1f{name}")
    col_bytes = _text_uint8("\x1e".join(parts))

    return digest_named_arrays(
        row_start_date_ord=row_start_date_ord,
        row_end_date_ord=row_end_date_ord,
        column_names=col_bytes,
        y=y,
        X=x,
    )


def spec_digest(
    *,
    engine_version: str,
    response_name: str,
    response_unit: str,
    response_transform: str,
    regressor_names: Sequence[str],
    regressor_units: Sequence[str],
    regressor_transforms: Sequence[str],
    intercept: bool,
    step: int,
    stride: int,
    start_ord: int,
    end_ord: int,
    covariance_kind: int,
    covariance_lag: int,
    rank_rtol: float,
    condition_threshold: float | None,
) -> str:
    """
    SHA-256 digest of the canonical specification bytes (A.6, C.16).

    Changes when the engine version, any name, unit, transform, intercept flag,
    step, stride, window, covariance kind, lag, ``rank_rtol`` or
    ``condition_threshold`` change.

    Parameters
    ----------
    engine_version
        Canonical engine string (``contracts.ENGINE_VERSION``).
    response_name, response_unit, response_transform
        Name, unit and transform-name (e.g. ``"LOG_RETURN"``) of the response.
    regressor_names, regressor_units, regressor_transforms
        Parallel sequences for the regressors in declared order.
    intercept
        True when the design includes a constant column.
    step, stride
        Row-construction parameters ``h`` and ``s``.
    start_ord, end_ord
        Alignment window bounds.
    covariance_kind
        Integer code (0=IID, 1=HC0, 2=HAC_BARTLETT).
    covariance_lag
        Lag ``L`` for HAC_BARTLETT; ``-1`` when not applicable (IID or HC0).
    rank_rtol
        Declared rank tolerance.
    condition_threshold
        Declared threshold or ``None``.  ``None`` digests as an empty
        ``float64`` array.

    Returns
    -------
    str
        64-character lowercase hex digest.

    Raises
    ------
    ValueError
        ``regressor_names``, ``regressor_units`` and ``regressor_transforms``
        differ in length.  A shorter sequence would drop terms from the
        digest without notice.

    Notes
    -----
    The ``terms`` uint8 array encodes:
    ``Y\\x1f{name}\\x1f{unit}\\x1f{TRANSFORM}`` for the response, then one
    ``X\\x1f{name}\\x1f{unit}\\x1f{TRANSFORM}`` line per regressor, all lines
    joined by ``\\x1e``.
    """
    lengths = (
        len(regressor_names),
        len(regressor_units),
        len(regressor_transforms),
    )
    if len(set(lengths)) != 1:
        raise ValueError(
            "regressor_names, regressor_units and regressor_transforms must "
            f"have equal length; got {lengths}"
        )

    # Build the terms text.
    lines = [f"Y\x1f{response_name}\x1f{response_unit}\x1f{response_transform}"]
    for n, u, t in zip(
        regressor_names, regressor_units, regressor_transforms, strict=True
    ):
        lines.append(f"X\x1f{n}\x1f{u}\x1f{t}")
    terms_arr = _text_uint8("\x1e".join(lines))

    engine_arr = _text_uint8(engine_version)
    intercept_arr = np.array([1 if intercept else 0], dtype=np.int8)
    step_arr = np.array([step], dtype=np.int64)
    stride_arr = np.array([stride], dtype=np.int64)
    window_arr = np.array([start_ord, end_ord], dtype=np.int32)
    cov_kind_arr = np.array([covariance_kind], dtype=np.int8)
    cov_lag_arr = np.array([covariance_lag], dtype=np.int64)
    rank_rtol_arr = np.array([rank_rtol], dtype=np.float64)
    if condition_threshold is None:
        ct_arr: NDArray[np.float64] = np.empty(0, dtype=np.float64)
    else:
        ct_arr = np.array([condition_threshold], dtype=np.float64)

    return digest_named_arrays(
        schema_version=_SCHEMA_VERSION,
        engine_version=engine_arr,
        terms=terms_arr,
        intercept=intercept_arr,
        step=step_arr,
        stride=stride_arr,
        window=window_arr,
        covariance_kind=cov_kind_arr,
        covariance_lag=cov_lag_arr,
        rank_rtol=rank_rtol_arr,
        condition_threshold=ct_arr,
    )


def restriction_digest(
    restriction: NDArray[np.float64],
    rhs: NDArray[np.float64],
) -> str:
    """
    SHA-256 digest of a Wald restriction pair (A.6, C.18).

    Hashes the arrays named ``"restriction"`` (float64, C-order) and
    ``"rhs"`` (float64, C-order) in this order.  Any change of a single
    element, shape or the order of the two arrays changes the digest.

    Parameters
    ----------
    restriction
        ``float64`` ``(J, K)`` restriction matrix.
    rhs
        ``float64`` ``(J,)`` right-hand side vector.

    Returns
    -------
    str
        64-character lowercase hex digest.

    Raises
    ------
    TypeError
        An array is not exactly ``numpy.ndarray`` with native ``float64``.
        Shapes are not checked here: shape errors belong to the Wald core
        (C.18).
    """
    _require_array("restriction", restriction, np.float64)
    _require_array("rhs", rhs, np.float64)
    return digest_named_arrays(restriction=restriction, rhs=rhs)


# ===================================================================== S4
# Digests for the unit-root test (A.3, C.9).

# Schema version for the unit-root spec bytes.  Increment only when the
# canonical byte layout of unit_root_spec_digest changes.
_UNIT_ROOT_SCHEMA_VERSION: Final = np.array([1], dtype=np.int64)

# TEST_ADF code, baked in.  The digest always encodes ADF; the schema
# version changes if a second test kind is ever added.
_TEST_ADF_ARR: Final = np.array([int(codes.TEST_ADF)], dtype=np.int8)


def unit_root_spec_digest(
    *,
    engine_version: str,
    term_names: Sequence[str],
    term_units: Sequence[str],
    weights: NDArray[np.float64],
    scale: int,
    deterministic: int,
    lag: int,
    rank_rtol: float,
    start_ord: int,
    end_ord: int,
    critical_value_source: str,
) -> str:
    """
    SHA-256 digest of the unit-root specification bytes (A.3, C.9).

    Changes when the engine version, any term name or unit, the weight
    vector, the scale, the deterministic case, the lag, ``rank_rtol``, the
    window, or the critical-value source change.

    The ``terms`` uint8 array encodes one line
    ``T\x1f{name}\x1f{unit}`` per term, joined by ``\x1e``.

    Parameters
    ----------
    engine_version
        Canonical engine string (``contracts.ENGINE_VERSION``).
    term_names, term_units
        Term names and units in declared order.  Must have the same length
        as ``weights``.
    weights
        Exactly ``numpy.ndarray``, native ``float64``, ``(n_terms,)``:
        the declared weight of each term.
    scale
        Integer scale code (``codes.SCALE_LEVEL`` or ``codes.SCALE_LOG``).
    deterministic
        Integer deterministic code (``codes.DET_CONSTANT`` or
        ``codes.DET_CONSTANT_TREND``).
    lag
        Declared ADF lag ``p``.
    rank_rtol
        Declared rank tolerance.
    start_ord, end_ord
        Alignment window bounds from ``AlignedLevels``.
    critical_value_source
        Fixed text string from ``labels.CRITICAL_VALUE_SOURCES``.

    Returns
    -------
    str
        64-character lowercase hex digest.

    Raises
    ------
    TypeError
        ``weights`` is not exactly ``numpy.ndarray`` with native ``float64``.
    ValueError
        ``term_names`` and ``term_units`` do not have the same length as
        ``weights``.
    """
    # Exact-type check runs before any reduction (note 4, C.9).
    _require_array("weights", weights, np.float64)
    if weights.ndim != 1:
        raise ValueError(f"weights must be 1-D; got ndim {weights.ndim}")
    n_terms = int(weights.shape[0])
    if len(term_names) != n_terms or len(term_units) != n_terms:
        raise ValueError(
            "term_names, term_units and weights must have equal length; "
            f"got {len(term_names)}, {len(term_units)}, {n_terms}"
        )

    terms_text = "\x1e".join(f"T\x1f{n}\x1f{u}" for n, u in zip(term_names, term_units))

    return digest_named_arrays(
        schema_version=_UNIT_ROOT_SCHEMA_VERSION,
        engine_version=_text_uint8(engine_version),
        test_kind=_TEST_ADF_ARR,
        terms=_text_uint8(terms_text),
        weights=weights,
        scale=np.array([scale], dtype=np.int8),
        deterministic=np.array([deterministic], dtype=np.int8),
        lag=np.array([lag], dtype=np.int64),
        rank_rtol=np.array([rank_rtol], dtype=np.float64),
        window=np.array([start_ord, end_ord], dtype=np.int32),
        critical_value_source=_text_uint8(critical_value_source),
    )


def unit_root_sample_digest(
    *,
    row_start_date_ord: NDArray[np.int32],
    row_end_date_ord: NDArray[np.int32],
    y: NDArray[np.float64],
    x: NDArray[np.float64],
    regressor_names: Sequence[str],
) -> str:
    """
    SHA-256 digest of the unit-root sample arrays (A.3, C.9).

    Delegates to ``sample_digest`` with ``response_name="dz"`` and
    ``intercept=True``, so the ``column_names`` bytes are:
    ``Y\x1fdz``, ``C\x1f``, then ``X\x1f{label}`` per label in
    ``regressor_names``, all joined by ``\x1e``.

    Changes when the row dates, any column label, the response vector ``y``
    or the design matrix ``X`` change.  ``rank_rtol`` and other spec
    parameters are not included (those live in ``unit_root_spec_digest``).

    Parameters
    ----------
    row_start_date_ord, row_end_date_ord
        Exactly ``numpy.ndarray``, native ``int32``, ``(n,)``.
    y
        Exactly ``numpy.ndarray``, native ``float64``, ``(n,)``: ADF
        response ``dz``.
    x
        Exactly ``numpy.ndarray``, native ``float64``, ``(n, k)`` C-order:
        ADF design before column scaling.
    regressor_names
        ADF design column labels after the intercept (``"trend"`` when
        present, then ``"z_lag1"``, ``"dz_lag1"`` … ``"dz_lag<p>"``).

    Returns
    -------
    str
        64-character lowercase hex digest.

    Raises
    ------
    TypeError
        An array is not exactly ``numpy.ndarray`` with its declared dtype.
    ValueError
        Row arrays not 1-D of equal length; ``y`` not 1-D of that length;
        ``x`` not 2-D with that many rows.
    """
    return sample_digest(
        row_start_date_ord=row_start_date_ord,
        row_end_date_ord=row_end_date_ord,
        response_name="dz",
        intercept=True,
        regressor_names=regressor_names,
        y=y,
        x=x,
    )
