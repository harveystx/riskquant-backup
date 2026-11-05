# src/riskquant/credit/exposure/add_on/ingest.py
"""
Supervisory add-on ingest (the edge).

`compile_add_on_factors` turns a config-authored ``subclass -> factor`` table into
a validated, UIX-sorted `CompiledAddOnFactors`. Fail-fast on anything that would
corrupt the lookup or the exposure: a frame that is not ``pl.DataFrame``, missing
columns, a column of the wrong dtype, null cells, fractional or overflowing
subclass ids, non-finite or negative factors, duplicate subclasses.

The private column readers of this module (`_require_polars_frame`,
`_int_column`, `_float_column`) are also the column rules of
`riskquant.credit.exposure.saccr.ingest` (WS_CREDIT_POLARS C.7, C.8).
"""

from __future__ import annotations

__all__ = [
    "compile_add_on_factors",
]

from typing import Any, cast, overload

import numpy as np
import numpy.typing as npt
import polars as pl

from riskquant.credit.exposure.add_on.compiled import CompiledAddOnFactors


def _require_polars_frame(frame: object, *, entry: str, arg: str) -> pl.DataFrame:
    """Return ``frame`` when it is a ``pl.DataFrame``; raise TypeError otherwise.

    Runs before any column is read (D094 form). The caller converts at its own
    boundary (``pl.from_pandas``, ``pl.from_dict``); a ``pl.LazyFrame`` must be
    collected by the caller, because the entry does not decide when to run it.
    """
    if not isinstance(frame, pl.DataFrame):
        kind = f"{type(frame).__module__}.{type(frame).__qualname__}"
        raise TypeError(
            f"{entry}: '{arg}' must be a pl.DataFrame (polars.DataFrame); "
            f"got {kind}. Convert at the call boundary (pl.from_pandas, "
            "pl.from_dict)."
        )
    return frame


def _require_columns(
    frame: pl.DataFrame, cols: tuple[str, ...], *, entry: str, what: str
) -> None:
    for col in cols:
        if col not in frame.columns:
            raise ValueError(f"{entry}: {what} missing required column '{col}'.")


@overload
def _int_column(
    frame: pl.DataFrame, col: str, dtype: type[np.int32], *, entry: str
) -> npt.NDArray[np.int32]: ...
@overload
def _int_column(
    frame: pl.DataFrame, col: str, dtype: type[np.int64], *, entry: str
) -> npt.NDArray[np.int64]: ...
def _int_column(
    frame: pl.DataFrame,
    col: str,
    dtype: type[np.int32] | type[np.int64],
    *,
    entry: str,
) -> npt.NDArray[Any]:
    """A required integer column as a new writeable, C-contiguous ``dtype`` array.

    Admits an integer dtype, or a float dtype whose values are integral (C.8).
    A null (in a float column: null or NaN) raises ValueError. A fractional
    value, an infinite value or a value outside the range of ``dtype`` raises
    TypeError: a silent ``astype`` would truncate 1.5 to 1 or wrap 2**40 to 0
    (F-11). Any other dtype (``pl.Boolean``, ``pl.Utf8``, ``pl.Decimal``,
    ``pl.Null``) raises TypeError.
    """
    series = frame.get_column(col)
    info = np.iinfo(dtype)
    if series.dtype.is_integer():
        if series.null_count():
            raise ValueError(f"{entry}: null in required integer column '{col}'.")
        if series.len():
            low, high = cast("int", series.min()), cast("int", series.max())
            if not (info.min <= low and high <= info.max):
                raise TypeError(
                    f"{entry}: column '{col}' holds a value outside the range of "
                    f"{np.dtype(dtype).name} (min {low}, max {high})."
                )
        values = series.to_numpy()
    elif series.dtype.is_float():
        if series.null_count() or bool(series.is_nan().any()):
            raise ValueError(
                f"{entry}: null or NaN in required integer column '{col}'."
            )
        values = series.to_numpy()
        with np.errstate(all="ignore"):
            integral = np.isfinite(values) & (values == np.trunc(values))
            # Upper bound as "< max + 1": float(int64 max) rounds up to 2**63,
            # so "<= max" would admit 2**63, which wraps on the cast.
            in_range = (values >= float(info.min)) & (values < float(info.max) + 1.0)
        if not integral.all():
            raise TypeError(
                f"{entry}: column '{col}' holds a non-integral value "
                f"{values[~integral][0]!r}; an integer column admits integral "
                "floats only."
            )
        if not in_range.all():
            raise TypeError(
                f"{entry}: column '{col}' holds a value outside the range of "
                f"{np.dtype(dtype).name} ({values[~in_range][0]!r})."
            )
    else:
        raise TypeError(
            f"{entry}: column '{col}' has dtype {series.dtype!s}; an integer column "
            "admits an integer dtype or a float dtype with integral values."
        )
    # copy=True: Series.to_numpy can return a read-only view (F-25, R-10).
    return np.array(values, dtype=dtype, order="C", copy=True)


def _float_column(
    frame: pl.DataFrame, col: str, *, entry: str
) -> npt.NDArray[np.float64]:
    """A required float column as a new writeable, C-contiguous float64 array.

    Admits an integer or a float dtype. A null or a NaN raises ValueError (in a
    float column, null means null or NaN). Any other dtype (``pl.Boolean``,
    ``pl.Utf8``, ``pl.Decimal``, ``pl.Null``) raises TypeError: Boolean is not
    numeric here (C.8).
    """
    series = frame.get_column(col)
    if series.dtype.is_integer():
        if series.null_count():
            raise ValueError(f"{entry}: null in required float column '{col}'.")
    elif series.dtype.is_float():
        if series.null_count() or bool(series.is_nan().any()):
            raise ValueError(f"{entry}: null or NaN in required float column '{col}'.")
    else:
        raise TypeError(
            f"{entry}: column '{col}' has dtype {series.dtype!s}; a float column "
            "admits an integer or a float dtype."
        )
    return np.array(series.to_numpy(), dtype=np.float64, order="C", copy=True)


def compile_add_on_factors(
    df: pl.DataFrame,
    *,
    subclass_col: str = "subclass_uix",
    factor_col: str = "factor",
) -> CompiledAddOnFactors:
    """Compile a subclass + factor table into a validated `CompiledAddOnFactors`.

    Rows are sorted by subclass here, so callers need not pre-sort. ``df`` must
    be a ``pl.DataFrame``; the caller's frame is not changed.

    Raises
    ------
    TypeError
        ``df`` is not a ``pl.DataFrame``; a column has a dtype that its rule
        does not admit; a ``subclass_col`` value is fractional or outside int32.
    ValueError
        A column is missing; a null (or NaN) cell; a non-finite or negative
        factor; a duplicate subclass.
    """
    entry = "compile_add_on_factors"
    frame = _require_polars_frame(df, entry=entry, arg="df")
    _require_columns(frame, (subclass_col, factor_col), entry=entry, what="df")

    subclass = _int_column(frame, subclass_col, np.int32, entry=entry)
    factor = _float_column(frame, factor_col, entry=entry)

    if not np.isfinite(factor).all():
        raise ValueError(
            f"compile_add_on_factors: non-finite value in factor column '{factor_col}'."
        )
    if (factor < 0.0).any():
        raise ValueError(
            f"compile_add_on_factors: negative add-on factor in '{factor_col}' "
            "(supervisory factors must be >= 0)."
        )
    if subclass.shape[0] != np.unique(subclass).shape[0]:
        raise ValueError("compile_add_on_factors: duplicate subclass_uix.")

    order = np.argsort(subclass, kind="stable")
    return CompiledAddOnFactors(
        subclass_uix=np.ascontiguousarray(subclass[order]),
        factor=np.ascontiguousarray(factor[order]),
    )
