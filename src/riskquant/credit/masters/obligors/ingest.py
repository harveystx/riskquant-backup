# src/riskquant/credit/masters/obligors/ingest.py
"""
Obligor master frame -> `CompiledObligorCredit`.

The obligor edge: a human-facing ``pl.DataFrame`` is read column by column into
strict dtypes, sorted by uid ascending, validated as one self-consistent
universe, and collapsed into a single immutable SoA container.

`uid` ascending sorted is a structural invariant the downstream layers rely on
for vectorized joins (`searchsorted` in `compile_jtd_portfolio`,
`aggregate_ead_to_obligor`, and `_resolve_collateral`). One stable permutation,
``np.argsort(uid, kind="stable")``, is applied to every column so the SoA stays
aligned row-wise (WS_CREDIT_POLARS C.9).

Column rules (WS_CREDIT_POLARS C.8): ``uid`` is an integer column (an integer
dtype, or a float dtype with integral values); ``pd`` and ``lgd`` are float
columns; ``rating_id``, ``sector_id``, ``country_id`` and ``parent_uid`` are
optional integer columns whose nulls compile to the sentinels of
`riskquant.constants`.
"""

from __future__ import annotations

__all__ = [
    "compile_obligor_master",
]

from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl

from riskquant.constants import (
    NO_COUNTRY_ID,
    NO_PARENT_OBLIGOR,
    NO_RATING_ID,
    NO_SECTOR_ID,
)
from riskquant.credit.exposure.add_on.ingest import (
    _float_column,
    _int_column,
    _require_polars_frame,
)
from riskquant.credit.masters.obligors.compiled import CompiledObligorCredit
from riskquant.credit.masters.obligors.validate import validate_obligor_master_arrays

_ENTRY = "compile_obligor_master"

_REQUIRED_COLS: tuple[str, ...] = (
    "uid",
    "pd",
    "lgd",
)


def _check_required_columns(frame: pl.DataFrame) -> None:
    missing = [c for c in _REQUIRED_COLS if c not in frame.columns]
    if missing:
        raise ValueError(f"{_ENTRY}: missing required column(s) {missing}.")


def _optional_int_column(
    frame: pl.DataFrame,
    col: str,
    dtype: type[np.int8] | type[np.int32] | type[np.int64],
    sentinel: int,
) -> npt.NDArray[Any]:
    """An optional integer column as a new writeable, C-contiguous array.

    An absent column or an all-null ``pl.Null`` column compiles to ``sentinel``
    for every row. A null cell (in a float column: null or NaN) compiles to
    ``sentinel``. An integral float compiles to its integer. A fractional or an
    infinite value, or a value outside the range of ``dtype``, raises TypeError:
    the pandas ``astype("Int8")`` path raised for 3.5, and a silent cast would
    truncate or wrap (C.8, F-11). Any other dtype raises TypeError.
    """
    n = frame.height
    out = np.full(n, sentinel, dtype=dtype)
    if col not in frame.columns:
        return out
    series = frame.get_column(col)
    if series.dtype == pl.Null:
        return out
    info = np.iinfo(dtype)
    if series.dtype.is_integer():
        present = ~series.is_null().to_numpy()
        values = series.drop_nulls()
        if values.len():
            low, high = int(values.min()), int(values.max())  # type: ignore[arg-type]
            if not (info.min <= low and high <= info.max):
                raise TypeError(
                    f"{_ENTRY}: column '{col}' holds a value outside the range of "
                    f"{np.dtype(dtype).name} (min {low}, max {high})."
                )
        out[present] = values.to_numpy().astype(dtype)
        return out
    if series.dtype.is_float():
        raw = np.asarray(series.to_numpy(), dtype=np.float64)  # null -> NaN (F-12)
        present = ~np.isnan(raw)
        kept = raw[present]
        with np.errstate(all="ignore"):
            integral = np.isfinite(kept) & (kept == np.trunc(kept))
            # "< max + 1": float(int64 max) rounds up to 2**63 (see _int_column).
            in_range = (kept >= float(info.min)) & (kept < float(info.max) + 1.0)
        if not integral.all():
            raise TypeError(
                f"{_ENTRY}: column '{col}' holds a non-integral value "
                f"{kept[~integral][0]!r}; an integer column admits integral "
                "floats only."
            )
        if not in_range.all():
            raise TypeError(
                f"{_ENTRY}: column '{col}' holds a value outside the range of "
                f"{np.dtype(dtype).name} ({kept[~in_range][0]!r})."
            )
        out[present] = kept.astype(dtype)
        return out
    raise TypeError(
        f"{_ENTRY}: column '{col}' has dtype {series.dtype!s}; an optional integer "
        "column admits an integer dtype, a float dtype with integral values, or "
        "pl.Null."
    )


def compile_obligor_master(
    obligor_df: pl.DataFrame,
) -> CompiledObligorCredit:
    """
    Compile a validated obligor-master frame into `CompiledObligorCredit`.

    Parameters
    ----------
    obligor_df
        ``pl.DataFrame``, one row per obligor. Required columns: ``uid``,
        ``pd``, ``lgd``. Optional columns: ``rating_id``, ``sector_id``,
        ``country_id``, ``parent_uid``; an absent column, a ``pl.Null`` column
        or a null cell maps to ``NO_RATING_ID``, ``NO_SECTOR_ID``,
        ``NO_COUNTRY_ID`` or ``NO_PARENT_OBLIGOR``. Other columns are ignored.
        The caller's frame is not changed.

    Returns
    -------
    CompiledObligorCredit
        `uid` sorted ascending; all other columns reordered by the same
        permutation to stay row-aligned. Every array is C-contiguous and
        writeable.

    Raises
    ------
    TypeError
        ``obligor_df`` is not a ``pl.DataFrame``; a column dtype that its rule
        does not admit; a fractional or out-of-range integer value.
    ValueError
        An empty frame; a missing required column; a null in a required
        column; a NaN in ``pd`` or ``lgd``; a failed master validation.
    """
    frame = _require_polars_frame(obligor_df, entry=_ENTRY, arg="obligor_df")
    if frame.height == 0:
        raise ValueError(f"{_ENTRY}: cannot compile an empty frame.")

    _check_required_columns(frame)

    uid_raw = _int_column(frame, "uid", np.int64, entry=_ENTRY)
    pd_raw = _float_column(frame, "pd", entry=_ENTRY)
    lgd_raw = _float_column(frame, "lgd", entry=_ENTRY)
    rating_raw = _optional_int_column(frame, "rating_id", np.int8, int(NO_RATING_ID))
    sector_raw = _optional_int_column(frame, "sector_id", np.int32, int(NO_SECTOR_ID))
    country_raw = _optional_int_column(
        frame, "country_id", np.int32, int(NO_COUNTRY_ID)
    )
    parent_raw = _optional_int_column(
        frame, "parent_uid", np.int64, int(NO_PARENT_OBLIGOR)
    )

    # One stable permutation by uid, applied to every column (C.9). Fancy
    # indexing returns new, writeable, C-contiguous arrays.
    order = np.argsort(uid_raw, kind="stable")
    uid = np.ascontiguousarray(uid_raw[order])
    pd_ = np.ascontiguousarray(pd_raw[order])
    lgd = np.ascontiguousarray(lgd_raw[order])
    rating_id = np.ascontiguousarray(rating_raw[order])
    sector_id = np.ascontiguousarray(sector_raw[order])
    country_id = np.ascontiguousarray(country_raw[order])
    parent_uid = np.ascontiguousarray(parent_raw[order])

    validate_obligor_master_arrays(
        uid=uid,
        pd=pd_,
        lgd=lgd,
        rating_id=rating_id,
        sector_id=sector_id,
        country_id=country_id,
        parent_uid=parent_uid,
    )

    return CompiledObligorCredit(
        uid=uid,
        pd=pd_,
        lgd=lgd,
        rating_id=rating_id,
        sector_id=sector_id,
        country_id=country_id,
        parent_uid=parent_uid,
    )
