# src/riskquant/credit/masters/factor_universe/ingest.py
"""
Credit factor master frame -> `CompiledCreditFactorUniverse`.

The factor edge: a human-facing ``pl.DataFrame`` is read column by column into
strict dtypes,
sorted by `factor_id` ascending, validated as one self-consistent universe,
and collapsed into a single immutable SoA container.

`factor_id` ascending sorted fixes the identity axis; UIX is the row position
after the sort and is owned by the engine. The same permutation is applied to
every column so the SoA stays aligned row-wise.

Authoring table
---------------
One row per factor. Columns:
    factor_id    : int   — stable identity, unique. Sorted ascending -> UIX.
                           An integer dtype, or a float dtype with integral
                           values; a value outside int32 raises TypeError.
    factor_name  : str   — human identity obligor ingest references, unique.
    factor_kind  : str   — a `CreditFactorKind` name (SYSTEMIC, CHI, ...).
                           String columns admit pl.Utf8, pl.Categorical and
                           pl.Enum; a null raises ValueError.
    nu           : float — OPTIONAL column. Required and > 0 for CHI rows;
                           null / NaN elsewhere. Mapped to the NO_CHI_NU
                           sentinel for non-CHI factors.
    alpha, beta  : float — OPTIONAL columns. Required and > 0 for BETA_PD rows;
                           null / NaN elsewhere. Mapped to the NO_BETA_PARAM
                           sentinel for non-BETA_PD factors.
"""

from __future__ import annotations

__all__ = [
    "compile_credit_factor_universe",
]

import numpy as np
import numpy.typing as npt
import polars as pl

from riskquant.constants import NO_BETA_PARAM, NO_CHI_NU
from riskquant.credit.exposure.add_on.ingest import (
    _int_column,
    _require_polars_frame,
)
from riskquant.credit.masters.enums.credit_factor_kind import (
    VALID_CREDIT_FACTOR_KIND_NAMES,
    CreditFactorKind,
)
from riskquant.credit.masters.factor_universe.compiled import (
    CompiledCreditFactorUniverse,
)
from riskquant.credit.masters.factor_universe.validate import (
    validate_credit_factor_universe_arrays,
)

_REQUIRED_COLS: tuple[str, ...] = (
    "factor_id",
    "factor_name",
    "factor_kind",
)
_PREVIEW = 10
_ENTRY = "compile_credit_factor_universe"

# Authored kind string -> int8 id (single source of truth = the enum).
_KIND_NAME_TO_ID: dict[str, int] = {k.name: np.int8(k.value) for k in CreditFactorKind}


def _check_required_columns(frame: pl.DataFrame) -> None:
    missing = [c for c in _REQUIRED_COLS if c not in frame.columns]
    if missing:
        raise ValueError(f"{_ENTRY}: missing required column(s) {missing}.")


def _str_values(frame: pl.DataFrame, col: str) -> list[str]:
    """A required string column as a list of Python str.

    Admits ``pl.Utf8``, ``pl.Categorical`` and ``pl.Enum`` (the factor-name
    rule of C.7, D094). A null raises ValueError; any other dtype raises
    TypeError. The pandas ``astype(str)`` path turned a null into the text
    "None" and an integer into its digits; both now fail at the edge.
    """
    series = frame.get_column(col)
    if not isinstance(series.dtype, pl.String | pl.Categorical | pl.Enum):
        raise TypeError(
            f"{_ENTRY}: column '{col}' has dtype {series.dtype!s}; a string column "
            "admits pl.Utf8, pl.Categorical or pl.Enum."
        )
    if series.null_count():
        raise ValueError(f"{_ENTRY}: null in required string column '{col}'.")
    return [str(v) for v in series.cast(pl.String).to_list()]


def _names(values: list[str]) -> npt.NDArray[np.str_]:
    # Fixed-width unicode SoA column (the v0.2.0 dtype, C.9).
    return np.ascontiguousarray(np.asarray(values, dtype=np.str_))


def _kind_ids(raw: list[str]) -> npt.NDArray[np.int8]:
    """Map authored kind strings to int8 ids; fail-fast on unknown names."""
    unknown = sorted(set(raw) - VALID_CREDIT_FACTOR_KIND_NAMES)
    if unknown:
        raise ValueError(
            f"{_ENTRY}: unknown factor_kind(s) "
            f"{unknown[:_PREVIEW]}. Valid: {sorted(VALID_CREDIT_FACTOR_KIND_NAMES)}."
        )
    return np.ascontiguousarray(
        np.array([_KIND_NAME_TO_ID[s] for s in raw], dtype=np.int8)
    )


def _optional_float(frame: pl.DataFrame, col: str) -> npt.NDArray[np.float64]:
    """An optional float column as float64 with NaN for every missing value.

    An absent column or a ``pl.Null`` column is all missing; a null (and, in a
    float column, a NaN) is missing. An integer dtype is admitted. Any other
    dtype (``pl.Boolean``, ``pl.Utf8``, ``pl.Decimal``) raises TypeError. The
    callers map NaN to the sentinel of the parameter, as the pandas path did.
    """
    n = frame.height
    if col not in frame.columns:
        return np.full(n, np.nan, dtype=np.float64)
    series = frame.get_column(col)
    if series.dtype == pl.Null:
        return np.full(n, np.nan, dtype=np.float64)
    if not (series.dtype.is_integer() or series.dtype.is_float()):
        raise TypeError(
            f"{_ENTRY}: column '{col}' has dtype {series.dtype!s}; an optional "
            "float column admits an integer or a float dtype, or pl.Null."
        )
    # Series.to_numpy maps null to NaN for numeric dtypes (F-12) and may return
    # a read-only view (F-25): copy into a new float64 array.
    return np.array(
        series.cast(pl.Float64).to_numpy(), dtype=np.float64, order="C", copy=True
    )


def _chi_nu(
    frame: pl.DataFrame, kind_ids: npt.NDArray[np.int8]
) -> npt.NDArray[np.float64]:
    """
    Map genuinely-missing nu to the NO_CHI_NU sentinel, preserving every
    authored value.

    Only a missing value (null or NaN) becomes the sentinel. A value authored on
    a non-CHI row is kept, NOT silently discarded, so `validate_chi_nu` can
    reject it as a likely kind-mislabel. A missing nu on a CHI row becomes the
    sentinel and is likewise rejected downstream (the sentinel is not > 0), with
    the offending factor id reported.
    """
    nu_raw = _optional_float(frame, "nu")
    return np.ascontiguousarray(np.where(np.isnan(nu_raw), NO_CHI_NU, nu_raw))


def _require_param_columns_for_kinds(
    frame: pl.DataFrame, factor_kind: npt.NDArray[np.int8]
) -> None:
    """Fail fast if a kind that needs a param column is present but the column
    is absent. Per-row value validity is still enforced downstream by
    `validate_chi_nu` / `validate_beta_params`; this just turns a "forgot the
    column entirely" schema mistake into a crisp, specific error."""
    if (
        np.any(factor_kind == np.int8(CreditFactorKind.CHI))
        and "nu" not in frame.columns
    ):
        raise ValueError(
            "compile_credit_factor_universe: universe has CHI factor(s) but no "
            "'nu' column. CHI factors require their degrees of freedom."
        )
    if np.any(factor_kind == np.int8(CreditFactorKind.BETA_PD)):
        missing = [c for c in ("alpha", "beta") if c not in frame.columns]
        if missing:
            raise ValueError(
                "compile_credit_factor_universe: universe has BETA_PD factor(s) "
                f"but no {missing} column(s). BETA_PD factors require their Beta "
                "shape parameters."
            )


def _beta_param(frame: pl.DataFrame, col: str) -> npt.NDArray[np.float64]:
    """
    Map a genuinely-missing Beta shape parameter to the NO_BETA_PARAM sentinel,
    preserving every authored value.

    Only a missing value (null or NaN) becomes the sentinel. A value authored on
    a non-BETA_PD row is kept, NOT silently discarded, so `validate_beta_params`
    can reject it as a likely kind-mislabel; a missing value on a BETA_PD row
    becomes the sentinel and is likewise rejected downstream.
    """
    raw = _optional_float(frame, col)
    return np.ascontiguousarray(np.where(np.isnan(raw), NO_BETA_PARAM, raw))


def compile_credit_factor_universe(
    factor_df: pl.DataFrame,
) -> CompiledCreditFactorUniverse:
    """
    Compile a validated credit factor-universe frame into
    `CompiledCreditFactorUniverse`.

    Parameters
    ----------
    factor_df
        ``pl.DataFrame``, one row per factor. Required columns: factor_id,
        factor_name, factor_kind. Optional `nu` column carries the chi degrees
        of freedom (required for CHI rows, null otherwise); optional `alpha`
        and `beta` columns carry the Beta shape parameters (required for
        BETA_PD rows, null otherwise). The caller's frame is not changed.

    Returns
    -------
    CompiledCreditFactorUniverse
        `factor_ids` sorted ascending; all other columns reordered by the same
        permutation. UIX == row position. Every array is C-contiguous and
        writeable.

    Raises
    ------
    TypeError
        ``factor_df`` is not a ``pl.DataFrame``; a column dtype that its rule
        does not admit; a fractional `factor_id` or one outside int32.
    ValueError
        An empty frame; a missing required column; a null in a required
        column; an unknown factor kind; a failed universe validation.
    """
    frame = _require_polars_frame(factor_df, entry=_ENTRY, arg="factor_df")
    if frame.height == 0:
        raise ValueError(f"{_ENTRY}: cannot compile an empty frame.")

    _check_required_columns(frame)

    ids_raw = _int_column(frame, "factor_id", np.int32, entry=_ENTRY)
    names_raw = _str_values(frame, "factor_name")
    kinds_raw = _str_values(frame, "factor_kind")

    # One stable permutation by the identity axis applied to every column (C.9).
    order = np.argsort(ids_raw, kind="stable")
    factor_ids = np.ascontiguousarray(ids_raw[order])
    factor_names = _names([names_raw[i] for i in order])
    factor_kind = _kind_ids([kinds_raw[i] for i in order])

    # Param columns stay OPTIONAL, but a column is REQUIRED the moment a factor
    # of its kind appears: a CHI factor without a 'nu' column, or a BETA_PD
    # factor without 'alpha'/'beta', is a schema mistake worth a crisp error
    # here rather than the value-level "missing/non-positive" downstream. The
    # structural requirement follows from the content, so simple universes
    # (no chi / no beta) need carry no param columns at all.
    _require_param_columns_for_kinds(frame, factor_kind)

    factor_chi_nu = np.ascontiguousarray(_chi_nu(frame, factor_kind)[order])
    factor_beta_alpha = np.ascontiguousarray(_beta_param(frame, "alpha")[order])
    factor_beta_beta = np.ascontiguousarray(_beta_param(frame, "beta")[order])

    validate_credit_factor_universe_arrays(
        factor_ids=factor_ids,
        factor_names=factor_names,
        factor_kind=factor_kind,
        factor_chi_nu=factor_chi_nu,
        factor_beta_alpha=factor_beta_alpha,
        factor_beta_beta=factor_beta_beta,
    )

    return CompiledCreditFactorUniverse(
        factor_ids=factor_ids,
        factor_names=factor_names,
        factor_kind=factor_kind,
        factor_chi_nu=factor_chi_nu,
        factor_beta_alpha=factor_beta_alpha,
        factor_beta_beta=factor_beta_beta,
    )
