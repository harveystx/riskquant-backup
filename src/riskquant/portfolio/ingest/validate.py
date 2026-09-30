# src/riskquant/portfolio/ingest/validate.py
"""
Semantic validation layer for materialized ingest dataframes.

Receives a ``pl.DataFrame`` that has been processed by:
    canonicalize_raw_ingest_df → materialize_canonicalized_ingest_df

Validates:
    - required column presence per instrument family
    - non-null constraints per family
    - currency format (alpha, 3 chars)
    - non-zero quantities
    - unique UIDs
    - instrument name resolution

Does NOT:
    - mutate the dataframe
    - coerce dtypes
    - perform RF mapping
    - perform market-data linkage validation (that is identity.py)

``validate_materialized_ingest_df`` (pandas-era dead code) has been removed.
The dispatcher supersedes it: see dispatcher.py.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import polars as pl

from riskquant.constants import POSITION_DUST_TOLERANCE
from riskquant.portfolio.instruments import (
    INSTRUMENT_NAME_TO_ID_REGISTRY,
    INSTRUMENT_TYPE_TO_FAMILY_ARRAY,
)

# ==============================================================================
# INTERNAL HELPERS
# ==============================================================================


def _require_cols(df: pl.DataFrame, cols: tuple[str, ...]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns:\n{missing}")


def _validate_non_null_masked(
    df: pl.DataFrame,
    mask: npt.NDArray[np.bool_],
    cols: tuple[str, ...],
    ctx: str,
) -> None:
    for col in cols:
        # is_null() → numpy bool array; mask out rows not in this family
        arr = df[col].is_null().to_numpy()
        bad = mask & arr
        if np.any(bad):
            idx = np.flatnonzero(bad)[:10].tolist()
            raise ValueError(f"{ctx} missing '{col}' at rows: {idx}")


def _validate_ccy_series(
    df: pl.DataFrame,
    col: str,
    mask: npt.NDArray[np.bool_],
    ctx: str,
) -> None:
    """
    Validate that non-null masked rows in a currency column are valid 3-letter
    ISO codes.

    After canonicalize, these strings are already uppercased; the alpha check
    accepts both upper and lower to be robust.
    """
    ser = df[col]
    not_null_arr = ~ser.is_null().to_numpy()
    valid_idx = np.flatnonzero(mask & not_null_arr)

    if len(valid_idx) == 0:
        return

    # Gather the relevant rows as a Polars sub-series.
    # Series.gather(int_array) selects by positional index.
    subset = ser.gather(valid_idx)

    # ALPHA CHECK
    alpha = subset.str.contains(r"^[A-Za-z]+$", strict=False)
    if not alpha.all():
        bad_local = np.flatnonzero(~alpha.to_numpy())
        idx = valid_idx[bad_local].tolist()[:10]
        raise ValueError(f"{ctx} invalid currency (non-alpha) at rows: {idx}")

    # LENGTH CHECK
    length = subset.str.len_chars() == 3
    if not length.all():
        bad_local = np.flatnonzero(~length.to_numpy())
        idx = valid_idx[bad_local].tolist()[:10]
        raise ValueError(f"{ctx} invalid currency (len!=3) at rows: {idx}")


def _validate_non_zero_quantity(
    df: pl.DataFrame,
    mask: npt.NDArray[np.bool_],
    ctx: str,
) -> None:
    # Float64 with no nulls after materialize; NaN guard inside errstate
    qty = df["quantity"].to_numpy()

    with np.errstate(invalid="ignore"):
        dust = np.abs(qty) < POSITION_DUST_TOLERANCE

    bad = mask & dust
    if np.any(bad):
        idx = np.flatnonzero(bad)[:25].tolist()
        raise ValueError(
            f"{ctx} contain zero / dust quantities.\n"
            f"Tolerance={POSITION_DUST_TOLERANCE}\nRows: {idx}"
        )


# ==============================================================================
# FAST CORE: INSTRUMENT → ID → FAMILY
# ==============================================================================


def _extract_instrument_id_array(
    df: pl.DataFrame,
) -> npt.NDArray[np.int32]:
    """
    Map instrument name strings to numeric IDs.

    Uses a numpy dict lookup over the instrument column rather than a pandas
    map. The sentinel -1 is safe because valid instrument IDs are non-negative.
    """
    # Polars Utf8 → numpy object array (Python strings, None for null)
    names = df["instrument"].to_numpy()

    _MISS = -1
    id_arr = np.array(
        [INSTRUMENT_NAME_TO_ID_REGISTRY.get(n, _MISS) for n in names],
        dtype=np.int32,
    )
    invalid = id_arr == _MISS

    if invalid.any():
        idx = np.flatnonzero(invalid)[:10].tolist()
        vals = sorted({str(n) for n in names[invalid]})[:10]
        raise ValueError(f"Invalid instruments:\n{vals}\nRows: {idx}")

    return id_arr


def _derive_family_array(
    inst_id_arr: npt.NDArray[np.int32],
) -> npt.NDArray[np.int8]:
    return np.take(INSTRUMENT_TYPE_TO_FAMILY_ARRAY, inst_id_arr)


def _mask_eq(
    arr: npt.NDArray,
    val: int,
) -> npt.NDArray[np.bool_]:
    return arr == val


def _mask_in(
    arr: npt.NDArray,
    vals: tuple[int, ...],
) -> npt.NDArray[np.bool_]:
    mask = np.zeros_like(arr, dtype=bool)
    for v in vals:
        mask |= arr == v
    return mask


# ==============================================================================
# FAMILY-LEVEL VALIDATION
# ==============================================================================


def validate_spot_family(
    df: pl.DataFrame,
    mask: npt.NDArray[np.bool_],
) -> None:
    if not np.any(mask):
        return

    cols = (
        "quantity",
        "underlying",
        "settlement_ccy",
    )

    _require_cols(df, cols)
    _validate_non_null_masked(df, mask, cols, "Spot")
    _validate_ccy_series(df, "settlement_ccy", mask, "Spot")
    _validate_non_zero_quantity(df, mask, "Spot")


def validate_forward_family(
    df: pl.DataFrame,
    mask: npt.NDArray[np.bool_],
) -> None:
    if not np.any(mask):
        return

    cols = (
        "quantity",
        "underlying",
        "settlement_ccy",
        "strike_price",
        "strike_ccy",
        # D062: every coordinate a forward block packs into an effective-factor
        # key is mandatory. A missing one used to pack as a sentinel and alias
        # another trade's row (MD-03).
        "delivery_date",
        "settlement_date",
        "under_discount_curve",
        "strike_discount_curve",
    )

    _require_cols(df, cols)
    _validate_non_null_masked(df, mask, cols, "Forward")
    _validate_ccy_series(df, "settlement_ccy", mask, "Forward")
    _validate_ccy_series(df, "strike_ccy", mask, "Forward")
    _validate_non_zero_quantity(df, mask, "Forward")


def validate_option_family(
    df: pl.DataFrame,
    mask: npt.NDArray[np.bool_],
) -> None:
    if not np.any(mask):
        return

    cols = (
        "quantity",
        "underlying",
        "settlement_ccy",
        "strike_price",
        "strike_ccy",
        "expiry_date",
        "is_call",
        # OPTION rows MUST point at a vol container; non-option rows
        # may leave the column None. The compiler resolves the name to
        # an int32 container id and derives the y-axis-type id off
        # `surface_meta_dict[id].y_axis_kind`.
        "volatility_container_name",
        # D062: settlement discounting and both discount curves are mandatory.
        # `delivery_date` is NOT required: the option's value is a function of
        # expiry (forward measure, vol, TTE) and settlement (payoff discount)
        # only, and its underlying is an exact market point that no instrument
        # date selects (D065, option half).
        "settlement_date",
        "under_discount_curve",
        "strike_discount_curve",
    )

    _require_cols(df, cols)
    _validate_non_null_masked(df, mask, cols, "Option")
    _validate_ccy_series(df, "settlement_ccy", mask, "Option")
    _validate_ccy_series(df, "strike_ccy", mask, "Option")
    _validate_non_zero_quantity(df, mask, "Option")


# ==============================================================================
# GLOBAL VALIDATORS
# ==============================================================================


def validate_unique_uids(df: pl.DataFrame) -> None:
    ser = df["uid"]
    if ser.n_unique() != len(ser):
        dupes = ser.filter(ser.is_duplicated()).unique().to_list()[:10]
        raise ValueError(f"Duplicate UIDs:\n{dupes[:10]}")
