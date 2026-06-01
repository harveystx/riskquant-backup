# src/riskquant/portfolio/ingest/validate.py

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from riskquant.constants import POSITION_DUST_TOLERANCE
from riskquant.portfolio.ingest.schema import STRING_COLS
from riskquant.portfolio.instruments import (
    INSTRUMENT_NAME_TO_ID_REGISTRY,
    INSTRUMENT_TO_FAMILY_ARRAY,
    InstrumentFamily,
)
from riskquant.utils.dataframes import validate_string_cols

# ==============================================================================
# INTERNAL HELPERS
# ==============================================================================


def _require_cols(df: pd.DataFrame, cols: tuple[str, ...]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns:\n{missing}")


def _validate_non_null_masked(
    df: pd.DataFrame,
    mask: npt.NDArray[np.bool_],
    cols: tuple[str, ...],
    ctx: str,
) -> None:
    for col in cols:
        arr = df[col].isna().to_numpy(copy=False)
        bad = mask & arr
        if np.any(bad):
            idx = df.index[bad].tolist()[:10]
            raise ValueError(f"{ctx} missing '{col}' at rows: {idx}")


def _validate_ccy_series(
    df: pd.DataFrame,
    col: str,
    mask: npt.NDArray[np.bool_],
    ctx: str,
) -> None:
    ser = df[col]
    valid_idx = np.flatnonzero(mask & ser.notna().to_numpy(copy=False))

    if len(valid_idx) == 0:
        return

    subset = ser.iloc[valid_idx]

    alpha = subset.str.isalpha()
    if not alpha.all():
        idx = subset.index[~alpha].tolist()[:10]
        raise ValueError(f"{ctx} invalid currency (non-alpha) at rows: {idx}")

    length = subset.str.len().eq(3)
    if not length.all():
        idx = subset.index[~length].tolist()[:10]
        raise ValueError(f"{ctx} invalid currency (len!=3) at rows: {idx}")


def _validate_non_zero_quantity(
    df: pd.DataFrame,
    mask: npt.NDArray[np.bool_],
    ctx: str,
) -> None:
    qty = df["quantity"].to_numpy(copy=False)

    with np.errstate(invalid="ignore"):
        dust = np.abs(qty) < POSITION_DUST_TOLERANCE

    bad = mask & dust
    if np.any(bad):
        idx = df.index[bad].tolist()[:25]
        raise ValueError(
            f"{ctx} contain zero / dust quantities.\n"
            f"Tolerance={POSITION_DUST_TOLERANCE}\nRows: {idx}"
        )


# ==============================================================================
# FAST CORE: INSTRUMENT → ID → FAMILY
# ==============================================================================


def _extract_instrument_id_array(
    df: pd.DataFrame,
) -> npt.NDArray[np.int32]:
    ser = df["instrument"]

    id_arr = ser.map(INSTRUMENT_NAME_TO_ID_REGISTRY).to_numpy()

    if np.any(pd.isna(id_arr)):
        bad_mask = pd.isna(id_arr)
        idx = ser.index[bad_mask].tolist()[:10]
        vals = sorted(ser[bad_mask].dropna().unique().tolist())

        raise ValueError(f"Invalid instruments:\n{vals}\nRows: {idx}")

    return id_arr.astype(np.int32, copy=False)


def _derive_family_array(
    inst_id_arr: npt.NDArray[np.int32],
) -> npt.NDArray[np.int8]:
    return np.take(INSTRUMENT_TO_FAMILY_ARRAY, inst_id_arr)


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
    df: pd.DataFrame,
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
    df: pd.DataFrame,
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
    )

    _require_cols(df, cols)
    _validate_non_null_masked(df, mask, cols, "Forward")
    _validate_ccy_series(df, "settlement_ccy", mask, "Forward")
    _validate_ccy_series(df, "strike_ccy", mask, "Forward")
    _validate_non_zero_quantity(df, mask, "Forward")


def validate_option_family(
    df: pd.DataFrame,
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
    )

    _require_cols(df, cols)
    _validate_non_null_masked(df, mask, cols, "Option")
    _validate_ccy_series(df, "settlement_ccy", mask, "Option")
    _validate_ccy_series(df, "strike_ccy", mask, "Option")
    _validate_non_zero_quantity(df, mask, "Option")


# ==============================================================================
# INSTRUMENT-LEVEL VALIDATION (SECONDARY DISPATCH)
# ==============================================================================

# ---> SPOT: N/A
# ---> VANILLA FORWARD NODE / CURVE: TODO
# ---> VANILLA EUROPEAN OPTION NODE / CURVE: TODO


# ==============================================================================
# GLOBAL VALIDATORS
# ==============================================================================


def validate_unique_uids(df: pd.DataFrame) -> None:
    ser = df["uid"]
    if not ser.is_unique:
        dupes = ser[ser.duplicated()].unique().tolist()
        raise ValueError(f"Duplicate UIDs:\n{dupes[:10]}")


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================


def validate_materialized_ingest_df(df: pd.DataFrame) -> None:
    """
    High-performance semantic validation.
    """

    # ------------------------------------------------------------------
    # 1. Global
    # ------------------------------------------------------------------
    validate_unique_uids(df)
    validate_string_cols(df, STRING_COLS)

    # ------------------------------------------------------------------
    # 2. Instrument → ID
    # ------------------------------------------------------------------
    inst_id = _extract_instrument_id_array(df)

    # ------------------------------------------------------------------
    # 3. ID → Family
    # ------------------------------------------------------------------
    family = _derive_family_array(inst_id)

    # ------------------------------------------------------------------
    # 4. Masks (pure NumPy)
    # ------------------------------------------------------------------

    # --- FAMILY MASKS (PRIMARY DISPATCH) ---
    spot_mask = _mask_eq(family, InstrumentFamily.SPOT.value)
    fwd_mask = _mask_eq(family, InstrumentFamily.FORWARD.value)
    opt_mask = _mask_eq(family, InstrumentFamily.OPTION.value)

    # --- INSTRUMENT MASKS (SECONDARY DISPATCH) ---
    # (none yet, but this is where they'd go once we have any instrument-level validators)

    # ------------------------------------------------------------------
    # 5. Validation passes
    # ------------------------------------------------------------------

    # --- FAMILY LAYER (PRIMARY DISPATCH) ---

    # Family layer
    validate_spot_family(df, spot_mask)
    validate_forward_family(df, fwd_mask)
    validate_option_family(df, opt_mask)

    # --- INSTRUMENT REFINEMENT (SECONDARY DISPATCH) ---
