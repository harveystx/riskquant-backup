# src/riskquant/portfolio/ingest/batch_builder.py
"""
Builds the pristine SoA IngestBatch from raw Pandas DataFrames.
Acts as the extraction and translation layer before trade generation,
ensuring all "Shift Left" validations pass before allocating memory.
"""

import numpy as np
import pandas as pd

from riskquant.portfolio.instruments import Instrument
from riskquant.constants import (
    # ID sentinels
    NO_PRICE_ID,
    NO_PRICE_CURVE_ID,
    NO_VOL_ID,
    # No discount curve sentinel
    NO_DISCOUNT_CURVE_ID,
    # No FX required sentinel
    NO_FX_RF_REQUIRED,
    # Date sentinels
    NO_DELIVERY_DATE_ORDINAL32,
    NO_SETTLEMENT_DATE_ORDINAL32,
    NO_EXPIRY_DATE_ORDINAL32,
    # Not option sentinel
    NOT_OPTION,
    # Volatility risk factor prefix
    VOL_RISK_FACTOR_PREFIX,
)
from riskquant.portfolio.ingest.batch import IngestBatch
from riskquant.portfolio.ingest.schema import REQUIRED_BASE_COLS
from riskquant.portfolio.ingest.validator import (
    require_cols_in_df,
    require_non_nan_in_series,
    require_non_zero_values,
    require_positive_values,
    validate_instruments,
    validate_is_call_flags,
    validate_conditional_option_fields,
    validate_uids_unique,
)
from riskquant.portfolio.ingest.parser import (
    parse_optional_date_series_to_ordinal32,
)
from riskquant.portfolio.ingest.fx_resolver import resolve_fx_rf_ids_inplace


def build_ingest_batch(
    *,
    df: pd.DataFrame,
    rf_name_to_rf_id: dict[str, int],
    container_name_to_container_id: dict[str, int],
    rf_id_to_ccy: dict[int, str],
    container_id_to_ccy: dict[int, str],
) -> IngestBatch:
    """
    Consume a raw positions DataFrame, validate it, translate strings to
    system integer IDs, and output a highly optimized IngestBatch.
    """
    n = len(df)
    if n == 0:
        raise ValueError("Cannot build IngestBatch from an empty DataFrame.")

    # ==========================================
    # 1. BULK VALIDATION ("SHIFT LEFT")
    # ==========================================

    # Check base schema and NaNs
    require_cols_in_df(df, REQUIRED_BASE_COLS, ctx="Positions Ingest")
    for col in REQUIRED_BASE_COLS:
        require_non_nan_in_series(df[col], ctx=f"Column '{col}'")

    # Domain logic validations
    validate_instruments(df["instrument"], ctx="Instrument Validation")
    validate_conditional_option_fields(df, ctx="Option Fields Validation")

    # Core Identity and Economic Validations
    validate_uids_unique(df["uid"], ctx="UID Validation")
    require_non_zero_values(df["quantity"], ctx="Quantity Validation")

    # Optional but if present, must be valid
    if "is_call" in df.columns:
        validate_is_call_flags(df["is_call"], ctx="is_call Validation")
    if "strike_price" in df.columns:
        require_positive_values(
            df["strike_price"], ctx="Strike Price Validation"
        )

    # ==========================================
    # 2. TRANSLATE INSTRUMENTS & UIDs
    # ==========================================

    # Validate UIDs are unique before we convert to numpy arrays, since Pandas has nice tools for this.
    validate_uids_unique(df["uid"], ctx="UID Uniqueness Validation")

    # UIDs and quantities are core to all trades, so we require them and convert to arrays early.
    uids = df["uid"].to_numpy(dtype=np.int64)

    # Quntity can be positive or negative, but not zero or NaN/Inf, so we validate and convert to float64 for safety.
    qty = df["quantity"].to_numpy(dtype=np.float64)

    # We already validated the strings exist in the Enum, so this list comprehension is safe and fast
    instrument_ids = np.array(
        [Instrument[inst].value for inst in df["instrument"]],
        dtype=np.int32,
    )

    # ==========================================
    # 3. TRANSLATE RISK FACTOR MAPPINGS
    # ==========================================

    # Fast boolean masks
    is_container_mask = (
        df["instrument"].str.endswith("CURVE").to_numpy(dtype=bool)
    )
    is_option_mask = (
        df["instrument"].str.contains("OPTION").to_numpy(dtype=bool)
    )

    # --- PRICE MAPPINGS ---
    # Map Nodes (SPOT, FORWARD_NODE, VANILLA_EUROPEAN_OPTION_NODE)
    node_underlyings = df.loc[~is_container_mask, "underlying"]
    mapped_nodes = node_underlyings.map(rf_name_to_rf_id)
    require_non_nan_in_series(mapped_nodes, ctx="Mapped Node Underlying IDs")

    # Map Containers (FORWARD_CURVE, VANILLA_EUROPEAN_OPTION_CURVE)
    container_underlyings = df.loc[is_container_mask, "underlying"]
    mapped_containers = container_underlyings.map(
        container_name_to_container_id
    )
    require_non_nan_in_series(mapped_containers, ctx="Mapped Container IDs")

    # Pre-allocate arrays filled with the defualt no ID sentinels.
    price_rf_ids = np.full(n, NO_PRICE_ID, dtype=np.int32)
    price_curve_ids = np.full(n, NO_PRICE_CURVE_ID, dtype=np.int32)
    vol_rf_ids = np.full(n, NO_VOL_ID, dtype=np.int32)
    vol_surface_rf_ids = np.full(n, NO_VOL_ID, dtype=np.int32)

    # Slot the translated PRICE integer IDs
    price_rf_ids[~is_container_mask] = mapped_nodes.to_numpy(dtype=np.int32)
    price_curve_ids[is_container_mask] = mapped_containers.to_numpy(
        dtype=np.int32
    )

    # --- VOLATILITY MAPPINGS (Options Only) ---
    if is_option_mask.any():
        option_nodes_mask = is_option_mask & ~is_container_mask
        option_curves_mask = is_option_mask & is_container_mask

        if option_nodes_mask.any():
            # Vectorized string prepend
            vol_nodes = (
                VOL_RISK_FACTOR_PREFIX + df.loc[option_nodes_mask, "underlying"]
            )
            mapped_vol_nodes = vol_nodes.map(rf_name_to_rf_id)
            require_non_nan_in_series(
                mapped_vol_nodes,
                ctx="Mapped Vol Node IDs",
            )
            vol_rf_ids[option_nodes_mask] = mapped_vol_nodes.to_numpy(
                dtype=np.int32
            )

        if option_curves_mask.any():
            # Vectorized string prepend
            vol_curves = (
                VOL_RISK_FACTOR_PREFIX
                + df.loc[option_curves_mask, "underlying"]
            )
            mapped_vol_curves = vol_curves.map(container_name_to_container_id)
            require_non_nan_in_series(
                mapped_vol_curves,
                ctx="Mapped Vol Surface IDs",
            )
            vol_surface_rf_ids[option_curves_mask] = mapped_vol_curves.to_numpy(
                dtype=np.int32
            )

    # ==========================================
    # 4. HANDLE OPTIONAL COLUMNS (Economics)
    # ==========================================

    # Strike price
    if "strike_price" in df.columns:
        strike_price = df["strike_price"].fillna(0.0).to_numpy(dtype=np.float64)
    else:
        strike_price = np.zeros(n, dtype=np.float64)

    # is_call flag (only relevant for options, but we allow it on all rows for simplicity)
    if "is_call" in df.columns:
        # Fill missing with NOT_OPTION sentinel
        is_call = df["is_call"].fillna(NOT_OPTION).to_numpy(dtype=np.int8)
    else:
        is_call = np.full(n, NOT_OPTION, dtype=np.int8)

    # ==========================================
    # 5. PARSE DATES TO ORDINAL32
    # ==========================================
    # The parser function handles Sentinels automatically if the column is missing or blank
    delivery_ord = parse_optional_date_series_to_ordinal32(
        df.get("delivery_date", pd.Series(index=df.index, dtype=object)),
        col_name="delivery_date",
        sentinel=NO_DELIVERY_DATE_ORDINAL32,
    )
    settle_ord = parse_optional_date_series_to_ordinal32(
        df.get("settlement_date", pd.Series(index=df.index, dtype=object)),
        col_name="settlement_date",
        sentinel=NO_SETTLEMENT_DATE_ORDINAL32,
    )
    expiry_ord = parse_optional_date_series_to_ordinal32(
        df.get("expiry_date", pd.Series(index=df.index, dtype=object)),
        col_name="expiry_date",
        sentinel=NO_EXPIRY_DATE_ORDINAL32,
    )

    # ==========================================
    # 6. RESOLVE FX LINKAGES IN BULK
    # ==========================================
    # Derive underlying currency from the resolved Risk Factor and Container IDs
    underlying_ccy_series = pd.Series(index=df.index, dtype=object)
    underlying_ccy_series.loc[~is_container_mask] = mapped_nodes.map(
        rf_id_to_ccy
    )
    underlying_ccy_series.loc[is_container_mask] = mapped_containers.map(
        container_id_to_ccy
    )

    require_non_nan_in_series(
        underlying_ccy_series, ctx="Underlying Currency Config Resolution"
    )

    underlying_ccy = underlying_ccy_series.to_numpy(dtype=object)
    settlement_ccy = df["settlement_ccy"].to_numpy(dtype=object)

    # If strike_ccy is provided, fill the blank/NaN rows with the underlying_ccy.
    # If the column is entirely missing, default to underlying_ccy.
    if "strike_ccy" in df.columns:
        strike_ccy = (
            df["strike_ccy"]
            .fillna(underlying_ccy_series)
            .to_numpy(dtype=object)
        )
    else:
        strike_ccy = underlying_ccy

    fx_under_to_settle_ids = np.empty(n, dtype=np.int32)
    fx_under_to_settle_invert = np.empty(n, dtype=np.int32)
    resolve_fx_rf_ids_inplace(
        base_ccy=underlying_ccy,
        settlement_ccy=settlement_ccy,
        out_fx_rf_ids=fx_under_to_settle_ids,
        out_fx_invert=fx_under_to_settle_invert,
        rf_name_to_rf_id=rf_name_to_rf_id,
    )

    fx_strike_to_settle_ids = np.empty(n, dtype=np.int32)
    fx_strike_to_settle_invert = np.empty(n, dtype=np.int32)
    resolve_fx_rf_ids_inplace(
        base_ccy=strike_ccy,
        settlement_ccy=settlement_ccy,
        out_fx_rf_ids=fx_strike_to_settle_ids,
        out_fx_invert=fx_strike_to_settle_invert,
        rf_name_to_rf_id=rf_name_to_rf_id,
    )

    # ==========================================
    # 6.1 CLEAN UP NON-STRIKE INSTRUMENTS
    # ==========================================

    # SPOT trades do not have strikes. Mask out their strike FX linkages.
    is_spot_mask = (df["instrument"] == "SPOT").to_numpy(dtype=bool)
    fx_strike_to_settle_ids[is_spot_mask] = NO_FX_RF_REQUIRED
    fx_strike_to_settle_invert[is_spot_mask] = 0

    # ==========================================
    # 6.5 RESOLVE DISCOUNT CURVES (TRADE LEVEL)
    # ==========================================

    # If the trade file doesn't provide the curves, then there is no discounting and we can fill with NO_DISCOUNT_CURVE_ID sentinel. If the columns are present, we map the names to IDs and validate they exist in the system.

    # Discounting for underlying
    if "under_discount_curve" in df.columns:
        mapped_under_dc = df["under_discount_curve"].map(
            container_name_to_container_id
        )
        under_discount_curve_ids = mapped_under_dc.fillna(
            NO_DISCOUNT_CURVE_ID
        ).to_numpy(dtype=np.int32)
    else:
        under_discount_curve_ids = np.full(
            n,
            NO_DISCOUNT_CURVE_ID,
            dtype=np.int32,
        )

    # Dicounting for strike.
    if "strike_discount_curve" in df.columns:
        mapped_strike_dc = df["strike_discount_curve"].map(
            container_name_to_container_id
        )
        strike_discount_curve_ids = mapped_strike_dc.fillna(
            NO_DISCOUNT_CURVE_ID
        ).to_numpy(dtype=np.int32)
    else:
        strike_discount_curve_ids = np.full(
            n,
            NO_DISCOUNT_CURVE_ID,
            dtype=np.int32,
        )

    # ==========================================
    # 7. PACK THE DUMB BATCH
    # ==========================================
    return IngestBatch(
        size=n,
        # --- IDS ---
        instrument_ids=instrument_ids,
        uids=uids,
        # --- ECONOMICS ---
        qty=qty,
        strike_price=strike_price,
        is_call=is_call,
        # --- DATES ---
        settle_ord=settle_ord,
        delivery_ord=delivery_ord,
        expiry_ord=expiry_ord,
        # --- FX LINKAGES ---
        fx_under_to_settle_ids=fx_under_to_settle_ids,
        fx_strike_to_settle_ids=fx_strike_to_settle_ids,
        # --- FX INVERT FLAGS ---
        fx_under_to_settle_invert=fx_under_to_settle_invert,
        fx_strike_to_settle_invert=fx_strike_to_settle_invert,
        # --- UNDERLYING LINKAGES
        price_rf_ids=price_rf_ids,
        price_curve_ids=price_curve_ids,
        # --- VOL LINKAGES
        vol_rf_ids=vol_rf_ids,
        vol_surface_rf_ids=vol_surface_rf_ids,
        # --- DISCOUNT CURVE LINKAGES ---
        under_discount_curve_ids=under_discount_curve_ids,
        strike_discount_curve_ids=strike_discount_curve_ids,
    )
