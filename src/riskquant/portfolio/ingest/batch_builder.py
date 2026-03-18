# src/riskquant/portfolio/ingest/batch_builder.py
import numpy as np
import pandas as pd

from riskquant.portfolio.instruments import (
    Instrument,
    FORWARD_CURVE_ID,
    VANILLA_EUROPEAN_OPTION_CURVE_ID,
    VANILLA_EUROPEAN_OPTION_NODE_ID,
    SPOT_ID,
)
from riskquant.constants import (
    NO_DATE_UIX,
    NO_PRICE_ID,
    NO_PRICE_CURVE_ID,
    NO_VOL_ID,
    NO_DISCOUNT_CURVE_ID,
    NO_FX_RF_REQUIRED,
    NO_DELIVERY_DATE_ORDINAL32,
    NO_SETTLEMENT_DATE_ORDINAL32,
    NO_EXPIRY_DATE_ORDINAL32,
    NOT_OPTION,
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
from riskquant.market_data.surfaces.y_axes_types import (
    SURFACE_Y_AXIS_TYPE_REGISTRY,
)


def build_ingest_batch(
    *,
    df: pd.DataFrame,
    rf_name_to_rf_id: dict[str, int],
    container_name_to_container_id: dict[str, int],
    rf_id_to_ccy: dict[int, str],
    container_id_to_ccy: dict[int, str],
    container_name_to_y_axis_type_name: dict[str, str],
) -> IngestBatch:
    n = len(df)
    if n == 0:
        raise ValueError("Cannot build IngestBatch from an empty DataFrame.")

    # ==========================================
    # 1. BULK VALIDATION ("SHIFT LEFT")
    # ==========================================
    require_cols_in_df(df, REQUIRED_BASE_COLS, ctx="Positions Ingest")
    for col in REQUIRED_BASE_COLS:
        require_non_nan_in_series(df[col], ctx=f"Column '{col}'")

    validate_instruments(df["instrument"], ctx="Instrument Validation")
    validate_conditional_option_fields(df, ctx="Option Fields Validation")
    validate_uids_unique(df["uid"], ctx="UID Validation")
    require_non_zero_values(df["quantity"], ctx="Quantity Validation")

    if "is_call" in df.columns:
        validate_is_call_flags(df["is_call"], ctx="is_call Validation")
    if "strike_price" in df.columns:
        require_positive_values(
            df["strike_price"], ctx="Strike Price Validation"
        )

    # ==========================================
    # 2. TRANSLATE INSTRUMENTS & UIDs
    # ==========================================
    uids = df["uid"].to_numpy(dtype=np.int64)
    qty = df["quantity"].to_numpy(dtype=np.float64)

    instrument_ids = np.array(
        [Instrument[inst].value for inst in df["instrument"]],
        dtype=np.int32,
    )

    # ==========================================
    # 3. TRANSLATE RISK FACTOR MAPPINGS
    # ==========================================
    # OPTIMIZATION: Integer logic instead of string matching
    is_container_mask = (instrument_ids == FORWARD_CURVE_ID) | (
        instrument_ids == VANILLA_EUROPEAN_OPTION_CURVE_ID
    )
    is_option_mask = (instrument_ids == VANILLA_EUROPEAN_OPTION_NODE_ID) | (
        instrument_ids == VANILLA_EUROPEAN_OPTION_CURVE_ID
    )

    # --- PRICE MAPPINGS ---
    node_underlyings = df.loc[~is_container_mask, "underlying"]
    mapped_nodes = node_underlyings.map(rf_name_to_rf_id)
    require_non_nan_in_series(mapped_nodes, ctx="Mapped Node Underlying IDs")

    container_underlyings = df.loc[is_container_mask, "underlying"]
    mapped_containers = container_underlyings.map(
        container_name_to_container_id
    )
    require_non_nan_in_series(mapped_containers, ctx="Mapped Container IDs")

    price_rf_ids = np.full(n, NO_PRICE_ID, dtype=np.int32)
    price_curve_ids = np.full(n, NO_PRICE_CURVE_ID, dtype=np.int32)
    vol_rf_ids = np.full(n, NO_VOL_ID, dtype=np.int32)
    vol_surface_rf_ids = np.full(n, NO_VOL_ID, dtype=np.int32)

    price_rf_ids[~is_container_mask] = mapped_nodes.to_numpy(dtype=np.int32)
    price_curve_ids[is_container_mask] = mapped_containers.to_numpy(
        dtype=np.int32
    )

    # DEFAULT Fallback mapping config
    default_y_axis = SURFACE_Y_AXIS_TYPE_REGISTRY["STRIKE"].value
    vol_surface_y_axis_type_ids = np.full(n, default_y_axis, dtype=np.int8)

    # --- VOLATILITY MAPPINGS (Options Only) ---
    if is_option_mask.any():
        option_nodes_mask = is_option_mask & ~is_container_mask
        option_curves_mask = is_option_mask & is_container_mask

        if option_nodes_mask.any():
            vol_nodes = (
                VOL_RISK_FACTOR_PREFIX + df.loc[option_nodes_mask, "underlying"]
            )
            mapped_vol_nodes = vol_nodes.map(rf_name_to_rf_id)
            require_non_nan_in_series(
                mapped_vol_nodes, ctx="Mapped Vol Node IDs"
            )
            vol_rf_ids[option_nodes_mask] = mapped_vol_nodes.to_numpy(
                dtype=np.int32
            )

        if option_curves_mask.any():
            vol_curves = (
                VOL_RISK_FACTOR_PREFIX
                + df.loc[option_curves_mask, "underlying"]
            )
            mapped_vol_curves = vol_curves.map(container_name_to_container_id)
            require_non_nan_in_series(
                mapped_vol_curves, ctx="Mapped Vol Surface IDs"
            )
            vol_surface_rf_ids[option_curves_mask] = mapped_vol_curves.to_numpy(
                dtype=np.int32
            )

            mapped_y_axis_type_strs = vol_curves.map(
                container_name_to_y_axis_type_name
            )
            mapped_y_axis_type_ids = mapped_y_axis_type_strs.map(
                lambda x: (
                    SURFACE_Y_AXIS_TYPE_REGISTRY[x.upper()].value
                    if pd.notna(x)
                    else default_y_axis
                )
            )
            vol_surface_y_axis_type_ids[option_curves_mask] = (
                mapped_y_axis_type_ids.to_numpy(dtype=np.int8)
            )

    # ==========================================
    # 4. HANDLE OPTIONAL COLUMNS (Economics)
    # ==========================================
    if "strike_price" in df.columns:
        strike_price = df["strike_price"].fillna(0.0).to_numpy(dtype=np.float64)
    else:
        strike_price = np.zeros(n, dtype=np.float64)

    if "is_call" in df.columns:
        is_call = df["is_call"].fillna(NOT_OPTION).to_numpy(dtype=np.int8)
    else:
        is_call = np.full(n, NOT_OPTION, dtype=np.int8)

    # ==========================================
    # 5. PARSE DATES TO ORDINAL32
    # ==========================================
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
    is_spot_mask = instrument_ids == SPOT_ID
    fx_strike_to_settle_ids[is_spot_mask] = NO_FX_RF_REQUIRED
    fx_strike_to_settle_invert[is_spot_mask] = 0

    # ==========================================
    # 6.5 RESOLVE DISCOUNT CURVES (TRADE LEVEL)
    # ==========================================
    if "under_discount_curve" in df.columns:
        mapped_under_dc = df["under_discount_curve"].map(
            container_name_to_container_id
        )
        under_discount_curve_ids = mapped_under_dc.fillna(
            NO_DISCOUNT_CURVE_ID
        ).to_numpy(dtype=np.int32)
    else:
        under_discount_curve_ids = np.full(
            n, NO_DISCOUNT_CURVE_ID, dtype=np.int32
        )

    if "strike_discount_curve" in df.columns:
        mapped_strike_dc = df["strike_discount_curve"].map(
            container_name_to_container_id
        )
        strike_discount_curve_ids = mapped_strike_dc.fillna(
            NO_DISCOUNT_CURVE_ID
        ).to_numpy(dtype=np.int32)
    else:
        strike_discount_curve_ids = np.full(
            n, NO_DISCOUNT_CURVE_ID, dtype=np.int32
        )

    # ==========================================
    # 7. EXTRACT UNIQUE DATES & MAP TO UIX GRID
    # ==========================================
    all_dates = np.concatenate([settle_ord, delivery_ord, expiry_ord])

    valid_mask = (
        (all_dates != NO_SETTLEMENT_DATE_ORDINAL32)
        & (all_dates != NO_DELIVERY_DATE_ORDINAL32)
        & (all_dates != NO_EXPIRY_DATE_ORDINAL32)
    )

    unique_dates_ordinal32 = np.unique(all_dates[valid_mask])

    settle_date_uix = np.where(
        settle_ord == NO_SETTLEMENT_DATE_ORDINAL32,
        NO_DATE_UIX,
        np.searchsorted(unique_dates_ordinal32, settle_ord),
    ).astype(np.int32)

    delivery_date_uix = np.where(
        delivery_ord == NO_DELIVERY_DATE_ORDINAL32,
        NO_DATE_UIX,
        np.searchsorted(unique_dates_ordinal32, delivery_ord),
    ).astype(np.int32)

    expiry_date_uix = np.where(
        expiry_ord == NO_EXPIRY_DATE_ORDINAL32,
        NO_DATE_UIX,
        np.searchsorted(unique_dates_ordinal32, expiry_ord),
    ).astype(np.int32)

    # ==========================================
    # 8. PACK THE DUMB BATCH
    # ==========================================
    return IngestBatch(
        size=n,
        instrument_ids=instrument_ids,
        uids=uids,
        qty=qty,
        strike_price=strike_price,
        is_call=is_call,
        settle_ord=settle_ord,
        delivery_ord=delivery_ord,
        expiry_ord=expiry_ord,
        fx_under_to_settle_ids=fx_under_to_settle_ids,
        fx_strike_to_settle_ids=fx_strike_to_settle_ids,
        fx_under_to_settle_invert=fx_under_to_settle_invert,
        fx_strike_to_settle_invert=fx_strike_to_settle_invert,
        price_rf_ids=price_rf_ids,
        price_curve_ids=price_curve_ids,
        vol_rf_ids=vol_rf_ids,
        vol_surface_rf_ids=vol_surface_rf_ids,
        under_discount_curve_ids=under_discount_curve_ids,
        strike_discount_curve_ids=strike_discount_curve_ids,
        unique_dates_ordinal32=unique_dates_ordinal32,
        settle_date_uix=settle_date_uix,
        delivery_date_uix=delivery_date_uix,
        expiry_date_uix=expiry_date_uix,
        vol_surface_y_axis_type_ids=vol_surface_y_axis_type_ids,
    )
