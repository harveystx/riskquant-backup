# src/riskquant/portfolio/ingest.py

from __future__ import annotations

from typing import Iterator, Tuple, Final, Iterable

import numpy as np
import pandas as pd

from riskquant.config.deprecated_configs import RiskStaticConfig
from riskquant.exceptions import PrettyKeyError
from riskquant.portfolio.constants import (
    NO_FX_RF_REQUIRED,
    NO_EXPIRY_DATE_ORDINAL32,
    NO_SETTLEMENT_DATE_ORDINAL32,
)
from riskquant.portfolio.fx import (
    get_fx_pair_for_under_to_settle_conversion,
    validate_ccys_unique,
)
from riskquant.portfolio.payloads.forward import ForwardPayload
from riskquant.portfolio.payloads.spot import SpotPayload
from riskquant.portfolio.payloads.vanilla_european_option import (
    VanillaEuropeanOptionPayload,
)
from riskquant.portfolio.trades import TradeRecord
from riskquant.utils.dates import (
    is_date_str_valid,
    datetime64D_to_ordinal32,
)
from riskquant.utils.printer import print_as_list_str


def iter_trades_from_positions_df(
    *,
    positions_df: pd.DataFrame,
    cfg: RiskStaticConfig,
) -> Iterator[TradeRecord]:
    """
    Unified ingest for mixed instruments.

    Supports:
      - SPOT
      - FORWARD
      - VANILLA_EUROPEAN_OPTION

    Vol convention for VANILLA_EUROPEAN_OPTION:
      vol_rf_name = "vol_" + underlying_rf_name
    """
    # --- BASE SMOKE TESTS --- #
    _require_cols(positions_df, _REQUIRED_BASE_COLS, ctx="positions_df")

    # Normalize instrument strings (no copy of df)
    inst = positions_df["instrument"].astype(str).str.upper()
    uids = positions_df["uid"].to_numpy()
    qty = positions_df["quantity"].to_numpy(dtype=np.float64, copy=False)

    # Smoke test - NaN quantities.
    if np.isnan(qty).any():
        bad_uids = sorted(uids[np.isnan(qty)].tolist())[:25]
        raise PrettyKeyError(
            f"Trade UIDs with NaN quantities:\n {_pl(bad_uids)}"
        )

    # --- Resolve instrument IDs from config (fail fast) --- #
    spot_id = cfg.get_instrument_id("SPOT")
    fwd_id = cfg.get_instrument_id("FORWARD")
    opt_id = cfg.get_instrument_id("VANILLA_EUROPEAN_OPTION")

    # Smoke test - instrument IDs.
    if spot_id is None:
        raise PrettyKeyError("cfg.instruments_dict missing key 'SPOT'!")
    if fwd_id is None:
        raise PrettyKeyError("cfg.instruments_dict missing key 'FORWARD'!")
    if opt_id is None:
        raise PrettyKeyError(
            "cfg.instruments_dict missing key 'VANILLA_EUROPEAN_OPTION'!"
        )

    # Convert to integers
    spot_id = np.int32(spot_id)
    fwd_id = np.int32(fwd_id)
    opt_id = np.int32(opt_id)

    # --- Underlying risk factor IDs (shared by all three instruments) --- #
    rf_names = positions_df["underlying"].astype(str)
    price_rf_ids_ser = rf_names.map(cfg.rf_name_to_rf_id)

    # Smoke test
    if price_rf_ids_ser.isna().any():
        unknown = sorted(rf_names[price_rf_ids_ser.isna()].unique().tolist())
        raise PrettyKeyError(
            f"Unknown risk_factor_name(s) not in config:\n {_pl(unknown)}"
        )

    # NumPy array
    price_rf_ids = price_rf_ids_ser.to_numpy(dtype=np.int32, copy=False)

    # --- CCYs (shared columns) --- #
    u_ser = positions_df["underlying_ccy"].astype(str).str.upper()
    s_ser = positions_df["settlement_ccy"].astype(str).str.upper()
    validate_ccys_unique(u_ser)
    validate_ccys_unique(s_ser)
    underlying_ccy = u_ser.to_numpy(dtype=object, copy=False)
    settlement_ccy = s_ser.to_numpy(dtype=object, copy=False)

    # --- Optional columns (pull safely; may not exist for SPOT-only datasets) --- #
    def _col_or_na(name: str) -> pd.Series:
        if name in positions_df.columns:
            return positions_df[name]
        # avoid allocating full object arrays later; but Series is required for consistent indexing
        return pd.Series([np.nan] * len(positions_df), index=positions_df.index)

    strike_price = _col_or_na("strike_price")
    strike_ccy_s = _col_or_na("strike_ccy").astype(str).str.upper()
    is_call_s = _col_or_na("is_call")
    expiry_date_s = _col_or_na("expiry_date")
    settlement_date_s = _col_or_na("settlement_date")

    # --- Precompute FX legs (vectorized masks, resolved only where needed) --- #
    n = len(positions_df)
    fx_under_to_settle = np.empty(n, dtype=np.int32)
    _resolve_fx_rf_ids_inplace(
        base_ccy=underlying_ccy,
        settlement_ccy=settlement_ccy,
        out_fx_rf_ids=fx_under_to_settle,
        rf_name_to_rf_id=cfg.rf_name_to_rf_id,
    )

    # strike_ccy defaults (FORWARD allows blank => default to underlying_ccy; OPTION requires strike_ccy)
    strike_ccy = strike_ccy_s.to_numpy(dtype=object, copy=False)

    # For strike->settle FX resolution we need a base_ccy array; we’ll build one that defaults to underlying_ccy
    strike_ccy_eff = np.empty(n, dtype=object)
    strike_ccy_eff[:] = underlying_ccy  # default

    # Fill explicit strike_ccy where non-blank/non-na
    for i in range(n):
        v = strike_ccy[i]
        if _is_na(v):
            continue
        vs = str(v).strip().upper()
        if not vs or vs == "NAN":
            continue
        strike_ccy_eff[i] = vs

    validate_ccys_unique(strike_ccy_eff.tolist())

    fx_strike_to_settle = np.empty(n, dtype=np.int32)
    _resolve_fx_rf_ids_inplace(
        base_ccy=strike_ccy_eff,
        settlement_ccy=settlement_ccy,
        out_fx_rf_ids=fx_strike_to_settle,
        rf_name_to_rf_id=cfg.rf_name_to_rf_id,
    )

    # --- Dates (ordinal32) --- #
    expiry_ord = _parse_optional_date_series_to_ordinal32(
        expiry_date_s,
        sentinel=np.int32(NO_EXPIRY_DATE_ORDINAL32),
        col_name="expiry_date",
    )
    settle_ord = _parse_optional_date_series_to_ordinal32(
        settlement_date_s,
        sentinel=np.int32(NO_SETTLEMENT_DATE_ORDINAL32),
        col_name="settlement_date",
    )

    # --- Iterate rows and yield TradeRecord (streaming, no list materialization) --- #
    for i in range(n):
        uid = int(uids[i])
        instrument = str(inst.iat[i])

        if instrument == "SPOT":
            yield TradeRecord(
                trade_uid=uid,
                instrument_id=spot_id,
                payload=SpotPayload(
                    price_rf_id=np.int32(price_rf_ids[i]),
                    quantity=np.float64(qty[i]),
                    fx_rf_id=np.int32(
                        fx_under_to_settle[i]
                    ),  # FX_NONE if same ccy
                ),
            )
            continue

        if instrument == "FORWARD":
            # strike_price required
            K = strike_price.iat[i]
            if _is_na(K):
                raise PrettyKeyError(
                    "FORWARD row missing strike_price:\n"
                    f"- uid={uid}\n"
                    f"- underlying={rf_names.iat[i]!r}\n"
                )

            yield TradeRecord(
                trade_uid=uid,
                instrument_id=fwd_id,
                payload=ForwardPayload(
                    price_rf_id=np.int32(price_rf_ids[i]),
                    quantity=np.float64(qty[i]),
                    strike=np.float64(K),
                    settlement_date_ordinal32=np.int32(settle_ord[i]),
                    fx_under_to_settle_rf_id=np.int32(fx_under_to_settle[i]),
                    fx_strike_to_settle_rf_id=np.int32(fx_strike_to_settle[i]),
                ),
            )
            continue

        if instrument == "VANILLA_EUROPEAN_OPTION":
            # strike_price required
            K = strike_price.iat[i]
            if _is_na(K):
                raise PrettyKeyError(
                    "VANILLA_EUROPEAN_OPTION row missing strike_price:\n"
                    f"- uid={uid}\n"
                    f"- underlying={rf_names.iat[i]!r}\n"
                )

            # is_call required (0/1)
            ic = is_call_s.iat[i]
            if _is_na(ic):
                raise PrettyKeyError(
                    "VANILLA_EUROPEAN_OPTION row missing is_call:\n"
                    f"- uid={uid}\n"
                    f"- underlying={rf_names.iat[i]!r}\n"
                )
            ic8 = np.int8(int(ic))

            # expiry_date required (for now)
            if int(expiry_ord[i]) == int(NO_EXPIRY_DATE_ORDINAL32):
                raise PrettyKeyError(
                    "VANILLA_EUROPEAN_OPTION row missing expiry_date:\n"
                    f"- uid={uid}\n"
                    f"- underlying={rf_names.iat[i]!r}\n"
                )

            # strike_ccy required for options (you said strike_ccy is explicit)
            vsc = str(strike_ccy_eff[i]).strip().upper()
            if not vsc:
                raise PrettyKeyError(
                    "VANILLA_EUROPEAN_OPTION row missing strike_ccy:\n"
                    f"- uid={uid}\n"
                    f"- underlying={rf_names.iat[i]!r}\n"
                )

            # VOL RF: deterministic convention vol_<underlying>
            under_name = str(rf_names.iat[i])
            vol_name = f"vol_{under_name}"
            try:
                vol_rf_id = np.int32(cfg.rf_name_to_rf_id[vol_name])
            except KeyError as e:
                raise PrettyKeyError(
                    "Missing implied volatility risk factor for vanilla option:\n"
                    f"- uid={uid}\n"
                    f"- underlying={under_name!r}\n"
                    f"- expected_vol_rf_name={vol_name!r}\n"
                    "Convention: vol_<underlying_risk_factor_name>\n"
                ) from e

            yield TradeRecord(
                trade_uid=uid,
                instrument_id=opt_id,
                payload=VanillaEuropeanOptionPayload(
                    underlying_price_rf_id=np.int32(price_rf_ids[i]),
                    quantity=np.float64(qty[i]),
                    strike=np.float64(K),
                    is_call=ic8,
                    vol_rf_id=vol_rf_id,
                    expiry_date_ordinal32=np.int32(expiry_ord[i]),
                    fx_under_to_settle_rf_id=np.int32(fx_under_to_settle[i]),
                    fx_strike_to_settle_rf_id=np.int32(fx_strike_to_settle[i]),
                ),
            )
            continue

        raise ValueError(
            "Unsupported instrument in positions_df:\n"
            f"- uid={uid}\n"
            f"- instrument={instrument!r}\n"
            "Supported: SPOT, FORWARD, VANILLA_EUROPEAN_OPTION\n"
        )


# Backwards-compatible wrappers (keep your existing API surface)


def iter_spot_trades_from_positions_df(
    *,
    positions_df: pd.DataFrame,
    cfg: RiskStaticConfig,
) -> Iterator[TradeRecord]:
    """
    Backwards-compatible SPOT-only iterator.
    Raises if any non-SPOT instruments are present.
    """
    inst = positions_df["instrument"].astype(str).str.upper()
    if not (inst == "SPOT").all():
        bad = sorted(inst[inst != "SPOT"].unique().tolist())
        raise ValueError(
            f"iter_spot_trades_from_positions_df expects SPOT-only rows; got:\n {_pl(bad)}"
        )
    yield from iter_trades_from_positions_df(positions_df=positions_df, cfg=cfg)


def list_spot_trades_from_positions_df(
    *,
    positions_df: pd.DataFrame,
    cfg: RiskStaticConfig,
) -> list[TradeRecord]:
    return list(
        iter_spot_trades_from_positions_df(positions_df=positions_df, cfg=cfg)
    )
