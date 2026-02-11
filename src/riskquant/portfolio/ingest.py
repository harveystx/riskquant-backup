# src/riskquant/portfolio/ingest.py

from __future__ import annotations

from typing import Iterator, Tuple, Final, Iterable

import numpy as np
import pandas as pd

from riskquant.config.configs import RiskStaticConfig
from riskquant.exceptions import PrettyKeyError
from riskquant.portfolio.constants import (
    FX_NONE,
    NO_EXPIRY_DATE,
    NO_SETTLEMENT_DATE,
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
from riskquant.utils.printer import as_list_str

_REQUIRED_BASE_COLS: Final[Tuple[str, ...]] = (
    "uid",
    "instrument",
    "quantity",
    "underlying",
    "underlying_ccy",
    "settlement_ccy",
)

# Optional “extended schema” cols (may be missing for SPOT-only sheets)
_OPTIONAL_COLS: Final[Tuple[str, ...]] = (
    "strike_price",
    "strike_ccy",
    "is_call",
    "expiry_date",
    "settlement_date",
)


def _pl(x: Iterable) -> str:
    return as_list_str(x)


def _is_na(x) -> bool:
    return x is None or pd.isna(x)


def _require_cols(
    df: pd.DataFrame,
    cols: Iterable[str],
    *,
    ctx: str,
) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{ctx}: missing required columns:\n {_pl(missing)}")


def _require_non_na(series: pd.Series, *, ctx: str) -> None:
    if series.isna().any():
        bad = series.index[series.isna()].tolist()
        # keep it bounded
        num_bad = len(bad)
        bad = bad[:25]
        raise PrettyKeyError(
            f"{ctx}: found {num_bad:_.0f} missing values at row indices: {bad}"
        )


def _parse_optional_date_series_to_ordinal32(
    ser: pd.Series,
    *,
    sentinel: np.int32,
    col_name: str,
) -> np.ndarray:
    """
    Convert a date-like column to ordinal32 days since DATE_EPOCH using the canonical
    riskquant date encoding (datetime64[D] -> ordinal32).

    Accepted inputs:
      - NaN / None / ""        -> sentinel
      - datetime64 / Timestamp -> encoded via datetime64D_to_ordinal32
      - str "YYYY-MM-DD"       -> validated via is_date_str_valid then encoded

    Rejected:
      - any other string format (e.g. "01/12/2026", "20261201", etc.)
      - ambiguous objects

    Output:
      - np.ndarray[int32] length n
    """
    n = int(len(ser))
    out = np.empty(n, dtype=np.int32)
    out.fill(np.int32(sentinel))

    # Fast exit
    if n == 0:
        return out

    # Work on a view
    sv = ser.to_numpy(copy=False)

    # Identify non-empty entries
    # Treat "" as empty (common in Excel exports)
    nn_mask = np.ones(n, dtype=bool)

    # pd.isna works on object arrays too
    nn_mask &= ~pd.isna(sv)

    # Empty strings -> treat as missing
    if sv.dtype == object:
        for i in range(n):
            if nn_mask[i]:
                v = sv[i]
                if isinstance(v, str) and not v.strip():
                    nn_mask[i] = False

    if not np.any(nn_mask):
        return out

    # Collect non-null values as a Series for dtype-aware conversion
    v = ser.loc[nn_mask]

    # Case 1: datetime-like column already
    if np.issubdtype(v.dtype, np.datetime64):
        d64 = v.to_numpy(dtype="datetime64[D]", copy=False)
        out[nn_mask] = datetime64D_to_ordinal32(d64)
        return out

    # Case 2: object column: may contain Timestamp / datetime / strings
    # We normalize row-wise but still encode via datetime64D_to_ordinal32 at the end.
    idx = np.nonzero(nn_mask)[0]

    # Build a small datetime64[D] array for just the present rows
    d64 = np.empty(idx.shape[0], dtype="datetime64[D]")

    for k, i in enumerate(idx):
        x = sv[i]

        # pandas Timestamp / numpy datetime64
        if isinstance(x, (pd.Timestamp, np.datetime64)):
            d64[k] = np.datetime64(x, "D")
            continue

        # strings must be strict YYYY-MM-DD
        if isinstance(x, str):
            xs = x.strip()
            if not is_date_str_valid(xs):
                raise ValueError(
                    f"Invalid {col_name}='{xs}' at row {int(i)} "
                    "(expected strict YYYY-MM-DD)."
                )
            d64[k] = np.datetime64(xs, "D")
            continue

        # Anything else: reject (keeps conventions deterministic)
        raise TypeError(
            f"Invalid {col_name} type at row {int(i)}: {type(x).__name__}. "
            "Expected YYYY-MM-DD string, pandas Timestamp, or datetime64."
        )

    out[idx] = datetime64D_to_ordinal32(d64)
    return out


def _resolve_fx_rf_ids_inplace(
    *,
    base_ccy: np.ndarray,  # (n,) object/str
    settlement_ccy: np.ndarray,  # (n,) object/str
    out_fx_rf_ids: np.ndarray,  # (n,) int32
    rf_name_to_id: dict[str, int],
) -> None:
    """
    Resolve FX RF IDs for rows where base_ccy != settlement_ccy.

    out_fx_rf_ids is filled with FX_NONE then overwritten for quanto rows.
    """
    out_fx_rf_ids.fill(np.int32(FX_NONE))

    mask = base_ccy != settlement_ccy
    if not np.any(mask):
        return

    qidx = np.nonzero(mask)[0]
    fx_pair_fn = get_fx_pair_for_under_to_settle_conversion
    rf_map = rf_name_to_id

    for i in qidx:
        fx_pair = fx_pair_fn(
            underlying_ccy=str(base_ccy[i]),
            settlement_ccy=str(settlement_ccy[i]),
        )
        # fx_pair_fn returns "" when equal, but mask excludes equals; still safe:
        if not fx_pair:
            out_fx_rf_ids[i] = np.int32(FX_NONE)
            continue

        try:
            out_fx_rf_ids[i] = np.int32(rf_map[fx_pair])
        except KeyError as e:
            raise PrettyKeyError(
                "Missing FX spot risk factor for currency conversion:\n"
                f"- fx_pair={fx_pair!r}\n"
                f"- base_ccy={base_ccy[i]!r}\n"
                f"- settlement_ccy={settlement_ccy[i]!r}\n"
            ) from e


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
        raise PrettyKeyError(f"Trade UIDs with NaN quantities:\n {_pl(bad_uids)}")

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
    price_rf_ids_ser = rf_names.map(cfg.rf_name_to_id)

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
        rf_name_to_id=cfg.rf_name_to_id,
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
        rf_name_to_id=cfg.rf_name_to_id,
    )

    # --- Dates (ordinal32) --- #
    expiry_ord = _parse_optional_date_series_to_ordinal32(
        expiry_date_s,
        sentinel=np.int32(NO_EXPIRY_DATE),
        col_name="expiry_date",
    )
    settle_ord = _parse_optional_date_series_to_ordinal32(
        settlement_date_s,
        sentinel=np.int32(NO_SETTLEMENT_DATE),
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
                    fx_rf_id=np.int32(fx_under_to_settle[i]),  # FX_NONE if same ccy
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
            if int(expiry_ord[i]) == int(NO_EXPIRY_DATE):
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
                vol_rf_id = np.int32(cfg.rf_name_to_id[vol_name])
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
                    expiry_rf_id=np.int32(expiry_ord[i]),
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
    return list(iter_spot_trades_from_positions_df(positions_df=positions_df, cfg=cfg))
