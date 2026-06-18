# src/riskquant/portfolio/ingest/compile.py
"""
DataFrame -> Struct-of-Arrays compiler.

This is the final hop of the ingest edge: a validated, materialized DataFrame
is collapsed into `CompiledIngestArrays`, a flat homogeneous SoA payload with
no strings and no Python objects. `IngestBatch.from_compiled` wraps the result
for the engine.

Refactor history
----------------
- Semantic Collapsing: `price_rf` and `price_curve` fused into dense `underlying_ids`.
- Semantic Collapsing: `vol_rf` and `vol_surface` fused into dense `volatility_ids`.
- Coordinate Purge: All `_uix` logic removed. Ingest acts purely as a global ID ledger.
- FX Linkages renamed to explicitly end in `_ids` to enforce boundary semantics.
- Underlying currency resolution optimized to occur concurrently with ID mapping.
- Real volatility-container resolution: the `volatility_container_name` schema
  column is mapped to int32 container ids and to the per-trade int8 y-axis-type
  id. Both fields land on `CompiledIngestSoA`; `IngestBatch.from_compiled` is
  a pure copy with no synthetic sentinels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from riskquant.config.static_config.static_config import RiskStaticConfig
from riskquant.constants import (
    CALL_OPTION_MULT,
    NO_DATE_ORD32,
    NO_FX_RF_REQUIRED,
    NO_ID,
    NOT_IMPLEMENTED_SENTINEL,
    NOT_OPTION_MULT,
    PUT_OPTION_MULT,
)
from riskquant.exceptions import PrettyKeyError
from riskquant.market_data.surfaces.y_axes_types import surface_y_axis_kind_for_kernel
from riskquant.portfolio.ingest.fx_resolver import resolve_fx_rf_ids_inplace
from riskquant.portfolio.ingest.parser import (
    parse_optional_date_series_to_ord32,
)
from riskquant.portfolio.ingest.rf_resolver import resolve_optional_ids

# ==============================================================================
# OUTPUT STRUCTURE
# ==============================================================================


@dataclass(slots=True, frozen=True, kw_only=True)
class CompiledIngestSoA:
    """
    Flat SoA payload emitted by the compiler. All arrays length == size.
    Contains ONLY pure economics, date ordinals, and global IDs.
    Strictly zero UIXs and fully dense semantic arrays.
    """

    size: int

    # --- ROUTING & IDENTITY ---
    instrument_ids: npt.NDArray[np.int32]
    uids: npt.NDArray[np.int64]  # external trade ids stay 64-bit

    # --- ECONOMICS ---
    qty: npt.NDArray[np.float64]
    strike: npt.NDArray[np.float64]
    cp_mult: npt.NDArray[np.int8]

    # --- DATES (Ordinals Only) ---
    settle_ord: npt.NDArray[np.int32]
    delivery_ord: npt.NDArray[np.int32]
    expiry_ord: npt.NDArray[np.int32]

    # --- MARKET DATA LINKAGES (Dense Global IDs) ---
    underlying_ids: npt.NDArray[np.int32]
    volatility_ids: npt.NDArray[np.int32]

    # --- FX LINKAGES (Global IDs) ---
    fx_u2s_ids: npt.NDArray[np.int32]
    fx_u2s_inv: npt.NDArray[np.int8]

    fx_k2s_ids: npt.NDArray[np.int32]
    fx_k2s_inv: npt.NDArray[np.int8]

    # --- DISCOUNT CURVE LINKAGES (Global IDs) ---
    under_discount_curve_ids: npt.NDArray[np.int32]
    strike_discount_curve_ids: npt.NDArray[np.int32]

    # --- VOL-SURFACE Y-AXIS TYPE (int8 per trade) ---
    # Resolved at compile time from `surface_meta_dict[vol_container_id]
    # .y_axis_kind`. Non-option rows and option rows whose volatility
    # container is not a surface (e.g. a future vol-node container) get
    # `NOT_IMPLEMENTED_SENTINEL`. The option-curve pricer's pre-flight
    # check refuses to invoke its kernel with sentinel ids.
    vol_surface_y_axis_type_ids: npt.NDArray[np.int8]

    # --- NETTING SET LINKAGE ---
    netting_set_id: npt.NDArray[np.int64]

    def print_soa(
        self,
        *,
        max_rows: int = 50,
        int_fmt: str = ">10_.0f",
        float_fmt: str = ">16_.5f",
    ) -> None:
        """Print the SoA as a dense matrix."""
        n = min(self.size, max_rows)

        headers = [
            "uid",
            "inst_id",
            "qty",
            "k",
            "cp_mult",
            "set_ord",
            "del_ord",
            "exp_ord",
            "und_id",
            "vol_id",
            "vol_yax",
            "fx_u2s_id",
            "fx_u2s_inv",
            "fx_k2s_id",
            "fx_k2s_inv",
            "disc_und_id",
            "disc_k_id",
            "net_set_id",
        ]

        data = np.column_stack(
            [
                self.uids[:n],
                self.instrument_ids[:n],
                self.qty[:n],
                self.strike[:n],
                self.cp_mult[:n],
                self.settle_ord[:n],
                self.delivery_ord[:n],
                self.expiry_ord[:n],
                self.underlying_ids[:n],
                self.volatility_ids[:n],
                self.vol_surface_y_axis_type_ids[:n],
                self.fx_u2s_ids[:n],
                self.fx_u2s_inv[:n],
                self.fx_k2s_ids[:n],
                self.fx_k2s_inv[:n],
                self.under_discount_curve_ids[:n],
                self.strike_discount_curve_ids[:n],
                self.netting_set_id[:n],
            ]
        )

        fmts = [int_fmt] * len(headers)
        fmts[2] = fmts[3] = float_fmt  # qty, strike

        # Parse field width from format spec (e.g. ">16_.5f" -> 16)
        import re

        def _fmt_width(fmt: str) -> int:
            m = re.match(r"[<>^]?(\d+)", fmt)
            return int(m.group(1)) if m else len(fmt)

        widths = [_fmt_width(f) for f in fmts]

        header = " ".join(f"{h:>{w}}" for h, w in zip(headers, widths, strict=True))
        print(header)
        print("-" * len(header))

        for i in range(n):
            print(" ".join(f"{data[i, j]:{fmts[j]}}" for j in range(len(headers))))

        if self.size > max_rows:
            print(f"... ({self.size - max_rows} more rows)")


# ==============================================================================
# CORE HELPERS
# ==============================================================================


def _zero_series(index: pd.Index) -> pd.Series:
    """Float64 zero-fill series for absent numeric columns."""
    return pd.Series(0.0, index=index, dtype=np.float64)


def _blank_series(index: pd.Index) -> pd.Series:
    """NA-filled string series for absent string columns."""
    return pd.Series(pd.NA, index=index, dtype="string")


def _to_numpy_str(ser: pd.Series) -> npt.NDArray[np.str_]:
    """Extract a C-contiguous str array; NA collapses to the empty string."""
    return np.ascontiguousarray(ser.fillna("").to_numpy(dtype=str, copy=False))


def _to_f64(ser: pd.Series) -> npt.NDArray[np.float64]:
    """Extract a C-contiguous float64 array; NA collapses to 0.0."""
    return np.ascontiguousarray(ser.fillna(0.0).to_numpy(dtype=np.float64))


def _compile_option_multiplier(
    df: pd.DataFrame,
    n: int,
) -> npt.NDArray[np.int8]:
    """
    Pack `is_call` into the branchless int8 multiplier.

    +1 call, -1 put, 0 not-an-option. The encoding lets the pricer form the
    payoff sign by multiplication instead of branching.
    """
    cp_mult = np.empty(n, dtype=np.int8)
    cp_mult.fill(NOT_OPTION_MULT)

    if "is_call" not in df.columns:
        return cp_mult

    # is_call is materialized to nullable float64 (1.0 / 0.0 / NaN).
    raw = df["is_call"].to_numpy(copy=False)
    present = pd.notna(raw)
    flags = np.where(present, raw, 0.0).astype(np.int8)

    cp_mult[present & (flags == 1)] = CALL_OPTION_MULT
    cp_mult[present & (flags == 0)] = PUT_OPTION_MULT
    return cp_mult


def _compile_netting_set_id(
    df: pd.DataFrame,
    n: int,
) -> npt.NDArray[np.int64]:
    """
    Compile the effective-netting-set linkage into a flat int64 array.
    Absent column or NA row -> `NO_NETTING_SET_ID`.
    """
    out = np.empty(n, dtype=np.int64)
    if "netting_set_id" not in df.columns:
        out.fill(np.int64(NO_ID))
        return out

    filled = df["netting_set_id"].fillna(np.int64(NO_ID))
    return np.ascontiguousarray(filled.to_numpy(dtype=np.int64))


def _resolve_underlying(
    underlying_names: npt.NDArray[np.str_],
    static_config: RiskStaticConfig,
    n: int,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int8]]:
    """
    Classify each underlying name as a risk factor or a container and return
    a single collapsed semantic underlying_ids array, alongside the ccy_id.

    Vectorized: one pd.Series.map per universe, boolean-mask scatter.
    """
    name_to_rf = static_config.rel_name_to_rf_id
    name_to_container = static_config.container_name_to_id

    und_ser = pd.Series(underlying_names)
    rf_hits = und_ser.map(name_to_rf)
    container_hits = und_ser.map(name_to_container)

    is_rf = rf_hits.notna().to_numpy(copy=False)
    is_container = container_hits.notna().to_numpy(copy=False)

    both = is_rf & is_container
    if both.any():
        bad = sorted({str(v) for v in underlying_names[both]})[:10]
        raise PrettyKeyError(
            "Underlying name resolves to BOTH a risk factor AND a container - "
            f"the static config has a namespace collision. Offending: {bad}"
        )

    neither = ~(is_rf | is_container)
    if neither.any():
        bad = sorted({str(v) for v in underlying_names[neither] if str(v)})[:10]
        blanks = int((underlying_names[neither] == "").sum())
        raise PrettyKeyError(
            f"Underlying name resolves to NEITHER a risk factor NOR a container.\n"
            f"  unknown names ({len(bad)} of {int(neither.sum())} unique): {bad}\n"
            f"  blank/empty values: {blanks}"
        )

    # Dense semantic ID array
    underlying_ids = np.full(n, NO_ID, dtype=np.int32)
    # Currency ID array
    ccy_ids = np.full(n, -1, dtype=np.int8)

    if is_rf.any():
        rf_int32 = rf_hits[is_rf].to_numpy(dtype=np.int32, copy=False)
        underlying_ids[is_rf] = rf_int32
        ccy_ids[is_rf] = static_config.rf_id_to_ccy_id[rf_int32]

    if is_container.any():
        cont_int32 = container_hits[is_container].to_numpy(dtype=np.int32, copy=False)
        underlying_ids[is_container] = cont_int32
        cont_slots = static_config.container_slots(cont_int32)
        ccy_ids[is_container] = static_config.container_ccy_id[cont_slots]

    return underlying_ids, ccy_ids


def _resolve_underlying_ccy_strings(
    ccy_ids: npt.NDArray[np.int8],
    static_config: RiskStaticConfig,
    fallback_ccy: npt.NDArray[np.str_],
) -> npt.NDArray[np.str_]:
    """
    Derive the underlying's quote currency NAME per row.
    Maps the ccy_ids derived during underlying resolution back to strings.
    """
    n = ccy_ids.shape[0]
    ccy_id_to_name = static_config.ccy_id_to_name  # list[str], indexed by ccy_id
    out = np.empty(n, dtype=object)

    for i in range(n):
        cid = int(ccy_ids[i])
        if cid < 0:
            out[i] = fallback_ccy[i]
        else:
            out[i] = ccy_id_to_name[cid]

    return np.ascontiguousarray(out.astype(str))


# ==============================================================================
# VOLATILITY-CONTAINER RESOLUTION
# ==============================================================================


def _resolve_volatility_ids(
    names: npt.NDArray[np.str_],
    static_config: RiskStaticConfig,
) -> npt.NDArray[np.int32]:
    """
    Resolve the `volatility_container_name` column to int32 semantic ids.

    A volatility name addresses ONE of two universes, exactly like
    `underlying`:
      - a vol-NODE rf-id (e.g. ``AAPL_ATM_VOL``), read by the option-NODE
        kernel as ``vol_nodes[v_uix]`` — lives in ``rel_name_to_rf_id``;
      - a vol-SURFACE container id (e.g. ``AAPL_VOL_SURF``), interpolated
        by the option-CURVE kernel — lives in ``container_name_to_id``.

    Resolving against only ``container_name_to_id`` (the prior behaviour)
    made every node-typed ATM vol structurally unresolvable and hard-failed
    the ingest. We mirror `_resolve_underlying`: map BOTH universes, scatter
    into one collapsed id array, guard the namespace collision, and let
    blank rows (non-option trades) collapse to ``NO_ID``.

    Downstream is already node-aware: the resolved id flows into
    ``IngestBatch.volatility_ids`` -> ``market_id_universe`` (node-ids and
    container-ids coexist in that sorted-unique space), and
    ``_resolve_vol_surface_y_axis_type_ids`` returns
    ``NOT_IMPLEMENTED_SENTINEL`` for any id that is not a surface (the
    option-NODE kernel never reads y-axis type). So this is the only site
    that needed the dual lookup.
    """
    n = names.shape[0]
    blank_mask = names == ""

    ser = pd.Series(names)
    rf_hits = ser.map(static_config.rel_name_to_rf_id)
    container_hits = ser.map(static_config.container_name_to_id)

    is_rf = rf_hits.notna().to_numpy(copy=False)
    is_container = container_hits.notna().to_numpy(copy=False)

    both = is_rf & is_container
    if both.any():
        bad = sorted({str(v) for v in names[both]})[:10]
        raise PrettyKeyError(
            "volatility_container_name resolves to BOTH a risk factor AND a "
            "container — the static config has a namespace collision. "
            f"Offending: {bad}"
        )

    # Populated rows that hit neither universe are typos / stale universe.
    unresolved = ~(is_rf | is_container) & ~blank_mask
    if unresolved.any():
        bad = sorted({str(v) for v in names[unresolved]})[:10]
        raise PrettyKeyError(
            f"volatility_container_name: {int(unresolved.sum())} name(s) could "
            f"not be resolved to a risk-factor universe index "
            f"(checked node and container universes).\nUnresolved names:\n{bad}"
        )

    out = np.empty(n, dtype=np.int32)
    out.fill(np.int32(NO_ID))
    if is_rf.any():
        out[is_rf] = rf_hits[is_rf].to_numpy(dtype=np.int32, copy=False)
    if is_container.any():
        out[is_container] = container_hits[is_container].to_numpy(
            dtype=np.int32, copy=False
        )
    return out


def _resolve_vol_surface_y_axis_type_ids(
    volatility_ids: npt.NDArray[np.int32],
    static_config: RiskStaticConfig,
) -> npt.NDArray[np.int8]:
    """
    Map each non-sentinel `volatility_id` to its surface y-axis-type id (int8).

    The kernel's per-trade y-axis dispatch (STRIKE / LOG_STRIKE / MONEYNESS
    / LOG_MONEYNESS) reads from this column. Resolving here keeps the
    option-curve block compiler a pure slicer — no static-config coupling
    past the ingest edge.

    Post-calibration values
    -----------------------
    DELTA-quoted surfaces are calibrated to MONEYNESS_FWD at builder
    time (see `market_data.calibrate.delta_to_moneyness`). The helper
    `surface_y_axis_kind_for_kernel` translates the static-config kind
    string to the post-calibration int8 id, keeping the per-trade view
    here in lockstep with the per-surface view consumed by the runtime
    builder.

    Sentinel rows (`vol_id == NO_ID`) and rows whose `vol_id` is NOT a
    surface (e.g. an ATM-vol NODE container) collapse to
    `NOT_IMPLEMENTED_SENTINEL`. The option-curve pricer's pre-flight
    check refuses the sentinel — that's the contract; option-NODE rows
    never reach that pricer.

    Performance
    -----------
    O(n) Python loop. Vol-bearing rows are a small minority of the
    portfolio; the surface-meta dict lookup is O(1) and cached per
    container id.
    """
    n = volatility_ids.shape[0]
    out = np.empty(n, dtype=np.int8)
    out.fill(NOT_IMPLEMENTED_SENTINEL)

    surface_meta_dict = static_config.surface_meta_dict
    # Per-container cache: identical surface ids (the common case) only
    # hit the helper lookup once. Cached `NOT_IMPLEMENTED_SENTINEL`
    # records non-surface containers so we don't re-probe `.get`.
    sid_to_y_axis_id: dict[int, np.int8] = {}

    for i in range(n):
        vid = int(volatility_ids[i])
        if vid < 0:
            continue

        cached = sid_to_y_axis_id.get(vid)
        if cached is not None:
            out[i] = cached
            continue

        sm = surface_meta_dict.get(vid)
        if sm is None:
            # Resolved container is not a surface — e.g. an ATM-vol node
            # container consumed by the option-NODE kernel, which does
            # not read y-axis-type at all.
            sid_to_y_axis_id[vid] = np.int8(NOT_IMPLEMENTED_SENTINEL)
            continue

        try:
            y_id = surface_y_axis_kind_for_kernel(sm.y_axis_kind)
        except KeyError as exc:
            raise PrettyKeyError(
                f"surface_meta_dict[{vid}].y_axis_kind={sm.y_axis_kind!r} "
                f"is not a SurfaceYAxisType member. The static config and "
                f"the runtime enum have drifted."
            ) from exc

        sid_to_y_axis_id[vid] = y_id
        out[i] = y_id

    return out


# ==============================================================================
# MAIN COMPILER
# ==============================================================================


def compile_ingest_soa_payload_from_df(
    *,
    df: pd.DataFrame,
    instrument_ids: npt.NDArray[np.int32],
    static_config: RiskStaticConfig,
) -> CompiledIngestSoA:
    """
    Compile a validated ingest frame into a flat SoA payload.

    Parameters
    ----------
    df
        Canonicalized, materialized, validated ingest frame (index 0..n-1).
    instrument_ids
        Per-row int32 instrument ids from `dispatch_ingest_validation`.
    static_config
        Compiled RiskStaticConfig. Single source of truth for every name
        lookup, ccy translation, and FX-pair resolution.

    Returns
    -------
    CompiledIngestSoA
        Flat SoA payload.
    """
    n = len(df)
    if n == 0:
        raise ValueError("Empty ingest dataframe.")

    index = df.index

    # --------------------------------------------------------------------------
    # CORE ECONOMICS
    # --------------------------------------------------------------------------
    uids = np.ascontiguousarray(df["uid"].to_numpy(dtype=np.int64, copy=False))
    qty = _to_f64(df.get("quantity", _zero_series(index)))
    strike = _to_f64(df.get("strike_price", _zero_series(index)))
    cp_mult = _compile_option_multiplier(df, n)

    # --------------------------------------------------------------------------
    # DATES -> ORDINAL32 (NO UIX LOGIC)
    # --------------------------------------------------------------------------
    settle_ord = parse_optional_date_series_to_ord32(
        df.get("settlement_date", _blank_series(index)),
        col_name="settlement_date",
        sentinel=NO_DATE_ORD32,
    )
    delivery_ord = parse_optional_date_series_to_ord32(
        df.get("delivery_date", _blank_series(index)),
        col_name="delivery_date",
        sentinel=NO_DATE_ORD32,
    )
    expiry_ord = parse_optional_date_series_to_ord32(
        df.get("expiry_date", _blank_series(index)),
        col_name="expiry_date",
        sentinel=NO_DATE_ORD32,
    )

    # --------------------------------------------------------------------------
    # MARKET LINKAGES (DENSE SEMANTIC IDS)
    # --------------------------------------------------------------------------

    # --- RESOLVE UNDERLYING ---
    underlying_names = _to_numpy_str(df.get("underlying", _blank_series(index)))
    underlying_ids, under_ccy_ids = _resolve_underlying(
        underlying_names, static_config, n
    )

    # --- RESOLVE DISCOUNT CURVE CONTAINERS ---
    container_name_to_container_id = static_config.container_name_to_id

    under_df_curve_ids = resolve_optional_ids(
        _to_numpy_str(df.get("under_discount_curve", _blank_series(index))),
        container_name_to_container_id,
        sentinel=NO_ID,
        ctx="under_discount_curve",
    )
    strike_df_curve_ids = resolve_optional_ids(
        _to_numpy_str(df.get("strike_discount_curve", _blank_series(index))),
        container_name_to_container_id,
        sentinel=NO_ID,
        ctx="strike_discount_curve",
    )

    # --- RESOLVE VOL CONTAINERS + SURFACE Y-AXIS TYPE ---
    volatility_names = _to_numpy_str(
        df.get("volatility_container_name", _blank_series(index))
    )
    volatility_ids = _resolve_volatility_ids(volatility_names, static_config)
    vol_surface_y_axis_type_ids = _resolve_vol_surface_y_axis_type_ids(
        volatility_ids, static_config
    )

    # --------------------------------------------------------------------------
    # FX RESOLUTION (underlying / strike -> settlement)
    # --------------------------------------------------------------------------
    settle_ccy = _to_numpy_str(df.get("settlement_ccy", _blank_series(index)))

    # Derive the underlying CCY string
    under_ccy = _resolve_underlying_ccy_strings(
        ccy_ids=under_ccy_ids,
        static_config=static_config,
        fallback_ccy=settle_ccy,
    )

    # Strike ccy defaults to underlying ccy when blank.
    if "strike_ccy" in df.columns:
        strike_ccy = np.ascontiguousarray(
            df["strike_ccy"]
            .fillna(pd.Series(under_ccy, index=index))
            .to_numpy(dtype=str)
        )
    else:
        strike_ccy = under_ccy

    # ---- FX: UNDERLYING -> SETTLEMENT ----
    fx_u2s_ids = np.full(n, np.int32(NO_FX_RF_REQUIRED), dtype=np.int32)
    fx_u2s_inv = np.zeros(n, dtype=np.int8)
    resolve_fx_rf_ids_inplace(
        base_ccy=under_ccy,
        settlement_ccy=settle_ccy,
        out_fx_rf_ids=fx_u2s_ids,
        out_fx_invert=fx_u2s_inv,
        static_config=static_config,
    )

    # ---- FX: STRIKE -> SETTLEMENT ----
    fx_k2s_ids = np.full(n, np.int32(NO_FX_RF_REQUIRED), dtype=np.int32)
    fx_k2s_inv = np.zeros(n, dtype=np.int8)
    resolve_fx_rf_ids_inplace(
        base_ccy=strike_ccy,
        settlement_ccy=settle_ccy,
        out_fx_rf_ids=fx_k2s_ids,
        out_fx_invert=fx_k2s_inv,
        static_config=static_config,
    )

    # --------------------------------------------------------------------------
    # NETTING SET LINKAGE
    # --------------------------------------------------------------------------
    netting_set_id = _compile_netting_set_id(df, n)

    # --------------------------------------------------------------------------
    # ASSEMBLE
    # --------------------------------------------------------------------------
    return CompiledIngestSoA(
        size=n,
        instrument_ids=np.ascontiguousarray(instrument_ids, dtype=np.int32),
        uids=uids,
        qty=qty,
        strike=strike,
        cp_mult=cp_mult,
        settle_ord=settle_ord,
        delivery_ord=delivery_ord,
        expiry_ord=expiry_ord,
        underlying_ids=underlying_ids,
        volatility_ids=volatility_ids,
        fx_u2s_ids=fx_u2s_ids,
        fx_u2s_inv=fx_u2s_inv,
        fx_k2s_ids=fx_k2s_ids,
        fx_k2s_inv=fx_k2s_inv,
        under_discount_curve_ids=under_df_curve_ids,
        strike_discount_curve_ids=strike_df_curve_ids,
        vol_surface_y_axis_type_ids=vol_surface_y_axis_type_ids,
        netting_set_id=netting_set_id,
    )
