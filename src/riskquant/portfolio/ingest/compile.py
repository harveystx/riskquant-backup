# src/riskquant/portfolio/ingest/compile.py
"""
DataFrame -> Struct-of-Arrays compiler.

This is the final hop of the ingest edge: a validated, materialized Polars
DataFrame is collapsed into ``CompiledIngestSoA``, a flat homogeneous SoA
payload with no strings and no Python objects. ``IngestBatch.from_compiled``
wraps the result for the engine.

Refactor history
----------------
- Semantic Collapsing: `price_rf` and `price_curve` fused into dense `underlying_ids`.
- Semantic Collapsing: `vol_rf` and `vol_surface` fused into dense `volatility_ids`.
- Coordinate Purge: All `_uix` logic removed. Ingest acts purely as a global ID ledger.
- FX Linkages renamed to explicitly end in `_ids` to enforce boundary semantics.
- Underlying currency resolution optimized to occur concurrently with ID mapping.
- Real volatility-container resolution: the `volatility_container_name` schema
  column is mapped to int32 container ids and to the per-trade int8 y-axis-type
  id. Both fields land on ``CompiledIngestSoA``; ``IngestBatch.from_compiled`` is
  a pure copy with no synthetic sentinels.
- Pandas removal: pd.DataFrame -> pl.DataFrame. All pd.Index / pd.Series /
  pd.Series.map helpers are replaced with direct numpy dict lookups and Polars
  Series extraction. No pandas at import time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

import numpy as np
import numpy.typing as npt
import polars as pl

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
from riskquant.portfolio.ingest.identity import validate_market_linkage_identity
from riskquant.portfolio.ingest.parser import (
    parse_optional_date_series_to_ord32,
)
from riskquant.portfolio.ingest.rf_resolver import resolve_optional_ids
from riskquant.utils.dates import date_to_ord32

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

    # --- RUN AS-OF (D067) ---
    as_of_ord32: int = int(NO_DATE_ORD32)

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

        data = np.column_stack([
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
        ])

        fmts = [int_fmt] * len(headers)
        fmts[2] = fmts[3] = float_fmt  # qty, strike

        # Parse field width from format spec (e.g. ">16_.5f" -> 16)
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


def _get_f64_col(df: pl.DataFrame, col: str, n: int) -> npt.NDArray[np.float64]:
    """
    Extract C-contiguous float64 numpy array.

    Absent column -> zeros. Null values -> 0.0.
    Replaces the prior ``_to_f64(df.get(col, _zero_series(index)))`` pattern.
    """
    if col not in df.columns:
        return np.zeros(n, dtype=np.float64)
    return np.ascontiguousarray(df[col].fill_null(0.0).to_numpy().astype(np.float64))


def _get_str_col(df: pl.DataFrame, col: str, n: int) -> npt.NDArray[np.str_]:
    """
    Extract C-contiguous str numpy array.

    Absent column -> empty strings. Null values -> "".
    Replaces the prior ``_to_numpy_str(df.get(col, _blank_series(index)))`` pattern.
    """
    if col not in df.columns:
        return np.full(n, "", dtype=object).astype(str)
    return np.ascontiguousarray(
        df[col].cast(pl.Utf8).fill_null("").to_numpy().astype(str)
    )


def _get_date_col(df: pl.DataFrame, col: str, n: int) -> pl.Series:
    """
    Return date column as pl.Series, or an all-null Utf8 series if absent.

    parse_optional_date_series_to_ord32 treats all-null Utf8 as sentinel-fill.
    """
    if col not in df.columns:
        return pl.Series([None] * n, dtype=pl.Utf8)
    return df[col]


def _compile_option_multiplier(
    df: pl.DataFrame,
    n: int,
) -> npt.NDArray[np.int8]:
    """
    Pack ``is_call`` into the branchless int8 multiplier.

    +1 call, -1 put, 0 not-an-option. The encoding lets the pricer form the
    payoff sign by multiplication instead of branching.

    ``is_call`` is materialized to pl.Float64 (null for absent rows).
    Polars extracts null-Float64 as NaN in numpy, preserving identical null
    detection semantics to the prior pandas path.
    """
    cp_mult = np.empty(n, dtype=np.int8)
    cp_mult.fill(NOT_OPTION_MULT)

    if "is_call" not in df.columns:
        return cp_mult

    # pl.Float64 with nulls -> numpy float64 with NaN for nulls
    raw = df["is_call"].to_numpy()
    present = ~np.isnan(raw)
    flags = np.where(present, raw, 0.0).astype(np.int8)

    cp_mult[present & (flags == 1)] = CALL_OPTION_MULT
    cp_mult[present & (flags == 0)] = PUT_OPTION_MULT
    return cp_mult


def _compile_netting_set_id(
    df: pl.DataFrame,
    n: int,
) -> npt.NDArray[np.int64]:
    """
    Compile the effective-netting-set linkage into a flat int64 array.

    Absent column or null row -> ``NO_ID`` sentinel.

    pl.Int64 preserves nulls natively; ``fill_null`` replaces them with the
    sentinel before narrowing to numpy. This avoids the pandas 3.0
    ``IntCastingNaNError`` that the prior pandas path encountered when
    nullable float64 columns were narrowed to np.int64.
    """
    out = np.empty(n, dtype=np.int64)
    if "netting_set_id" not in df.columns:
        out.fill(np.int64(NO_ID))
        return out

    filled = df["netting_set_id"].fill_null(np.int64(NO_ID))
    return np.ascontiguousarray(filled.to_numpy().astype(np.int64))


_MISS = object()  # sentinel for "name not in dict"; unique object, never == int


def _resolve_underlying(
    underlying_names: npt.NDArray[np.str_],
    static_config: RiskStaticConfig,
    n: int,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int8]]:
    """
    Classify each underlying name as a risk factor or a container.

    Returns a collapsed ``underlying_ids`` array and the per-row ``ccy_id``
    (int8), both length n.

    Replaces the prior ``pd.Series(names).map(dict)`` dual-lookup with numpy
    dict comprehensions. Sentinel -1 is safe: valid RF/container IDs are
    non-negative.
    """
    name_to_rf = static_config.rel_name_to_rf_id
    name_to_container = static_config.container_name_to_id

    rf_hits = np.array(
        [name_to_rf.get(n_str, _MISS) for n_str in underlying_names], dtype=object
    )
    cont_hits = np.array(
        [name_to_container.get(n_str, _MISS) for n_str in underlying_names],
        dtype=object,
    )

    is_rf = np.array([v is not _MISS for v in rf_hits], dtype=bool)
    is_container = np.array([v is not _MISS for v in cont_hits], dtype=bool)

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

    underlying_ids = np.full(n, NO_ID, dtype=np.int32)
    ccy_ids = np.full(n, -1, dtype=np.int8)

    if is_rf.any():
        rf_int32 = np.array([rf_hits[i] for i in np.flatnonzero(is_rf)], dtype=np.int32)
        underlying_ids[is_rf] = rf_int32
        ccy_ids[is_rf] = static_config.rf_id_to_ccy_id[rf_int32]

    if is_container.any():
        cont_int32 = np.array(
            [cont_hits[i] for i in np.flatnonzero(is_container)], dtype=np.int32
        )
        underlying_ids[is_container] = cont_int32
        cont_slots = static_config.container_slots(
            cont_int32.astype(np.int32, copy=False)
        )
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
    Rows where ccy_id < 0 (underlying is absent) fall back to settlement ccy.
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
    Resolve the ``volatility_container_name`` column to int32 semantic ids.

    Mirrors ``_resolve_underlying``: maps both the RF universe
    (``rel_name_to_rf_id``) and the container universe
    (``container_name_to_id``). Blank rows (non-option trades) collapse to
    ``NO_ID``.
    """
    n = names.shape[0]
    blank_mask = names == ""

    rf_hits = np.array(
        [static_config.rel_name_to_rf_id.get(n_str, _MISS) for n_str in names],
        dtype=object,
    )
    cont_hits = np.array(
        [static_config.container_name_to_id.get(n_str, _MISS) for n_str in names],
        dtype=object,
    )

    is_rf = np.array([v is not _MISS for v in rf_hits], dtype=bool)
    is_container = np.array([v is not _MISS for v in cont_hits], dtype=bool)

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
        idx_rf = np.flatnonzero(is_rf)
        out[idx_rf] = np.array([rf_hits[i] for i in idx_rf], dtype=np.int32)

    if is_container.any():
        idx_cont = np.flatnonzero(is_container)
        out[idx_cont] = np.array([cont_hits[i] for i in idx_cont], dtype=np.int32)

    return out


def _resolve_vol_surface_y_axis_type_ids(
    volatility_ids: npt.NDArray[np.int32],
    static_config: RiskStaticConfig,
) -> npt.NDArray[np.int8]:
    """
    Map each non-sentinel ``volatility_id`` to its surface y-axis-type id (int8).

    The kernel's per-trade y-axis dispatch (STRIKE / LOG_STRIKE / MONEYNESS
    / LOG_MONEYNESS) reads from this column. Resolving here keeps the
    option-curve block compiler a pure slicer — no static-config coupling
    past the ingest edge.

    Post-calibration values
    -----------------------
    DELTA-quoted surfaces are calibrated to MONEYNESS_FWD at builder
    time (see ``market_data.calibrate.delta_to_moneyness``). The helper
    ``surface_y_axis_kind_for_kernel`` translates the static-config kind
    string to the post-calibration int8 id, keeping the per-trade view
    here in lockstep with the per-surface view consumed by the runtime
    builder.

    Sentinel rows (``vol_id == NO_ID``) and rows whose ``vol_id`` is NOT a
    surface (e.g. an ATM-vol NODE container) collapse to
    ``NOT_IMPLEMENTED_SENTINEL``. The option-curve pricer's pre-flight
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
    # hit the helper lookup once. Cached ``NOT_IMPLEMENTED_SENTINEL``
    # records non-surface containers so we don't re-probe ``.get``.
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
    df: pl.DataFrame,
    instrument_ids: npt.NDArray[np.int32],
    static_config: RiskStaticConfig,
    as_of_date: date | None = None,
) -> CompiledIngestSoA:
    """
    Compile a validated ingest frame into a flat SoA payload.

    Parameters
    ----------
    df
        Canonicalized, materialized, validated ingest frame (``pl.DataFrame``).
    instrument_ids
        Per-row int32 instrument ids from ``dispatch_ingest_validation``.
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

    # D067 / AUD-04: the run as-of must be a real date. A date whose ordinal
    # equals NO_DATE_ORD32 would be stored as the sentinel and read downstream
    # as "no as-of", silently erasing the date that fixes relative identity.
    as_of_ord32 = int(NO_DATE_ORD32)
    if as_of_date is not None:
        as_of_ord32 = int(date_to_ord32(as_of_date))
        if as_of_ord32 == int(NO_DATE_ORD32):
            raise ValueError(
                f"as_of_date {as_of_date} has ordinal {as_of_ord32}, the "
                "NO_DATE_ORD32 sentinel; it cannot be a run as-of (D067)."
            )

    # --------------------------------------------------------------------------
    # CORE ECONOMICS
    # --------------------------------------------------------------------------
    uids = np.ascontiguousarray(df["uid"].to_numpy().astype(np.int64))
    qty = _get_f64_col(df, "quantity", n)
    strike = _get_f64_col(df, "strike_price", n)
    cp_mult = _compile_option_multiplier(df, n)

    # --------------------------------------------------------------------------
    # DATES -> ORDINAL32 (NO UIX LOGIC)
    # --------------------------------------------------------------------------
    # _get_date_col returns an all-null Utf8 series for absent columns, which
    # parse_optional_date_series_to_ord32 treats as sentinel-fill.
    settle_ord = parse_optional_date_series_to_ord32(
        _get_date_col(df, "settlement_date", n),
        col_name="settlement_date",
        sentinel=NO_DATE_ORD32,
    )
    delivery_ord = parse_optional_date_series_to_ord32(
        _get_date_col(df, "delivery_date", n),
        col_name="delivery_date",
        sentinel=NO_DATE_ORD32,
    )
    expiry_ord = parse_optional_date_series_to_ord32(
        _get_date_col(df, "expiry_date", n),
        col_name="expiry_date",
        sentinel=NO_DATE_ORD32,
    )

    # --------------------------------------------------------------------------
    # MARKET LINKAGES (DENSE SEMANTIC IDS)
    # --------------------------------------------------------------------------

    # --- RESOLVE UNDERLYING ---
    underlying_names = _get_str_col(df, "underlying", n)
    underlying_ids, under_ccy_ids = _resolve_underlying(
        underlying_names, static_config, n
    )

    # --- RESOLVE DISCOUNT CURVE CONTAINERS ---
    container_name_to_container_id = static_config.container_name_to_id

    under_df_curve_ids = resolve_optional_ids(
        _get_str_col(df, "under_discount_curve", n),
        container_name_to_container_id,
        sentinel=NO_ID,
        ctx="under_discount_curve",
    )
    strike_df_curve_ids = resolve_optional_ids(
        _get_str_col(df, "strike_discount_curve", n),
        container_name_to_container_id,
        sentinel=NO_ID,
        ctx="strike_discount_curve",
    )

    # --- RESOLVE VOL CONTAINERS + SURFACE Y-AXIS TYPE ---
    volatility_names = _get_str_col(df, "volatility_container_name", n)
    volatility_ids = _resolve_volatility_ids(volatility_names, static_config)
    vol_surface_y_axis_type_ids = _resolve_vol_surface_y_axis_type_ids(
        volatility_ids, static_config
    )

    # --------------------------------------------------------------------------
    # FX RESOLUTION (underlying / strike -> settlement)
    # --------------------------------------------------------------------------
    settle_ccy = _get_str_col(df, "settlement_ccy", n)

    # Derive the underlying CCY string
    under_ccy = _resolve_underlying_ccy_strings(
        ccy_ids=under_ccy_ids,
        static_config=static_config,
        fallback_ccy=settle_ccy,
    )

    # Strike ccy defaults to underlying ccy where blank (null in Polars).
    if "strike_ccy" in df.columns:
        # Fill nulls with corresponding under_ccy elements.
        # Cannot use pl.Series.fill_null(another_pl.Series); use numpy.
        sc_raw = df["strike_ccy"].fill_null("").to_numpy().astype(str)
        blank_sc = sc_raw == ""
        sc_raw[blank_sc] = under_ccy[blank_sc]
        strike_ccy = np.ascontiguousarray(sc_raw)
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
    # SEMANTIC IDENTITY GATE (D063 source kind, D065 contract, D066 discounting)
    # --------------------------------------------------------------------------
    # Last layer that still knows what each integer id IS. Everything after this
    # point indexes untyped ids.
    validate_market_linkage_identity(
        instrument_ids=np.ascontiguousarray(instrument_ids, dtype=np.int32),
        uids=uids,
        underlying_ids=underlying_ids,
        volatility_ids=volatility_ids,
        under_ccy=under_ccy,
        strike_ccy=strike_ccy,
        under_discount_curve_ids=under_df_curve_ids,
        strike_discount_curve_ids=strike_df_curve_ids,
        delivery_ord=delivery_ord,
        as_of_date=as_of_date,
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
        as_of_ord32=as_of_ord32,
    )
