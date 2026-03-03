# riskquant/simlab/resolve/proxy.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Final

import numpy as np

from riskquant.config.deprecated_configs import RiskStaticConfig
from riskquant.exceptions import PrettyKeyError
from riskquant.utils.tenors import parse_bracket_token

# =============================== #
# --- SOURCE SELECTION CODES ---- #
# =============================== #

# --- RESOLUTION CODES --- #
RES_NOT_SIMULATED: Final[np.int8] = np.int8(-1)
RES_DIRECT: Final[np.int8] = np.int8(0)
RES_TENOR_FALLBACK: Final[np.int8] = np.int8(1)
RES_ID_PROXY: Final[np.int8] = np.int8(2)
RES_CURVE_PROXY: Final[np.int8] = np.int8(3)

# ======================================================= #
# --- COMPILED OUTPUT: REQUIRED -> SOURCE (PROXY) MAP --- #
# ======================================================= #


@dataclass(frozen=True, slots=True)
class ProxyResolution:
    """
    Compiled proxy mapping between portfolio-required risk factors and
    available market data risk factors.

    This object defines, for each required risk factor ID referenced by
    instruments in the compiled portfolio:

      - which historical time series should be used to generate shocks
      - how those shocks should be scaled
      - how to efficiently reuse shared source series across multiple exposures

    It is a pure mapping layer:
      - no current levels
      - no shift types
      - no currency logic
      - no valuation logic

    Attributes
    ----------
    required_factor_ids : (N,) int32
        Ordered list of risk factor IDs required by portfolio exposures.
        This is the domain being resolved (portfolio universe A).

    source_factor_ids : (N,) int32
        For each required_factor_id, the resolved risk factor ID whose
        historical time series will be used as the shock source.
        Each entry is guaranteed to exist in RiskFactorMatrix.factor_ids
        (market data universe B).

    required_to_source_index : (N,) int32
        For each required_factor_id, an integer index into unique_source_factor_ids.
        Enables:
          - computing returns/shocks once per unique source series (K series),
          - then broadcasting those shocks to all required IDs that map to it.

    shock_scale : (N,) float32
        Volatility scaling factor applied to the shock derived from the source
        factor for this required exposure. Defaults to 1.0 when no scaling is
        configured.

        IMPORTANT: scaling applies to the shock (return/diff), not to levels.

    unique_source_factor_ids : (K,) int32
        Deduplicated list of source factor IDs required from the RiskFactorMatrix.
        K <= N. The ordering is deterministic: first-seen order during resolution.

    resolution_code : (N,) int8
        Per-required-id resolution type code:

          - RES_DIRECT        (0): required_id exists directly in market data universe B
          - RES_ID_PROXY      (2): explicit ID-level proxy (risk_factor_proxies) applied
          - RES_CURVE_PROXY   (3): explicit curve-level proxy (curve_proxies) applied
          - RES_TENOR_FALLBACK(1): same-curve nearest-tenor fallback within B applied
    """

    # Attribute list
    required_factor_ids: np.ndarray
    source_factor_ids: np.ndarray
    required_to_source_index: np.ndarray
    shock_scale: np.ndarray
    unique_source_factor_ids: np.ndarray
    resolution_code: np.ndarray


# ========================================== #
# --- SMALL HOT HELPERS (NO ALLOCATIONS) --- #
# ========================================== #


def _nearest_offset_match(
    *,
    offsets_sorted: np.ndarray,
    rf_ids_sorted: np.ndarray,
    target: int,
) -> int:
    """
    Nearest neighbor selection on signed OFFSET tenors.

    - offsets_sorted: sorted int32 offsets for proxy curve
    - rf_ids_sorted:  rf_id aligned with offsets_sorted
    - target:         required signed offset to match

    Deterministic tie-break:
      - if equally close to two offsets, choose the lower offset (j-1).
    """
    j = int(np.searchsorted(offsets_sorted, np.int32(target), side="left"))
    if j <= 0:
        return int(rf_ids_sorted[0])
    if j >= offsets_sorted.shape[0]:
        return int(rf_ids_sorted[-1])

    lo = int(offsets_sorted[j - 1])
    hi = int(offsets_sorted[j])
    if (target - lo) <= (hi - target):
        return int(rf_ids_sorted[j - 1])
    return int(rf_ids_sorted[j])


def _build_available_curve_tenor_index(
    *,
    cfg: RiskStaticConfig,
    market_data_factor_ids: np.ndarray,  # (Nb,) int32 (B)
) -> Dict[str, Tuple[np.ndarray, np.ndarray, Dict[str, int]]]:
    """
    Build a per-curve tenor lookup restricted to the IDs that actually exist in market data (B).

    Output:
        curve_name -> (offsets_sorted, rf_ids_sorted, label_to_rf_id)

    This is used for the default "same-curve extrapolation" behavior:
      - if required_id is missing in B but the curve exists in B, match to nearest tenor on the same curve.

    Notes (CPU/RAM):
      - Allocates small per-curve Python lists/dicts, proportional to Nb (market data size), not config size.
      - Uses cfg.rf_id_to_risk_factor for O(1) id -> (curve, tenor_token).
      - Sorting is per-curve and typically small.
    """
    # --- COLLECT OFFSETS + LABELS FOR IDS PRESENT IN MARKET DATA --- #
    offsets: Dict[str, list[int]] = {}
    offset_ids: Dict[str, list[int]] = {}
    labels: Dict[str, Dict[str, int]] = {}

    rf_by_id = cfg.rf_id_to_risk_factor  # local binding

    for mid in market_data_factor_ids:
        rid = int(mid)
        rf = rf_by_id.get(rid)
        if rf is None:
            # Market data contains an id unknown to config: fail-fast, because
            # tenor/curve matching is undefined without config metadata.
            raise PrettyKeyError(
                f"market_data_factor_id not found in config: {rid}"
            )

        curve = rf.container_name
        tok = rf.tenor_token

        if tok is None:
            # NONE tenor => empty label key
            labels.setdefault(curve, {})[""] = rid
            continue

        unit, off = parse_bracket_token(tok)
        if off is None:
            # LABEL tenor => exact label match
            labels.setdefault(curve, {})[unit] = rid
        else:
            # OFFSET tenor => nearest neighbor match
            offsets.setdefault(curve, []).append(int(off))
            offset_ids.setdefault(curve, []).append(rid)

    # --- MATERIALIZE SORTED ARRAYS PER CURVE --- #
    out: Dict[str, Tuple[np.ndarray, np.ndarray, Dict[str, int]]] = {}

    for curve, olist in offsets.items():
        ids_list = offset_ids[curve]
        o = np.asarray(olist, dtype=np.int32)
        r = np.asarray(ids_list, dtype=np.int32)
        idx = np.argsort(o, kind="mergesort")  # deterministic ties
        out[curve] = (o[idx], r[idx], labels.get(curve, {}))

    for curve, d in labels.items():
        if curve not in out:
            out[curve] = (
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                d,
            )

    return out


# ============================================================= #
# --- MAIN RESOLVER: REQUIRED IDS -> MARKET DATA SOURCE IDS --- #
# ============================================================= #


def resolve_required_factor_ids_to_market_data_factor_ids(
    *,
    cfg: RiskStaticConfig,
    required_factor_ids: np.ndarray,  # (N,) int32 (A)
    market_data_factor_ids: np.ndarray,  # (Nb,) int32 (B)
) -> ProxyResolution:
    """
    Resolve each required risk factor ID onto a source ID present in market data.

    Input universes:
      - required_factor_ids: portfolio-required factor IDs (A)
      - market_data_factor_ids: factor IDs present in RiskFactorMatrix.factor_ids (B)

    Resolution priority (first match wins):
      (1) Direct: required_id exists in B

      (2) ID-level proxy (explicit, highest override):
            - cfg.rf_id_to_proxy_rf_id[required_id] -> (proxy_id, scale)

      (3) Curve-level proxy (explicit, second override):
            - map required curve -> proxy curve via cfg.curve_proxies_dict
            - match tenor on proxy curve (LABEL exact match; OFFSET nearest match)
            - apply curve-level scaling factor

      (4) Same-curve tenor fallback (default extrapolation, lowest priority):
            - if required_id missing but the curve exists in B, match tenor on the *same* curve
              (LABEL exact match; OFFSET nearest match, including extrapolation to end points)
            - scale = 1.0

    IMPORTANT DESIGN RULE:
      - This resolver selects *data sources only*.
      - Any metadata (ccy / shift type / risk factor group) must come from the
        *required* curve later, not from the proxy curve.
    """

    # --- SMOKE TESTS --- #
    if required_factor_ids.dtype != np.int32:
        raise TypeError(
            f"required_factor_ids must be int32; got {required_factor_ids.dtype}"
        )
    if market_data_factor_ids.dtype != np.int32:
        raise TypeError(
            f"market_data_factor_ids must be int32; got {market_data_factor_ids.dtype}"
        )

    # --- FAST MEMBERSHIP STRUCTURE FOR AVAILABLE MARKET DATA IDS --- #
    #
    # avail_get(rf_id) is not None <=> rf_id exists in market data universe B
    avail: Dict[int, int] = {}
    for mid in market_data_factor_ids:
        avail[int(mid)] = 1
    avail_get = avail.get

    # --- CONFIG LOOKUPS (LOCAL BINDINGS) --- #
    #
    # Local bindings reduce attribute resolution overhead inside the per-id loop.
    id_proxy_get = cfg.rf_id_to_proxy_rf_id.get
    rf_by_id = cfg.rf_id_to_risk_factor
    curve_proxy_get = cfg.curve_proxies_dict.get

    # --- BUILD MARKET-DATA-RESTRICTED CURVE INDEX (FOR CURVE PROXY + TENOR FALLBACK) --- #
    #
    # Critical design rule:
    #   Any chosen source_id must exist in market data universe B.
    # Therefore:
    #   - same-curve fallback matches tenors using only IDs that exist in B
    #   - curve-level proxy also matches tenors using only IDs that exist in B
    #
    # This prevents "config-only" proxy IDs that are absent from market data.
    avail_curve_index = _build_available_curve_tenor_index(
        cfg=cfg,
        market_data_factor_ids=market_data_factor_ids,
    )
    avail_curve_index_get = avail_curve_index.get

    # --- OUTPUT ARRAYS --- #
    num_required_ids = int(required_factor_ids.shape[0])
    source_factor_ids = np.empty(num_required_ids, dtype=np.int32)
    shock_scale = np.empty(num_required_ids, dtype=np.float32)
    resolution_code = np.empty(num_required_ids, dtype=np.int8)

    # --- UNIQUE SOURCE UNIVERSE (FIRST-SEEN ORDER) --- #
    #
    # We build:
    #   src_to_uix: source_id -> universe index (0..K-1)
    #   unique_src: list of unique source ids in first-seen order
    src_to_uix: Dict[int, int] = {}
    unique_src: list[np.int32] = []
    src_to_uix_get = src_to_uix.get
    src_to_uix_set = src_to_uix.__setitem__
    unique_append = unique_src.append

    # --- RESOLVE EACH REQUIRED ID --- #
    for i in range(num_required_ids):
        required_id = int(required_factor_ids[i])

        # ============================================================ #
        # --- (0) REQUIRED METADATA LOOKUP (CURVE + TENOR TOKEN) --- #
        # ============================================================ #
        #
        # We need curve_name and tenor_token for:
        #   - curve-level proxy tenor matching
        #   - same-curve tenor fallback matching
        rf = rf_by_id.get(required_id)
        if rf is None:
            raise PrettyKeyError(
                f"Required factor_id not found in config: {required_id}"
            )

        required_curve = rf.container_name
        tok = rf.tenor_token
        if tok is None:
            target_unit = ""
            target_off = None
        else:
            target_unit, target_off = parse_bracket_token(tok)

        # ================================= #
        # --- (1) DIRECT (NO PROXYING) --- #
        # ================================= #
        if avail_get(required_id) is not None:
            source_id = required_id
            sc = 1.0
            rc = RES_DIRECT

        else:
            # =================================================== #
            # --- (2) ID-LEVEL PROXY (EXPLICIT OVERRIDE) --- #
            # =================================================== #
            #
            # Highest priority override: if the user explicitly proxied the point,
            # we use it even if other fallbacks would be possible.
            pr = id_proxy_get(required_id)
            if pr is not None:
                source_id = int(pr[0])
                sc = float(pr[1])
                rc = RES_ID_PROXY

                # Must exist in market data universe B.
                if avail_get(source_id) is None:
                    raise PrettyKeyError(
                        "ID-level proxy resolved to missing market data factor_id:\n"
                        f"- required_id={required_id}\n"
                        f"- required_curve={required_curve!r}\n"
                        f"- proxy_id={source_id}\n"
                    )

            else:
                # ===================================================== #
                # --- (3) CURVE-LEVEL PROXY (EXPLICIT OVERRIDE) --- #
                # ===================================================== #
                #
                # Second priority override: if the curve has been configured to proxy
                # onto another curve, use that proxy curve's data + tenor matching.
                cp = curve_proxy_get(required_curve)
                if cp is not None:
                    proxy_curve = cp.proxy_curve_name
                    curve_scale = float(
                        cp.proxy_curve_vol_scaling_factor or 1.0
                    )

                    # --- MARKET-DATA-RESTRICTED TENOR MATCH ON PROXY CURVE --- #
                    #
                    # We match on proxy_curve using only IDs that exist in B, otherwise
                    # we'd select an ID not present in the matrix (the original bug).
                    ci_px = avail_curve_index_get(proxy_curve)
                    if ci_px is None:
                        raise PrettyKeyError(
                            "Curve-level proxy curve has no market data coverage:\n"
                            f"- required_id={required_id}\n"
                            f"- required_curve={required_curve!r}\n"
                            f"- proxy_curve={proxy_curve!r}\n"
                            f"- tenor_token={tok!r}\n"
                        )

                    offsets_sorted_px, rf_ids_sorted_px, label_map_px = ci_px

                    # LABEL/NONE tenor: exact label match on proxy curve (within B)
                    if target_off is None:
                        sid2 = label_map_px.get(target_unit)
                        if sid2 is None:
                            raise PrettyKeyError(
                                "Cannot proxy LABEL/NONE tenor onto proxy curve (no matching label in market data):\n"
                                f"- required_id={required_id}\n"
                                f"- required_curve={required_curve!r}\n"
                                f"- proxy_curve={proxy_curve!r}\n"
                                f"- tenor_token={tok!r}\n"
                                f"- label={target_unit!r}\n"
                            )
                        source_id = int(sid2)

                    # OFFSET tenor: nearest signed offset match on proxy curve (within B)
                    else:
                        if offsets_sorted_px.shape[0] == 0:
                            raise PrettyKeyError(
                                "Proxy curve has no OFFSET tenors in market data to match required tenor:\n"
                                f"- required_id={required_id}\n"
                                f"- required_curve={required_curve!r}\n"
                                f"- proxy_curve={proxy_curve!r}\n"
                                f"- tenor_token={tok!r}\n"
                            )
                        source_id = _nearest_offset_match(
                            offsets_sorted=offsets_sorted_px,
                            rf_ids_sorted=rf_ids_sorted_px,
                            target=int(target_off),
                        )

                    # Defensive: should always be in B by construction.
                    if avail_get(source_id) is None:
                        raise PrettyKeyError(
                            "Internal error: curve-level proxy produced id not in market data:\n"
                            f"- required_id={required_id}\n"
                            f"- required_curve={required_curve!r}\n"
                            f"- proxy_curve={proxy_curve!r}\n"
                            f"- proxy_id={source_id}\n"
                        )

                    sc = curve_scale
                    rc = RES_CURVE_PROXY

                else:
                    # =============================================================== #
                    # --- (4) SAME-CURVE TENOR FALLBACK (DEFAULT EXTRAPOLATION) --- #
                    # =============================================================== #
                    #
                    # Lowest priority behavior: if we have the curve in market data but
                    # not the exact tenor point, map to the nearest available tenor on
                    # the *same curve within B* (end-point extrapolation by selection).
                    ci_av = avail_curve_index_get(required_curve)
                    if ci_av is None:
                        raise PrettyKeyError(
                            "Missing history for required_id and no proxy configured:\n"
                            f"- required_id={required_id}\n"
                            f"- required_curve={required_curve!r}\n"
                            f"- tenor_token={tok!r}\n"
                            "No ID proxy, no curve proxy, and curve not present in market data universe.\n"
                        )

                    offsets_sorted, rf_ids_sorted, label_map = ci_av

                    # LABEL/NONE tenor: exact label match on same curve (within B)
                    if target_off is None:
                        sid2 = label_map.get(target_unit)
                        if sid2 is None:
                            raise PrettyKeyError(
                                "Same-curve fallback failed for LABEL/NONE tenor:\n"
                                f"- required_id={required_id}\n"
                                f"- required_curve={required_curve!r}\n"
                                f"- tenor_token={tok!r}\n"
                                f"- label={target_unit!r}\n"
                                "Curve exists in market data, but no matching label exists.\n"
                            )
                        source_id = int(sid2)

                    # OFFSET tenor: nearest signed offset match on same curve (within B)
                    else:
                        if offsets_sorted.shape[0] == 0:
                            raise PrettyKeyError(
                                "Same-curve fallback failed: curve has no OFFSET tenors in market data:\n"
                                f"- required_id={required_id}\n"
                                f"- required_curve={required_curve!r}\n"
                                f"- tenor_token={tok!r}\n"
                            )
                        source_id = _nearest_offset_match(
                            offsets_sorted=offsets_sorted,
                            rf_ids_sorted=rf_ids_sorted,
                            target=int(target_off),
                        )

                    # Defensive: should always be in B by construction.
                    if avail_get(source_id) is None:
                        raise PrettyKeyError(
                            "Internal error: same-curve fallback produced id not in market data:\n"
                            f"- required_id={required_id}\n"
                            f"- required_curve={required_curve!r}\n"
                            f"- resolved_source_id={source_id}\n"
                        )

                    sc = 1.0
                    rc = RES_TENOR_FALLBACK

        # --- WRITE PER-REQUIRED OUTPUTS --- #
        source_factor_ids[i] = np.int32(source_id)
        shock_scale[i] = np.float32(sc)
        resolution_code[i] = np.int8(rc)

        # --- MAINTAIN UNIQUE SOURCE UNIVERSE --- #
        uix = src_to_uix_get(source_id)
        if uix is None:
            uix = len(unique_src)
            src_to_uix_set(source_id, uix)
            unique_append(np.int32(source_id))

    unique_source_factor_ids = np.asarray(unique_src, dtype=np.int32)

    # --- BUILD REQUIRED -> UNIQUE SOURCE INDEX VECTOR --- #
    required_to_source_index = np.empty(num_required_ids, dtype=np.int32)
    for i in range(num_required_ids):
        required_to_source_index[i] = np.int32(
            src_to_uix[int(source_factor_ids[i])]
        )

    # build and return
    return ProxyResolution(
        required_factor_ids=required_factor_ids,
        source_factor_ids=source_factor_ids,
        required_to_source_index=required_to_source_index,
        shock_scale=shock_scale,
        unique_source_factor_ids=unique_source_factor_ids,
        resolution_code=resolution_code,
    )
