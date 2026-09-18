# src/riskquant/portfolio/compilers/block_compilers/vanilla_european_option.py
"""
Vanilla European option block compiler.

Translates the option sub-range of an `IngestBatch` (global IDs) into a
`VanillaEuropeanOptionBlock` (execution UIXs) via the run's `UixMapper`.

Supported economic contract
---------------------------
European option on the STRIKE-CURRENCY forward, Black-76, cash settled in the
strike currency at `settle_ord` and reported in settlement currency:

    FV(T_exp) = cp * [ F_K(T_exp) Phi(cp d1) - K Phi(cp d2) ]     (strike ccy)
    PV(0)     = Q * FV(T_exp) * DF_K(0, T_settle) * FX_spot(K -> settle)

    d1 = (ln(F_K / K) + 0.5 sigma^2 T_exp) / (sigma sqrt(T_exp))
    d2 = d1 - sigma sqrt(T_exp)

`cp = +1` call, `cp = -1` put; `Q` is signed, so a short call is `Q < 0` with
`cp = +1`. The four economic dates are distinct and must stay so: the forward
measure and the volatility use T_exp; the discount factor uses T_settle.

What `F_K` is
-------------
NOT a new market quantity. It is the ratio of two effective rows the mapper
already carries, both taken at EXPIRY:

    F_K(T) = effective_forward_prices[efp] / effective_discount_factors[edf]
           = F_ccy(T) DF_ccy(0,T) FX(ccy->settle)
             / ( DF_K(0,T) FX(K->settle) )

which is the covered-interest-parity construction `market_data.state`
documents: discount the asset-currency forward to PV, translate through the
settlement numeraire, re-accumulate on the strike curve.

VAL-11 pairing
--------------
`forward_price_strike_ccy_uix` and `option_log_forward_price_strike_ccy_uix`
index two arrays that are generated from ONE table --
`_fold_forward_prices_strike_ccy` returns the forward and its elementwise log
together, row for row. The two UIX are therefore the SAME integer, stamped once
here, and `L[k] == log(F[k])` holds by construction. The kernel never searches
for a matching forward, recomputes a log, or validates the pair.

Both arrays are retained rather than collapsed to one: the log is a hot-path
cache computed once per scenario batch instead of once per trade per scenario,
which is the same reason `log_strike`, `sqrt_tte` and `half_tte` are
precomputed. Collapsing the two block fields into one is a possible later
simplification; it would change the kernel signature and is out of scope here.

Volatility identity
-------------------
`vol_uix` must name the volatility of the SAME stochastic quantity that
`forward_price_strike_ccy_uix` carries -- the strike-currency forward.

When the asset and strike currencies differ, the strike-currency forward is a
product of the asset forward and an FX forward, so its volatility is not the
asset forward's. Nothing in `IngestBatch` or `MarketDataMatrix` supplies an FX
volatility or a correlation, so no quanto adjustment is constructible here. The
contract is therefore a requirement on the FEED: `volatility_ids` must name a
surface already quoted in the option's own measure. This compiler binds the row
and cannot verify the claim (D045/X-07, still open).

Source kind
-----------
NODE-sourced options only. `VANILLA_EUROPEAN_OPTION_CURVE_ID` reads
`MarketDataMatrix.vol_surfaces` and needs a surface interpolation the flat
`vols` universe does not model; it stays unregistered so the engine fails loud
rather than pricing a surface option off a node row.

UIX translation
---------------
    underlying + under_curve + expiry + fx_u2s  -> efp_uix    (intermediate)
    strike_curve + expiry     + fx_k2s          -> edf_exp    (intermediate)
    (efp_uix, edf_exp)                          -> forward_price_strike_ccy_uix
                                                   AND the log uix (same value)
    volatility_ids                              -> vol_uix
    strike_curve + settle     + fx_k2s          -> effective_discount_factor_uix

Fail-fast ownership (D045/VAL-04, option path)
----------------------------------------------
This compiler proves every UIX it stamps is a valid dense row of a POPULATED
universe, and proves the block's own precomputed invariants (`strike > 0`,
`log_strike == log(strike)`, `half_tte == 0.5 * sqrt_tte**2`) before emitting.
The kernel runs under `boundscheck=False` and performs no validation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from riskquant.portfolio.ingest.batch import IngestBatch
from riskquant.valuation.blocks.vanilla_european_option_block import (
    VanillaEuropeanOptionBlock,
)

if TYPE_CHECKING:
    from riskquant.universe import UixMapper


def _require_resolved(
    uix: np.ndarray,
    *,
    n_rows: int,
    leg: str,
    uids: np.ndarray,
) -> None:
    """Prove `0 <= uix < n_rows` for every row before the block is emitted."""
    bad = (uix < 0) | (uix >= n_rows)
    if not bool(bad.any()):
        return
    offenders = uids[bad][:10].tolist()
    raise ValueError(
        f"vanilla european option: {int(bad.sum())} row(s) have an unresolved "
        f"or out-of-range {leg} coordinate against a universe of {n_rows} "
        f"row(s). First offending trade uid(s): {offenders}. An option cannot "
        "be compiled until its market-data legs are present in the run's "
        "effective-factor universes."
    )


def compile_vanilla_european_option_block(
    batch: IngestBatch,
    start: int,
    end: int,
    uix_mapper: UixMapper,
) -> VanillaEuropeanOptionBlock | None:
    """Slice + UIX-translate a vanilla European option run from `batch[start:end]`."""
    if start == end:
        return None

    uids = batch.uids[start:end]
    strike = batch.strike_price[start:end]

    # ---- ECONOMIC DOMAIN --------------------------------------------------
    if not bool(np.all(strike > 0.0)):
        bad = uids[strike <= 0.0][:10].tolist()
        raise ValueError(
            "vanilla european option: strike must be strictly positive "
            f"(log-moneyness is undefined otherwise). Offending uid(s): {bad}"
        )

    expiry_ord = batch.expiry_ord[start:end]
    settle_ord = batch.settle_ord[start:end]
    if not bool(np.all(settle_ord >= expiry_ord)):
        bad = uids[settle_ord < expiry_ord][:10].tolist()
        raise ValueError(
            "vanilla european option: settlement precedes expiry on uid(s): "
            f"{bad}. Black-76 takes the forward measure at expiry and discounts "
            "from settlement; a settlement before expiry is not a deferred "
            "settlement, it is a malformed trade."
        )

    # ---- EXPIRED-OPTION POLICY (D068) -------------------------------------
    # An option whose expiry precedes the run as-of is not a live Black-76
    # position: it has either exercised or lapsed, and its settlement belongs to
    # the lifecycle feed, not to the pricer. It is rejected, not floored to
    # TTE_FLOOR. Expiry ON the as-of stays priceable: TTE_FLOOR is then the
    # documented representation of the intrinsic-value limit.
    if uix_mapper.as_of_ord32 is None:
        raise ValueError(
            "vanilla european option: UixMapper was built without `as_of_ord32`, so "
            "expiry cannot be checked or converted to time (D067)."
        )
    expired = expiry_ord < np.int32(uix_mapper.as_of_ord32)
    if bool(expired.any()):
        raise ValueError(
            "vanilla european option: expiry precedes the run as-of on uid(s) "
            f"{uids[expired][:10].tolist()} (D068: expired options are rejected)."
        )

    # ---- TIME: one canonical T, two precomputed forms ---------------------
    tte = uix_mapper.tte_years(expiry_ord)
    sqrt_tte = np.ascontiguousarray(np.sqrt(tte))
    half_tte = np.ascontiguousarray(0.5 * tte)

    # ---- STRIKE-CURRENCY FORWARD (and, by construction, its log) ----------
    expiry_uix = uix_mapper.map_dates_to_uix(expiry_ord)
    fx_u2s_eff_uix = uix_mapper.map_fx_legs_to_effective_uix(
        raw_fx_ids=batch.fx_u2s_ids[start:end],
        raw_fx_inv=batch.fx_u2s_inv[start:end],
    )
    fx_k2s_eff_uix = uix_mapper.map_fx_legs_to_effective_uix(
        raw_fx_ids=batch.fx_k2s_ids[start:end],
        raw_fx_inv=batch.fx_k2s_inv[start:end],
    )
    strike_curve_uix = uix_mapper.map_discount_ids_to_uix(
        batch.strike_discount_curve_ids[start:end]
    )

    efp_uix = uix_mapper.map_efp_to_effective_uix(
        price_uix=uix_mapper.map_market_ids_to_uix(batch.underlying_ids[start:end]),
        curve_uix=uix_mapper.map_discount_ids_to_uix(
            batch.under_discount_curve_ids[start:end]
        ),
        date_uix=expiry_uix,
        fx_eff_uix=fx_u2s_eff_uix,
    )
    edf_expiry_uix = uix_mapper.map_edf_to_effective_uix(
        curve_uix=strike_curve_uix,
        date_uix=expiry_uix,
        fx_eff_uix=fx_k2s_eff_uix,
    )
    forward_uix = uix_mapper.map_fk_to_effective_uix(
        efp_uix=efp_uix,
        edf_expiry_uix=edf_expiry_uix,
    )
    _require_resolved(
        forward_uix,
        n_rows=uix_mapper.fk_effective_pairs.shape[0],
        leg="forward_price_strike_ccy",
        uids=uids,
    )

    # ---- VOLATILITY -------------------------------------------------------
    vol_uix = uix_mapper.map_vol_to_effective_uix(
        uix_mapper.map_market_ids_to_uix(batch.volatility_ids[start:end])
    )
    _require_resolved(
        vol_uix,
        n_rows=uix_mapper.vol_effective_uix.shape[0],
        leg="volatility",
        uids=uids,
    )

    # ---- SETTLEMENT DISCOUNTING ------------------------------------------
    effective_discount_factor_uix = uix_mapper.map_edf_to_effective_uix(
        curve_uix=strike_curve_uix,
        date_uix=uix_mapper.map_dates_to_uix(settle_ord),
        fx_eff_uix=fx_k2s_eff_uix,
    )
    _require_resolved(
        effective_discount_factor_uix,
        n_rows=uix_mapper.edf_effective_triples.shape[0],
        leg="effective_discount_factor",
        uids=uids,
    )

    return VanillaEuropeanOptionBlock(
        trade_uids=uids,
        qty=batch.qty[start:end],
        strike=strike,
        log_strike=np.ascontiguousarray(np.log(strike)),
        cp_mult=batch.cp_mult[start:end],
        sqrt_tte=sqrt_tte,
        half_tte=half_tte,
        # ONE resolved coordinate, stamped into both fields: the log cache is
        # generated row-for-row from the forward table, so these are the same
        # row of two materialisations of one universe (VAL-11).
        forward_price_strike_ccy_uix=forward_uix,
        option_log_forward_price_strike_ccy_uix=forward_uix,
        vol_uix=vol_uix,
        effective_discount_factor_uix=effective_discount_factor_uix,
    )
