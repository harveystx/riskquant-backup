# tests/valuation/oracle/economic_oracle.py
"""
Independent economic valuation oracle.

Structural independence contract (D045 Phase 3)
-----------------------------------------------
This module MUST NOT import anything from `riskquant`. It is the reference
side of the acceptance test; the production valuation path is the SUT.

It computes PV from ECONOMIC IDENTITY, not from array position: every market
quantity is looked up by its factor id out of a dict, and every trade is
described by the ids it references. There is no array ordering anywhere in
this file, so an oracle/production agreement cannot be produced by both sides
sharing a row layout.

Black-76 uses `scipy.stats.norm`, which is a structurally different normal CDF
from the Abramowitz-Stegun polynomial the production kernel inlines.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

# =========================================================================
# ECONOMIC MARKET STATE -- keyed by factor identity, never by position
# =========================================================================


@dataclass(frozen=True)
class EconomicMarket:
    """Scenario-wise market quotes, addressed by factor id."""

    n_scenarios: int
    # factor_id -> (n_scenarios,) asset price in its NATIVE quote currency
    spot_native: dict[int, np.ndarray]
    # factor_id -> (n_scenarios,) forward price in its NATIVE quote currency
    forward_native: dict[int, np.ndarray]
    # factor_id -> (n_scenarios,) discount factor, native currency curve
    discount: dict[int, np.ndarray]
    # (ccy_from, ccy_to) -> (n_scenarios,) rate. Multiply a `ccy_from` amount by it.
    fx: dict[tuple[str, str], np.ndarray]
    # factor_id -> (n_scenarios,) lognormal vol of the strike-ccy forward
    vol: dict[int, np.ndarray]
    # factor_id -> (n_scenarios,) annualised simple forward rate
    forward_rate: dict[int, np.ndarray]

    def fx_rate(self, frm: str, to: str) -> np.ndarray:
        """FX(frm -> to), derived by inversion when only the reverse is quoted."""
        if frm == to:
            return np.ones(self.n_scenarios, dtype=np.float64)
        if (frm, to) in self.fx:
            return self.fx[(frm, to)]
        if (to, frm) in self.fx:
            return 1.0 / self.fx[(to, frm)]
        raise KeyError(f"no FX quote for {frm}->{to}")


# =========================================================================
# ECONOMIC TRADES
# =========================================================================


@dataclass(frozen=True)
class SpotTrade:
    trade_uid: int
    qty: float
    spot_factor_id: int
    quote_ccy: str


@dataclass(frozen=True)
class ForwardTrade:
    """Physically settled forward: receive the asset at T, pay `strike` at T."""

    trade_uid: int
    qty: float
    forward_factor_id: int
    asset_ccy: str
    asset_discount_factor_id: int
    strike: float
    strike_ccy: str
    strike_discount_factor_id: int


@dataclass(frozen=True)
class OptionTrade:
    """European option on the strike-currency forward, Black-76."""

    trade_uid: int
    qty: float
    is_call: bool
    strike: float
    strike_ccy: str
    tte_years: float
    forward_strike_ccy_factor_id: int
    vol_factor_id: int
    settle_discount_factor_id: int


@dataclass(frozen=True)
class FixedCashflowTrade:
    trade_uid: int
    notional: float
    accrual_factor: float
    fixed_rate: float
    is_receive: bool
    ccy: str
    discount_factor_id: int


@dataclass(frozen=True)
class FloatingCashflowTrade:
    trade_uid: int
    notional: float
    accrual_factor: float
    spread: float
    is_receive: bool
    ccy: str
    forward_rate_factor_id: int
    discount_factor_id: int


@dataclass(frozen=True)
class PrincipalCashflowTrade:
    trade_uid: int
    amount: float  # SIGNED: receive > 0, pay < 0
    ccy: str
    discount_factor_id: int


# =========================================================================
# INDEPENDENT PV MATHEMATICS
# =========================================================================


def _black76_fv_scalar(
    fwd: float, strike: float, vol: float, tte: float, cp: int
) -> float:
    """FV = cp * (F * Phi(cp*d1) - K * Phi(cp*d2)). scipy CDF; no A&S anywhere."""
    sT = vol * math.sqrt(tte)
    d1 = (math.log(fwd / strike) + 0.5 * vol * vol * tte) / sT
    d2 = d1 - sT
    return cp * (fwd * float(norm.cdf(cp * d1)) - strike * float(norm.cdf(cp * d2)))


def pv_spot(t: SpotTrade, m: EconomicMarket, settle_ccy: str) -> np.ndarray:
    """PV = Q * S_native * FX(native -> settle). No discounting: spot has no time value."""
    s = m.spot_native[t.spot_factor_id]
    return t.qty * s * m.fx_rate(t.quote_ccy, settle_ccy)


def pv_forward(t: ForwardTrade, m: EconomicMarket, settle_ccy: str) -> np.ndarray:
    """PV = Q * [ F_a DF_a FX(a->s)  -  K DF_k FX(k->s) ]."""
    leg_asset = (
        m.forward_native[t.forward_factor_id]
        * m.discount[t.asset_discount_factor_id]
        * m.fx_rate(t.asset_ccy, settle_ccy)
    )
    leg_strike = (
        t.strike
        * m.discount[t.strike_discount_factor_id]
        * m.fx_rate(t.strike_ccy, settle_ccy)
    )
    return t.qty * (leg_asset - leg_strike)


def pv_option(t: OptionTrade, m: EconomicMarket, settle_ccy: str) -> np.ndarray:
    """PV = Q * FV_Black76(strike ccy, at expiry) * DF_k(0,T) * FX(k -> settle)."""
    fwd = m.forward_native[t.forward_strike_ccy_factor_id]
    vol = m.vol[t.vol_factor_id]
    cp = 1 if t.is_call else -1
    fv = np.array(
        [
            _black76_fv_scalar(float(fwd[s]), t.strike, float(vol[s]), t.tte_years, cp)
            for s in range(m.n_scenarios)
        ],
        dtype=np.float64,
    )
    df = m.discount[t.settle_discount_factor_id]
    return t.qty * fv * df * m.fx_rate(t.strike_ccy, settle_ccy)


def pv_fixed_cashflow(
    t: FixedCashflowTrade, m: EconomicMarket, settle_ccy: str
) -> np.ndarray:
    """PV = N * alpha * r * eps * DF * FX."""
    eps = 1.0 if t.is_receive else -1.0
    cf = t.notional * t.accrual_factor * t.fixed_rate * eps
    return cf * m.discount[t.discount_factor_id] * m.fx_rate(t.ccy, settle_ccy)


def pv_floating_cashflow(
    t: FloatingCashflowTrade, m: EconomicMarket, settle_ccy: str
) -> np.ndarray:
    """PV = N * alpha * (f + spread) * eps * DF * FX."""
    eps = 1.0 if t.is_receive else -1.0
    rate = m.forward_rate[t.forward_rate_factor_id] + t.spread
    cf = t.notional * t.accrual_factor * rate * eps
    return cf * m.discount[t.discount_factor_id] * m.fx_rate(t.ccy, settle_ccy)


def pv_principal_cashflow(
    t: PrincipalCashflowTrade, m: EconomicMarket, settle_ccy: str
) -> np.ndarray:
    """PV = A * DF * FX. `amount` is already signed."""
    return t.amount * m.discount[t.discount_factor_id] * m.fx_rate(t.ccy, settle_ccy)


@dataclass(frozen=True)
class CrossCurrencyOptionTrade:
    """
    European option priced off a STRIKE-CURRENCY forward the oracle builds
    itself from the asset forward, both discount curves and both FX legs.

    Nothing about the production construction is reused: the oracle is told the
    ASSET forward and derives `F_K` from covered interest parity independently.
    """

    trade_uid: int
    qty: float
    is_call: bool
    strike: float
    # asset leg
    asset_forward_factor_id: int
    asset_ccy: str
    asset_expiry_discount_id: int
    # strike leg
    strike_ccy: str
    strike_expiry_discount_id: int
    strike_settle_discount_id: int
    # model
    vol_factor_id: int
    tte_years: float


def forward_in_strike_ccy(
    t: CrossCurrencyOptionTrade, m: EconomicMarket, settle_ccy: str
) -> np.ndarray:
    """
    Covered interest parity, derived here rather than read from production:

        F_K(T) = F_a(T) DF_a(0,T) FX(a->s) / ( DF_K(0,T) FX(K->s) )

    Discount the asset-currency forward to a PV, translate it through the
    settlement numeraire into strike-currency terms, then re-accumulate to
    expiry on the strike curve.
    """
    pv_asset_ccy = (
        m.forward_native[t.asset_forward_factor_id]
        * m.discount[t.asset_expiry_discount_id]
    )
    pv_settle = pv_asset_ccy * m.fx_rate(t.asset_ccy, settle_ccy)
    pv_strike_ccy = pv_settle / m.fx_rate(t.strike_ccy, settle_ccy)
    return pv_strike_ccy / m.discount[t.strike_expiry_discount_id]


def pv_cross_currency_option(
    t: CrossCurrencyOptionTrade, m: EconomicMarket, settle_ccy: str
) -> np.ndarray:
    """
    PV = Q * FV_Black76(F_K, K, sigma, T_exp) * DF_K(0, T_settle) * FX(K -> settle).

    The forward measure and the volatility use EXPIRY; the discount factor uses
    SETTLEMENT. `cp` comes from the economic `is_call` flag, never from a
    production `cp_mult` field.
    """
    fwd_k = forward_in_strike_ccy(t, m, settle_ccy)
    vol = m.vol[t.vol_factor_id]
    cp = 1 if t.is_call else -1
    fv = np.array(
        [
            _black76_fv_scalar(
                float(fwd_k[s]),
                t.strike,
                float(vol[s]),
                t.tte_years,
                cp,
            )
            for s in range(m.n_scenarios)
        ],
        dtype=np.float64,
    )
    return (
        t.qty
        * fv
        * m.discount[t.strike_settle_discount_id]
        * m.fx_rate(t.strike_ccy, settle_ccy)
    )
