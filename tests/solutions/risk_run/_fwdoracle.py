# tests/solutions/risk_run/_fwdoracle.py
"""
Independent oracle for the WS-B VANILLA_FORWARD_NODE FULL_REVAL evidence.

Workstream specification revision 8, section 5. This module imports only the
standard library (`math`, `decimal`) and `numpy`. It implements no call into
RiskQuant: no valuation, no effective-factor fold, no sensitizer, no
`run_valuation`, no tail helper, no `compute_var_es`. The static import check
in `test_vanilla_forward_node_production.py` enforces this.

Every array operation below is one elementwise IEEE-754 binary64 operation
(NumPy does not contract a multiply and an add into an FMA), so each formula is
the prescribed oracle form of section 5.2:

    V    = fl(Q * fl(fl(F * X_a) - fl(K * X_k)))       (identity leg: no product)
    T    = fl(fl(|Q| * |a|) + fl(|Q| * |b|))
    dW_o = fl(fl(dV_o * r_s) + fl(V0_o * d_r)),  d_r = fl(r_s - r_0)
"""

from __future__ import annotations

import math
from decimal import Decimal, localcontext

import numpy as np

U = 2.0**-24  # float32 unit roundoff
EPS = 2.0**-53  # float64 unit roundoff
NP_EPS = 2.0**-52  # NumPy machine epsilon (tail snap rule only)
PHI = 2.0**-149  # float32 floor

IDENTITY, DIRECT, INVERTED = "identity", "direct", "inverted"
LOG, SIMPLE, DIFF = "LOG_RETURN", "SIMPLE_RETURN", "DIFFERENCE"


def gamma(k: int) -> float:
    return k * EPS / (1.0 - k * EPS)


# ---- valuation (section 5.4) ------------------------------------------------------


def effective_rate(x: np.ndarray, kind: str) -> np.ndarray | None:
    """Exact effective rate in its oracle form: x, fl(1.0 / x), or no multiplication."""
    if kind == IDENTITY:
        return None
    x = np.asarray(x, dtype=np.float64)
    if kind == DIRECT:
        return x
    if kind == INVERTED:
        return 1.0 / x
    raise ValueError(kind)


def forward_value(
    q: float,
    f: np.ndarray,
    k: float,
    x_a: np.ndarray | None,
    x_k: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """(V, T) of one NODE forward row in DiscountingMode.NONE (DF = 1 exactly)."""
    f = np.asarray(f, dtype=np.float64)
    a = f if x_a is None else f * x_a
    b = np.full_like(f, k) if x_k is None else k * x_k
    value = q * (a - b)
    scale = abs(q) * np.abs(a) + abs(q) * np.abs(b)
    return value, scale


def spot_value(
    q: float, s: np.ndarray, x: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """(V, T) of one SPOT row: q * (S * X); T = |V| (one product chain)."""
    s = np.asarray(s, dtype=np.float64)
    e = s if x is None else s * x
    value = q * e
    return value, np.abs(value)


# ---- tolerances (sections 5.4, 5.5, 5.7) -------------------------------------------


def tol_v0(t0: np.ndarray) -> np.ndarray:
    return 2.0**-48 * t0


def tol_dv_full(dv: np.ndarray, ts: np.ndarray, t0: np.ndarray) -> np.ndarray:
    return 2.0 * U * np.abs(dv) + 2.0**-48 * (ts + t0) + PHI


def reporting_dw(
    dv: np.ndarray, v0: np.ndarray, rs: np.ndarray, r0: np.ndarray
) -> np.ndarray:
    d_r = rs - r0
    return dv * rs + v0 * d_r


def tol_dw(
    rs: np.ndarray,
    r0: np.ndarray,
    tol_dv: np.ndarray,
    tol_v: np.ndarray,
    dv: np.ndarray,
    v0: np.ndarray,
) -> np.ndarray:
    return (
        np.abs(rs) * tol_dv
        + np.abs(rs - r0) * tol_v
        + 2.0**-48 * (np.abs(dv * rs) + np.abs(v0) * (np.abs(rs) + np.abs(r0)))
    )


# ---- tail geometry, VaR, ES (section 5.8) and portfolio (section 5.9) ---------------


def tail_geometry(confidence: float, scenarios: int) -> tuple[int, int, float, float]:
    """(var_index, f, w, m) by the frozen rule, including the snap."""
    alpha = 1.0 - confidence
    m = alpha * float(scenarios)
    nearest = float(np.rint(m))
    if abs(m - nearest) <= 16.0 * NP_EPS * max(1.0, abs(m)):
        m = nearest
    f = math.floor(m)
    w = m - float(f)
    var_index = f - 1 if w == 0.0 else f
    return var_index, f, w, m


def var_es(
    p: np.ndarray, geometry: tuple[int, int, float, float]
) -> tuple[float, float]:
    """Lower-tail order statistic and tail mean of mass m (no interpolation)."""
    var_index, f, w, m = geometry
    ordered = np.sort(np.asarray(p, dtype=np.float64))
    tail = float(np.sum(ordered[:f])) if f else 0.0
    if w > 0.0:
        tail = tail + w * float(ordered[f])
    return float(ordered[var_index]), tail / m


def tol_es(eps_in: float, p_inf: float, f: int) -> float:
    return eps_in + 2.0 * gamma(f + 2) * (p_inf + eps_in)


def portfolio_eps(tol_dw_rows: np.ndarray, dw_rows: np.ndarray) -> np.ndarray:
    """eps_P,s over M = number of rows summed, per scenario."""
    m = tol_dw_rows.shape[0]
    g = gamma(m - 1)
    return (1.0 + 2.0 * g) * tol_dw_rows.sum(axis=0) + 2.0 * g * np.abs(dw_rows).sum(
        axis=0
    )


# ---- historical levels (section 5.3) -----------------------------------------------


def chart_return(v1: np.ndarray, v0: np.ndarray, chart: str) -> np.ndarray:
    """Y64 with the A-L2 forms: one float64 division or subtraction, CPython log."""
    v1 = np.asarray(v1, dtype=np.float64)
    v0 = np.asarray(v0, dtype=np.float64)
    if chart == LOG:
        return np.array([math.log(r) for r in (v1 / v0).tolist()], dtype=np.float64)
    if chart == SIMPLE:
        return v1 / v0 - 1.0
    if chart == DIFF:
        return v1 - v0
    raise ValueError(chart)


def reconstruct(x0: float, x: np.ndarray, chart: str) -> np.ndarray:
    """x_hat = T^-1(x0, X) with the A-L2 forms (CPython exp)."""
    x = np.asarray(x, dtype=np.float64)
    if chart == LOG:
        return np.array([x0 * math.exp(v) for v in x.tolist()], dtype=np.float64)
    if chart == SIMPLE:
        return x0 * (1.0 + x)
    if chart == DIFF:
        return x0 + x
    raise ValueError(chart)


def tol_level(
    chart: str,
    beta: float,
    y64: np.ndarray,
    x0: float,
    x_o: np.ndarray,
    x_hat: np.ndarray,
) -> np.ndarray:
    if chart == LOG:
        bar = abs(beta) * np.abs(y64) * (2.0 * U + 2.0**-47)
        return np.abs(x_hat) * ((1.0 + 2.0**-48) * bar * np.exp(bar) + 19.0 * EPS)
    if chart == SIMPLE:
        return 2.0 * gamma(2) * abs(x0) * (1.0 + np.abs(x_o))
    if chart == DIFF:
        return 2.0 * gamma(2) * (abs(x0) + np.abs(x_o))
    raise ValueError(chart)


def level_oracle(
    history: np.ndarray, x0: float, beta: float, chart: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(Y64, Y32_o, x_hat, tol_level) for horizon 1: scenario s uses days s, s+1."""
    y64 = chart_return(history[1:], history[:-1], chart)
    y32_o = y64.astype(np.float32).astype(np.float64)
    x_o = beta * y32_o
    x_hat = reconstruct(x0, x_o, chart)
    return y64, y32_o, x_hat, tol_level(chart, beta, y64, x0, x_o, x_hat)


# ---- A-L1 reference (section 5.3) ---------------------------------------------------


def ulps_from_decimal_reference(args: np.ndarray, values: np.ndarray, fn: str) -> float:
    """Max distance of `values` from the precision-40 Decimal ln/exp of `args`,
    in binary64 ulps of the reference. The float argument converts exactly."""
    worst = 0.0
    with localcontext() as ctx:
        ctx.prec = 40
        for a, v in zip(args.tolist(), values.tolist(), strict=True):
            ref = Decimal(a).ln() if fn == "ln" else Decimal(a).exp()
            unit = Decimal(math.ulp(float(ref)))
            worst = max(worst, float(abs(Decimal(v) - ref) / unit))
    return worst
