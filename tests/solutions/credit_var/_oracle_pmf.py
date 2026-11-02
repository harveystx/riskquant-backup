# tests/solutions/credit_var/_oracle_pmf.py
"""
WS_CREDIT_VAR_API S4: independent loss distributions for the oracle tests.

Each function writes out the closed form or the quadrature of
A_ORACLES.md section 2 for one catalog key. The module imports only numpy,
scipy and the standard library (C.27). It calls no riskquant function.

Every loss in this module is an integer: EAD and LGD are integers or 1, so the
portfolio loss L is a count or a weighted count (A_ORACLES rule 2). A
distribution is a dense probability vector `probs` with `probs[l] = P(L = l)`
for l = 0, 1, ..., len(probs) - 1. An integer with no probability mass has
probability 0.0 and does not change F.

The tail statistics use the population definitions of A_ORACLES rule 3 and
the bounds of rule 5. The atom-gap rule is rule 4.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import special, stats

FloatArray = npt.NDArray[np.float64]

#: A_ORACLES rule 5: absolute slack in loss units.
TAU = 1e-9

#: A_ORACLES rules 4 and 5: number of standard deviations.
N_SIGMA = 6.0


# ---------------------------------------------------------------------------
# Distribution and population statistics (A_ORACLES rule 3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LossPmf:
    """Distribution of an integer portfolio loss L on 0, 1, ..., len - 1."""

    probs: FloatArray

    def __post_init__(self) -> None:
        if self.probs.ndim != 1 or self.probs.size == 0:
            raise ValueError("probs must be a non-empty 1-D array")
        if self.probs.dtype != np.float64:
            raise ValueError(f"probs must be float64, got {self.probs.dtype}")
        if not np.all(np.isfinite(self.probs)) or np.any(self.probs < 0.0):
            raise ValueError("probs must be finite and non-negative")

    @property
    def atoms(self) -> FloatArray:
        return np.arange(self.probs.size, dtype=np.float64)

    @property
    def mass(self) -> float:
        return float(math.fsum(self.probs.tolist()))

    def cdf(self, loss: int) -> float:
        """F(loss) = P(L <= loss). F(-1) = 0."""
        if loss < 0:
            return 0.0
        return float(math.fsum(self.probs[: loss + 1].tolist()))

    def mean(self) -> float:
        """EL = E[L]."""
        return float(math.fsum((self.atoms * self.probs).tolist()))

    def second_moment(self) -> float:
        return float(math.fsum((self.atoms**2 * self.probs).tolist()))

    def variance(self) -> float:
        """Var_L = max(E[L^2] - EL^2, 0)."""
        return max(self.second_moment() - self.mean() ** 2, 0.0)

    def var(self, alpha: float) -> int:
        """VaR_a = min{ l : F(l) >= a }."""
        cumulative = 0.0
        for loss, p in enumerate(self.probs.tolist()):
            cumulative += p
            if cumulative >= alpha:
                return loss
        raise ValueError(f"F never reaches {alpha}: total mass {self.mass}")

    def _upper_tail(self, alpha: float, power: int) -> tuple[int, float, float]:
        """(VaR_a, E[L^power 1{L > VaR_a}], F(VaR_a) - a)."""
        v = self.var(alpha)
        above = self.atoms[v + 1 :]
        tail = float(math.fsum((above**power * self.probs[v + 1 :]).tolist()))
        return v, tail, self.cdf(v) - alpha

    def es(self, alpha: float) -> float:
        """ES_a = (E[L 1{L > VaR_a}] + VaR_a (F(VaR_a) - a)) / (1 - a)."""
        v, tail, excess = self._upper_tail(alpha, 1)
        return (tail + v * excess) / (1.0 - alpha)

    def es_tail_variance(self, alpha: float) -> float:
        """s_ES^2 = max((E[L^2 1{L > VaR_a}] + VaR_a^2 (F(VaR_a) - a)) / (1 - a)
        - ES_a^2, 0): the variance of L under the tail measure."""
        v, tail2, excess = self._upper_tail(alpha, 2)
        es = self.es(alpha)
        return max((tail2 + v * v * excess) / (1.0 - alpha) - es * es, 0.0)


# ---------------------------------------------------------------------------
# Atom-gap rule and assertion bounds (A_ORACLES rules 4 and 5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AtomGap:
    """Rule 4 diagnostics at one level."""

    alpha: float
    var: int
    cdf_below: float
    cdf_at: float
    d: float

    @property
    def holds(self) -> bool:
        """F(l_prev) <= a - d and F(VaR_a) >= a + d."""
        return self.cdf_below <= self.alpha - self.d and (
            self.cdf_at >= self.alpha + self.d
        )


def atom_gap(pmf: LossPmf, alpha: float, n_scenarios: int) -> AtomGap:
    """Rule 4 at level a. l_prev is the integer below VaR_a: F has no step
    between it and the atom with mass below VaR_a, so F(l_prev) is the same.
    With no atom below VaR_a, F(l_prev) = 0."""
    v = pmf.var(alpha)
    d = N_SIGMA * math.sqrt(alpha * (1.0 - alpha) / n_scenarios)
    return AtomGap(alpha=alpha, var=v, cdf_below=pmf.cdf(v - 1), cdf_at=pmf.cdf(v), d=d)


def el_bound(pmf: LossPmf, n_scenarios: int) -> float:
    """6 sqrt(Var_L / S) + tau."""
    return N_SIGMA * math.sqrt(pmf.variance() / n_scenarios) + TAU


def es_bound(pmf: LossPmf, alpha: float, n_scenarios: int) -> float:
    """6 sqrt((s_ES^2 + a (ES - VaR_a)^2) / ((1 - a) S)) + tau."""
    es = pmf.es(alpha)
    v = pmf.var(alpha)
    spread = pmf.es_tail_variance(alpha) + alpha * (es - v) ** 2
    return N_SIGMA * math.sqrt(spread / ((1.0 - alpha) * n_scenarios)) + TAU


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _pmf(probs: npt.ArrayLike) -> LossPmf:
    return LossPmf(np.ascontiguousarray(np.asarray(probs, dtype=np.float64)))


def _scaled(count_probs: FloatArray, weight: int) -> FloatArray:
    """pmf of weight x K from the pmf of K: mass on multiples of weight."""
    out = np.zeros(weight * (count_probs.size - 1) + 1, dtype=np.float64)
    out[::weight] = count_probs
    return out


def weighted_sum(
    count_a: FloatArray, weight_a: int, count_b: FloatArray, weight_b: int
) -> LossPmf:
    """pmf of weight_a K_A + weight_b K_B for independent K_A and K_B."""
    return _pmf(np.convolve(_scaled(count_a, weight_a), _scaled(count_b, weight_b)))


def _binomial_rows(n: int, p: FloatArray) -> FloatArray:
    """Bin(k; n, p) = C(n, k) p^k (1 - p)^(n - k) for k = 0..n, one row per
    value of p. Shape (len(p), n + 1).

    The formula is written out: `scipy.stats.binom.pmf` raises OverflowError
    for a denormal p (scipy 1.17.1, sandbox), and the far chi2 nodes of K9,
    K10 and K12 give such p. numpy gives 0.0 ** 0 = 1.0, so p = 0 and p = 1
    give the correct point masses.
    """
    k = np.arange(n + 1, dtype=np.float64)
    coefficients = np.array([math.comb(n, i) for i in range(n + 1)], np.float64)
    col = p[:, np.newaxis]
    rows = coefficients * col**k * (1.0 - col) ** (n - k)
    return np.ascontiguousarray(rows, dtype=np.float64)


def _hermite(n_nodes: int) -> tuple[FloatArray, FloatArray]:
    """Probabilists' Gauss-Hermite: E[f(Z)] = sum w f(x), Z ~ N(0, 1)."""
    x, w = np.polynomial.hermite_e.hermegauss(n_nodes)
    return np.asarray(x, dtype=np.float64), np.asarray(
        w / math.sqrt(2.0 * math.pi), dtype=np.float64
    )


def _chi2_laguerre(nu: float, n_nodes: int) -> tuple[FloatArray, FloatArray]:
    """E[f(W)], W ~ chi2(nu), as sum w f(2 u): generalized Gauss-Laguerre in
    u = W / 2 with exponent nu / 2 - 1, weights divided by Gamma(nu / 2)."""
    u, w = special.roots_genlaguerre(n_nodes, nu / 2.0 - 1.0)
    u = np.asarray(u, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64) / special.gamma(nu / 2.0)
    if not (np.all(np.isfinite(u)) and np.all(np.isfinite(w))):
        raise ValueError(f"non-finite Gauss-Laguerre nodes at n = {n_nodes}")
    return 2.0 * u, w


def _mixture_count(n: int, p: FloatArray, weights: FloatArray) -> FloatArray:
    """sum_j weights[j] Bin(.; n, p[j])."""
    return np.ascontiguousarray(weights @ _binomial_rows(n, p))


def _two_sector_sum(
    n_a: int, p_a: FloatArray, n_b: int, p_b: FloatArray, weights: FloatArray
) -> LossPmf:
    """P(L = l) = sum_j w_j [Bin(.; n_a, p_a[j]) * Bin(.; n_b, p_b[j])](l), where
    `*` is discrete convolution and EAD = 1 in both sectors."""
    rows_a = _binomial_rows(n_a, p_a)
    rows_b = _binomial_rows(n_b, p_b)
    joint = (rows_a * weights[:, np.newaxis]).T @ rows_b  # P(K_A = i, K_B = j)
    probs = np.zeros(n_a + n_b + 1, dtype=np.float64)
    for i in range(n_a + 1):
        probs[i : i + n_b + 1] += joint[i]
    return _pmf(probs)


# ---------------------------------------------------------------------------
# One oracle per catalog key (A_ORACLES section 2)
# ---------------------------------------------------------------------------


def independent_bernoulli(ead: Sequence[int], pd: Sequence[float]) -> LossPmf:
    """K2: enumerate every default vector D; L = sum(EAD_i D_i); P = prod of
    PD_i or (1 - PD_i)."""
    if len(ead) != len(pd):
        raise ValueError("ead and pd differ in length")
    probs = np.zeros(sum(ead) + 1, dtype=np.float64)
    for defaults in itertools.product((0, 1), repeat=len(ead)):
        loss = sum(e * d for e, d in zip(ead, defaults, strict=True))
        p = math.prod(q if d else 1.0 - q for q, d in zip(pd, defaults, strict=True))
        probs[loss] += p
    return _pmf(probs)


def beta_binomial_count(n: int, a: float, b: float) -> FloatArray:
    """K ~ BetaBinomial(n, a, b)."""
    k = np.arange(n + 1)
    return np.asarray(stats.betabinom.pmf(k, n, a, b), dtype=np.float64)


def beta_binomial(n: int, a: float, b: float) -> LossPmf:
    """K3: L = K."""
    return _pmf(beta_binomial_count(n, a, b))


def stress_regime_count(n: int, pd: float, q: float, m: float) -> FloatArray:
    """K ~ (1 - q) Bin(n, PD) + q Bin(n, min(1, m PD)): one regime per group."""
    p = np.array([pd, min(1.0, m * pd)], dtype=np.float64)
    return _mixture_count(n, p, np.array([1.0 - q, q], dtype=np.float64))


def stress_regime_bernoulli(n: int, pd: float, q: float, m: float) -> LossPmf:
    """K5: L = K."""
    return _pmf(stress_regime_count(n, pd, q, m))


def basel_corporate_rho(pd: float) -> float:
    """Basel IRB corporate correlation R(PD), written out (A_ORACLES K8)."""
    weight = (1.0 - math.exp(-50.0 * pd)) / (1.0 - math.exp(-50.0))
    return 0.12 * weight + 0.24 * (1.0 - weight)


def sf_gaussian(n: int, pd: float, rho: float, n_nodes: int = 200) -> LossPmf:
    """K7, K8: p(z) = Phi((Phi^-1(PD) - sqrt(rho) z) / sqrt(1 - rho));
    P(K = k) = integral Bin(k; n, p(z)) phi(z) dz."""
    z, w = _hermite(n_nodes)
    p = special.ndtr((special.ndtri(pd) - math.sqrt(rho) * z) / math.sqrt(1.0 - rho))
    return _pmf(_mixture_count(n, np.asarray(p, dtype=np.float64), w))


def sf_students_t(
    n: int, pd: float, rho: float, nu: float, n_z: int = 120, n_w: int = 300
) -> LossPmf:
    """K9, K10: p(z, w) = Phi((t_nu^-1(PD) sqrt(w / nu) - sqrt(rho) z)
    / sqrt(1 - rho)); W ~ chi2(nu) shared."""
    z, wz = _hermite(n_z)
    chi, ww = _chi2_laguerre(nu, n_w)
    threshold = float(stats.t.ppf(pd, nu)) * np.sqrt(chi / nu)
    p = special.ndtr(
        (threshold[:, np.newaxis] - math.sqrt(rho) * z[np.newaxis, :])
        / math.sqrt(1.0 - rho)
    )
    weights = (ww[:, np.newaxis] * wz[np.newaxis, :]).ravel()
    return _pmf(_mixture_count(n, np.asarray(p, dtype=np.float64).ravel(), weights))


def _two_factor_probability(
    threshold: FloatArray,
    loadings: tuple[float, float],
    z1: FloatArray,
    z2: FloatArray,
) -> FloatArray:
    """Phi((threshold - b1 z1 - b2 z2) / sqrt(1 - b1^2 - b2^2)), elementwise."""
    b1, b2 = loadings
    idio = math.sqrt(1.0 - b1 * b1 - b2 * b2)
    p = special.ndtr((threshold - b1 * z1 - b2 * z2) / idio)
    return np.ascontiguousarray(np.asarray(p, dtype=np.float64))


def mf_gaussian(
    n_a: int,
    loadings_a: tuple[float, float],
    n_b: int,
    loadings_b: tuple[float, float],
    pd: float,
    n_nodes: int = 80,
) -> LossPmf:
    """K11: two systemic factors F1, F2; sector loadings (on F1, on F2)."""
    x, w = _hermite(n_nodes)
    z1, z2 = (a.ravel() for a in np.meshgrid(x, x, indexing="ij"))
    weights = np.outer(w, w).ravel()
    threshold = np.full(z1.shape, special.ndtri(pd), dtype=np.float64)
    p_a = _two_factor_probability(threshold, loadings_a, z1, z2)
    p_b = _two_factor_probability(threshold, loadings_b, z1, z2)
    return _two_sector_sum(n_a, p_a, n_b, p_b, weights)


def mf_students_t(
    n_a: int,
    loadings_a: tuple[float, float],
    n_b: int,
    loadings_b: tuple[float, float],
    pd: float,
    nu: float,
    n_z: int = 60,
    n_w: int = 300,
) -> LossPmf:
    """K12: K11 with one shared W ~ chi2(nu); each threshold Phi^-1(PD)
    becomes t_nu^-1(PD) sqrt(w / nu)."""
    x, wx = _hermite(n_z)
    chi, ww = _chi2_laguerre(nu, n_w)
    grid_w, grid_z1, grid_z2 = np.meshgrid(chi, x, x, indexing="ij")
    weights = (
        ww[:, np.newaxis, np.newaxis] * wx[np.newaxis, :, np.newaxis] * wx
    ).ravel()
    threshold = float(stats.t.ppf(pd, nu)) * np.sqrt(grid_w.ravel() / nu)
    z1, z2 = grid_z1.ravel(), grid_z2.ravel()
    p_a = _two_factor_probability(threshold, loadings_a, z1, z2)
    p_b = _two_factor_probability(threshold, loadings_b, z1, z2)
    return _two_sector_sum(n_a, p_a, n_b, p_b, weights)


def marshall_olkin(n: int, pd: float, systemic_pd: float) -> LossPmf:
    """K13: p_idio = 1 - (1 - PD) / (1 - s);
    P(K = k) = s 1{k = n} + (1 - s) Bin(k; n, p_idio)."""
    p_idio = 1.0 - (1.0 - pd) / (1.0 - systemic_pd)
    probs = (1.0 - systemic_pd) * _binomial_rows(n, np.array([p_idio]))[0]
    probs[n] += systemic_pd
    return _pmf(probs)
