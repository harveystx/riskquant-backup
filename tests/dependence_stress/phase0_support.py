# tests/dependence_stress/phase0_support.py
"""
Phase 0 test support: shapes grid, transformer catalog, and an exact replica of
the repository VaR/ES lower-tail convention.

The VaR/ES helper mirrors ``riskquant.scenario_metrics._computer.
_compute_lower_tail_parameters`` and ``_kernel_lower_tail_var_es`` in pure NumPy.
It exists so the benchmark identities are locked as executable specification even
where the Numba scenario_metrics kernel is not importable. When the real API is
importable, the identity tests additionally cross-check against it.

Sign convention (repository-authoritative, do NOT change):
    VaR = signed lower P&L quantile = ceil(alpha*S)-th smallest scenario value.
    More-negative = more risk. ES = corresponding lower-tail expected shortfall.
"""

from __future__ import annotations

import numpy as np

from riskquant.dependence_stress.copulas import GaussianCopula, TCopula
from riskquant.dependence_stress.tails import (
    Comonotonic,
    DualTailAlignment,
    LowerTailAlignment,
    UpperTailAlignment,
)

# ---- shape grid (Phase 0 §4.1) -------------------------------------------------
K_GRID = (2, 3, 8)
S_GRID = (2, 3, 5, 7, 10, 101)
SEED = 20260914


def equicorrelation(k: int, rho: float = 0.5) -> np.ndarray:
    """Strictly positive-definite K x K equicorrelation matrix (min eig = 1-rho)."""
    r = np.full((k, k), rho, dtype=np.float64)
    np.fill_diagonal(r, 1.0)
    return r


def heterogeneous_matrix(k: int, s: int, *, seed: int = SEED) -> np.ndarray:
    """
    K x S matrix with heterogeneous, skewed, fat-tailed, mixed-sign rows.

    Deterministic for a fixed (k, s, seed). Guarantees per-row non-degeneracy for
    every S in the grid, including S = 2.
    """
    rng = np.random.default_rng(seed + 1000 * k + s)
    rows = []
    for i in range(k):
        kind = i % 4
        if kind == 0:
            row = rng.standard_normal(s)
        elif kind == 1:
            row = rng.standard_t(3.0, s) * 0.8
        elif kind == 2:
            row = rng.lognormal(0.0, 0.6, s) - 1.0
        else:
            row = -rng.exponential(1.5, s)  # mixed sign: short book
        # perturb to break exact ties across rows without changing shape family
        row = row + 1e-9 * np.arange(s)
        rows.append(row)
    m = np.ascontiguousarray(np.vstack(rows), dtype=np.float64)
    # guard: no zero-variance row (copula/spearman contracts need it)
    if np.any(np.var(m, axis=1) <= 0.0):
        m = m + rng.standard_normal(m.shape) * 1e-6
    return np.ascontiguousarray(m, dtype=np.float64)


def safe_transformers(k: int):
    """
    Every current transformer at overlap-SAFE parameters (n_lower+n_upper<=S for
    all S in the grid). Returns [(id, transformer, needs_rng)].

    NOTE (Phase 4 rename): tail-alignment class names become *Comonotone. These
    identities survive the rename; only the imports change.
    """
    corr = equicorrelation(k, 0.5)
    q = 0.05
    return [
        ("gaussian_copula", GaussianCopula(latent_correlation=corr), True),
        ("t_copula", TCopula(latent_correlation=corr, degrees_of_freedom=4.0), True),
        ("comonotonic", Comonotonic(), False),
        ("lower_comono", LowerTailAlignment(tail_probability=q, tail_order="comonotonic"), False),
        ("lower_random", LowerTailAlignment(tail_probability=q, tail_order="random"), True),
        ("upper_comono", UpperTailAlignment(tail_probability=q, tail_order="comonotonic"), False),
        ("upper_random", UpperTailAlignment(tail_probability=q, tail_order="random"), True),
        ("dual_comono", DualTailAlignment(lower_probability=q, upper_probability=q, tail_order="comonotonic"), False),
        ("dual_random", DualTailAlignment(lower_probability=q, upper_probability=q, tail_order="random"), True),
    ]


# ---- exact repository VaR/ES convention (replica) ------------------------------
def lower_tail_parameters(s: int, confidence_level: float):
    """Replica of scenario_metrics._computer._compute_lower_tail_parameters."""
    alpha = 1.0 - confidence_level
    tail_mass = alpha * float(s)
    nearest = np.rint(tail_mass)
    tol = 16.0 * np.finfo(np.float64).eps * max(1.0, abs(tail_mass))
    if abs(tail_mass - nearest) <= tol:
        tail_mass = float(nearest)
    full_tail_count = int(np.floor(tail_mass))
    fractional_weight = tail_mass - float(full_tail_count)
    if fractional_weight == 0.0:
        var_index = max(0, full_tail_count - 1)
    else:
        var_index = full_tail_count
    return var_index, full_tail_count, fractional_weight, tail_mass


def var_es_lower_tail(pnl: np.ndarray, confidence_level: float):
    """Signed lower-tail empirical VaR and ES (repository convention)."""
    pnl = np.ascontiguousarray(pnl, dtype=np.float64).ravel()
    s = pnl.shape[0]
    var_index, full_tail_count, fractional_weight, tail_mass = lower_tail_parameters(
        s, confidence_level
    )
    ordered = np.sort(pnl)
    var_value = float(ordered[var_index])
    tail_sum = float(ordered[:full_tail_count].sum())
    if fractional_weight > 0.0:
        tail_sum += fractional_weight * var_value
    es_value = tail_sum / tail_mass
    return var_value, es_value


def standalone_sum_var_es(data: np.ndarray, confidence_level: float):
    """Sum over rows of each row's standalone VaR and ES (undiversified)."""
    vs = 0.0
    es = 0.0
    for k in range(data.shape[0]):
        v, e = var_es_lower_tail(data[k], confidence_level)
        vs += v
        es += e
    return vs, es
