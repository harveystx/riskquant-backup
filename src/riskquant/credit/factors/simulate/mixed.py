# src/riskquant/credit/factors/simulate/mixed.py
r"""Mixed credit-factor generator (zero-allocation, streaming).

The general-case generator: fills ONE `CreditFactorMatrix` whose rows carry
different marginals, so a portfolio spanning multiple JTD model families
(copula + Bernoulli + stress-regime) draws from a single coherent factor pool.

`GaussianCopulaGenerator`, `IndependentUniformGenerator` and `TCopulaGenerator`
are special cases of this one — kept separate because their single-marginal
fill is one contiguous draw with no per-column dispatch. Reach for this
generator only when a matrix must hold more than one marginal at once.

Row marginals
-------------
Every column not listed below is an independent standard normal (systemic Z,
idiosyncratic eps — the copula factor rows). The listed columns are overwritten:

    uniform_columns        : U(0, 1)                 — independent-Bernoulli rows
    chi_columns            : 1 / sqrt(W), W ~ chi^2(nu)
                             — t-copula mixing rows (same precompute as
                             `TCopulaGenerator`; the kernel multiplies, never
                             takes a sqrt in the hot loop)
    indicator_columns      : {0.0, 1.0}, P(=1) = p   — stress-regime rows
    beta_columns           : P ~ Beta(alpha, beta)   — beta-binomial common-PD
                             rows (the per-scenario stochastic default
                             probability the kernel thresholds each obligor's
                             independent uniform against)

Dependence
----------
NONE is enforced here. Obligor-to-obligor correlation is built in the JTD
kernels from factor loadings against SHARED row indices (`A_i = beta_i * Z +
sqrt(1 - beta_i^2) * eps_i`; two obligors are correlated iff they reference the
same systemic row). Coupling a uniform row to a systemic factor would convert
the independent-Bernoulli model into a Gaussian copula — that is what
`COPULA_SF_GAUSSIAN` is for. The one correlation that legitimately belongs at
the generator level — covariance AMONG systemic factors in a multi-factor
copula (a Cholesky over a designated systemic sub-block, with every other row
left independent and PIT-transformed off that core) — is deferred until
multi-factor copula support lands.

CONTRACT: yielded `latent_factors` are views into a reused buffer.

LAYOUT
------
- Main buffer is C-order ``(batch_size, n_factors)`` so the partial-batch slice
  ``buffer[:bs, :]`` is C-contiguous — required by ``rng.standard_normal(out=
  ...)`` and ``rng.random(out=...)``.
- A 1D ``scratch`` buffer (length ``batch_size``) backs each non-normal column
  fill: the draw targets the contiguous scratch, then the strided write
  ``buffer[:bs, col] = scratch[:bs]`` lands it in the factor column. Filling a
  factor column directly would require a non-contiguous ``out=``, which numpy
  rejects. ``bool_scratch`` backs the indicator threshold the same way.
- Both scratch buffers are allocated once at setup and reused every batch — no
  heap allocation inside the streaming loop.
"""

from __future__ import annotations

__all__ = [
    "MixedCreditFactorGenerator",
]

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt

from riskquant.credit.factors.simulate.base import (
    CreditFactorBatch,
    CreditFactorGenerator,
)

if TYPE_CHECKING:
    from riskquant.credit.masters.factor_universe.compiled import (
        CompiledCreditFactorUniverse,
    )


def _empty_i32() -> npt.NDArray[np.int32]:
    return np.empty(0, dtype=np.int32)


def _empty_f64() -> npt.NDArray[np.float64]:
    return np.empty(0, dtype=np.float64)


@dataclass(frozen=True, slots=True, kw_only=True)
class MixedCreditFactorGenerator(CreditFactorGenerator):
    """Zero-allocation heterogeneous-marginal latent-factor generator.

    Parameters
    ----------
    available_factor_ids
        (F,) int32 — the full credit-factor universe. Its length sets the
        number of columns; column position == factor UIX.
    total_scenarios
        Number of scenarios to stream.
    uniform_columns
        (n_u,) int32 — factor columns filled with U(0, 1).
    chi_columns, chi_nu
        (n_c,) int32 / (n_c,) float64 — chi columns and their aligned degrees
        of freedom. Each column emits ``1 / sqrt(W)`` with ``W ~ chi^2(nu)``.
        ``nu`` MUST match the obligor ``nu`` of every t-copula obligor that
        references the column (the kernel's ``sqrt(nu_i / W)`` transform breaks
        the t-marginal otherwise).
    indicator_columns, indicator_p
        (n_i,) int32 / (n_i,) float64 — stress-regime columns and their aligned
        stress probabilities in [0, 1]. Each column emits {0.0, 1.0} with
        ``P(=1) = p``.
    seed
        Optional RNG seed.

    Columns not listed in any of the three column arrays are independent
    standard normals. The three column sets must be pairwise disjoint.
    """

    available_factor_ids: npt.NDArray[np.int32]
    total_scenarios: int
    uniform_columns: npt.NDArray[np.int32] = field(default_factory=_empty_i32)
    chi_columns: npt.NDArray[np.int32] = field(default_factory=_empty_i32)
    chi_nu: npt.NDArray[np.float64] = field(default_factory=_empty_f64)
    indicator_columns: npt.NDArray[np.int32] = field(default_factory=_empty_i32)
    indicator_p: npt.NDArray[np.float64] = field(default_factory=_empty_f64)
    beta_columns: npt.NDArray[np.int32] = field(default_factory=_empty_i32)
    beta_alpha: npt.NDArray[np.float64] = field(default_factory=_empty_f64)
    beta_beta: npt.NDArray[np.float64] = field(default_factory=_empty_f64)
    seed: int | None = field(default=None)

    @classmethod
    def from_universe(
        cls,
        universe: CompiledCreditFactorUniverse,
        *,
        total_scenarios: int,
        seed: int | None = None,
        stress_p_by_uix: Mapping[int, float] | None = None,
    ) -> MixedCreditFactorGenerator:
        """Build from the credit factor-universe master.

        Every structural column set — chi, uniform, stress-indicator — is
        derived from the universe's kind data, and the chi dof come from
        `chi_nu_vector`. This produces a matrix with the correct marginal on
        EVERY universe row, so it cannot disagree with what JTD ingest expects
        for any model family in the portfolio.

        `stress_p_by_uix` supplies the per-run stress probability for each
        STRESS_INDICATOR factor (keyed by UIX) — the one knob deliberately kept
        off the static universe. Required iff the universe declares any
        stress-indicator factors.
        """
        indicator_columns = universe.stress_indicator_uix
        if indicator_columns.size == 0:
            indicator_p = _empty_f64()
        else:
            if stress_p_by_uix is None:
                raise ValueError(
                    "from_universe: stress_p_by_uix is required — the universe "
                    f"declares STRESS_INDICATOR factor(s) at UIX "
                    f"{indicator_columns.tolist()}, and stress probability is a "
                    "per-run parameter, not a static universe attribute."
                )
            try:
                indicator_p = np.array(
                    [float(stress_p_by_uix[int(c)]) for c in indicator_columns],
                    dtype=np.float64,
                )
            except KeyError as missing:
                raise ValueError(
                    "from_universe: stress_p_by_uix has no entry for "
                    f"STRESS_INDICATOR UIX {missing}."
                ) from None

        return cls(
            available_factor_ids=universe.factor_ids,
            total_scenarios=total_scenarios,
            uniform_columns=universe.uniform_uix,
            chi_columns=universe.chi_uix,
            chi_nu=universe.chi_nu_vector,
            indicator_columns=indicator_columns,
            indicator_p=indicator_p,
            beta_columns=universe.beta_pd_uix,
            beta_alpha=universe.beta_alpha,
            beta_beta=universe.beta_beta,
            seed=seed,
        )

    def chi_nu_by_uix(self) -> dict[int, float]:
        """{chi factor UIX -> dof} — the reconciliation map JTD ingest checks
        obligor ``nu`` against. Pass the result as
        ``compile_jtd_portfolio_from_df(..., chi_nu_by_uix=...)``."""
        return dict(zip(self.chi_columns.tolist(), self.chi_nu.tolist(), strict=True))

    def _validate(self, n_factors: int) -> None:
        if self.chi_columns.shape != self.chi_nu.shape:
            raise ValueError("chi_columns and chi_nu must be aligned")
        if self.indicator_columns.shape != self.indicator_p.shape:
            raise ValueError("indicator_columns and indicator_p must be aligned")
        if not (
            self.beta_columns.shape == self.beta_alpha.shape == self.beta_beta.shape
        ):
            raise ValueError("beta_columns, beta_alpha and beta_beta must be aligned")
        if self.chi_nu.size and not np.all(self.chi_nu > 0.0):
            raise ValueError("chi_nu must be strictly positive")
        if self.indicator_p.size and not np.all(
            (self.indicator_p >= 0.0) & (self.indicator_p <= 1.0)
        ):
            raise ValueError("indicator_p must lie in [0, 1]")
        if self.beta_alpha.size and not np.all(
            (self.beta_alpha > 0.0) & (self.beta_beta > 0.0)
        ):
            raise ValueError("beta_alpha and beta_beta must be strictly positive")

        special = np.concatenate(
            (
                self.uniform_columns,
                self.chi_columns,
                self.indicator_columns,
                self.beta_columns,
            )
        )
        if special.size == 0:
            return
        if special.min() < 0 or special.max() >= n_factors:
            raise ValueError(
                f"factor column out of range [0, {n_factors}): "
                f"{special.min()}..{special.max()}"
            )
        if np.unique(special).size != special.size:
            raise ValueError(
                "a factor column is assigned more than one marginal "
                "(uniform / chi / indicator sets must be disjoint)"
            )

    def iter_batches(self, *, batch_size: int) -> Iterator[CreditFactorBatch]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        n_factors = int(self.available_factor_ids.shape[0])
        self._validate(n_factors)

        uniform_cols = self.uniform_columns
        chi_cols = self.chi_columns
        chi_shape_halved = self.chi_nu * 0.5  # chi^2(nu) = 2 * Gamma(nu/2)
        ind_cols = self.indicator_columns
        ind_p = self.indicator_p
        beta_cols = self.beta_columns
        beta_alpha = self.beta_alpha
        beta_beta = self.beta_beta

        n_uniform = int(uniform_cols.shape[0])
        n_chi = int(chi_cols.shape[0])
        n_ind = int(ind_cols.shape[0])
        n_beta = int(beta_cols.shape[0])

        rng = np.random.default_rng(self.seed)

        # Main buffer: C-order so [:bs, :] is C-contiguous for any bs.
        buffer = np.empty((batch_size, n_factors), dtype=np.float64, order="C")
        # Reused scratch (allocated once): contiguous `out=` target for the
        # per-column draws, plus a bool scratch for the indicator threshold.
        scratch = np.empty(batch_size, dtype=np.float64)
        scratch2 = np.empty(batch_size, dtype=np.float64)
        bool_scratch = np.empty(batch_size, dtype=np.bool_)

        scenario_offset = 0
        total = self.total_scenarios

        while scenario_offset < total:
            bs = min(batch_size, total - scenario_offset)
            view = buffer[:bs, :]

            # Fill the whole batch with independent standard normals; the
            # normal (systemic / idiosyncratic) columns keep these draws.
            rng.standard_normal(out=cast(np.ndarray, view))

            # Overwrite uniform columns with U(0, 1).
            for c in range(n_uniform):
                s = scratch[:bs]
                rng.random(out=cast(np.ndarray, s))
                buffer[:bs, uniform_cols[c]] = s

            # Overwrite chi columns with 1 / sqrt(W), W ~ chi^2(nu).
            for c in range(n_chi):
                s = scratch[:bs]
                rng.standard_gamma(chi_shape_halved[c], out=cast(np.ndarray, s))
                s *= 2.0  # W ~ chi^2(nu)
                np.reciprocal(np.sqrt(s), out=s)  # store 1 / sqrt(W)
                buffer[:bs, chi_cols[c]] = s

            # Overwrite indicator columns with {0.0, 1.0}, P(=1) = p.
            for c in range(n_ind):
                s = scratch[:bs]
                rng.random(out=cast(np.ndarray, s))
                b = bool_scratch[:bs]
                np.less(s, ind_p[c], out=b)  # u < p  -> stressed
                buffer[:bs, ind_cols[c]] = b  # bool -> float64 on assignment

            # Overwrite beta columns with P ~ Beta(alpha, beta). Drawn via the
            # Gamma ratio Beta = G1 / (G1 + G2), G1 ~ Gamma(alpha), G2 ~
            # Gamma(beta): standard_gamma supports out=, so the whole fill stays
            # zero-allocation (rng.beta has no out= and would allocate). This is
            # the per-scenario stochastic common PD the beta-binomial kernel
            # thresholds each obligor's independent uniform against.
            for c in range(n_beta):
                g1 = scratch[:bs]
                g2 = scratch2[:bs]
                rng.standard_gamma(beta_alpha[c], out=cast(np.ndarray, g1))
                rng.standard_gamma(beta_beta[c], out=cast(np.ndarray, g2))
                np.add(g1, g2, out=g2)  # denom = G1 + G2
                np.divide(g1, g2, out=g1)  # G1 / (G1 + G2) ~ Beta(alpha, beta)
                buffer[:bs, beta_cols[c]] = g1

            yield CreditFactorBatch(
                latent_factors=view,
                scenario_offset=scenario_offset,
                scenario_count_total=total,
            )
            scenario_offset += bs
