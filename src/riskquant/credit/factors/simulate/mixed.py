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
left independent and PIT-transformed off that core) — remains UNIMPLEMENTED.
The multi-factor models therefore run against mutually independent systemic
factors, which is exactly what their `w_i = sqrt(1 - sum beta^2)` normalisation
assumes; loadings calibrated against a correlated factor model would break that
normalisation. Correlated systemic factors are a model extension, not a gap in
the current contract (see STABLE_API.md, C-07).

REPRODUCIBILITY (D-C15): BATCH-INVARIANT BY CONSTRUCTION
--------------------------------------------------------
At a fixed ``seed``, factor universe and ``total_scenarios``, the generated
matrix is bitwise identical for EVERY ``batch_size``. That is a contract, not an
accident, and it is achieved by having exactly ONE consumer of the random
stream: the ``standard_normal`` fill of the whole ``(bs, n_factors)`` buffer.
Concatenating sequential fills of the same generator reproduces one large fill,
so batching cannot move the stream.

Every non-normal marginal is then derived from the standard normal already
sitting in its own cell by an exact probability-integral transform (PIT):

    uniform    U = Phi(Z)
    indicator  1{Z < Phi^-1(p)}                    == 1{Phi(Z) < p}
    chi        W = F_chi2(nu)^-1(Phi(Z)) = 2 * Gamma(nu/2)^-1(Phi(Z));
               the column stores 1 / sqrt(W)
    beta       P = F_Beta(alpha, beta)^-1(Phi(Z))

The marginals are exactly those the previous per-column ``rng`` draws produced,
and the columns remain mutually independent because the underlying normals are.
The PREVIOUS design drew each special column from the shared generator INSIDE
the batch loop, so a second batch interleaved those draws with the next normal
fill and the whole stream shifted: same seed, different ``batch_size``,
different scenarios. Nothing downstream detected it.

Deriving the special columns off the normal core is also the construction the
deferred multi-factor work needs (a Cholesky over a systemic sub-block with
every other row PIT-transformed off that core), so this does not spend a design
decision that belongs to a later tranche.

CONTRACT: yielded `latent_factors` are views into a reused buffer.

LAYOUT
------
- Main buffer is C-order ``(batch_size, n_factors)`` so the partial-batch slice
  ``buffer[:bs, :]`` is C-contiguous — required by ``rng.standard_normal(out=
  ...)``.
- A 1D ``scratch`` buffer (length ``batch_size``) backs each non-normal column
  transform: the column is copied into the contiguous scratch, transformed in
  place with ``out=`` ufuncs, then written back with the strided assignment
  ``buffer[:bs, col] = scratch[:bs]``. Transforming the strided column view
  directly would work but at strided-access cost. ``bool_scratch`` backs the
  indicator threshold the same way.
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
from scipy.special import betaincinv, gammaincinv, ndtr, ndtri

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


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
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

        special = np.concatenate((
            self.uniform_columns,
            self.chi_columns,
            self.indicator_columns,
            self.beta_columns,
        ))
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
        """Stream `(batch_size, F)` latent batches; bitwise batch-invariant.

        The only draw is the whole-buffer ``standard_normal`` fill. Non-normal
        columns are exact PIT transforms of the normal already in the cell, so
        the random stream advances by ``bs * n_factors`` variates per batch and
        by nothing else — see the module docstring (D-C15).
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        n_factors = int(self.available_factor_ids.shape[0])
        self._validate(n_factors)

        uniform_cols = self.uniform_columns
        chi_cols = self.chi_columns
        chi_shape_halved = self.chi_nu * 0.5  # chi^2(nu) = 2 * Gamma(nu/2)
        ind_cols = self.indicator_columns
        # 1{U < p} with U = Phi(Z) is exactly 1{Z < Phi^-1(p)}; hoisting the
        # threshold keeps the indicator a single comparison on the raw normal.
        # p = 0 -> -inf (never stressed), p = 1 -> +inf (always): both exact.
        ind_threshold = ndtri(self.indicator_p)
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
        # Reused scratch (allocated once): contiguous target for the per-column
        # transforms, plus a bool scratch for the indicator threshold.
        scratch = np.empty(batch_size, dtype=np.float64)
        bool_scratch = np.empty(batch_size, dtype=np.bool_)

        scenario_offset = 0
        total = self.total_scenarios

        while scenario_offset < total:
            bs = min(batch_size, total - scenario_offset)
            view = buffer[:bs, :]

            # THE ONLY CONSUMER OF THE RANDOM STREAM. Everything below is a
            # deterministic transform of these draws, which is what makes the
            # batch size irrelevant to the result.
            rng.standard_normal(out=cast(np.ndarray, view))

            # Uniform columns: U = Phi(Z).
            for c in range(n_uniform):
                s_buf = scratch[:bs]
                np.copyto(s_buf, buffer[:bs, uniform_cols[c]])
                ndtr(s_buf, out=s_buf)
                buffer[:bs, uniform_cols[c]] = s_buf

            # Chi columns: W = 2 * Gamma(nu/2)^-1(Phi(Z)) ~ chi^2(nu); store
            # 1 / sqrt(W) so the t-copula kernel never takes a sqrt.
            for c in range(n_chi):
                s_buf = scratch[:bs]
                np.copyto(s_buf, buffer[:bs, chi_cols[c]])
                ndtr(s_buf, out=s_buf)
                gammaincinv(chi_shape_halved[c], s_buf, out=s_buf)
                s_buf *= 2.0  # W ~ chi^2(nu)
                np.sqrt(s_buf, out=s_buf)
                np.reciprocal(s_buf, out=s_buf)  # store 1 / sqrt(W)
                if not bool(np.isfinite(s_buf).all()):
                    # Only reachable if Phi(Z) underflows to exactly 0 (needs
                    # Z < -37), which would make W = 0 and 1/sqrt(W) infinite.
                    # Fail loud rather than ship an infinite latent scale.
                    raise ValueError(
                        "MixedCreditFactorGenerator: non-finite 1/sqrt(W) in "
                        f"chi column {int(chi_cols[c])} (nu="
                        f"{float(self.chi_nu[c])}) at scenario offset "
                        f"{scenario_offset}."
                    )
                buffer[:bs, chi_cols[c]] = s_buf

            # Indicator columns: {0.0, 1.0} with P(=1) = p.
            for c in range(n_ind):
                s_buf = scratch[:bs]
                np.copyto(s_buf, buffer[:bs, ind_cols[c]])
                b = bool_scratch[:bs]
                np.less(s_buf, ind_threshold[c], out=b)  # Z < Phi^-1(p)
                buffer[:bs, ind_cols[c]] = b  # bool -> float64 on assignment

            # Beta columns: P = F_Beta(alpha, beta)^-1(Phi(Z)). This is the
            # per-scenario stochastic common PD the beta-binomial kernel
            # thresholds each obligor's independent uniform against.
            for c in range(n_beta):
                s_buf = scratch[:bs]
                np.copyto(s_buf, buffer[:bs, beta_cols[c]])
                ndtr(s_buf, out=s_buf)
                betaincinv(beta_alpha[c], beta_beta[c], s_buf, out=s_buf)
                buffer[:bs, beta_cols[c]] = s_buf

            yield CreditFactorBatch(
                latent_factors=view,
                scenario_offset=scenario_offset,
                scenario_count_total=total,
            )
            scenario_offset += bs
