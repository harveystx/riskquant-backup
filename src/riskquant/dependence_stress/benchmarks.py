# src/riskquant/dependence_stress/benchmarks.py

"""
Extremal dependence benchmarks.

These transformers are explicit deterministic (or, in the randomized tail
mode, semi-deterministic) extremal rearrangements of the fixed empirical
marginals. They are reference/corner-case constructions, not calibrated
dependence models: they engineer empirical co-occurrence directly and never
fit or sample a parametric dependence structure.

Contrast with ``copulas.py`` (GaussianCopula, TCopula), which are stochastic
DEPENDENCE STRESSES: fully specified alternative dependence models applied to
the same fixed marginals. A benchmark answers "what is the extremal corner?";
a dependence stress answers "what does this dependence model imply?".

Every construction here preserves each marginal exactly; only the
cross-sectional pairing of scenarios changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .permutations import _comonotonic_permutation_unchecked, _resolve_data_order
from .types import FloatArray, IntArray, TailOrder
from .validation import (
    validate_disjoint_tail_counts,
    validate_probability,
    validate_rng,
)


@dataclass(frozen=True, slots=True)
class LowerTailComonotone:
    """
    Extremal lower-tail benchmark.

    The lowest ``n_tail = max(1, ceil(q * S))`` observations of every marginal
    occupy the common lower block ``0 .. n_tail-1``, so all rows realise their
    worst outcomes in the same scenarios. This is a benchmark construction, not
    a calibrated tail-dependence model, and not a general-purpose estimator.

    Risk identity
    -------------
    Under RiskQuant's empirical signed lower-tail P&L VaR convention, for any
    reporting level ``alpha <= q``::

        VaR_benchmark(alpha) = sum_k VaR_k(alpha)

    because every row's ``alpha``-tail scenarios coincide. The benchmark
    therefore INTENTIONALLY destroys diversification at the target lower tail;
    it reproduces the undiversified sum, and is a VaR reference at that tail,
    not a tunable dependence stress.

    Parameters
    ----------
    tail_probability:
        ``q`` in the open interval (0, 1). Sets the lower block size.

    tail_order:
        ``"comonotonic"``
            Align equal empirical severity ranks: worst with worst, second
            worst with second worst, inside the common tail block.
            Deterministic.

        ``"random"``
            Randomize severity pairing WITHIN the selected lower block. Tail
            membership is preserved; only the within-tail cross-row pairing is
            redrawn. This is a research/scenario mode: it is not a calibrated
            copula dependence specification and provides no target VaR
            parameter. Its previously established invariants hold — the tail
            SET is unchanged, so the lower-tail ES is invariant to the draw,
            while the tail VaR is not calibrated to any level. Stochastic;
            requires an explicit generator.

    preserve_non_tail_order:
        True
            Non-tail observations keep their relative scenario ordering; only
            the tail block is rearranged.
        False
            Non-tail observations are also placed ascending, making the whole
            matrix comonotonic.

    Notes
    -----
    Marginal preservation is exact in every mode; the transformer only permutes
    scenario indices.

    Tie convention at the tail boundary
    -----------------------------------
    The tail block is the first ``n_tail`` entries of a stable argsort of the
    row. When observations straddle the boundary at equal value, the tied
    scenario with the lower index enters the tail. This is a fixed, documented
    consequence of ``kind="stable"``; there is no random tie-breaking. Two runs
    on identical data select identical tail scenarios, in ``"comonotonic"`` and
    ``"random"`` mode alike.
    """

    tail_probability: float = 0.05
    tail_order: TailOrder = "comonotonic"
    preserve_non_tail_order: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tail_probability",
            validate_probability(self.tail_probability),
        )

        if self.tail_order not in ("comonotonic", "random"):
            raise ValueError(
                f"unsupported tail_order={self.tail_order!r}; "
                "expected 'comonotonic' or 'random'."
            )

    @property
    def requires_rng(self) -> bool:
        """Only the ``"random"`` tail order consumes randomness."""
        return self.tail_order == "random"

    def build_permutation(
        self,
        data: FloatArray,
        rng: np.random.Generator | None = None,
        *,
        data_order: IntArray | None = None,
    ) -> IntArray:
        """
        Construct the lower-tail-comonotone permutation.

        Parameters
        ----------
        data:
            Validated K x S observation matrix.
        rng:
            Explicit generator. Required only when
            ``tail_order == "random"``.

        Returns
        -------
        IntArray
            K x S int32 row-wise permutation matrix.

        Notes
        -----
        The non-tail block under ``preserve_non_tail_order=True`` is
        obtained by re-sorting the non-tail scenario indices ascending.
        Ascending scenario index order is exactly the original relative
        ordering, so this reproduces a boolean-mask selection without
        allocating a per-row K x S mask.
        """
        k, s = data.shape

        n_tail = max(
            1,
            int(np.ceil(self.tail_probability * s)),
        )

        sorted_indices = _resolve_data_order(data, data_order)

        permutations = np.empty(
            (k, s),
            dtype=np.int32,
            order="C",
        )

        tail_block = sorted_indices[:, :n_tail]

        if self.tail_order == "random":
            generator = validate_rng(rng)

            # rng.permuted shuffles each row independently in one call.
            # Benchmarked against argsort-of-uniform-keys and a Python
            # loop of rng.permutation over K rows (K=100, n_tail=250:
            # 257us vs 1145us vs 506us; K=100, n_tail=2500: 3.1ms vs
            # 18.5ms vs 3.6ms). It is the fastest of the three at every
            # size measured, allocates one int32 block instead of a
            # K x n_tail float64 key matrix, and reads more directly.
            permutations[:, :n_tail] = generator.permuted(
                tail_block,
                axis=1,
            )
        else:
            permutations[:, :n_tail] = tail_block

        if n_tail < s:
            non_tail_block = sorted_indices[:, n_tail:]

            if self.preserve_non_tail_order:
                permutations[:, n_tail:] = np.sort(
                    non_tail_block,
                    axis=1,
                )
            else:
                permutations[:, n_tail:] = non_tail_block

        return permutations


@dataclass(frozen=True, slots=True)
class UpperTailComonotone:
    """
    Extremal upper-tail benchmark.

    The highest ``n_tail = max(1, ceil(q * S))`` observations of every marginal
    occupy the common upper block ``S - n_tail .. S-1``, so all rows realise
    their highest outcomes in the same scenarios. "Upper" refers to the
    NUMERICAL P&L tail.

    Under a signed lower-tail P&L VaR convention the upper P&L tail is
    generally the benign direction, so this construction is not the loss tail.
    It is nonetheless useful as an extremal upper-tail benchmark, a dependence
    scenario, and a diagnostic/reference construction. It is not a generic
    "risk tail" and not a calibrated dependence model.

    ``tail_order`` and ``preserve_non_tail_order`` behave exactly as documented
    on ``LowerTailComonotone``, reflected to the upper block; the ``"random"``
    mode is the same research severity-pairing randomization with the same
    tail-set / ES invariance.
    """

    tail_probability: float = 0.05
    tail_order: TailOrder = "comonotonic"
    preserve_non_tail_order: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tail_probability",
            validate_probability(self.tail_probability),
        )

        if self.tail_order not in ("comonotonic", "random"):
            raise ValueError(
                f"unsupported tail_order={self.tail_order!r}; "
                "expected 'comonotonic' or 'random'."
            )

    @property
    def requires_rng(self) -> bool:
        return self.tail_order == "random"

    def build_permutation(
        self,
        data: FloatArray,
        rng: np.random.Generator | None = None,
        *,
        data_order: IntArray | None = None,
    ) -> IntArray:

        k, s = data.shape
        n_tail = max(1, int(np.ceil(self.tail_probability * s)))

        sorted_indices = _resolve_data_order(data, data_order)

        permutations = np.empty((k, s), dtype=np.int32, order="C")

        # Upper tail occupies the last n_tail slots
        tail_block = sorted_indices[:, s - n_tail :]

        if self.tail_order == "random":
            generator = validate_rng(rng)
            permutations[:, s - n_tail :] = generator.permuted(tail_block, axis=1)
        else:
            permutations[:, s - n_tail :] = tail_block

        # Non-tail block occupies the first S - n_tail slots
        if n_tail < s:
            non_tail_block = sorted_indices[:, : s - n_tail]

            if self.preserve_non_tail_order:
                permutations[:, : s - n_tail] = np.sort(non_tail_block, axis=1)
            else:
                permutations[:, : s - n_tail] = non_tail_block

        return permutations


@dataclass(frozen=True, slots=True)
class DualTailComonotone:
    """
    Extremal dual-tail benchmark.

    The lowest ``n_lower`` observations move into slots ``0 .. n_lower-1`` and
    the highest ``n_upper`` observations into ``S - n_upper .. S-1``, so both
    ends of every marginal co-occur simultaneously. This is the benchmark for
    factors with mixed long/short exposures.

    Discrete allocation contract
    ----------------------------
    The discretized blocks must satisfy::

        n_lower + n_upper <= S

    Overlap (``n_lower + n_upper > S``) is a hard contract violation and raises
    ``ValueError`` before any permutation is built. Equality
    (``n_lower + n_upper == S``) is valid: the two blocks exactly tile the
    scenario space and leave no middle block.

    Benchmark status is explicit: this is an extremal reference construction,
    not a calibrated dependence model.
    """

    lower_probability: float = 0.05
    upper_probability: float = 0.05
    tail_order: TailOrder = "comonotonic"
    preserve_non_tail_order: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "lower_probability",
            validate_probability(self.lower_probability),
        )
        object.__setattr__(
            self,
            "upper_probability",
            validate_probability(self.upper_probability),
        )

        if self.lower_probability + self.upper_probability >= 1.0:
            raise ValueError(
                "The sum of lower and upper tail probabilities must be "
                f"strictly less than 1.0; got {self.lower_probability + self.upper_probability}"
            )

        if self.tail_order not in ("comonotonic", "random"):
            raise ValueError(
                f"unsupported tail_order={self.tail_order!r}; "
                "expected 'comonotonic' or 'random'."
            )

    @property
    def requires_rng(self) -> bool:
        return self.tail_order == "random"

    def build_permutation(
        self,
        data: FloatArray,
        rng: np.random.Generator | None = None,
        *,
        data_order: IntArray | None = None,
    ) -> IntArray:

        k, s = data.shape

        n_lower = max(1, int(np.ceil(self.lower_probability * s)))
        n_upper = max(1, int(np.ceil(self.upper_probability * s)))

        # Discrete allocation contract. ceil() and the max(1, ...) floor can
        # make the tail blocks overlap even when the probabilities sum below
        # one. S is known only here, so the guard runs before any allocation
        # or write, for both comonotonic and random modes.
        validate_disjoint_tail_counts(n_lower, n_upper, s)

        sorted_indices = _resolve_data_order(data, data_order)

        permutations = np.empty((k, s), dtype=np.int32, order="C")

        # 1. Lower Tail Block
        lower_block = sorted_indices[:, :n_lower]

        # 2. Upper Tail Block
        upper_block = sorted_indices[:, s - n_upper :]

        if self.tail_order == "random":
            generator = validate_rng(rng)
            permutations[:, :n_lower] = generator.permuted(lower_block, axis=1)
            permutations[:, s - n_upper :] = generator.permuted(upper_block, axis=1)
        else:
            permutations[:, :n_lower] = lower_block
            permutations[:, s - n_upper :] = upper_block

        # 3. Non-Tail (Middle) Block
        if n_lower + n_upper < s:
            middle_block = sorted_indices[:, n_lower : s - n_upper]

            if self.preserve_non_tail_order:
                # Re-sorting the original indices ascending restores their original relative ordering
                permutations[:, n_lower : s - n_upper] = np.sort(middle_block, axis=1)
            else:
                permutations[:, n_lower : s - n_upper] = middle_block

        return permutations


@dataclass(frozen=True, slots=True)
class Comonotonic:
    """
    Global comonotonic benchmark: maximal positive rank dependence.

    Every row is sorted ascending and aligned column by column, so all rows
    move together across the whole distribution. This is the extremal
    positive-dependence corner, a reference construction rather than a
    calibrated dependence model. Deterministic; it never consumes randomness.

    Risk identities (RiskQuant empirical signed lower-tail convention)
    ------------------------------------------------------------------
    - Exact marginal preservation: applying the permutation gives exactly
      ``np.sort(data, axis=1)``.
    - Exact ES additivity: ``ES_comonotone = sum_k ES_k`` at any level, and the
      comonotonic ES is the MAXIMUM over all row-wise rearrangements of the
      empirical portfolio ES.
    - VaR identity::

          VaR_comonotone = sum_k VaR_k

      This is a VaR REFERENCE, not a VaR upper bound: VaR is not subadditive,
      so a dependence structure can pierce the comonotonic sum. (ES, being
      subadditive/coherent, does have the comonotone value as its upper bound.)

    Correlation
    -----------
    - Spearman correlation of the output is 1 for every pair of non-degenerate
      rows.
    - Pearson correlation is MAXIMAL over all row-wise permutations by the
      rearrangement inequality (Hardy-Littlewood-Polya), NOT necessarily 1.
      Comonotonicity is a rank property; Pearson reaches 1 only when the rows
      are affine transforms of one another. For rows with different marginal
      shapes the comonotonic Pearson correlation is strictly below 1 while the
      Spearman correlation is exactly 1. The test suite asserts the maximality,
      not equality with 1.

    The transformation is idempotent: ``T(T(X)) == T(X)``.
    """

    @property
    def requires_rng(self) -> bool:
        """Comonotonic alignment is deterministic."""
        return False

    def build_permutation(
        self,
        data: FloatArray,
        rng: np.random.Generator | None = None,
        *,
        data_order: IntArray | None = None,
    ) -> IntArray:
        """
        Construct the deterministic comonotonic permutation.

        ``rng`` is accepted for protocol uniformity and ignored.
        """
        return _comonotonic_permutation_unchecked(data, data_order=data_order)
