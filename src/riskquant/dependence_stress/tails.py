# src/riskquant/dependence_stress/tails.py

"""
Non-parametric tail and extreme-dependence policy.

These transformers engineer empirical co-occurrence directly. They do not
fit or sample a parametric dependence model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .permutations import _comonotonic_permutation_unchecked
from .types import FloatArray, IntArray, TailOrder
from .validation import (
    validate_disjoint_tail_counts,
    validate_probability,
    validate_rng,
)


@dataclass(frozen=True, slots=True)
class LowerTailAlignment:
    """
    Non-parametric lower-tail alignment transformer.

    The lowest ``n_tail = max(1, ceil(q * S))`` observations of every row
    move into the same scenario slots ``0 .. n_tail-1``. The rows
    therefore realize their worst outcomes together.

    Parameters
    ----------
    tail_probability:
        ``q`` in the open interval (0, 1).

    tail_order:
        ``"comonotonic"``
            Worst with worst, second worst with second worst, and so on
            inside the common tail block. Deterministic.

        ``"random"``
            Randomize severity matching inside the tail block. The same
            observations still occupy the tail slots; only their pairing
            across rows changes. This is a research severity-randomization
            mode, not a canonical statistical dependence model. It is
            stochastic and requires an explicit generator.

    preserve_non_tail_order:
        True
            The non-tail observations keep their relative scenario
            ordering. They are not sorted. Only the tail block is
            rearranged.

        False
            The non-tail observations are also placed in ascending order,
            which makes the complete matrix comonotonic.

    Notes
    -----
    Marginal preservation is exact in every mode. The transformer only
    permutes scenario indices.

    Tie convention at the tail boundary
    -----------------------------------
    The tail block is the first ``n_tail`` entries of a stable argsort of
    the row. When observations straddle the boundary at equal value, the
    tied scenario with the lower index enters the tail. This is a fixed,
    documented consequence of ``kind="stable"``; there is no random
    tie-breaking. Two runs on identical data therefore select identical
    tail scenarios, in ``"comonotonic"`` and ``"random"`` mode alike.
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
    ) -> IntArray:
        """
        Construct the lower-tail-alignment permutation.

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
            K x S int64 row-wise permutation matrix.

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

        sorted_indices = np.argsort(
            data,
            axis=1,
            kind="stable",
        ).astype(np.int64, copy=False)

        permutations = np.empty(
            (k, s),
            dtype=np.int64,
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
            # size measured, allocates one int64 block instead of a
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
class UpperTailAlignment:
    """
    Non-parametric upper-tail alignment transformer.

    The highest ``n_tail = max(1, ceil(q * S))`` observations of every row
    move into the same scenario slots ``S - n_tail .. S-1``. The rows
    therefore realize their highest outcomes together.
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
    ) -> IntArray:

        k, s = data.shape
        n_tail = max(1, int(np.ceil(self.tail_probability * s)))

        sorted_indices = np.argsort(
            data,
            axis=1,
            kind="stable",
        ).astype(np.int64, copy=False)

        permutations = np.empty((k, s), dtype=np.int64, order="C")

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
class DualTailAlignment:
    """
    Non-parametric dual-tail alignment transformer.

    The lowest `n_lower` observations move into slots `0 .. n_lower-1`.
    The highest `n_upper` observations move into slots `S - n_upper .. S-1`.
    This enforces extreme co-occurrence at both ends of the marginals,
    essential for risk factors with mixed long/short exposures.
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
    ) -> IntArray:

        k, s = data.shape

        n_lower = max(1, int(np.ceil(self.lower_probability * s)))
        n_upper = max(1, int(np.ceil(self.upper_probability * s)))

        # Discrete allocation contract. ceil() and the max(1, ...) floor can
        # make the tail blocks overlap even when the probabilities sum below
        # one. S is known only here, so the guard runs before any allocation
        # or write, for both comonotonic and random modes.
        validate_disjoint_tail_counts(n_lower, n_upper, s)

        sorted_indices = np.argsort(
            data,
            axis=1,
            kind="stable",
        ).astype(np.int64, copy=False)

        permutations = np.empty((k, s), dtype=np.int64, order="C")

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
    Extreme positive-dependence empirical benchmark.

    Every row is sorted ascending and aligned column by column.

    Invariants
    ----------
    - Applying the permutation gives exactly ``np.sort(data, axis=1)``.
    - Spearman correlation of the output is 1 for every pair of
      non-degenerate rows.
    - Pearson correlation is *maximal* over all row-wise permutations,
      by the rearrangement inequality (Hardy-Littlewood-Polya).
    - The transformation is idempotent: ``T(T(X)) == T(X)``.

    Deterministic. It never consumes randomness.

    Notes
    -----
    Pearson correlation of the comonotonic arrangement is not 1 in
    general. Comonotonicity is a rank property; Pearson reaches 1 only
    when the two rows are affine transforms of one another. For rows
    with different marginal shapes the comonotonic Pearson correlation
    is strictly below 1 while the Spearman correlation is exactly 1.
    The correct Pearson statement is the maximality above, and that is
    what the test suite asserts.
    """

    @property
    def requires_rng(self) -> bool:
        """Comonotonic alignment is deterministic."""
        return False

    def build_permutation(
        self,
        data: FloatArray,
        rng: np.random.Generator | None = None,
    ) -> IntArray:
        """
        Construct the deterministic comonotonic permutation.

        ``rng`` is accepted for protocol uniformity and ignored.
        """
        return _comonotonic_permutation_unchecked(data)
