# src/riskquant/dependence_stress/copulas.py

"""
Copula-based dependence-stress specifications.

Both transformers are stochastic. They draw one finite copula sample and
transfer its rank structure onto the empirical observations. The caller
therefore owns reproducibility through the explicit generator.

``latent_correlation`` is the correlation of the latent Gaussian
variables. It is not the Pearson correlation and not the Spearman
correlation of the stressed empirical matrix. The finite-sample
observable dependence is a realization, not the specification.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .permutations import _build_rank_permutation_unchecked
from .types import FloatArray, IntArray
from .validation import validate_correlation, validate_rng


def _own_correlation_and_factor(
    correlation: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """
    Validate, copy, and factorize a latent correlation matrix.

    The copy exists so that internal state is owned. Freezing a caller's
    array would be an observable side effect on caller memory.

    The Cholesky factor is computed once at construction. The hot path
    only multiplies.
    """
    validated = validate_correlation(correlation)

    owned = np.array(
        validated,
        dtype=np.float64,
        order="C",
        copy=True,
    )

    cholesky = np.ascontiguousarray(
        np.linalg.cholesky(owned),
        dtype=np.float64,
    )

    owned.setflags(write=False)
    cholesky.setflags(write=False)

    return owned, cholesky


def _require_finite_scores(
    scores: FloatArray,
) -> FloatArray:
    """
    Reject a non-finite score matrix.

    Very small ``nu`` can produce a chi-square draw that underflows to
    zero, which sends the scaled t score to infinity. Sorting such a
    matrix still yields a valid permutation, so the failure would be
    silent. Fail loudly instead. Cost is O(K S), below the O(K S log S)
    sort that follows.
    """
    if not np.isfinite(scores).all():
        raise ValueError(
            "copula score matrix contains NaN or infinite values; "
            "the latent draw is degenerate."
        )

    return scores


def _check_dimension(
    correlation: FloatArray,
    k: int,
) -> None:
    """
    Reject a latent correlation matrix that does not match K.

    An explicit contract error is required. A raw broadcasting or
    matrix-multiplication exception is not an acceptable contract.
    """
    if correlation.shape != (k, k):
        raise ValueError(
            f"latent correlation shape {correlation.shape} does not match K={k}."
        )


@dataclass(frozen=True, slots=True, eq=False)
class GaussianCopula:
    """
    Gaussian-copula empirical dependence transformer.

    The transformer draws K x S correlated Gaussian scores and transfers
    their row-wise rank structure onto the empirical observations.

    Parameters
    ----------
    latent_correlation:
        K x K strictly positive-definite correlation matrix of the latent
        Gaussian variables.

    Notes
    -----
    Stochastic. The same specification with a different generator state
    produces a different finite-sample permutation. This is Monte Carlo
    realization noise, not a change of specification.

    The latent correlation does not equal the resulting empirical Pearson
    or Spearman correlation. For a Gaussian copula the population
    Spearman correlation is ``(6/pi) * arcsin(rho/2)``, and the
    finite-sample estimate scatters around it.
    """

    latent_correlation: FloatArray

    _cholesky: FloatArray = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        owned, cholesky = _own_correlation_and_factor(
            self.latent_correlation,
        )

        object.__setattr__(self, "latent_correlation", owned)
        object.__setattr__(self, "_cholesky", cholesky)

    @property
    def requires_rng(self) -> bool:
        """Gaussian copula sampling is stochastic."""
        return True

    def build_permutation(
        self,
        data: FloatArray,
        rng: np.random.Generator | None = None,
        *,
        data_order: IntArray | None = None,
    ) -> IntArray:
        """
        Build the Gaussian-copula rank permutation.

        Parameters
        ----------
        data:
            Validated K x S observation matrix.
        rng:
            Explicit generator. Required.

        Returns
        -------
        IntArray
            K x S int32 row-wise permutation matrix.
        """
        generator = validate_rng(rng)

        k, s = data.shape

        _check_dimension(self.latent_correlation, k)

        scores = _require_finite_scores(
            self._generate_scores(
                rng=generator,
                num_scenarios=s,
            )
        )

        return _build_rank_permutation_unchecked(
            data,
            scores,
            data_order=data_order,
        )

    def _generate_scores(
        self,
        *,
        rng: np.random.Generator,
        num_scenarios: int,
    ) -> FloatArray:
        """
        Draw K x S correlated Gaussian scores as ``Z = L @ eps``.

        Both operands are C-contiguous, so the product is C-contiguous
        and no layout conversion copy occurs.
        """
        k = self.latent_correlation.shape[0]

        iid = rng.standard_normal(
            size=(k, num_scenarios),
        )

        return self._cholesky @ iid


@dataclass(frozen=True, slots=True, eq=False)
class TCopula:
    """
    Student-t copula empirical dependence transformer.

    Construction:

        Z = L @ eps,        eps ~ N(0, I)
        X_s = Z_s / sqrt(chi2_nu_s / nu)

    One chi-square scale variable is drawn per scenario and shared across
    all K dimensions in that scenario. That common scale is what creates
    symmetric tail dependence. Drawing a K x S chi-square array instead
    would produce independently scaled marginals and destroy the
    construction.

    Parameters
    ----------
    latent_correlation:
        K x K strictly positive-definite latent correlation matrix.
    degrees_of_freedom:
        Finite and strictly positive. Lower values give stronger
        symmetric tail dependence for a fixed latent correlation.

    Notes
    -----
    Stochastic. ``nu = inf`` is not accepted: the contract requires a
    finite value, and the Gaussian limiting case has its own explicit
    transformer. The two methods are not identical for any finite ``nu``.
    """

    latent_correlation: FloatArray
    degrees_of_freedom: float

    _cholesky: FloatArray = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        degrees_of_freedom = float(self.degrees_of_freedom)

        if not np.isfinite(degrees_of_freedom):
            raise ValueError(
                f"degrees_of_freedom must be finite; got {self.degrees_of_freedom!r}."
            )

        if degrees_of_freedom <= 0.0:
            raise ValueError(
                f"degrees_of_freedom must be positive; got {degrees_of_freedom!r}."
            )

        owned, cholesky = _own_correlation_and_factor(
            self.latent_correlation,
        )

        object.__setattr__(self, "latent_correlation", owned)
        object.__setattr__(self, "degrees_of_freedom", degrees_of_freedom)
        object.__setattr__(self, "_cholesky", cholesky)

    @property
    def requires_rng(self) -> bool:
        """t-copula sampling is stochastic."""
        return True

    def build_permutation(
        self,
        data: FloatArray,
        rng: np.random.Generator | None = None,
        *,
        data_order: IntArray | None = None,
    ) -> IntArray:
        """
        Build the t-copula rank permutation.

        Parameters
        ----------
        data:
            Validated K x S observation matrix.
        rng:
            Explicit generator. Required.

        Returns
        -------
        IntArray
            K x S int32 row-wise permutation matrix.
        """
        generator = validate_rng(rng)

        k, s = data.shape

        _check_dimension(self.latent_correlation, k)

        scores = _require_finite_scores(
            self._generate_scores(
                rng=generator,
                num_scenarios=s,
            )
        )

        return _build_rank_permutation_unchecked(
            data,
            scores,
            data_order=data_order,
        )

    def _generate_scores(
        self,
        *,
        rng: np.random.Generator,
        num_scenarios: int,
    ) -> FloatArray:
        """
        Draw K x S multivariate-t scores with one common scale per scenario.

        The division is in place on the Gaussian block to avoid a second
        K x S allocation.
        """
        k = self.latent_correlation.shape[0]
        nu = self.degrees_of_freedom

        iid = rng.standard_normal(
            size=(k, num_scenarios),
        )

        scores = self._cholesky @ iid

        chi_squared = rng.chisquare(
            nu,
            size=num_scenarios,
        )

        scale = np.sqrt(chi_squared / nu)

        # A host process may run under np.seterr(all="raise"). A chi-square
        # draw that underflows to zero would then raise FloatingPointError
        # here, an incidental exception from the wrong layer. Let the
        # division produce inf and report it through the explicit
        # finiteness contract in _require_finite_scores instead.
        with np.errstate(all="ignore"):
            scores /= scale[None, :]

        return scores
