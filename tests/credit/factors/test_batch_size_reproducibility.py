# tests/credit/factors/test_batch_size_reproducibility.py
"""
Regression tests for audit C-15: at a fixed seed, universe and scenario count,
the generated factor matrix must be BITWISE identical for every batch size, and
so must the public credit VaR built on top of it.

The old `MixedCreditFactorGenerator` drew each non-normal column from the shared
generator inside the batch loop, so a second batch interleaved those draws with
the next whole-buffer normal fill and the entire stream shifted. `solutions.
credit_var` batches at `_FACTOR_BATCH = 25_000`, so retuning that constant — or
crossing it with `n_scenarios` — silently moved published risk numbers.

Marginals are checked against `scipy.stats` reference distributions, never
against the generator's own output, so a batch-invariant but WRONG stream cannot
pass.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import riskquant.solutions.credit_var.run as credit_var_run
from riskquant.credit.factors.simulate.base import build_credit_factor_matrix
from riskquant.credit.factors.simulate.mixed import MixedCreditFactorGenerator
from riskquant.solutions.credit_var.run import run_credit_var
from riskquant.solutions.credit_var.schemas import CreditVarRequest

N_FACTORS = 7
SEED = 20260918

# column -> marginal. 0 and 6 stay standard normal (systemic Z / idiosyncratic).
UNIFORM_COL, CHI_COL, IND_COL, BETA_COL = 1, 2, 3, 4
CHI_NU = 5.0
IND_P = 0.2
BETA_A, BETA_B = 2.0, 38.0


def _generator(total_scenarios: int) -> MixedCreditFactorGenerator:
    """Every supported marginal at once — the case the old design broke worst."""
    return MixedCreditFactorGenerator(
        available_factor_ids=np.arange(N_FACTORS, dtype=np.int32),
        total_scenarios=total_scenarios,
        seed=SEED,
        uniform_columns=np.array([UNIFORM_COL], dtype=np.int32),
        chi_columns=np.array([CHI_COL], dtype=np.int32),
        chi_nu=np.array([CHI_NU]),
        indicator_columns=np.array([IND_COL], dtype=np.int32),
        indicator_p=np.array([IND_P]),
        beta_columns=np.array([BETA_COL], dtype=np.int32),
        beta_alpha=np.array([BETA_A]),
        beta_beta=np.array([BETA_B]),
    )


def _matrix(total_scenarios: int, batch_size: int) -> np.ndarray:
    gen = _generator(total_scenarios)
    return build_credit_factor_matrix(
        gen.iter_batches(batch_size=batch_size),
        num_factors=N_FACTORS,
        num_scenarios=total_scenarios,
    ).latent_factors


@pytest.mark.parametrize("batch_size", [96, 64, 37, 13, 7, 2, 1])
def test_batch_size_reproducibility(batch_size: int) -> None:
    """Bitwise, not approximately: the stochastic contract is exact equality."""
    reference = _matrix(96, 96)  # one batch covers everything
    candidate = _matrix(96, batch_size)
    assert np.array_equal(reference, candidate), (
        f"batch_size={batch_size} changed the factor matrix; first differing "
        f"column {int(np.argmax((reference != candidate).any(axis=0)))}"
    )


def test_every_column_kind_is_batch_invariant_individually() -> None:
    """Isolate each marginal so a regression names the offending column kind."""
    reference = _matrix(96, 96)
    candidate = _matrix(96, 7)
    for col, kind in (
        (0, "normal"),
        (UNIFORM_COL, "uniform"),
        (CHI_COL, "chi"),
        (IND_COL, "indicator"),
        (BETA_COL, "beta"),
    ):
        assert np.array_equal(reference[col], candidate[col]), f"{kind} column drifted"


def test_a_different_seed_still_changes_the_stream() -> None:
    """Batch-invariance must not have been bought by ignoring the seed."""
    other = MixedCreditFactorGenerator(
        available_factor_ids=np.arange(N_FACTORS, dtype=np.int32),
        total_scenarios=96,
        seed=SEED + 1,
        uniform_columns=np.array([UNIFORM_COL], dtype=np.int32),
    )
    matrix = build_credit_factor_matrix(
        other.iter_batches(batch_size=96), num_factors=N_FACTORS, num_scenarios=96
    ).latent_factors
    assert not np.array_equal(matrix[0], _matrix(96, 96)[0])


def test_marginals_match_independent_reference_distributions() -> None:
    """Independent oracle: the PIT-derived columns must still be exactly the
    distributions the kernels assume. `scipy.stats` is the reference; the
    generator is never compared against itself."""
    n = 200_000
    matrix = _matrix(n, 4_096)

    assert stats.kstest(matrix[0], "norm").pvalue > 0.001  # systemic Z
    assert stats.kstest(matrix[UNIFORM_COL], "uniform").pvalue > 0.001

    # The chi column stores 1 / sqrt(W); recover W and test it.
    w = 1.0 / (matrix[CHI_COL] ** 2)
    assert stats.kstest(w, "chi2", args=(CHI_NU,)).pvalue > 0.001

    indicator = matrix[IND_COL]
    assert set(np.unique(indicator).tolist()) <= {0.0, 1.0}
    assert abs(indicator.mean() - IND_P) < 4.0 * np.sqrt(IND_P * (1.0 - IND_P) / n)

    assert stats.kstest(matrix[BETA_COL], "beta", args=(BETA_A, BETA_B)).pvalue > 0.001


def test_public_credit_var_is_invariant_to_the_internal_factor_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The contract that actually matters: `_FACTOR_BATCH` is a tuning knob, not
    a risk parameter. `independent_bernoulli` drives a uniform column, the kind
    the old generator broke."""
    obligors = pd.DataFrame({
        "obligor_uid": [1, 2, 3, 4, 5],
        "ead": [1e6, 2e6, 3e6, 1.5e6, 5e5],
        "pd": [0.01, 0.02, 0.05, 0.03, 0.10],
        "lgd": [0.6, 0.4, 0.5, 0.45, 0.6],
    })
    request = CreditVarRequest.from_dataframe(
        obligor_table=obligors,
        model="independent_bernoulli",
        n_scenarios=5_000,
        alphas=[0.99, 0.999],
        seed=99,
    )

    results = []
    for factor_batch in (5_000, 1_024, 97):
        monkeypatch.setattr(credit_var_run, "_FACTOR_BATCH", factor_batch)
        summary = run_credit_var(request).summary
        results.append((dict(summary.var), dict(summary.es), summary.loss_mean))

    assert results[0] == results[1] == results[2]
