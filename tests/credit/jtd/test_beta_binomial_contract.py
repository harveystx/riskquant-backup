# tests/credit/jtd/test_beta_binomial_contract.py
"""
C-18: the beta-binomial contract — master PD is NOT a model parameter.

Adjudication (D-C18, Contract A). The model's marginal default probability is
`alpha / (alpha + beta)`. The obligor master's PD column is not an input to it:
`contracts.py` does not list `pd` among the model's required columns, the
builder joins the master for LGD only, and the catalog discloses the exception
on both the base and grouped entries. Heterogeneous cohorts are expressed by
`beta_binomial_grouped`, one `(alpha, beta)` per pool.

These tests therefore prove two different things and must not be confused:

  * the MATHEMATICS — realised marginal and pairwise dependence against
    independent analytic results;
  * the CONTRACT — a master PD that disagrees with the Beta mean is *ignored by
    design*, not *rejected as invalid*. A test that merely asserted portfolio EL
    could not tell those two apart.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from riskquant.credit.jtd.ingest.contracts import JTD_MODEL_CONTRACTS
from riskquant.credit.jtd.kernels.beta_binomial import kernel_beta_binomial_jtd
from riskquant.credit.jtd.models import JtdModel
from riskquant.solutions.credit_var.catalog import model_entry
from riskquant.solutions.credit_var.run import run_credit_var
from riskquant.solutions.credit_var.schemas import CreditVarRequest

ALPHA, BETA = 2.0, 38.0
BETA_MEAN = ALPHA / (ALPHA + BETA)  # 0.05
S = 200_000
SEED = 90210


def _z(hat: float, target: float, n: int = S) -> float:
    return (hat - target) / math.sqrt(target * (1.0 - target) / n)


# ==========================================================================
# 1. ANALYTIC IDENTITIES — written out, not taken from production
# ==========================================================================
def test_beta_mean_is_the_declared_marginal() -> None:
    assert pytest.approx(0.05) == BETA_MEAN
    assert pytest.approx(0.05) == ALPHA / (ALPHA + BETA)


def test_beta_binomial_default_correlation_identity() -> None:
    """For P ~ Beta(a,b) and D|P ~ Bernoulli(P):
    Var(P) = ab / ((a+b)^2 (a+b+1))
    corr(D_i, D_j) = Var(P) / (m(1-m)) = 1 / (a+b+1)
    """
    m = BETA_MEAN
    var_p = ALPHA * BETA / ((ALPHA + BETA) ** 2 * (ALPHA + BETA + 1.0))
    assert var_p / (m * (1.0 - m)) == pytest.approx(1.0 / (ALPHA + BETA + 1.0))


# ==========================================================================
# 2. THE MODEL — realised marginal and dependence
# ==========================================================================
@pytest.fixture(scope="module")
def realised():
    """Drive the kernel directly: row 0/1 independent uniforms, row 2 the shared
    Beta common PD. EAD*LGD = 1 so loss is the default indicator."""
    rng = np.random.default_rng(SEED)
    latent = np.ascontiguousarray(
        np.vstack([rng.random(S), rng.random(S), rng.beta(ALPHA, BETA, S)])
    )
    out = np.empty((2, S), dtype=np.float64, order="C")
    kernel_beta_binomial_jtd(
        out,
        latent,
        np.ascontiguousarray(np.ones((2, S))),
        np.ones(2),
        np.array([0, 1], dtype=np.int32),
        np.array([2, 2], dtype=np.int32),
    )
    return out


def test_realised_marginal_equals_the_beta_mean(realised) -> None:
    for i in range(2):
        assert abs(_z(realised[i].mean(), BETA_MEAN)) < 4.0, realised[i].mean()


def test_realised_pairwise_joint_matches_the_analytic_mixture(realised) -> None:
    """P(both default) = E[P^2] = m^2 + Var(P)."""
    var_p = ALPHA * BETA / ((ALPHA + BETA) ** 2 * (ALPHA + BETA + 1.0))
    joint_theory = BETA_MEAN**2 + var_p
    assert abs(_z(float((realised[0] * realised[1]).mean()), joint_theory)) < 4.0


def test_realised_default_correlation_matches_one_over_a_plus_b_plus_one(
    realised,
) -> None:
    empirical = float(np.corrcoef(realised[0], realised[1])[0, 1])
    assert empirical == pytest.approx(1.0 / (ALPHA + BETA + 1.0), abs=3e-3)


# ==========================================================================
# 3. THE CONTRACT — master PD is ignored, NOT rejected
# ==========================================================================
def test_model_contract_does_not_require_a_pd_column() -> None:
    """Structural proof that PD is not an input: the ingest contract lists the
    columns the model consumes, and `pd` is not among them."""
    contract = JTD_MODEL_CONTRACTS[JtdModel.BETA_BINOMIAL]
    assert contract.required_columns == frozenset({
        "uid",
        "uniform_factor",
        "beta_factor",
    })
    assert "pd" not in contract.required_columns


def test_catalog_discloses_that_master_pd_is_unused() -> None:
    """The disclosure is the public contract; it must not silently disappear."""
    for key in ("beta_binomial", "beta_binomial_grouped"):
        description = model_entry(key).description.lower()
        assert "pd column is not used" in description
    assert "alpha/(alpha+beta)" in model_entry("beta_binomial").description.lower()


def _mismatched_book() -> pd.DataFrame:
    """Master PDs 60x apart and both far from the Beta mean of 0.05."""
    return pd.DataFrame({
        "obligor_uid": [1, 2],
        "ead": [1_000_000.0, 1_000_000.0],
        "pd": [0.001, 0.300],
        "lgd": [0.5, 0.5],
    })


def test_contradictory_master_pd_is_accepted_not_rejected() -> None:
    """The distinguishing test. Under Contract A a PD that disagrees with the
    Beta mean is IGNORED; it is not a contract error. If a future tranche adopts
    reconciliation instead, this test must be replaced deliberately, not by
    accident."""
    request = CreditVarRequest.from_dataframe(
        obligor_table=_mismatched_book(),
        model="beta_binomial",
        alpha=ALPHA,
        beta=BETA,
        n_scenarios=20_000,
        alphas=[0.99],
        seed=SEED,
    )
    assert request.model == "beta_binomial"
    result = run_credit_var(request)
    assert result.summary.loss_mean > 0.0


def test_both_obligors_receive_the_same_marginal_despite_60x_pd_spread() -> None:
    """The operative consequence of Contract A, measured rather than asserted:
    identical EAD and LGD plus identical pool membership give identical expected
    loss, whatever the master PDs say."""
    result = run_credit_var(
        CreditVarRequest.from_dataframe(
            obligor_table=_mismatched_book(),
            model="beta_binomial",
            alpha=ALPHA,
            beta=BETA,
            n_scenarios=200_000,
            alphas=[0.99],
            seed=SEED,
        )
    )
    els = [c.expected_loss for c in result.obligor_contributions]
    assert els[0] == pytest.approx(els[1], rel=0.05)


# ==========================================================================
# 4. PUBLIC PATH — analytic EL follows the Beta mean, not the PD column
# ==========================================================================
def test_public_expected_loss_matches_the_beta_mean_contract() -> None:
    book = pd.DataFrame({
        "obligor_uid": [1, 2, 3, 4, 5],
        "ead": [1e6, 2e6, 3e6, 1.5e6, 5e5],
        "pd": [0.01, 0.02, 0.05, 0.03, 0.10],
        "lgd": [0.6, 0.4, 0.5, 0.45, 0.6],
    })
    exposure_at_loss = float((book["ead"] * book["lgd"]).sum())

    # The contract's EL, and the EL a PD-driven model would produce.
    el_contract = exposure_at_loss * BETA_MEAN
    el_if_pd_were_used = float((book["pd"] * book["ead"] * book["lgd"]).sum())
    assert el_contract != pytest.approx(el_if_pd_were_used)  # the two disagree

    result = run_credit_var(
        CreditVarRequest.from_dataframe(
            obligor_table=book,
            model="beta_binomial",
            alpha=ALPHA,
            beta=BETA,
            n_scenarios=200_000,
            alphas=[0.99],
            seed=SEED,
        )
    )
    assert result.summary.loss_mean == pytest.approx(el_contract, rel=0.02)
    assert result.summary.loss_mean != pytest.approx(el_if_pd_were_used, rel=0.02)


def test_provenance_records_the_parameters_that_define_the_marginal() -> None:
    result = run_credit_var(
        CreditVarRequest.from_dataframe(
            obligor_table=_mismatched_book(),
            model="beta_binomial",
            alpha=ALPHA,
            beta=BETA,
            n_scenarios=1_000,
            alphas=[0.99],
            seed=SEED,
        )
    )
    assert result.provenance.model == "beta_binomial"
    assert result.provenance.params["alpha"] == ALPHA
    assert result.provenance.params["beta"] == BETA


# ==========================================================================
# 5. GROUPED — heterogeneous cohorts are expressed by pools, not by PD
# ==========================================================================
GROUPS = {"IG": (1.0, 99.0), "HY": (4.0, 16.0)}  # means 0.01 and 0.20


def test_grouped_pools_carry_different_marginals_and_preserve_membership() -> None:
    book = pd.DataFrame({
        "obligor_uid": [1, 2, 3, 4],
        "ead": [1_000_000.0] * 4,
        "pd": [0.07, 0.07, 0.07, 0.07],  # identical and wrong for both pools
        "lgd": [1.0] * 4,
        "group": ["IG", "IG", "HY", "HY"],
    })
    result = run_credit_var(
        CreditVarRequest.from_dataframe(
            obligor_table=book,
            model="beta_binomial_grouped",
            group_params={g: {"alpha": a, "beta": b} for g, (a, b) in GROUPS.items()},
            n_scenarios=200_000,
            alphas=[0.99],
            seed=SEED,
        )
    )
    el = {c.obligor_uid: c.expected_loss for c in result.obligor_contributions}

    ig_mean = GROUPS["IG"][0] / sum(GROUPS["IG"])
    hy_mean = GROUPS["HY"][0] / sum(GROUPS["HY"])

    # Membership preserved: uids 1,2 -> IG pool; uids 3,4 -> HY pool.
    for uid in (1, 2):
        assert el[uid] == pytest.approx(1_000_000.0 * ig_mean, rel=0.08), uid
    for uid in (3, 4):
        assert el[uid] == pytest.approx(1_000_000.0 * hy_mean, rel=0.05), uid

    # The pools are genuinely different, and neither follows the PD column.
    assert el[3] > 10.0 * el[1]
    assert el[1] != pytest.approx(1_000_000.0 * 0.07, rel=0.2)


def test_grouped_portfolio_expected_loss_is_the_pool_weighted_sum() -> None:
    book = pd.DataFrame({
        "obligor_uid": [1, 2, 3],
        "ead": [1e6, 2e6, 4e6],
        "pd": [0.5, 0.5, 0.5],  # deliberately absurd, and ignored
        "lgd": [0.5, 0.5, 0.25],
        "group": ["IG", "HY", "HY"],
    })
    ig_mean = GROUPS["IG"][0] / sum(GROUPS["IG"])
    hy_mean = GROUPS["HY"][0] / sum(GROUPS["HY"])
    el_contract = 1e6 * 0.5 * ig_mean + 2e6 * 0.5 * hy_mean + 4e6 * 0.25 * hy_mean

    result = run_credit_var(
        CreditVarRequest.from_dataframe(
            obligor_table=book,
            model="beta_binomial_grouped",
            group_params={g: {"alpha": a, "beta": b} for g, (a, b) in GROUPS.items()},
            n_scenarios=200_000,
            alphas=[0.99],
            seed=SEED,
        )
    )
    assert result.summary.loss_mean == pytest.approx(el_contract, rel=0.03)


def test_shared_latent_factor_does_not_mean_perfectly_dependent_defaults(
    realised,
) -> None:
    """C-07A terminology. Both obligors in the `realised` fixture read the SAME
    beta column, so their latent PD is identical scenario by scenario. Their
    DEFAULT INDICATORS are nevertheless conditionally independent Bernoulli
    draws, and unconditionally correlate at 1/(alpha+beta+1) = 0.0244 — nowhere
    near 1. Describing shared-factor cohorts as 'perfectly dependent' would be
    wrong by a factor of forty here."""
    empirical = float(np.corrcoef(realised[0], realised[1])[0, 1])
    identity = 1.0 / (ALPHA + BETA + 1.0)
    assert empirical == pytest.approx(identity, abs=3e-3)
    assert empirical < 0.05  # emphatically not perfect dependence
    assert not np.array_equal(realised[0], realised[1])  # indicators differ

    # ... while the latent factor they share is identical by construction.
    disagreements = int(np.sum(realised[0] != realised[1]))
    assert disagreements > 0
