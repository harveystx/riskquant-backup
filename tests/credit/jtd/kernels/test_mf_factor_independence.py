# tests/credit/jtd/kernels/test_mf_factor_independence.py
"""
C-07: what independence structure do the multi-factor copulas actually guarantee?

Four claims are separated here because conflating them is the whole finding:

  1. raw stochastic coordinates          Z_k, eps_i, W are iid / mutually independent
  2. latent obligor variables            Corr(A_i, A_j) = sum_k beta_ik beta_jk
  3. obligor independence                zero latent correlation => independent (GAUSSIAN ONLY)
  4. default indicators                  NOT implied by (3) under Student-t

Independent systemic factors do NOT mean independent obligors: shared loadings
couple them. And under Student-t, obligors with completely DISJOINT loadings are
uncorrelated but still dependent through the shared chi-square mixing variable.

Oracles are scipy / closed forms written here. No production function is
compared against another production function.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtri
from scipy.stats import chi2, norm
from scipy.stats import t as student_t

from riskquant.credit.jtd.ingest.contracts import JTD_MODEL_CONTRACTS
from riskquant.credit.jtd.kernels.mf_copula_gaussian import (
    kernel_mf_gaussian_copula_jtd,
)
from riskquant.credit.jtd.kernels.mf_copula_students_t import (
    kernel_mf_students_t_copula_jtd,
)
from riskquant.credit.jtd.models import JtdModel

S = 400_000
NU = 5.0
SEED = 20260921
I32, I64 = np.int32, np.int64


def _z(hat: float, target: float, n: int = S) -> float:
    return (hat - target) / math.sqrt(max(target * (1.0 - target), 1e-18) / n)


@pytest.fixture(scope="module")
def latent():
    """rows 0,1,2 = systemic Z ; 3,4 = idiosyncratic eps ; 5,6 = 1/sqrt(W)."""
    rng = np.random.default_rng(SEED)
    w1 = 2.0 * rng.standard_gamma(NU / 2.0, S)
    w2 = 2.0 * rng.standard_gamma(NU / 2.0, S)
    return np.ascontiguousarray(
        np.vstack([
            rng.standard_normal(S),  # 0 systemic F1
            rng.standard_normal(S),  # 1 systemic F2
            rng.standard_normal(S),  # 2 systemic F3
            rng.standard_normal(S),  # 3 eps_A
            rng.standard_normal(S),  # 4 eps_B
            1.0 / np.sqrt(w1),  # 5 shared W
            1.0 / np.sqrt(w2),  # 6 a SECOND, independent W
        ])
    )


def _run_gauss(latent, betas, sys_rows, idio_rows, pds):
    """betas: list of per-obligor loading lists, aligned to sys_rows."""
    indptr = np.cumsum([0] + [len(b) for b in betas]).astype(I64)
    values = np.ascontiguousarray(np.array([v for b in betas for v in b], np.float64))
    sys_uix = np.ascontiguousarray(np.array([r for rs in sys_rows for r in rs], I32))
    sum_sq = np.array([float(np.sum(np.square(b))) for b in betas])
    out = np.empty((len(betas), S), dtype=np.float64, order="C")
    kernel_mf_gaussian_copula_jtd(
        out,
        latent,
        np.ascontiguousarray(np.ones((len(betas), S))),
        np.ones(len(betas)),
        indptr,
        values,
        sys_uix,
        np.ascontiguousarray(np.array(idio_rows, I32)),
        np.ascontiguousarray(np.sqrt(np.maximum(0.0, 1.0 - sum_sq))),
        np.ascontiguousarray(ndtri(np.asarray(pds, dtype=np.float64))),
    )
    return out


def _run_t(latent, betas, sys_rows, idio_rows, chi_rows, pds):
    indptr = np.cumsum([0] + [len(b) for b in betas]).astype(I64)
    values = np.ascontiguousarray(np.array([v for b in betas for v in b], np.float64))
    sys_uix = np.ascontiguousarray(np.array([r for rs in sys_rows for r in rs], I32))
    sum_sq = np.array([float(np.sum(np.square(b))) for b in betas])
    out = np.empty((len(betas), S), dtype=np.float64, order="C")
    kernel_mf_students_t_copula_jtd(
        out,
        latent,
        np.ascontiguousarray(np.ones((len(betas), S))),
        np.ones(len(betas)),
        indptr,
        values,
        sys_uix,
        np.ascontiguousarray(np.array(idio_rows, I32)),
        np.ascontiguousarray(np.sqrt(np.maximum(0.0, 1.0 - sum_sq))),
        np.ascontiguousarray(student_t.ppf(np.asarray(pds, np.float64), df=NU)),
        np.ascontiguousarray(np.full(len(betas), math.sqrt(NU))),
        np.ascontiguousarray(np.array(chi_rows, I32)),
    )
    return out


# ==========================================================================
# 1. RAW COORDINATES — statistical verification, explicitly NOT proof
# ==========================================================================
def test_raw_systemic_coordinates_are_standard_and_uncorrelated(latent) -> None:
    """Statistical check on the fixture's own draws. The STRUCTURAL argument —
    distinct factors occupy distinct columns of one i.i.d. fill — is tested in
    `test_distinct_factor_names_cannot_alias_to_one_column` below."""
    for row in (0, 1, 2):
        assert abs(float(latent[row].mean())) < 5.0 / math.sqrt(S)
        assert float(latent[row].var()) == pytest.approx(1.0, abs=0.02)
    for a, b in ((0, 1), (0, 2), (1, 2)):
        assert abs(float(np.corrcoef(latent[a], latent[b])[0, 1])) < 5.0 / math.sqrt(S)


# ==========================================================================
# 2. LATENT CORRELATION IDENTITY  rho_ij = sum_k beta_ik beta_jk
# ==========================================================================
@pytest.mark.parametrize(
    "beta_a,beta_b,rows_a,rows_b,expected",
    [
        ([0.6], [0.6], [0], [0], 0.36),  # same single factor
        ([0.6], [0.6], [0], [1], 0.00),  # DISJOINT factors -> zero
        ([0.5, 0.5], [0.5, 0.5], [0, 1], [0, 1], 0.50),
        ([0.5, 0.5], [0.8, 0.0], [0, 1], [0, 2], 0.40),
        ([0.6], [-0.6], [0], [0], -0.36),  # sign is carried
        ([0.0], [0.7], [0], [1], 0.00),  # zero loading has no effect
    ],
)
def test_latent_correlation_matches_the_loading_inner_product(
    latent, beta_a, beta_b, rows_a, rows_b, expected
) -> None:
    """The identity the whole model rests on, checked against a latent rebuilt
    independently from the same fixture rows — and the kernel's indicators
    checked bit-for-bit against a threshold on that rebuilt latent."""
    pds = [0.5, 0.5]
    out = _run_gauss(latent, [beta_a, beta_b], [rows_a, rows_b], [3, 4], pds)

    w_a = math.sqrt(max(0.0, 1.0 - sum(v * v for v in beta_a)))
    w_b = math.sqrt(max(0.0, 1.0 - sum(v * v for v in beta_b)))
    a = (
        sum(v * latent[r] for v, r in zip(beta_a, rows_a, strict=True))
        + w_a * latent[3]
    )
    b = (
        sum(v * latent[r] for v, r in zip(beta_b, rows_b, strict=True))
        + w_b * latent[4]
    )

    assert float(np.corrcoef(a, b)[0, 1]) == pytest.approx(expected, abs=0.006)
    assert float(a.var()) == pytest.approx(1.0, abs=0.02)  # unit variance
    assert float(b.var()) == pytest.approx(1.0, abs=0.02)

    np.testing.assert_array_equal(out[0], (a < ndtri(pds[0])).astype(np.float64))
    np.testing.assert_array_equal(out[1], (b < ndtri(pds[1])).astype(np.float64))


def test_joint_default_matches_the_bivariate_normal_oracle(latent) -> None:
    from scipy.stats import multivariate_normal

    pds = [0.02, 0.15]
    out = _run_gauss(latent, [[0.5, 0.5], [0.8, 0.0]], [[0, 1], [0, 2]], [3, 4], pds)
    rho = 0.5 * 0.8
    thr = ndtri(np.array(pds))
    oracle = float(multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).cdf(thr))
    assert abs(_z(float((out[0] * out[1]).mean()), oracle)) < 4.0


# ==========================================================================
# 3. FACTOR INDEPENDENCE  !=  OBLIGOR INDEPENDENCE
# ==========================================================================
def test_independent_factors_still_give_correlated_obligors_when_loadings_overlap(
    latent,
) -> None:
    """The point of the model. F1 and F2 are independent coordinates, yet two
    obligors both loading 0.6 on F1 have latent correlation 0.36 and a joint
    default probability well above the product."""
    pds = [0.05, 0.05]
    shared = _run_gauss(latent, [[0.6], [0.6]], [[0], [0]], [3, 4], pds)
    disjoint = _run_gauss(latent, [[0.6], [0.6]], [[0], [1]], [3, 4], pds)

    joint_shared = float((shared[0] * shared[1]).mean())
    joint_disjoint = float((disjoint[0] * disjoint[1]).mean())
    product = pds[0] * pds[1]

    assert joint_shared > 1.5 * product  # coupling through the loading
    assert abs(_z(joint_disjoint, product)) < 4.0  # Gaussian: uncorrelated => indep.


# ==========================================================================
# 4. STUDENT-T — conditional vs UNCONDITIONAL independence
# ==========================================================================
def _t_joint_disjoint(t_i, t_j, nu):
    """Disjoint loadings, SHARED W. Conditional on W the two latents are
    independent N(0,1), and obligor i defaults iff A_i < t_i*sqrt(W/nu):

        P(both) = E_W[ Phi(t_i sqrt(W/nu)) * Phi(t_j sqrt(W/nu)) ]

    Both factors are increasing in W, so by the covariance inequality this
    STRICTLY exceeds P(i)P(j) whenever both t are finite. Zero correlation does
    not give independence here."""

    def integrand(w):
        s = math.sqrt(w / nu)
        return norm.cdf(t_i * s) * norm.cdf(t_j * s) * chi2.pdf(w, nu)

    total, _ = quad(integrand, 1e-9, 200.0, limit=300)
    return total


def test_students_t_disjoint_loadings_are_uncorrelated_but_dependent(latent) -> None:
    """THE C-07 headline. Disjoint systemic loadings, one shared chi column."""
    pds = [0.02, 0.02]
    out = _run_t(latent, [[0.6], [0.6]], [[0], [1]], [3, 4], [5, 5], pds)

    joint = float((out[0] * out[1]).mean())
    product = pds[0] * pds[1]
    oracle = _t_joint_disjoint(
        student_t.ppf(pds[0], df=NU), student_t.ppf(pds[1], df=NU), NU
    )

    assert oracle > product  # analytic: dependence exists
    assert abs(_z(joint, oracle)) < 4.0  # the kernel reproduces it
    assert joint > 1.5 * product  # and it is material


def test_students_t_distinct_chi_columns_restore_independence(latent) -> None:
    """Independence across cohorts needs BOTH disjoint loadings AND distinct
    mixing variables. Row 6 is a second, independent W."""
    pds = [0.02, 0.02]
    out = _run_t(latent, [[0.6], [0.6]], [[0], [1]], [3, 4], [5, 6], pds)
    assert abs(_z(float((out[0] * out[1]).mean()), pds[0] * pds[1])) < 4.0


def test_students_t_marginal_is_preserved_under_the_shared_mixing(latent) -> None:
    """The shared W changes dependence, not the marginals."""
    pds = [0.02, 0.15]
    out = _run_t(
        latent, [[0.5, 0.5], [0.8, 0.0]], [[0, 1], [0, 2]], [3, 4], [5, 5], pds
    )
    for i, pd_i in enumerate(pds):
        assert abs(_z(float(out[i].mean()), pd_i)) < 4.0


def test_students_t_conditional_on_w_the_latents_are_conditionally_independent(
    latent,
) -> None:
    """Conditioning restores what the mixture destroys. Slice the scenarios into
    a narrow W-band and check the residual dependence collapses."""
    pds = [0.10, 0.10]
    out = _run_t(latent, [[0.6], [0.6]], [[0], [1]], [3, 4], [5, 5], pds)
    inv_sqrt_w = latent[5]
    edges = np.quantile(inv_sqrt_w, [0.4, 0.5])  # a narrow central W band
    band = (inv_sqrt_w >= edges[0]) & (inv_sqrt_w <= edges[1])

    a, b = out[0][band], out[1][band]
    unconditional = float((out[0] * out[1]).mean()) - float(out[0].mean()) * float(
        out[1].mean()
    )
    conditional = float((a * b).mean()) - float(a.mean()) * float(b.mean())
    assert unconditional > 0.0
    assert abs(conditional) < 0.5 * unconditional


# ==========================================================================
# 5. STRUCTURAL — aliasing, and the guard that was missing
# ==========================================================================
def test_distinct_factor_names_cannot_alias_to_one_column() -> None:
    """Name -> uix resolution is a dict lookup against the factor universe, and
    the universe validator forbids duplicate factor ids, so two distinct names
    cannot resolve to one stochastic coordinate."""
    from riskquant.credit.jtd.ingest._shared import _resolve_factor_uix

    mapping = {"F1": 0, "F2": 1, "F3": 2}
    got = _resolve_factor_uix(np.array(["F1", "F2", "F3", "F1"]), mapping, ctx="t")
    assert got.tolist() == [0, 1, 2, 0]
    assert got.dtype == np.int32
    with pytest.raises(ValueError, match="Unknown credit factor"):
        _resolve_factor_uix(np.array(["F1", "NOPE"]), mapping, ctx="t")


COPULA_MODELS = [
    JtdModel.COPULA_SF_GAUSSIAN,
    JtdModel.COPULA_SF_STUDENTS_T,
    JtdModel.COPULA_MF_GAUSSIAN,
    JtdModel.COPULA_MF_STUDENTS_T,
]


@pytest.mark.parametrize("model", COPULA_MODELS, ids=lambda m: m.name)
def test_every_copula_model_requires_a_private_idiosyncratic_factor(model) -> None:
    """C-07 remediation. `assert_uniforms_distinct` exists precisely to stop
    obligors sharing a private coordinate, and was wired only to
    `uniform_factor` for the four Bernoulli-family models. The copula family
    needs the same guarantee on `idiosyncratic_factor`, because its correlation
    contract assumes eps_i is private."""
    contract = JTD_MODEL_CONTRACTS[model]
    assert "idiosyncratic_factor" in contract.factor_name_columns
    assert "idiosyncratic_factor" in contract.distinct_columns


@pytest.mark.parametrize("model", COPULA_MODELS, ids=lambda m: m.name)
def test_ingest_rejects_a_shared_idiosyncratic_factor(model) -> None:
    """The guard must actually fire on the contract's declared column, naming
    the role rather than calling an epsilon a 'uniform'."""
    import polars as pl

    from riskquant.credit.jtd.ingest.validate import _assert_distinct_columns

    frame = pl.DataFrame({
        "uid": [1, 2],
        "idiosyncratic_factor": ["EPS_SHARED", "EPS_SHARED"],
    })
    mapping = {"EPS_SHARED": 7}
    with pytest.raises(ValueError, match="idiosyncratic factor UIX"):
        _assert_distinct_columns(
            frame, JTD_MODEL_CONTRACTS[model], mapping, ctx=model.name
        )

    private = pl.DataFrame({"uid": [1, 2], "idiosyncratic_factor": ["EPS_1", "EPS_2"]})
    _assert_distinct_columns(
        private, JTD_MODEL_CONTRACTS[model], {"EPS_1": 7, "EPS_2": 8}, ctx=model.name
    )


def test_the_uniform_guard_still_names_uniforms() -> None:
    """The shared validator is reused, not repurposed: the Bernoulli-family
    message must not start calling uniforms 'idiosyncratic'."""
    import polars as pl

    from riskquant.credit.jtd.ingest.validate import _assert_distinct_columns

    frame = pl.DataFrame({"uid": [1, 2], "uniform_factor": ["U_SHARED", "U_SHARED"]})
    with pytest.raises(ValueError, match="uniform factor UIX"):
        _assert_distinct_columns(
            frame,
            JTD_MODEL_CONTRACTS[JtdModel.INDEPENDENT_BERNOULLI],
            {"U_SHARED": 3},
            ctx="INDEPENDENT_BERNOULLI",
        )


def test_a_shared_idiosyncratic_column_breaks_the_correlation_identity(latent) -> None:
    """The consequence, measured. Two obligors with DISJOINT systemic loadings
    should have latent correlation 0 by the identity; sharing one epsilon column
    gives w_i * w_j = 0.64 instead, and the joint default probability jumps."""
    pds = [0.05, 0.05]
    beta = [[0.6], [0.6]]
    correct = _run_gauss(latent, beta, [[0], [1]], [3, 4], pds)  # private eps
    shared = _run_gauss(latent, beta, [[0], [1]], [3, 3], pds)  # SHARED eps

    w = math.sqrt(1.0 - 0.36)
    implied_by_identity = 0.0
    implied_when_shared = w * w  # 0.64

    a = 0.6 * latent[0] + w * latent[3]
    b_priv = 0.6 * latent[1] + w * latent[4]
    b_shar = 0.6 * latent[1] + w * latent[3]
    assert float(np.corrcoef(a, b_priv)[0, 1]) == pytest.approx(
        implied_by_identity, abs=0.006
    )
    assert float(np.corrcoef(a, b_shar)[0, 1]) == pytest.approx(
        implied_when_shared, abs=0.01
    )

    joint_correct = float((correct[0] * correct[1]).mean())
    joint_shared = float((shared[0] * shared[1]).mean())
    assert joint_shared > 3.0 * joint_correct


# ==========================================================================
# 6. EDGE CASES
# ==========================================================================
def test_full_systematic_loading_gives_zero_idiosyncratic_weight(latent) -> None:
    """sum beta^2 == 1 is permitted (the validator's bound is inclusive), and
    the residual weight is exactly zero — no clipping, no epsilon fudge."""
    out = _run_gauss(latent, [[1.0]], [[0]], [3], [0.10])
    expected = (latent[0] < ndtri(0.10)).astype(np.float64)
    np.testing.assert_array_equal(out[0], expected)


def test_near_boundary_loading_is_not_clipped(latent) -> None:
    beta = math.sqrt(1.0 - 1e-12)
    out = _run_gauss(latent, [[beta]], [[0]], [3], [0.10])
    w = math.sqrt(max(0.0, 1.0 - beta * beta))
    a = beta * latent[0] + w * latent[3]
    np.testing.assert_array_equal(out[0], (a < ndtri(0.10)).astype(np.float64))
    assert 0.0 < w < 1e-5


def test_zero_loading_factor_has_no_effect(latent) -> None:
    with_zero = _run_gauss(latent, [[0.6, 0.0]], [[0, 1]], [3], [0.10])
    without = _run_gauss(latent, [[0.6]], [[0]], [3], [0.10])
    np.testing.assert_array_equal(with_zero[0], without[0])


def test_reordering_factors_leaves_the_economic_model_unchanged(latent) -> None:
    """Coordinate relabelling only: reorder the CSR entries and their uix
    together and the indicators must be bit-identical."""
    a = _run_gauss(latent, [[0.5, 0.3]], [[0, 1]], [3], [0.10])
    b = _run_gauss(latent, [[0.3, 0.5]], [[1, 0]], [3], [0.10])
    np.testing.assert_array_equal(a, b)


def test_single_factor_mf_reduces_to_the_single_factor_construction(latent) -> None:
    from riskquant.credit.jtd.kernels.sf_copula_gaussian import (
        kernel_sf_gaussian_copula_jtd,
    )

    pds = [0.08]
    mf = _run_gauss(latent, [[0.6]], [[0]], [3], pds)
    sf = np.empty((1, S), dtype=np.float64, order="C")
    kernel_sf_gaussian_copula_jtd(
        sf,
        latent,
        np.ascontiguousarray(np.ones((1, S))),
        np.ones(1),
        np.array([0.6]),
        ndtri(np.array(pds)),
        np.array([0], I32),
        np.array([3], I32),
    )
    np.testing.assert_array_equal(mf, sf)
