# tests/dependence/test_tail_coefficients.py

"""
Phase 1b-A acceptance tests.

Every mathematical claim is checked against an independent pure-Python oracle
transcribed from ``PHASE_1B_A_MATHEMATICAL_SPECIFICATION.md`` section 12. The
oracle uses ``math.log`` and plain Python arithmetic, calls no production code,
and is written so that changing the production implementation cannot change it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from riskquant.dependence.coefficients.tail_coefficients import (
    _derived_digest,
    tail_coefficients,
)
from riskquant.dependence.contracts import (
    TAIL_COEFFICIENT_METHODOLOGY_VERSION,
    DependenceProvenance,
    TailCoefficientResult,
    TailCountResult,
    TailSide,
)
from riskquant.dependence.estimators.tail_counts import tail_counts
from riskquant.observations.compile import compile_ranks, compile_sample
from riskquant.observations.orientation import ObservationOrientation

VARIABLES_ON_ROWS = ObservationOrientation.VARIABLES_ON_ROWS

#: Relative tolerance for the logarithmic statistics. ``lambda_hat`` is compared
#: exactly: it is a ratio of two integers and both sides compute it identically.
LOG_TOLERANCE = 1e-15


# ============================================================ independent oracle


def oracle_cell(
    n_observations: int, conditioning_count: int, joint_count: int
) -> tuple[float, bool, float, bool, float, bool]:
    """
    Specification section 12, transcribed. No production code is used.

    Returns
    -------
    (lambda_hat, estimable_lambda, chi_hat, estimable_chi,
     chi_bar_hat, estimable_chi_bar)
    """
    nan = float("nan")
    m_i = conditioning_count
    n_ij = joint_count
    n_obs = n_observations

    if m_i >= 1:
        lam, lam_ok = n_ij / m_i, True
    else:
        lam, lam_ok = nan, False

    marginal_interior = (m_i >= 1) and (m_i <= n_obs - 1)

    if marginal_interior and n_ij >= 1:
        p_i = m_i / n_obs
        p_ij = n_ij / n_obs
        chi, chi_ok = 2.0 - math.log(p_ij) / math.log(p_i), True
    else:
        chi, chi_ok = nan, False

    if marginal_interior:
        if n_ij >= 1:
            p_i = m_i / n_obs
            p_ij = n_ij / n_obs
            bar, bar_ok = 2.0 * math.log(p_i) / math.log(p_ij) - 1.0, True
        else:
            # Assigned, never computed through log(0).
            bar, bar_ok = -1.0, True
    else:
        bar, bar_ok = nan, False

    return lam, lam_ok, chi, chi_ok, bar, bar_ok


def assert_matches_oracle(counts: TailCountResult) -> None:
    """Cell-by-cell comparison of every array against the oracle."""
    result = tail_coefficients(counts)
    n_obs = counts.provenance.observation_count
    n_thresholds, n_variables, _ = counts.joint_count.shape

    for q in range(n_thresholds):
        for i in range(n_variables):
            for j in range(n_variables):
                lam, lam_ok, chi, chi_ok, bar, bar_ok = oracle_cell(
                    n_obs,
                    int(counts.marginal_count[q, i]),
                    int(counts.joint_count[q, i, j]),
                )
                cell = (q, i, j)

                assert bool(result.estimable_lambda[cell]) is lam_ok, cell
                assert bool(result.estimable_chi[cell]) is chi_ok, cell
                assert bool(result.estimable_chi_bar[cell]) is bar_ok, cell

                if lam_ok:
                    assert result.lambda_hat[cell] == lam, cell
                else:
                    assert math.isnan(result.lambda_hat[cell]), cell

                if chi_ok:
                    assert result.chi_hat[cell] == pytest.approx(chi, rel=LOG_TOLERANCE)
                else:
                    assert math.isnan(result.chi_hat[cell]), cell

                if bar_ok:
                    assert result.chi_bar_hat[cell] == pytest.approx(
                        bar, rel=LOG_TOLERANCE, abs=LOG_TOLERANCE
                    ), cell
                else:
                    assert math.isnan(result.chi_bar_hat[cell]), cell


# ================================================================ construction


def synthetic_counts(
    *,
    joint: list[list[list[int]]],
    marginal: list[list[int]],
    n_observations: int,
    thresholds: list[float] | None = None,
    tail: TailSide = TailSide.UPPER,
    upstream_digest: str | None = "upstream",
) -> TailCountResult:
    """
    A ``TailCountResult`` built directly from counts.

    Lets the domain boundaries be hit exactly, including combinations a real
    panel reaches only by construction. The Phase 1 invariants are respected:
    the joint matrix is symmetric and its diagonal is the marginal count.
    """
    grid = np.ascontiguousarray(
        thresholds if thresholds is not None else [0.9] * len(joint), dtype=np.float64
    )
    joint_array = np.ascontiguousarray(joint, dtype=np.int32)
    marginal_array = np.ascontiguousarray(marginal, dtype=np.int32)
    for array in (grid, joint_array, marginal_array):
        array.setflags(write=False)

    return TailCountResult(
        thresholds=grid,
        tail=tail,
        joint_count=joint_array,
        marginal_count=marginal_array,
        provenance=DependenceProvenance(
            observation_count=n_observations,
            parameter_digest=upstream_digest,
        ),
    )


def counts_from_panel(
    matrix: np.ndarray,
    thresholds: np.ndarray,
    tail: TailSide = TailSide.UPPER,
) -> TailCountResult:
    """Real Phase 1 counts from an observation panel."""
    sample = compile_sample(
        np.ascontiguousarray(matrix), orientation=VARIABLES_ON_ROWS
    )
    return tail_counts(
        sample, compile_ranks(sample), thresholds=thresholds, tail=tail
    )


# ============================================================== oracle equality


def test_oracle_equality_on_continuous_data() -> None:
    rng = np.random.default_rng(101)
    panel = rng.standard_normal((5, 300))
    assert_matches_oracle(counts_from_panel(panel, np.array([0.80, 0.90, 0.95])))


def test_oracle_equality_on_tied_data() -> None:
    rng = np.random.default_rng(102)
    panel = np.round(rng.standard_normal((5, 300)), 1)
    assert_matches_oracle(counts_from_panel(panel, np.array([0.80, 0.90, 0.95])))


def test_oracle_equality_on_zero_inflated_data() -> None:
    rng = np.random.default_rng(103)
    panel = np.where(rng.random((5, 300)) < 0.5, 0.0, rng.standard_normal((5, 300)))
    assert_matches_oracle(counts_from_panel(panel, np.array([0.80, 0.90, 0.95, 0.99])))


def test_oracle_equality_on_the_lower_tail() -> None:
    rng = np.random.default_rng(104)
    panel = np.where(rng.random((4, 200)) < 0.4, 0.0, rng.standard_normal((4, 200)))
    assert_matches_oracle(
        counts_from_panel(panel, np.array([0.85, 0.95]), tail=TailSide.LOWER)
    )


def test_oracle_equality_on_synthetic_boundary_counts() -> None:
    """Every domain boundary in one object, unreachable from a single panel."""
    n_obs = 20
    counts = synthetic_counts(
        joint=[
            [
                [0, 0, 0, 0],
                [0, 1, 1, 1],
                [0, 1, 10, 5],
                [0, 1, 5, 20],
            ]
        ],
        marginal=[[0, 1, 10, 20]],
        n_observations=n_obs,
    )
    assert_matches_oracle(counts)


# ================================================================== E1 .. E8


def test_e1_continuous_no_ties() -> None:
    """Specification section 13, E1. S = 200, m_i = 10."""
    n_obs = 200
    expected = {
        10: (1.0, 1.0, 1.0),
        5: (0.5, 0.7686217868, 0.6241963506),
        1: (0.1, 0.2313782132, 0.1308240206),
    }
    for joint, (lam, chi, bar) in expected.items():
        counts = synthetic_counts(
            joint=[[[10, joint], [joint, 10]]],
            marginal=[[10, 10]],
            n_observations=n_obs,
        )
        result = tail_coefficients(counts)
        assert result.lambda_hat[0, 0, 1] == pytest.approx(lam, abs=1e-12)
        assert result.chi_hat[0, 0, 1] == pytest.approx(chi, abs=1e-9)
        assert result.chi_bar_hat[0, 0, 1] == pytest.approx(bar, abs=1e-9)

    counts = synthetic_counts(
        joint=[[[10, 0], [0, 10]]], marginal=[[10, 10]], n_observations=n_obs
    )
    result = tail_coefficients(counts)
    assert result.lambda_hat[0, 0, 1] == 0.0
    assert bool(result.estimable_lambda[0, 0, 1]) is True
    assert math.isnan(result.chi_hat[0, 0, 1])
    assert bool(result.estimable_chi[0, 0, 1]) is False
    assert result.chi_bar_hat[0, 0, 1] == -1.0
    assert bool(result.estimable_chi_bar[0, 0, 1]) is True


def test_e2_asymmetric_tied_case() -> None:
    """Specification section 13, E2, built from a real panel."""
    tied = np.concatenate([np.arange(14.0), np.full(6, 99.0)])
    panel = np.ascontiguousarray(np.vstack([tied, np.arange(20.0)]))
    counts = counts_from_panel(panel, np.array([0.80]))

    assert counts.marginal_count[0].tolist() == [6, 4]
    assert int(counts.joint_count[0, 0, 1]) == 4

    result = tail_coefficients(counts)
    assert result.lambda_hat[0, 0, 1] == pytest.approx(0.6666666667, abs=1e-9)
    assert result.chi_hat[0, 0, 1] == pytest.approx(0.6632273531, abs=1e-9)
    assert result.chi_bar_hat[0, 0, 1] == pytest.approx(0.4961407272, abs=1e-9)

    assert result.lambda_hat[0, 1, 0] == pytest.approx(1.0, abs=1e-12)
    assert result.chi_hat[0, 1, 0] == pytest.approx(1.0, abs=1e-12)
    assert result.chi_bar_hat[0, 1, 0] == pytest.approx(1.0, abs=1e-12)


def test_e3_zero_marginal_makes_everything_undefined() -> None:
    counts = synthetic_counts(
        joint=[[[0, 0], [0, 5]]], marginal=[[0, 5]], n_observations=20
    )
    result = tail_coefficients(counts)
    for cell in ((0, 0, 0), (0, 0, 1)):
        assert math.isnan(result.lambda_hat[cell])
        assert math.isnan(result.chi_hat[cell])
        assert math.isnan(result.chi_bar_hat[cell])
        assert not result.estimable_lambda[cell]
        assert not result.estimable_chi[cell]
        assert not result.estimable_chi_bar[cell]


def test_e3_zero_marginal_is_reachable_from_a_real_panel() -> None:
    """A 60 per cent tie block at the maximum sits below a q = 0.95 cut."""
    n_obs = 100
    tied = np.concatenate([np.linspace(-1.0, 0.0, 40), np.full(60, 5.0)])
    panel = np.ascontiguousarray(np.vstack([tied, np.arange(float(n_obs))]))
    counts = counts_from_panel(panel, np.array([0.95]))

    assert int(counts.marginal_count[0, 0]) == 0
    result = tail_coefficients(counts)
    assert not result.estimable_lambda[0, 0, 1]
    assert math.isnan(result.lambda_hat[0, 0, 1])


def test_e4_zero_joint_gives_three_different_dispositions() -> None:
    """The clearest evidence that the three statistics were resolved apart."""
    counts = synthetic_counts(
        joint=[[[10, 0], [0, 10]]], marginal=[[10, 10]], n_observations=200
    )
    result = tail_coefficients(counts)
    assert result.lambda_hat[0, 0, 1] == 0.0 and result.estimable_lambda[0, 0, 1]
    assert math.isnan(result.chi_hat[0, 0, 1]) and not result.estimable_chi[0, 0, 1]
    assert result.chi_bar_hat[0, 0, 1] == -1.0 and result.estimable_chi_bar[0, 0, 1]


def test_e5_diagonal_is_exactly_one() -> None:
    counts = synthetic_counts(
        joint=[[[10, 3], [3, 10]]], marginal=[[10, 10]], n_observations=200
    )
    result = tail_coefficients(counts)
    for i in range(2):
        assert result.lambda_hat[0, i, i] == 1.0
        assert result.chi_hat[0, i, i] == pytest.approx(1.0, abs=1e-15)
        assert result.chi_bar_hat[0, i, i] == pytest.approx(1.0, abs=1e-15)


def test_e6_marginal_equal_to_the_sample_size() -> None:
    """lambda stays defined; chi and chi-bar do not."""
    n_obs = 20
    counts = synthetic_counts(
        joint=[[[20, 7], [7, 7]]], marginal=[[20, 7]], n_observations=n_obs
    )
    result = tail_coefficients(counts)

    assert result.estimable_lambda[0, 0, 1]
    assert result.lambda_hat[0, 0, 1] == pytest.approx(7 / 20)
    assert not result.estimable_chi[0, 0, 1]
    assert not result.estimable_chi_bar[0, 0, 1]
    assert math.isnan(result.chi_hat[0, 0, 1])
    assert math.isnan(result.chi_bar_hat[0, 0, 1])


def test_e6_marginal_equal_to_the_sample_size_is_reachable() -> None:
    """14 of 20 tied at the minimum gives p_marginal == 1 at low thresholds."""
    n_obs = 20
    tied = np.concatenate([np.full(14, 1.0), np.arange(2.0, 8.0)])
    panel = np.ascontiguousarray(np.vstack([tied, np.arange(float(n_obs))]))
    counts = counts_from_panel(panel, np.array([0.05, 0.10, 0.30]))

    assert counts.marginal_count[:, 0].tolist() == [n_obs, n_obs, n_obs]
    result = tail_coefficients(counts)
    assert not result.estimable_chi[:, 0, :].any()
    assert not result.estimable_chi_bar[:, 0, :].any()
    assert result.estimable_lambda[:, 0, :].all()


def test_e7_smallest_defined_counts() -> None:
    counts = synthetic_counts(
        joint=[[[1, 1], [1, 1]]], marginal=[[1, 1]], n_observations=200
    )
    result = tail_coefficients(counts)
    assert result.lambda_hat[0, 0, 1] == 1.0
    assert result.chi_hat[0, 0, 1] == pytest.approx(1.0, abs=1e-15)
    assert result.chi_bar_hat[0, 0, 1] == pytest.approx(1.0, abs=1e-15)
    # The effective conditioning count travels with the result, so a consumer
    # can see the estimate rests on a single observation.
    assert int(result.marginal_count[0, 0]) == 1


def test_e8_independence_gives_chi_exactly_zero() -> None:
    """p_ij == p_i * p_j implies chi == 0."""
    counts = synthetic_counts(
        joint=[[[100, 1], [1, 100]]], marginal=[[100, 100]], n_observations=10000
    )
    result = tail_coefficients(counts)
    assert result.chi_hat[0, 0, 1] == pytest.approx(0.0, abs=1e-12)


def test_e8_negative_tail_association_gives_negative_chi() -> None:
    counts = synthetic_counts(
        joint=[[[200, 1], [1, 200]]], marginal=[[200, 200]], n_observations=10000
    )
    result = tail_coefficients(counts)
    assert result.chi_hat[0, 0, 1] == pytest.approx(-0.3543676403, abs=1e-9)
    assert result.chi_hat[0, 0, 1] < 0.0


# ================================================================ domain grid


@pytest.mark.parametrize("conditioning_count", [0, 1, 9, 10])
@pytest.mark.parametrize("joint_count", [0, 1])
def test_every_domain_boundary(conditioning_count: int, joint_count: int) -> None:
    """m_i in {0, 1, S-1, S} against n_ij in {0, 1}, plus n_ij == m_i."""
    n_obs = 10
    joint = min(joint_count, conditioning_count)
    counts = synthetic_counts(
        joint=[[[conditioning_count, joint], [joint, joint]]],
        marginal=[[conditioning_count, joint]],
        n_observations=n_obs,
    )
    assert_matches_oracle(counts)


def test_joint_equal_to_marginal_gives_unit_values() -> None:
    counts = synthetic_counts(
        joint=[[[5, 5], [5, 5]]], marginal=[[5, 5]], n_observations=20
    )
    result = tail_coefficients(counts)
    assert result.lambda_hat[0, 0, 1] == 1.0
    assert result.chi_hat[0, 0, 1] == pytest.approx(1.0, abs=1e-15)
    assert result.chi_bar_hat[0, 0, 1] == pytest.approx(1.0, abs=1e-15)


# ================================================================== invariants


def _realistic_results() -> list[tuple[str, TailCoefficientResult]]:
    rng = np.random.default_rng(202)
    panels = {
        "continuous": rng.standard_normal((6, 400)),
        "tied": np.round(rng.standard_normal((6, 400)), 1),
        "zero_inflated": np.where(
            rng.random((6, 400)) < 0.5, 0.0, rng.standard_normal((6, 400))
        ),
    }
    grid = np.array([0.80, 0.90, 0.95, 0.99])
    return [
        (name, tail_coefficients(counts_from_panel(panel, grid)))
        for name, panel in panels.items()
    ]


@pytest.mark.parametrize("name,result", _realistic_results())
def test_invariant_lambda_is_bounded_in_the_unit_interval(name: str, result: TailCoefficientResult) -> None:
    values = result.lambda_hat[result.estimable_lambda]
    assert values.min() >= 0.0
    assert values.max() <= 1.0


@pytest.mark.parametrize("name,result", _realistic_results())
def test_invariant_chi_is_bounded_above_by_one(name: str, result: TailCoefficientResult) -> None:
    values = result.chi_hat[result.estimable_chi]
    assert values.max() <= 1.0


@pytest.mark.parametrize("name,result", _realistic_results())
def test_invariant_chi_bar_is_bounded_in_minus_one_to_one(name: str, result: TailCoefficientResult) -> None:
    values = result.chi_bar_hat[result.estimable_chi_bar]
    assert values.min() >= -1.0
    assert values.max() <= 1.0


@pytest.mark.parametrize("name,result", _realistic_results())
def test_invariant_nan_appears_exactly_where_the_mask_is_false(name: str, result: TailCoefficientResult) -> None:
    for values, mask in (
        (result.lambda_hat, result.estimable_lambda),
        (result.chi_hat, result.estimable_chi),
        (result.chi_bar_hat, result.estimable_chi_bar),
    ):
        assert np.isnan(values[~mask]).all()
        assert np.isfinite(values[mask]).all()


@pytest.mark.parametrize("name,result", _realistic_results())
def test_invariant_domain_inclusion(name: str, result: TailCoefficientResult) -> None:
    """chi implies chi_bar implies lambda."""
    assert not (result.estimable_chi & ~result.estimable_chi_bar).any()
    assert not (result.estimable_chi_bar & ~result.estimable_lambda).any()


@pytest.mark.parametrize("name,result", _realistic_results())
def test_invariant_diagonal_is_one_wherever_defined(name: str, result: TailCoefficientResult) -> None:
    size = result.variable_count
    for q in range(result.threshold_count):
        for i in range(size):
            if result.estimable_lambda[q, i, i]:
                assert result.lambda_hat[q, i, i] == 1.0
            if result.estimable_chi[q, i, i]:
                assert result.chi_hat[q, i, i] == pytest.approx(1.0, abs=1e-15)
            if result.estimable_chi_bar[q, i, i]:
                assert result.chi_bar_hat[q, i, i] == pytest.approx(1.0, abs=1e-15)


def test_repeated_evaluation_is_bit_identical() -> None:
    rng = np.random.default_rng(203)
    counts = counts_from_panel(rng.standard_normal((4, 250)), np.array([0.9, 0.95]))
    first = tail_coefficients(counts)
    second = tail_coefficients(counts)
    for name in ("lambda_hat", "chi_hat", "chi_bar_hat"):
        left, right = getattr(first, name), getattr(second, name)
        assert np.array_equal(left, right, equal_nan=True)


# =============================================================== (a)symmetry


def test_directional_results_differ_under_unequal_marginals() -> None:
    """E2 structure. Asymmetry is expected, not a defect."""
    tied = np.concatenate([np.arange(14.0), np.full(6, 99.0)])
    panel = np.ascontiguousarray(np.vstack([tied, np.arange(20.0)]))
    result = tail_coefficients(counts_from_panel(panel, np.array([0.80])))

    assert result.lambda_hat[0, 0, 1] != result.lambda_hat[0, 1, 0]
    assert result.chi_hat[0, 0, 1] != result.chi_hat[0, 1, 0]
    assert result.chi_bar_hat[0, 0, 1] != result.chi_bar_hat[0, 1, 0]


def test_symmetry_arises_as_a_consequence_of_equal_marginals() -> None:
    """Continuous margins give m_i == m_j, hence symmetric arrays."""
    rng = np.random.default_rng(204)
    counts = counts_from_panel(rng.standard_normal((5, 400)), np.array([0.90, 0.95]))
    assert len(set(counts.marginal_count[0].tolist())) == 1

    result = tail_coefficients(counts)
    for q in range(result.threshold_count):
        for name in ("lambda_hat", "chi_hat", "chi_bar_hat"):
            values = getattr(result, name)[q]
            assert np.array_equal(values, values.T, equal_nan=True)


# ================================================================== provenance


def test_identical_inputs_give_identical_derived_digests() -> None:
    counts = synthetic_counts(
        joint=[[[5, 2], [2, 5]]], marginal=[[5, 5]], n_observations=50
    )
    assert (
        tail_coefficients(counts).provenance.derived_digest
        == tail_coefficients(counts).provenance.derived_digest
    )


def test_a_one_bit_upstream_digest_change_changes_the_derived_digest() -> None:
    left = _derived_digest("abc", TAIL_COEFFICIENT_METHODOLOGY_VERSION)
    right = _derived_digest("abd", TAIL_COEFFICIENT_METHODOLOGY_VERSION)
    assert left != right


def test_a_methodology_version_change_changes_the_derived_digest() -> None:
    left = _derived_digest("abc", TAIL_COEFFICIENT_METHODOLOGY_VERSION)
    right = _derived_digest("abc", TAIL_COEFFICIENT_METHODOLOGY_VERSION + "x")
    assert left != right


def test_a_none_upstream_digest_is_encoded_as_empty_bytes() -> None:
    import hashlib

    expected = hashlib.blake2b(
        b"|" + TAIL_COEFFICIENT_METHODOLOGY_VERSION.encode("utf-8"), digest_size=16
    ).hexdigest()
    assert _derived_digest(None, TAIL_COEFFICIENT_METHODOLOGY_VERSION) == expected


def test_provenance_chains_the_upstream_digest() -> None:
    counts = synthetic_counts(
        joint=[[[5, 2], [2, 5]]],
        marginal=[[5, 5]],
        n_observations=50,
        upstream_digest="deadbeef",
    )
    provenance = tail_coefficients(counts).provenance
    assert provenance.upstream_digest == "deadbeef"
    assert provenance.methodology_version == TAIL_COEFFICIENT_METHODOLOGY_VERSION
    assert provenance.observation_count == 50
    assert provenance.derived_digest == _derived_digest(
        "deadbeef", TAIL_COEFFICIENT_METHODOLOGY_VERSION
    )


def test_the_digest_does_not_hash_counts_or_values() -> None:
    """Two different count sets under one upstream digest share a derived digest."""
    left = synthetic_counts(
        joint=[[[5, 2], [2, 5]]], marginal=[[5, 5]], n_observations=50
    )
    right = synthetic_counts(
        joint=[[[5, 4], [4, 5]]], marginal=[[5, 5]], n_observations=50
    )
    assert (
        tail_coefficients(left).provenance.derived_digest
        == tail_coefficients(right).provenance.derived_digest
    )


# ============================================================ ownership, layout


def test_result_arrays_are_non_writeable_and_canonical() -> None:
    counts = synthetic_counts(
        joint=[[[5, 2], [2, 5]]], marginal=[[5, 5]], n_observations=50
    )
    result = tail_coefficients(counts)

    for name, dtype in (
        ("lambda_hat", np.float64),
        ("chi_hat", np.float64),
        ("chi_bar_hat", np.float64),
        ("estimable_lambda", np.bool_),
        ("estimable_chi", np.bool_),
        ("estimable_chi_bar", np.bool_),
    ):
        array = getattr(result, name)
        assert array.shape == (1, 2, 2), name
        assert array.dtype == dtype, name
        assert array.flags.c_contiguous, name
        assert not array.flags.writeable, name

    assert result.thresholds.dtype == np.float64
    assert not result.thresholds.flags.writeable
    assert result.marginal_count.dtype == np.int32
    assert not result.marginal_count.flags.writeable


def test_result_does_not_alias_the_input() -> None:
    counts = synthetic_counts(
        joint=[[[5, 2], [2, 5]]], marginal=[[5, 5]], n_observations=50
    )
    result = tail_coefficients(counts)
    assert result.marginal_count is not counts.marginal_count
    assert result.thresholds is not counts.thresholds
    assert not np.shares_memory(result.marginal_count, counts.marginal_count)
    assert not np.shares_memory(result.thresholds, counts.thresholds)


def test_the_input_is_never_mutated() -> None:
    counts = synthetic_counts(
        joint=[[[5, 2], [2, 5]]], marginal=[[5, 5]], n_observations=50
    )
    joint_before = counts.joint_count.copy()
    marginal_before = counts.marginal_count.copy()
    thresholds_before = counts.thresholds.copy()

    tail_coefficients(counts)

    assert np.array_equal(counts.joint_count, joint_before)
    assert np.array_equal(counts.marginal_count, marginal_before)
    assert np.array_equal(counts.thresholds, thresholds_before)


def test_the_result_is_frozen() -> None:
    counts = synthetic_counts(
        joint=[[[5, 2], [2, 5]]], marginal=[[5, 5]], n_observations=50
    )
    result = tail_coefficients(counts)
    # The attribute name is bound to a variable rather than written inline: a
    # direct assignment is rejected by mypy (the field is read-only) and an
    # inline setattr with a literal is rejected by ruff B010. Going through a
    # variable reaches the dataclass's own FrozenInstanceError -- the contract
    # under test -- with neither a type: ignore nor a noqa.
    #
    # object.__setattr__ would be wrong here: it bypasses __setattr__ entirely
    # and silently succeeds on a frozen slots dataclass.
    read_only_field = "lambda_hat"
    with pytest.raises(AttributeError):
        setattr(result, read_only_field, np.zeros((1, 2, 2)))
    with pytest.raises(ValueError):
        result.lambda_hat[0, 0, 0] = 0.0


def test_thresholds_and_tail_are_carried_through_in_order() -> None:
    grid = [0.80, 0.90, 0.99]
    counts = synthetic_counts(
        joint=[[[5, 2], [2, 5]]] * 3,
        marginal=[[5, 5]] * 3,
        n_observations=50,
        thresholds=grid,
        tail=TailSide.LOWER,
    )
    result = tail_coefficients(counts)
    assert result.thresholds.tolist() == grid
    assert result.tail is TailSide.LOWER
    assert result.threshold_count == 3


# ==================================================================== rejection


def test_a_non_positive_observation_count_is_rejected() -> None:
    counts = synthetic_counts(
        joint=[[[0, 0], [0, 0]]], marginal=[[0, 0]], n_observations=0
    )
    with pytest.raises(ValueError, match="observation_count must be >= 1"):
        tail_coefficients(counts)


def test_a_joint_count_exceeding_its_marginal_is_rejected() -> None:
    counts = synthetic_counts(
        joint=[[[2, 5], [5, 9]]], marginal=[[2, 9]], n_observations=20
    )
    with pytest.raises(ValueError, match="must not exceed marginal_count"):
        tail_coefficients(counts)


def test_a_marginal_count_above_the_observation_count_is_rejected() -> None:
    counts = synthetic_counts(
        joint=[[[30, 1], [1, 1]]], marginal=[[30, 1]], n_observations=20
    )
    with pytest.raises(ValueError, match=r"marginal_count must lie in"):
        tail_coefficients(counts)


# ======================================================================= memory


def test_no_observation_sized_allocation() -> None:
    """O(Q K^2). No (K, S) array is created; S never enters an array shape."""
    import tracemalloc

    n_variables, n_observations = 20, 4000
    rng = np.random.default_rng(205)
    counts = counts_from_panel(
        rng.standard_normal((n_variables, n_observations)),
        np.array([0.80, 0.90, 0.95, 0.99]),
    )

    tail_coefficients(counts)  # warm
    tracemalloc.start()
    tail_coefficients(counts)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    n_thresholds = 4
    cell = n_thresholds * n_variables * n_variables
    # Three float64 coefficient arrays, three bool masks, plus working
    # temporaries of the same order. Generous headroom, and far below one
    # (K, S) float64 array.
    assert peak < 12 * cell * 8
    assert peak < n_variables * n_observations * 8


# ================================================================ phase boundary


def test_no_phase_one_b_b_object_exists() -> None:
    import riskquant.dependence.coefficients.tail_coefficients as module

    for name in (
        "bootstrap",
        "confidence_interval",
        "asymptotic_independence",
        "gaussian_copula_test",
        "minimum_credible_count",
    ):
        assert not hasattr(module, name), name


def test_no_asymptotic_coefficient_is_named() -> None:
    """Naming discipline: nothing is named for a limit."""
    import riskquant.dependence.contracts as contracts

    for name in ("lambda_U", "lambda_upper", "tail_dependence_coefficient"):
        assert not hasattr(contracts, name), name
