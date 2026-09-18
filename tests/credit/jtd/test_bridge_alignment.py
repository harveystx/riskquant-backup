# tests/credit/jtd/test_bridge_alignment.py
"""
Regression tests for audit C-01 (target-order-preserving obligor join) and
C-03 (a modelled obligor with no netting set is an input gap, not a zero).

C-01. `aggregate_ead_to_obligor` joined with `searchsorted` against
`obligor_uids_target`. That target is normally `jtd_portfolio.obligor_uids` —
model/band order, structurally NOT sorted — and `_arrays` documents the
sorted-unique universe as an unvalidated caller invariant. The join silently
dropped or mis-attributed exposure: the two-block reproduction below lost 60%
of portfolio EAD with no exception.

C-03. The same call zeroed any target obligor with no netting set, while its
sibling `align_obligor_ead_to_portfolio` raised for the identical condition.

Truth here is hand-constructed (uid -> EAD written out per obligor), never
re-derived from the implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from riskquant.credit.exposure.obligor_ead.align import (
    align_obligor_ead_to_portfolio,
)
from riskquant.credit.exposure.obligor_ead.compiled import CompiledObligorEad
from riskquant.credit.exposure.results import NettingSetExposurePaths
from riskquant.credit.jtd.blocks.forced_default import ForcedDefaultJtdBlock
from riskquant.credit.jtd.blocks.independent_bernoulli import (
    IndependentBernoulliJtdBlock,
)
from riskquant.credit.jtd.blocks.sf_copula_gaussian import SfGaussianCopulaJtdBlock
from riskquant.credit.jtd.bridge import aggregate_ead_to_obligor
from riskquant.credit.jtd.compiled import CompiledJtdPortfolio
from riskquant.credit.masters.netting_set.compiled import CompiledNettingSetMaster

# --- hand-written truth -----------------------------------------------------
# Four obligors, six netting sets, three obligors holding more than one set.
NS_UID = np.array([10, 11, 12, 13, 14, 15], dtype=np.int64)
NS_OBLIGOR = np.array([700, 700, 100, 101, 101, 700], dtype=np.int64)
NS_EAD = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
EXPECTED_EAD = {700: 1.0 + 2.0 + 32.0, 100: 4.0, 101: 8.0 + 16.0}


def _master() -> CompiledNettingSetMaster:
    return CompiledNettingSetMaster(
        uid=NS_UID, obligor_uid=NS_OBLIGOR, net_collateral=np.zeros(NS_UID.size)
    )


def _paths(n_scenarios: int = 3) -> NettingSetExposurePaths:
    # Scenario axis scaled 1x / 10x / 100x so an axis mix-up cannot pass.
    # C-contiguous: `_kernel_aggregate_ead` is typed `float64[:, ::1]` and
    # correctly refuses a strided column slice, so materialise the slice.
    exposure = np.ascontiguousarray(
        (NS_EAD[:, None] * np.array([1.0, 10.0, 100.0]))[:, :n_scenarios]
    )
    return NettingSetExposurePaths(uid=NS_UID, exposure=exposure)


def _band_order_portfolio() -> np.ndarray:
    """Two model blocks with interleaved uid ranges -> unsorted band order."""
    fd = ForcedDefaultJtdBlock(
        obligor_uids=np.array([700], dtype=np.int64), lgd=np.array([0.6])
    )
    ib = IndependentBernoulliJtdBlock(
        obligor_uids=np.array([101, 100], dtype=np.int64),
        lgd=np.array([0.5, 0.4]),
        pd=np.array([0.02, 0.03]),
        uniform_uix=np.array([0, 1], dtype=np.int32),
    )
    return CompiledJtdPortfolio(
        forced_default_jtd_block=fd, independent_bernoulli_jtd_block=ib
    ).obligor_uids


def test_band_order_is_not_sorted() -> None:
    """The precondition the defect depended on: band order really is unsorted,
    so a sorted-universe join is invalid on the documented composition."""
    band = _band_order_portfolio()
    assert band.tolist() == [700, 101, 100]
    assert not bool(np.all(np.diff(band) > 0))


def test_band_order_target_preserves_exact_per_obligor_ead() -> None:
    """C-01: the documented composition. Every row must carry its own obligor's
    EAD, in the caller's order."""
    band = _band_order_portfolio()
    out = aggregate_ead_to_obligor(
        ead_result=_paths(), netting_set_master=_master(), obligor_uids_target=band
    )

    assert out.shape == (band.size, 3)
    assert out.dtype == np.float64
    assert out.flags.c_contiguous

    for row, uid in enumerate(band):
        expected = EXPECTED_EAD[int(uid)]
        np.testing.assert_allclose(
            out[row], [expected, expected * 10.0, expected * 100.0]
        )


@pytest.mark.parametrize("seed", range(12))
def test_every_target_permutation_conserves_and_attributes_ead(seed: int) -> None:
    """Property: for ANY target order, each row gets its own obligor's EAD and
    the portfolio total is conserved."""
    rng = np.random.default_rng(seed)
    target = np.array(
        rng.permutation(sorted(EXPECTED_EAD)), dtype=np.int64
    )  # random order
    out = aggregate_ead_to_obligor(
        ead_result=_paths(1), netting_set_master=_master(), obligor_uids_target=target
    )

    for row, uid in enumerate(target):
        assert out[row, 0] == pytest.approx(EXPECTED_EAD[int(uid)])
    assert out[:, 0].sum() == pytest.approx(NS_EAD.sum())


def test_subset_target_drops_unmodelled_obligors_but_stays_exact() -> None:
    """The soft direction is unchanged: netting sets whose obligor is not in the
    target are dropped, and the retained rows are still exact."""
    target = np.array([101, 100], dtype=np.int64)  # 700 not modelled
    out = aggregate_ead_to_obligor(
        ead_result=_paths(1), netting_set_master=_master(), obligor_uids_target=target
    )
    assert out[:, 0].tolist() == [EXPECTED_EAD[101], EXPECTED_EAD[100]]


def test_duplicate_target_uid_is_rejected() -> None:
    """A repeated target uid makes row attribution ambiguous."""
    target = np.array([100, 101, 100], dtype=np.int64)
    with pytest.raises(ValueError, match="duplicate uid"):
        aggregate_ead_to_obligor(
            ead_result=_paths(1),
            netting_set_master=_master(),
            obligor_uids_target=target,
        )


def test_unknown_netting_set_still_fails_fast() -> None:
    """The strict direction is unchanged: an EAD row for an unknown netting set
    is an upstream bug."""
    paths = NettingSetExposurePaths(
        uid=np.array([10, 99], dtype=np.int64), exposure=np.array([[1.0], [2.0]])
    )
    with pytest.raises(ValueError, match="netting-set uid"):
        aggregate_ead_to_obligor(
            ead_result=paths,
            netting_set_master=_master(),
            obligor_uids_target=np.array([700], dtype=np.int64),
        )


def test_modelled_obligor_without_netting_set_fails_in_both_paths() -> None:
    """C-03: the two sibling paths must answer the same question the same way."""
    target = np.array([100, 101, 999], dtype=np.int64)  # 999 has no netting set

    with pytest.raises(ValueError, match="no netting set"):
        aggregate_ead_to_obligor(
            ead_result=_paths(1),
            netting_set_master=_master(),
            obligor_uids_target=target,
        )

    with pytest.raises(ValueError, match="absent from the universe"):
        align_obligor_ead_to_portfolio(
            obligor_ead=CompiledObligorEad(
                uid=np.array([100, 101], dtype=np.int64), ead=np.array([4.0, 24.0])
            ),
            obligor_uids_target=target,
        )


def test_align_preserves_band_order_for_a_modelled_target() -> None:
    """The uid-safe sibling was already order-correct; lock that in so the two
    paths cannot drift apart again."""
    target = np.array([101, 100], dtype=np.int64)
    aligned = align_obligor_ead_to_portfolio(
        obligor_ead=CompiledObligorEad(
            uid=np.array([100, 101], dtype=np.int64), ead=np.array([4.0, 24.0])
        ),
        obligor_uids_target=target,
    )
    assert aligned.tolist() == [24.0, 4.0]


def test_scenario_varying_ead_is_summed_per_scenario() -> None:
    """Multi-netting-set obligors must sum along the netting-set axis only."""
    band = _band_order_portfolio()
    out = aggregate_ead_to_obligor(
        ead_result=_paths(3), netting_set_master=_master(), obligor_uids_target=band
    )
    # 700 holds netting sets 10, 11, 15 -> 1 + 2 + 32 at each scenario scale.
    row = int(np.flatnonzero(band == 700)[0])
    np.testing.assert_allclose(out[row], [35.0, 350.0, 3500.0])


def test_sf_copula_block_target_also_round_trips() -> None:
    """A third model block shifts band order again; attribution must follow."""
    fd = ForcedDefaultJtdBlock(
        obligor_uids=np.array([700], dtype=np.int64), lgd=np.array([0.6])
    )
    sf_block = SfGaussianCopulaJtdBlock(
        obligor_uids=np.array([100, 101], dtype=np.int64),
        lgd=np.array([0.5, 0.5]),
        thresholds=np.array([-2.0, -2.0]),
        factor_loading=np.array([0.3, 0.3]),
        systemic_uix=np.array([0, 0], dtype=np.int32),
        idiosyncratic_uix=np.array([1, 2], dtype=np.int32),
    )
    band = CompiledJtdPortfolio(
        forced_default_jtd_block=fd, sf_gaussian_copula_jtd_block=sf_block
    ).obligor_uids
    assert band.tolist() == [700, 100, 101]

    out = aggregate_ead_to_obligor(
        ead_result=_paths(1), netting_set_master=_master(), obligor_uids_target=band
    )
    assert out[:, 0].tolist() == [
        EXPECTED_EAD[700],
        EXPECTED_EAD[100],
        EXPECTED_EAD[101],
    ]
