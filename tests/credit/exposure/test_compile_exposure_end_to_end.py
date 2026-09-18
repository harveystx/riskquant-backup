# tests/credit/exposure/test_compile_exposure_end_to_end.py
"""
End-to-end exposure tests: compile -> run_exposure, for all three EAD models.

Replaces a suite that had been uncollectable since the compiler was decoupled
from `CompiledPortfolio`/`IngestBatch` (audit P2-01). Nothing deleted has been
restored; the tests are written against the current flat-array signature.

Every expected EAD is produced by a reference formula written out here in plain
Python from the model definition, never by calling the production path. The MtM
matrix is supplied in TRADE (portfolio) row order, so the CSR permutation is
exercised rather than assumed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from riskquant.constants import (
    CEM_PFE_COEFFICIENT_DEFAULT,
    CEM_PFE_FLOOR_DEFAULT,
    SACCR_ALPHA_DEFAULT,
    SACCR_MULTIPLIER_FLOOR_DEFAULT,
)
from riskquant.credit.exposure.add_on.compiled import CompiledAddOnFactors
from riskquant.credit.exposure.compile import (
    compile_exposure_cem,
    compile_exposure_rc,
    compile_exposure_saccr,
)
from riskquant.credit.exposure.engine import run_exposure
from riskquant.credit.exposure.saccr.config import CompiledSaccrConfig
from riskquant.credit.masters.netting_set.compiled import CompiledNettingSetMaster

# ==========================================================================
# The book — trade rows deliberately NOT grouped by netting set
# ==========================================================================
#   row 0 -> ns 20      row 1 -> ns 10      row 2 -> ns 10
TRADE_UIDS = np.array([101, 102, 103], dtype=np.int64)
TRADE_NS = np.array([20, 10, 10], dtype=np.int64)

#                      s0      s1      s2
MTM = np.ascontiguousarray(
    np.array([
        [300.0, 100.0, -200.0],  # row 0, ns 20
        [100.0, -50.0, 200.0],  # row 1, ns 10
        [50.0, 150.0, -100.0],  # row 2, ns 10
    ])
)
S = 3

COLLATERAL = {10: 25.0, 20: 0.0}
EFF_NOTIONAL = np.array([3000.0, 1000.0, 2000.0])  # trade order
SUP_FACTOR = np.array([0.03, 0.01, 0.02])  # trade order
SUBCLASS = np.array([1, 0, 0], dtype=np.int32)  # trade order

# --- hand-derived netting-set aggregates (CSR order is [10, 20]) -----------
V = {10: [150.0, 100.0, 100.0], 20: [300.0, 100.0, -200.0]}
GRC = {10: [150.0, 150.0, 200.0], 20: [300.0, 100.0, 0.0]}  # sum of max(mtm, 0)
GROSS_PFE = {10: 1000.0 * 0.01 + 2000.0 * 0.02, 20: 3000.0 * 0.03}  # 50.0, 90.0
NS_ORDER = [10, 20]


def _master() -> CompiledNettingSetMaster:
    return CompiledNettingSetMaster(
        uid=np.array([10, 20], dtype=np.int64),
        obligor_uid=np.array([900, 910], dtype=np.int64),
        net_collateral=np.array([COLLATERAL[10], COLLATERAL[20]]),
    )


# ==========================================================================
# Reference formulas (independent of production)
# ==========================================================================
def ref_rc(ns: int, s: int) -> float:
    return max(V[ns][s] - COLLATERAL[ns], 0.0)


def ref_cem(ns: int, s: int) -> float:
    nrc = ref_rc(ns, s)
    grc = GRC[ns][s]
    ngr = 0.0 if grc == 0.0 else nrc / grc
    return nrc + GROSS_PFE[ns] * (
        CEM_PFE_FLOOR_DEFAULT + CEM_PFE_COEFFICIENT_DEFAULT * ngr
    )


def ref_saccr(ns: int, s: int, add_on: float) -> float:
    v_minus_c = V[ns][s] - COLLATERAL[ns]
    rc = max(v_minus_c, 0.0)
    f = SACCR_MULTIPLIER_FLOOR_DEFAULT
    if add_on <= 0.0:
        mult = 1.0
    else:
        mult = min(
            1.0, f + (1.0 - f) * math.exp(v_minus_c / (2.0 * (1.0 - f) * add_on))
        )
    return SACCR_ALPHA_DEFAULT * (rc + mult * add_on)


# ==========================================================================
# RC
# ==========================================================================
def test_rc_end_to_end_matches_reference() -> None:
    compiled = compile_exposure_rc(
        trade_uids=TRADE_UIDS,
        trade_netting_set_id=TRADE_NS,
        netting_set_master=_master(),
    )
    paths = run_exposure(exposure=compiled, mtm=MTM)

    assert paths.uid.tolist() == NS_ORDER
    assert paths.exposure.shape == (2, S)
    assert paths.exposure.dtype == np.float64
    assert paths.exposure.flags.c_contiguous

    for i, ns in enumerate(NS_ORDER):
        for s in range(S):
            assert paths.exposure[i, s] == pytest.approx(ref_rc(ns, s)), (ns, s)


def test_rc_nets_negative_marks_inside_the_netting_set() -> None:
    """s1: ns10 holds +100 and -50. Netting gives V = 50... no: V = 100 - 50 + 50.
    The point is that the -50 is NOT floored at trade level."""
    paths = run_exposure(
        exposure=compile_exposure_rc(
            trade_uids=TRADE_UIDS,
            trade_netting_set_id=TRADE_NS,
            netting_set_master=_master(),
        ),
        mtm=MTM,
    )
    # ns10 s1: (-50) + 150 = 100; RC = 100 - 25 = 75. Trade-level flooring would
    # have produced max(-50,0) + max(150,0) - 25 = 125.
    assert paths.exposure[0, 1] == pytest.approx(75.0)


def test_rc_floors_at_zero_when_collateral_exceeds_value() -> None:
    master = CompiledNettingSetMaster(
        uid=np.array([10, 20], dtype=np.int64),
        obligor_uid=np.array([900, 910], dtype=np.int64),
        net_collateral=np.array([10_000.0, 10_000.0]),
    )
    paths = run_exposure(
        exposure=compile_exposure_rc(
            trade_uids=TRADE_UIDS,
            trade_netting_set_id=TRADE_NS,
            netting_set_master=master,
        ),
        mtm=MTM,
    )
    assert (paths.exposure == 0.0).all()


# ==========================================================================
# CEM
# ==========================================================================
def test_cem_end_to_end_matches_reference() -> None:
    compiled = compile_exposure_cem(
        trade_uids=TRADE_UIDS,
        trade_netting_set_id=TRADE_NS,
        trade_effective_notional=EFF_NOTIONAL,
        trade_supervisory_factor=SUP_FACTOR,
        netting_set_master=_master(),
    )
    paths = run_exposure(exposure=compiled, mtm=MTM)

    assert paths.uid.tolist() == NS_ORDER
    for i, ns in enumerate(NS_ORDER):
        for s in range(S):
            assert paths.exposure[i, s] == pytest.approx(ref_cem(ns, s)), (ns, s)


def test_cem_zero_gross_replacement_cost_gives_zero_ngr() -> None:
    """ns20 at s2 has a single negative mark: GRC = 0, so NGR must be 0 and the
    add-on enters at the floor weight only."""
    paths = run_exposure(
        exposure=compile_exposure_cem(
            trade_uids=TRADE_UIDS,
            trade_netting_set_id=TRADE_NS,
            trade_effective_notional=EFF_NOTIONAL,
            trade_supervisory_factor=SUP_FACTOR,
            netting_set_master=_master(),
        ),
        mtm=MTM,
    )
    assert paths.exposure[1, 2] == pytest.approx(
        0.0 + GROSS_PFE[20] * CEM_PFE_FLOOR_DEFAULT
    )


def test_cem_always_dominates_rc() -> None:
    """The add-on term is non-negative, so CEM EAD >= RC EAD everywhere."""
    rc = run_exposure(
        exposure=compile_exposure_rc(
            trade_uids=TRADE_UIDS,
            trade_netting_set_id=TRADE_NS,
            netting_set_master=_master(),
        ),
        mtm=MTM,
    ).exposure
    cem = run_exposure(
        exposure=compile_exposure_cem(
            trade_uids=TRADE_UIDS,
            trade_netting_set_id=TRADE_NS,
            trade_effective_notional=EFF_NOTIONAL,
            trade_supervisory_factor=SUP_FACTOR,
            netting_set_master=_master(),
        ),
        mtm=MTM,
    ).exposure
    assert (cem >= rc - 1e-12).all()


# ==========================================================================
# SA-CCR
# ==========================================================================
def _saccr_inputs():
    factors = CompiledAddOnFactors(
        subclass_uix=np.array([0, 1], dtype=np.int32),
        factor=np.array([0.18, 0.40]),
    )
    config = CompiledSaccrConfig(
        subclass_uix=np.array([0, 1], dtype=np.int32),
        subclass_hedging_set=np.array([0, 0], dtype=np.int32),
        hedging_set_uix=np.array([0], dtype=np.int32),
        hedging_set_rho=np.array([0.5]),
        alpha=SACCR_ALPHA_DEFAULT,
        multiplier_floor=SACCR_MULTIPLIER_FLOOR_DEFAULT,
    )
    return factors, config


# ns 10 holds trades 102 (subclass 0, 1000) and 103 (subclass 0, 2000)
#   -> one subclass, D = 3000, A = 0.18 * 3000 = 540; single entry -> AddOn = 540
# ns 20 holds trade 101 (subclass 1, 3000) -> A = 0.40 * 3000 = 1200 -> AddOn = 1200
ADD_ON = {10: 0.18 * 3000.0, 20: 0.40 * 3000.0}


def test_saccr_end_to_end_matches_reference() -> None:
    factors, config = _saccr_inputs()
    compiled = compile_exposure_saccr(
        trade_uids=TRADE_UIDS,
        trade_netting_set_id=TRADE_NS,
        trade_effective_notional=EFF_NOTIONAL,
        trade_subclass_uix=SUBCLASS,
        netting_set_master=_master(),
        add_on_factors=factors,
        saccr_config=config,
    )
    assert compiled.saccr.add_on.tolist() == [ADD_ON[10], ADD_ON[20]]

    paths = run_exposure(exposure=compiled, mtm=MTM)
    assert paths.uid.tolist() == NS_ORDER
    for i, ns in enumerate(NS_ORDER):
        for s in range(S):
            assert paths.exposure[i, s] == pytest.approx(
                ref_saccr(ns, s, ADD_ON[ns])
            ), (ns, s)


def test_saccr_is_strictly_positive_even_when_deeply_out_of_the_money() -> None:
    """The multiplier floor keeps EAD above zero where RC is zero."""
    factors, config = _saccr_inputs()
    paths = run_exposure(
        exposure=compile_exposure_saccr(
            trade_uids=TRADE_UIDS,
            trade_netting_set_id=TRADE_NS,
            trade_effective_notional=EFF_NOTIONAL,
            trade_subclass_uix=SUBCLASS,
            netting_set_master=_master(),
            add_on_factors=factors,
            saccr_config=config,
        ),
        mtm=MTM,
    )
    # ns20 s2: V - C = -200 -> RC = 0, but EAD > 0 through mult * AddOn.
    assert paths.exposure[1, 2] > 0.0
    assert paths.exposure[1, 2] == pytest.approx(ref_saccr(20, 2, ADD_ON[20]))


# ==========================================================================
# Cross-model structural invariants
# ==========================================================================
def test_all_three_models_agree_on_netting_set_identity_and_shape() -> None:
    factors, config = _saccr_inputs()
    compiled = [
        compile_exposure_rc(
            trade_uids=TRADE_UIDS,
            trade_netting_set_id=TRADE_NS,
            netting_set_master=_master(),
        ),
        compile_exposure_cem(
            trade_uids=TRADE_UIDS,
            trade_netting_set_id=TRADE_NS,
            trade_effective_notional=EFF_NOTIONAL,
            trade_supervisory_factor=SUP_FACTOR,
            netting_set_master=_master(),
        ),
        compile_exposure_saccr(
            trade_uids=TRADE_UIDS,
            trade_netting_set_id=TRADE_NS,
            trade_effective_notional=EFF_NOTIONAL,
            trade_subclass_uix=SUBCLASS,
            netting_set_master=_master(),
            add_on_factors=factors,
            saccr_config=config,
        ),
    ]
    for c in compiled:
        paths = run_exposure(exposure=c, mtm=MTM)
        assert paths.uid.tolist() == NS_ORDER
        assert paths.exposure.shape == (2, S)
        assert paths.exposure.flags.c_contiguous
        assert np.isfinite(paths.exposure).all()
        assert (paths.exposure >= 0.0).all()  # EAD is never negative
