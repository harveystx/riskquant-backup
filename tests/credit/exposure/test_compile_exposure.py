# tests/credit/exposure/test_compile_exposure.py
"""
Unit tests for the exposure compiler, written against the CURRENT contract.

These tests replace a suite that had been uncollectable since the compiler was
decoupled from `CompiledPortfolio`/`IngestBatch`: it imported
`_join_netting_set_ids` and `_resolve_collateral`, neither of which exists, and
called `compile_exposure_rc(portfolio=..., batch=...)` rather than the flat-array
signature. The compiler therefore had zero effective coverage (audit P2-01).

Production is the source of truth: nothing deleted has been restored. Coverage
intent from the old suite is preserved — input validation, CSR construction, uid
ordering, missing-netting-set detection, collateral projection — and extended to
the CEM and SA-CCR compile paths, which were never covered at all.

Expected values are hand-derived. The fixture deliberately scrambles trade order
relative to netting-set id so a positional carry cannot pass where a uid join is
required.
"""

from __future__ import annotations

import numpy as np
import pytest

from riskquant.constants import NO_ID
from riskquant.credit.exposure.add_on.compiled import CompiledAddOnFactors
from riskquant.credit.exposure.compile import (
    _build_csr,
    _project_netting_set_attributes,
    _validate_inputs,
    _validate_master,
    _validate_no_missing_netting_sets,
    compile_exposure_cem,
    compile_exposure_rc,
    compile_exposure_saccr,
)
from riskquant.credit.exposure.saccr.config import CompiledSaccrConfig
from riskquant.credit.masters.netting_set.compiled import CompiledNettingSetMaster

# ==========================================================================
# Fixture book
# ==========================================================================
# Trades are deliberately NOT grouped by netting set in input order.
#   row 0 -> ns 30 | row 1 -> ns 10 | row 2 -> ns 30 | row 3 -> ns 20
TRADE_UIDS = np.array([101, 102, 103, 104], dtype=np.int64)
TRADE_NS = np.array([30, 10, 30, 20], dtype=np.int64)

MASTER_UID = np.array([10, 20, 30], dtype=np.int64)
MASTER_OBLIGOR = np.array([900, 910, 920], dtype=np.int64)
MASTER_COLLATERAL = np.array([11.0, 22.0, 33.0])


def _master(**overrides) -> CompiledNettingSetMaster:
    kwargs = {
        "uid": MASTER_UID,
        "obligor_uid": MASTER_OBLIGOR,
        "net_collateral": MASTER_COLLATERAL,
    }
    kwargs.update(overrides)
    return CompiledNettingSetMaster(**kwargs)


def _add_on_factors() -> CompiledAddOnFactors:
    return CompiledAddOnFactors(
        subclass_uix=np.array([0, 1], dtype=np.int32),
        factor=np.array([0.18, 0.40]),
    )


def _saccr_config() -> CompiledSaccrConfig:
    return CompiledSaccrConfig(
        subclass_uix=np.array([0, 1], dtype=np.int32),
        subclass_hedging_set=np.array([0, 0], dtype=np.int32),
        hedging_set_uix=np.array([0], dtype=np.int32),
        hedging_set_rho=np.array([0.5]),
        alpha=1.4,
        multiplier_floor=0.05,
    )


# ==========================================================================
# INPUT VALIDATION
# ==========================================================================
class TestValidateInputs:
    def test_accepts_aligned_int64_arrays(self) -> None:
        _validate_inputs(TRADE_UIDS, TRADE_NS)  # must not raise

    def test_rejects_non_int64_trade_uids(self) -> None:
        with pytest.raises(TypeError, match="trade_uids must be int64"):
            _validate_inputs(TRADE_UIDS.astype(np.int32), TRADE_NS)

    def test_rejects_non_int64_netting_set_id(self) -> None:
        with pytest.raises(TypeError, match="trade_netting_set_id must be int64"):
            _validate_inputs(TRADE_UIDS, TRADE_NS.astype(np.int32))

    def test_rejects_non_1d(self) -> None:
        with pytest.raises(ValueError, match="1D"):
            _validate_inputs(TRADE_UIDS.reshape(2, 2), TRADE_NS.reshape(2, 2))

    def test_rejects_misaligned_lengths(self) -> None:
        with pytest.raises(ValueError, match="must align"):
            _validate_inputs(TRADE_UIDS, TRADE_NS[:2])


class TestValidateMaster:
    def test_accepts_sorted_unique_master(self) -> None:
        _validate_master(_master())

    def test_rejects_non_int64_uid(self) -> None:
        with pytest.raises(TypeError, match="uid must be int64"):
            _validate_master(
                _master(
                    uid=MASTER_UID.astype(np.int32),
                    obligor_uid=MASTER_OBLIGOR,
                )
            )

    def test_rejects_non_float64_collateral(self) -> None:
        with pytest.raises(TypeError, match="net_collateral must be float64"):
            _validate_master(
                _master(net_collateral=MASTER_COLLATERAL.astype(np.float32))
            )

    def test_rejects_unsorted_or_duplicate_uid(self) -> None:
        for uid in (
            np.array([30, 10, 20], dtype=np.int64),
            np.array([10, 10, 20], dtype=np.int64),
        ):
            with pytest.raises(ValueError, match="strictly ascending and unique"):
                _validate_master(_master(uid=uid))


class TestMissingNettingSetDetection:
    def test_accepts_fully_assigned_trades(self) -> None:
        _validate_no_missing_netting_sets(TRADE_NS)

    def test_rejects_the_unassigned_sentinel(self) -> None:
        ns = np.array([30, int(NO_ID), 20], dtype=np.int64)
        with pytest.raises(ValueError, match="netting set"):
            _validate_no_missing_netting_sets(ns)

    def test_compile_rejects_unassigned_before_any_join(self) -> None:
        ns = np.array([30, int(NO_ID), 30, 20], dtype=np.int64)
        with pytest.raises(ValueError, match="netting set"):
            compile_exposure_rc(
                trade_uids=TRADE_UIDS,
                trade_netting_set_id=ns,
                netting_set_master=_master(),
            )


# ==========================================================================
# CSR CONSTRUCTION
# ==========================================================================
class TestBuildCsr:
    def test_groups_by_netting_set_with_sorted_uids(self) -> None:
        uids, offsets, perm = _build_csr(TRADE_NS)
        assert uids.tolist() == [10, 20, 30]  # ascending, deduplicated
        assert offsets.tolist() == [0, 1, 2, 4]  # 1 + 1 + 2 trades
        assert perm.tolist() == [1, 3, 0, 2]  # rows regrouped, order stable
        assert uids.dtype == offsets.dtype == perm.dtype == np.int64

    def test_offsets_partition_the_rows_exactly(self) -> None:
        _, offsets, perm = _build_csr(TRADE_NS)
        assert offsets[0] == 0
        assert offsets[-1] == TRADE_NS.size
        assert sorted(perm.tolist()) == list(range(TRADE_NS.size))
        assert np.all(np.diff(offsets) > 0)

    def test_ordering_within_a_netting_set_is_stable(self) -> None:
        ns = np.array([7, 7, 7, 7], dtype=np.int64)
        uids, offsets, perm = _build_csr(ns)
        assert uids.tolist() == [7]
        assert offsets.tolist() == [0, 4]
        assert perm.tolist() == [0, 1, 2, 3]  # input order preserved

    def test_single_trade(self) -> None:
        uids, offsets, perm = _build_csr(np.array([5], dtype=np.int64))
        assert (uids.tolist(), offsets.tolist(), perm.tolist()) == ([5], [0, 1], [0])


# ==========================================================================
# COLLATERAL PROJECTION
# ==========================================================================
class TestProjectNettingSetAttributes:
    def test_projects_by_uid_not_by_position(self) -> None:
        """Ask for the master rows out of order: the answer must follow the uid."""
        got = _project_netting_set_attributes(
            np.array([30, 10], dtype=np.int64), _master()
        )
        assert got.tolist() == [33.0, 11.0]  # NOT [11.0, 22.0]
        assert got.dtype == np.float64 and got.flags.c_contiguous

    def test_unknown_netting_set_uid_fails_fast(self) -> None:
        with pytest.raises(ValueError, match="netting_set_uids"):
            _project_netting_set_attributes(
                np.array([10, 99], dtype=np.int64), _master()
            )

    def test_subset_projection_is_exact(self) -> None:
        got = _project_netting_set_attributes(np.array([20], dtype=np.int64), _master())
        assert got.tolist() == [22.0]


# ==========================================================================
# RC COMPILE PATH
# ==========================================================================
class TestCompileExposureRc:
    def test_view_structure_and_projection(self) -> None:
        compiled = compile_exposure_rc(
            trade_uids=TRADE_UIDS,
            trade_netting_set_id=TRADE_NS,
            netting_set_master=_master(),
        )
        view = compiled.replacement_cost
        assert compiled.current_exposure_method is None
        assert compiled.saccr is None
        assert view.netting_set_uids.tolist() == [10, 20, 30]
        assert view.netting_set_offsets.tolist() == [0, 1, 2, 4]
        assert view.mtm_row_perm.tolist() == [1, 3, 0, 2]
        # collateral follows the netting-set uid, not the master row order
        assert view.collateral.tolist() == [11.0, 22.0, 33.0]

    def test_master_row_order_does_not_leak_into_the_view(self) -> None:
        """Only netting set 30 is traded: the view must carry ITS collateral."""
        compiled = compile_exposure_rc(
            trade_uids=np.array([101, 103], dtype=np.int64),
            trade_netting_set_id=np.array([30, 30], dtype=np.int64),
            netting_set_master=_master(),
        )
        assert compiled.replacement_cost.collateral.tolist() == [33.0]

    def test_unknown_netting_set_fails_fast(self) -> None:
        with pytest.raises(ValueError, match="netting_set_uids"):
            compile_exposure_rc(
                trade_uids=TRADE_UIDS,
                trade_netting_set_id=np.array([30, 10, 30, 77], dtype=np.int64),
                netting_set_master=_master(),
            )


# ==========================================================================
# CEM COMPILE PATH
# ==========================================================================
_EFF_NOTIONAL = np.array([1000.0, 2000.0, 3000.0, 4000.0])
_SUP_FACTOR = np.array([0.01, 0.02, 0.03, 0.04])


def _compile_cem(**overrides):
    kwargs = {
        "trade_uids": TRADE_UIDS,
        "trade_netting_set_id": TRADE_NS,
        "trade_effective_notional": _EFF_NOTIONAL,
        "trade_supervisory_factor": _SUP_FACTOR,
        "netting_set_master": _master(),
    }
    kwargs.update(overrides)
    return compile_exposure_cem(**kwargs)


class TestCompileExposureCem:
    def test_add_on_scalars_are_permuted_into_csr_order(self) -> None:
        view = _compile_cem().current_exposure_method
        # perm is [1, 3, 0, 2]; both scalars must follow it, together.
        assert view.effective_notional.tolist() == [2000.0, 4000.0, 1000.0, 3000.0]
        assert view.supervisory_factor.tolist() == [0.02, 0.04, 0.01, 0.03]
        assert view.effective_notional.flags.c_contiguous
        assert view.supervisory_factor.flags.c_contiguous

    def test_structure_matches_the_rc_projection(self) -> None:
        cem = _compile_cem().current_exposure_method
        rc = compile_exposure_rc(
            trade_uids=TRADE_UIDS,
            trade_netting_set_id=TRADE_NS,
            netting_set_master=_master(),
        ).replacement_cost
        assert cem.netting_set_uids.tolist() == rc.netting_set_uids.tolist()
        assert cem.netting_set_offsets.tolist() == rc.netting_set_offsets.tolist()
        assert cem.mtm_row_perm.tolist() == rc.mtm_row_perm.tolist()
        assert cem.collateral.tolist() == rc.collateral.tolist()

    @pytest.mark.parametrize(
        "field", ["trade_effective_notional", "trade_supervisory_factor"]
    )
    def test_rejects_wrong_dtype(self, field) -> None:
        with pytest.raises(TypeError, match="must be float64"):
            _compile_cem(**{field: np.array([1, 2, 3, 4], dtype=np.int64)})

    @pytest.mark.parametrize(
        "field", ["trade_effective_notional", "trade_supervisory_factor"]
    )
    def test_rejects_misaligned_length(self, field) -> None:
        with pytest.raises(ValueError, match="must align"):
            _compile_cem(**{field: np.array([1.0, 2.0])})

    @pytest.mark.parametrize("floor", [-0.01, 1.01])
    def test_rejects_pfe_floor_outside_unit_interval(self, floor) -> None:
        with pytest.raises(ValueError, match="pfe_floor"):
            _compile_cem(pfe_floor=floor)

    def test_rejects_negative_pfe_coefficient(self) -> None:
        with pytest.raises(ValueError, match="pfe_coefficient"):
            _compile_cem(pfe_coefficient=-0.1)

    def test_floor_and_coefficient_are_carried_onto_the_view(self) -> None:
        view = _compile_cem(
            pfe_floor=0.25, pfe_coefficient=0.75
        ).current_exposure_method
        assert view.pfe_floor == 0.25
        assert view.pfe_coefficient == 0.75


# ==========================================================================
# SA-CCR COMPILE PATH
# ==========================================================================
_SUBCLASS = np.array([0, 1, 0, 1], dtype=np.int32)


def _compile_saccr(**overrides):
    kwargs = {
        "trade_uids": TRADE_UIDS,
        "trade_netting_set_id": TRADE_NS,
        "trade_effective_notional": _EFF_NOTIONAL,
        "trade_subclass_uix": _SUBCLASS,
        "netting_set_master": _master(),
        "add_on_factors": _add_on_factors(),
        "saccr_config": _saccr_config(),
    }
    kwargs.update(overrides)
    return compile_exposure_saccr(**kwargs)


class TestCompileExposureSaccr:
    def test_structure_and_add_on_alignment(self) -> None:
        view = _compile_saccr().saccr
        assert view.netting_set_uids.tolist() == [10, 20, 30]
        assert view.netting_set_offsets.tolist() == [0, 1, 2, 4]
        assert view.collateral.tolist() == [11.0, 22.0, 33.0]
        assert view.add_on.shape == (3,)
        assert view.add_on.dtype == np.float64
        assert view.alpha == 1.4 and view.multiplier_floor == 0.05

    def test_add_on_is_hand_calculable_per_netting_set(self) -> None:
        """ns 10 holds trade row 1 only: subclass 1, notional 2000, SF 0.40.
        A single-entry hedging set collapses to A = SF*|D| for any rho."""
        view = _compile_saccr().saccr
        assert view.add_on[0] == pytest.approx(0.40 * 2000.0)
        # ns 20 holds row 3 only: subclass 1, notional 4000.
        assert view.add_on[1] == pytest.approx(0.40 * 4000.0)
        # ns 30 holds rows 0 and 2: subclass 0 both -> D = 1000 + 3000 = 4000.
        assert view.add_on[2] == pytest.approx(0.18 * 4000.0)

    def test_unknown_subclass_fails_fast(self) -> None:
        with pytest.raises(ValueError, match="subclass"):
            _compile_saccr(trade_subclass_uix=np.array([0, 1, 0, 9], dtype=np.int32))

    def test_only_the_saccr_view_is_populated(self) -> None:
        compiled = _compile_saccr()
        assert compiled.replacement_cost is None
        assert compiled.current_exposure_method is None
        assert compiled.saccr is not None
