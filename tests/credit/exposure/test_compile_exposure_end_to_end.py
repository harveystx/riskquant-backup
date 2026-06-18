# tests/credit/exposure/test_compile_exposure_end_to_end.py
"""
True end-to-end integration test for the credit EXPOSURE layer.

Every other exposure test hand-builds `IngestBatch` / `CompiledPortfolio`
fixtures and therefore skips the ingest edge entirely. This test closes that
gap: it drives the canonical `minimal_replacement_cost` data through the full
pipeline

    build_ingest_batch            (canonicalize -> materialize -> dispatch
                                   -> compile_ingest_df -> from_compiled)
        -> compile_portfolio_from_batch   (instrument-id argsort + reorder)
        -> compile_exposure               (uid join + CSR banding)
        -> run_exposure

and asserts the three netting-set EADs against the analytically known result.

The MtM matrix is synthesized directly in `CompiledPortfolio.trade_uids` row
order rather than produced by `run_valuation`: MtM == qty * price makes every
netting-set sum exact, so the RC floor (`max(V - C, 0)`) is checked against
closed-form values. No `MarketDataMatrix` is needed.

What only this test exercises (vs. the hand-built fixtures):
  - `netting_set_id` actually flows through `_materialize_nullable_int_col`,
    `canonicalize._validate_schema`, the `"Int64"` round-trip and
    `_compile_netting_set_id`;
  - `IngestBatch.reorder()` carries `netting_set_id` through the
    instrument-id argsort intact;
  - the `compile_exposure` uid join is non-positional — proven by the
    scrambled-row variant, which permutes the raw frame and still recovers
    identical EADs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskquant.credit.exposure.compile import compile_exposure
from riskquant.credit.exposure.engine import run_exposure
from riskquant.credit.exposure.results import NettingSetExposurePaths
from riskquant.ingest_fixtures.credit.credit_exposure.minimal_spot_with_netting_sets import (
    CANONICAL_REPLACEMENT_COST_COLLATERAL,
    CANONICAL_REPLACEMENT_COST_INGEST_DATA,
)
from riskquant.portfolio.engine import compile_portfolio_from_ingest_batch
from riskquant.portfolio.ingest.pipeline import compile_raw_df_into_ingest_batch

# ============================================================
# FIXED INPUTS
# ============================================================

# The minimal RC portfolio references a single underlying and is entirely
# USD-settled, so FX resolution is a no-op (under_ccy == settlement_ccy).
_RF_NAME_TO_RF_ID: dict[str, int] = {"TEST_BRENT_CRUDE_202701": 0}
_RF_ID_TO_CCY: dict[int, str] = {0: "USD"}

# Synthetic per-scenario price vector. Three distinct scenarios exercise the
# per-scenario independence of the RC kernel.
_PRICES: np.ndarray = np.array([100.0, 50.0, 250.0], dtype=np.float64)


# ============================================================
# HELPERS
# ============================================================


def _expected_ead_by_set() -> tuple[list[int], dict[int, np.ndarray]]:
    """
    Closed-form reference EAD per netting set, computed straight from the
    canonical data — independent of the RC kernel under test.
    """
    data = CANONICAL_REPLACEMENT_COST_INGEST_DATA
    qty_by_uid = dict(zip(data["uid"], data["quantity"], strict=True))
    ns_by_uid = dict(zip(data["uid"], data.get("netting_set_id", []), strict=True))

    sets = sorted({ns for ns in ns_by_uid.values() if ns is not None})

    expected: dict[int, np.ndarray] = {}
    for ns in sets:
        members = [u for u, n in ns_by_uid.items() if n == ns]
        netted = sum(qty_by_uid[u] for u in members) * _PRICES
        collateral = CANONICAL_REPLACEMENT_COST_COLLATERAL.get(ns, 0.0)
        expected[ns] = np.maximum(netted - collateral, 0.0)
    return sets, expected


def _run_pipeline(
    raw_df: pd.DataFrame,
    collateral: dict[int, float],
) -> NettingSetExposurePaths:
    """Full ingest -> portfolio -> exposure -> EAD pipeline for one raw frame."""
    batch = compile_raw_df_into_ingest_batch(
        raw_df=raw_df,
        rf_name_to_rf_id=_RF_NAME_TO_RF_ID,
        rf_id_to_ccy=_RF_ID_TO_CCY,
    )
    portfolio = compile_portfolio_from_ingest_batch(batch)

    # Synthesize MtM in CompiledPortfolio.trade_uids row order — the contract
    # run_valuation would otherwise fulfil. Keyed by uid so the matrix is
    # correct regardless of the block/portfolio row ordering.
    qty_by_uid = dict(
        zip(
            CANONICAL_REPLACEMENT_COST_INGEST_DATA["uid"],
            CANONICAL_REPLACEMENT_COST_INGEST_DATA["quantity"],
            strict=True,
        )
    )
    qty_vec = np.array(
        [qty_by_uid[int(uid)] for uid in portfolio.trade_uids],
        dtype=np.float64,
    )
    mtm = np.ascontiguousarray(qty_vec[:, None] * _PRICES[None, :])

    exposure = compile_exposure(
        portfolio=portfolio,
        batch=batch,
        collateral=collateral,
    )
    return run_exposure(exposure=exposure, mtm=mtm)


# ============================================================
# FIXTURES
# ============================================================


@pytest.fixture
def rc_result() -> NettingSetExposurePaths:
    """EAD result from the canonical RC frame in its natural row order."""
    return _run_pipeline(
        pd.DataFrame(CANONICAL_REPLACEMENT_COST_INGEST_DATA),
        CANONICAL_REPLACEMENT_COST_COLLATERAL,
    )


# ============================================================
# TESTS
# ============================================================


def test_end_to_end_replacement_cost_ead(rc_result: NettingSetExposurePaths) -> None:
    """Canonical RC data resolves to the closed-form EAD through the full edge."""
    sets, expected = _expected_ead_by_set()

    # The two NO_NETTING_SET trades (uid 1007, 1008) are dropped; only the
    # three netting sets survive, in ascending-uid band order.
    np.testing.assert_array_equal(rc_result.uid, np.array(sets, dtype=np.int64))
    assert rc_result.uid.dtype == np.int64
    assert rc_result.exposure.shape == (len(sets), _PRICES.shape[0])
    assert rc_result.exposure.dtype == np.float64

    for i, ns in enumerate(sets):
        np.testing.assert_allclose(
            rc_result.exposure[i],
            expected[ns],
            rtol=0.0,
            atol=1e-9,
            err_msg=f"EAD mismatch for netting set {ns}",
        )

    # Explicit scenario-0 (price 100.0) anchors — the RC floor and collateral
    # offset spelled out, decoupled from the reference helper above.
    ead_by_set = dict(zip(rc_result.uid.tolist(), rc_result.exposure, strict=True))
    assert ead_by_set[101][0] == pytest.approx(950.0)  # 12*100 - 250 collateral
    assert ead_by_set[102][0] == pytest.approx(0.0)  # nets short -> floored
    assert ead_by_set[103][0] == pytest.approx(0.0)  # over-collateralized


def test_scrambled_raw_rows_yield_identical_ead(
    rc_result: NettingSetExposurePaths,
) -> None:
    """
    A permuted raw frame produces byte-identical EADs.

    Proves the `compile_exposure` uid join is non-positional and that
    `IngestBatch.reorder()` carries `netting_set_id` through the
    instrument-id argsort without misalignment.
    """
    raw = pd.DataFrame(CANONICAL_REPLACEMENT_COST_INGEST_DATA)
    perm = [5, 0, 7, 2, 4, 1, 6, 3]
    scrambled = raw.iloc[perm].reset_index(drop=True)

    scrambled_result = _run_pipeline(
        scrambled,
        CANONICAL_REPLACEMENT_COST_COLLATERAL,
    )

    np.testing.assert_array_equal(scrambled_result.uid, rc_result.uid)
    np.testing.assert_allclose(
        scrambled_result.exposure,
        rc_result.exposure,
        rtol=0.0,
        atol=1e-9,
    )


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
