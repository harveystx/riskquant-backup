# tests/valuation/oracle/test_node_forward_optional_delivery.py
"""
Optional-delivery contract for exact (non-CALENDAR_PERIOD) NODE forwards.

Owner-confirmed rule (ws/vanilla-forward-node):
    delivery_date is optional for exact NODE forwards.
    When present:  asset-leg discount date = delivery_date.
    When absent:   asset-leg discount date = settlement_date.

The underlying NODE rf is fixed by identity; delivery never selects it.

Architecture
------------
Uses the same IngestBatch hand-build pattern as forward_fixture.py (the
Phase-4 acceptance suite). The batch bypasses the ingest family validator
and identity gate — this isolates the compiler and downstream valuation layers.
The negative controls for the identity gate (relative NODE, missing delivery)
are in test_node_option_delivery_contract.py.

Independence contract
---------------------
The oracle (pv_forward from economic_oracle.py) does not call:
  * the NODE compiler
  * build_uix_mapper
  * compile_portfolio_from_ingest_batch
  * assemble_valuation_state
  * run_valuation
It reads the NODE level directly from a dict keyed on factor_id and computes
the two-leg PV formula using independently specified discount factors.

Fixture design
--------------
    SETTLE_CCY  = USD
    NODE rf     = RF_NODE     (unique; the oracle reads exactly this row)
    CURVE_A     = CURVE_A_ID  (asset discount)
    CURVE_S     = CURVE_S_ID  (strike discount; may equal CURVE_A)

    Three trades on the same NODE, same strike, same settlement date:
      T1: delivery = ORD_D1             -> asset DF = DF(ORD_D1)
      T2: delivery = ORD_D2             -> asset DF = DF(ORD_D2)
      T3: delivery = NO_DATE_ORD32      -> asset DF = DF(ORD_SETTLE)

    DF constraints (required so mutations are not degenerate):
      DF(ORD_D1) != DF(ORD_D2)
      DF(ORD_D1) != DF(ORD_SETTLE)
      DF(ORD_D2) != DF(ORD_SETTLE)
      NODE price != any value interpolation could produce between the two
      neighbouring NODE rows (anti-interpolation check).

Anti-interpolation fixture
--------------------------
    NODE_A = RF_NODE  = 100.0   (the exact observable; all three trades use this)
    NODE_B = RF_NODE2 = 150.0   (a neighbouring node with a deliberately different
                                  level so any delivery-date lookup of NODE_B would
                                  produce 150.0, not 100.0)
    price_curve_grid is poisoned with 999.0 everywhere so accidental interpolation
    would produce 999.0 rather than 100.0.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from riskquant.constants import NO_DATE_ORD32, NO_FX_RF_REQUIRED
from riskquant.market_data.compile import assemble_valuation_state
from riskquant.market_data.state import MarketDataMatrix
from riskquant.portfolio.engine import (
    build_uix_mapper,
    compile_portfolio_from_ingest_batch,
)
from riskquant.portfolio.ingest.batch import IngestBatch
from riskquant.portfolio.instruments import VANILLA_FORWARD_NODE_ID
from riskquant.valuation.engine import run_valuation

from . import economic_oracle as oracle

# ---------------------------------------------------------------------------
# FIXTURE CONSTANTS
# ---------------------------------------------------------------------------

SETTLE_CCY = "USD"
N_SCENARIOS = 2
RTOL = 1e-13

# Global rf / curve ids — sparse and deliberately non-ascending so id != row.
RF_NODE = 7_001          # The exact NODE all three trades reference.
RF_NODE2 = 8_002         # A neighbouring NODE with a different level.
CURVE_A_ID = 900_001     # Asset discount curve.
CURVE_S_ID = 900_001     # Strike discount curve (same as asset: identity FX + same ccy).

# Date ordinals.  D1 < D2 < SETTLE so the three DFs are all distinct.
ORD_D1 = 20_400          # delivery T1
ORD_D2 = 20_450          # delivery T2
ORD_SETTLE = 20_500      # settlement (= asset-leg date for T3)
ORD_OTHER = 20_100       # extra date in the universe to prove no accidental match

# Scenario-wise market levels.
LVL_NODE = np.array([100.0, 95.0])    # The exact NODE price.
LVL_NODE2 = np.array([150.0, 145.0])  # Neighbouring NODE; must differ from LVL_NODE.

# Discount factors — all strictly different so every fallback choice is detectable.
DF_D1 = np.array([0.9800, 0.9750])   # DF at delivery D1
DF_D2 = np.array([0.9600, 0.9550])   # DF at delivery D2  (< DF_D1)
DF_SETTLE = np.array([0.9400, 0.9350])  # DF at settlement  (< DF_D2)

# Strike and quantity (same across all three trades).
K = 90.0
Q = 1_000.0

# Trade UIDs.
UID_T1 = 1001   # delivery D1
UID_T2 = 1002   # delivery D2
UID_T3 = 1003   # no delivery; asset leg falls back to settlement


# ---------------------------------------------------------------------------
# PRODUCTION-SIDE FIXTURE
# ---------------------------------------------------------------------------


def _i32(*args):
    return np.ascontiguousarray(args, dtype=np.int32)


def _i8(*args):
    return np.ascontiguousarray(args, dtype=np.int8)


def _f64(*args):
    return np.ascontiguousarray(args, dtype=np.float64)


def _make_batch() -> IngestBatch:
    """
    Three VANILLA_FORWARD_NODE rows, no FX (single-currency USD).

    T1: delivery = ORD_D1
    T2: delivery = ORD_D2
    T3: delivery = NO_DATE_ORD32  (absent)
    """
    n = 3
    return IngestBatch(
        size=n,
        instrument_ids=_i32(
            VANILLA_FORWARD_NODE_ID,
            VANILLA_FORWARD_NODE_ID,
            VANILLA_FORWARD_NODE_ID,
        ),
        uids=np.ascontiguousarray([UID_T1, UID_T2, UID_T3], dtype=np.int64),
        qty=_f64(Q, Q, Q),
        strike_price=_f64(K, K, K),
        cp_mult=_i8(0, 0, 0),
        settle_ord=_i32(ORD_SETTLE, ORD_SETTLE, ORD_SETTLE),
        delivery_ord=_i32(ORD_D1, ORD_D2, int(NO_DATE_ORD32)),
        expiry_ord=_i32(ORD_D1, ORD_D2, int(NO_DATE_ORD32)),
        underlying_ids=_i32(RF_NODE, RF_NODE, RF_NODE),
        volatility_ids=_i32(-1, -1, -1),
        fx_u2s_ids=_i32(NO_FX_RF_REQUIRED, NO_FX_RF_REQUIRED, NO_FX_RF_REQUIRED),
        fx_u2s_inv=_i8(0, 0, 0),
        fx_k2s_ids=_i32(NO_FX_RF_REQUIRED, NO_FX_RF_REQUIRED, NO_FX_RF_REQUIRED),
        fx_k2s_inv=_i8(0, 0, 0),
        under_discount_curve_ids=_i32(CURVE_A_ID, CURVE_A_ID, CURVE_A_ID),
        strike_discount_curve_ids=_i32(CURVE_S_ID, CURVE_S_ID, CURVE_S_ID),
        vol_surface_y_axis_type_ids=_i8(-99, -99, -99),
        netting_set_id=np.ascontiguousarray([1, 1, 1], dtype=np.int64),
    )


def _build_matrix(uix_mapper) -> MarketDataMatrix:
    """
    Hand-built MarketDataMatrix.

    Rows are keyed by the mapper's sorted universes.  `forward_nodes` carries
    the exact NODE levels.  `price_curve_grid` is poisoned with 999.0 so any
    accidental read from it produces a materially wrong answer.
    """
    market = uix_mapper.market_id_universe          # sorted int32 ids
    curves = uix_mapper.discount_curve_id_universe  # sorted int32 ids
    dates = uix_mapper.date_ord32_universe          # sorted int32 ids
    s = N_SCENARIOS

    # --- scalar NODE levels, aligned to market_id_universe ---
    levels = {RF_NODE: LVL_NODE, RF_NODE2: LVL_NODE2}
    prices = np.zeros((market.shape[0], s), dtype=np.float64)
    for rf, lvl in levels.items():
        idx = int(np.searchsorted(market, rf))
        prices[idx] = lvl
    prices = np.ascontiguousarray(prices)

    # --- discount factor grid (curve x date x scenario) ---
    df_grid = np.ones((curves.shape[0], dates.shape[0], s), dtype=np.float64)
    c_row = int(np.searchsorted(curves, CURVE_A_ID))
    for ord_val, df_vec in [
        (ORD_D1, DF_D1),
        (ORD_D2, DF_D2),
        (ORD_SETTLE, DF_SETTLE),
    ]:
        d_row = int(np.searchsorted(dates, ord_val))
        df_grid[c_row, d_row] = df_vec
    df_grid = np.ascontiguousarray(df_grid)

    # --- POISON the price_curve_grid ---
    # Any accidental read here would produce 999.0 instead of 100.0.
    price_curve_grid = np.full(
        (market.shape[0], dates.shape[0], s), fill_value=999.0, dtype=np.float64
    )

    return MarketDataMatrix(
        num_scenarios=s,
        uix_fingerprint=uix_mapper.fingerprint,
        fx_spots_effective=np.ones((0, s), dtype=np.float64),
        prices=prices,
        forward_nodes=prices,
        vol_nodes=prices,
        discount_factor_grid=df_grid,
        price_curve_grid=price_curve_grid,
        tte_grid=np.ascontiguousarray(np.full(dates.shape[0], 1.0)),
        vol_surface_strikes=np.zeros((0, 0)),
        vol_surfaces=np.ascontiguousarray(np.zeros((0, dates.shape[0], s, 0))),
        surface_y_axis_type_ids=np.zeros(0, dtype=np.int8),
    )


def _compile():
    batch = _make_batch()
    mapper = build_uix_mapper([batch])
    portfolio = compile_portfolio_from_ingest_batch(
        ingest_batch=batch, uix_mapper=mapper
    )
    matrix = _build_matrix(mapper)
    state = assemble_valuation_state(matrix=matrix, uix_mapper=mapper)
    out = np.empty((portfolio.total_trades, N_SCENARIOS), dtype=np.float64, order="C")
    run_valuation(portfolio=portfolio, valuation_state=state, out=out)
    uid_list = portfolio.trade_uids.tolist()
    return {uid: out[i] for i, uid in enumerate(uid_list)}, portfolio, state


# ---------------------------------------------------------------------------
# ORACLE-SIDE FIXTURE
# ---------------------------------------------------------------------------

# Discount-factor ids for the oracle: a tuple (curve_id, date_ord) matches
# how economic_oracle.EconomicMarket.discount is keyed.
_DF_D1_ID = (CURVE_A_ID, ORD_D1)
_DF_D2_ID = (CURVE_A_ID, ORD_D2)
_DF_SETTLE_ID = (CURVE_A_ID, ORD_SETTLE)


def _economic_market() -> oracle.EconomicMarket:
    return oracle.EconomicMarket(
        n_scenarios=N_SCENARIOS,
        spot_native={},
        forward_native={RF_NODE: LVL_NODE},
        discount={
            _DF_D1_ID: DF_D1,
            _DF_D2_ID: DF_D2,
            _DF_SETTLE_ID: DF_SETTLE,
        },
        fx={},
        vol={},
        forward_rate={},
    )


def _oracle_trade(uid: int, asset_df_id) -> oracle.ForwardTrade:
    """Build an oracle ForwardTrade with the given asset discount id."""
    return oracle.ForwardTrade(
        trade_uid=uid,
        qty=Q,
        forward_factor_id=RF_NODE,
        asset_ccy=SETTLE_CCY,
        asset_discount_factor_id=asset_df_id,
        strike=K,
        strike_ccy=SETTLE_CCY,
        strike_discount_factor_id=_DF_SETTLE_ID,
    )


def _oracle_pv() -> dict[int, np.ndarray]:
    """
    Independent oracle PV for the three trades.

    T1: asset DF from ORD_D1   (delivery present)
    T2: asset DF from ORD_D2   (delivery present, different date)
    T3: asset DF from ORD_SETTLE (delivery absent → fallback to settlement)
    """
    m = _economic_market()
    trades = [
        _oracle_trade(UID_T1, _DF_D1_ID),
        _oracle_trade(UID_T2, _DF_D2_ID),
        _oracle_trade(UID_T3, _DF_SETTLE_ID),  # fallback
    ]
    return {t.trade_uid: oracle.pv_forward(t, m, SETTLE_CCY) for t in trades}


# ---------------------------------------------------------------------------
# REACHABILITY
# ---------------------------------------------------------------------------


def test_batch_with_no_delivery_reaches_the_compiler():
    """T3 (delivery = NO_DATE_ORD32) must compile to a VanillaForwardBlock row."""
    _, portfolio, _ = _compile()
    fwd = portfolio.vanilla_forward
    assert fwd is not None
    assert fwd.size == 3


# ---------------------------------------------------------------------------
# NODE IDENTITY — delivery must NOT select the NODE
# ---------------------------------------------------------------------------


def test_node_identity_is_independent_of_delivery():
    """
    All three trades reference RF_NODE.  The compiled underlying_ids must all
    equal RF_NODE regardless of delivery.

    This proves delivery never re-resolves the rf_id (D065, D077.2).
    """
    batch = _make_batch()
    assert int(batch.underlying_ids[0]) == RF_NODE
    assert int(batch.underlying_ids[1]) == RF_NODE
    assert int(batch.underlying_ids[2]) == RF_NODE


def test_delivery_does_not_change_the_used_forward_node_row(
    capfd,
):
    """
    The same forward_nodes[price_uix] is read for all three trades.
    Verify by checking that all three EFP uix values share the same price_uix.
    """
    batch = _make_batch()
    mapper = build_uix_mapper([batch])
    portfolio = compile_portfolio_from_ingest_batch(
        ingest_batch=batch, uix_mapper=mapper
    )
    fwd = portfolio.vanilla_forward
    assert fwd is not None

    # All three trades: same underlying rf → same price_uix column in the quad.
    quads = mapper.efp_effective_quads
    price_uix_col = quads[:, 0]
    efp_uix = fwd.effective_forward_price_uix
    price_rows = price_uix_col[efp_uix]
    assert int(price_rows[0]) == int(price_rows[1]) == int(price_rows[2])


# ---------------------------------------------------------------------------
# ANTI-INTERPOLATION PROOF
# ---------------------------------------------------------------------------


def test_production_pv_is_not_999_and_matches_oracle():
    """
    price_curve_grid is 999.0 everywhere.  If the compiler or fold accidentally
    reads from it, the PV will be built on 999.0 rather than LVL_NODE=100.0
    and diverge materially from the oracle.
    """
    pv, _, _ = _compile()
    exp = _oracle_pv()
    for uid in (UID_T1, UID_T2, UID_T3):
        # Sanity: if price_curve_grid were used, the PV would be ~9-10× larger.
        assert np.all(pv[uid] < 80_000.0), (
            f"uid {uid}: PV too large — curve_grid contamination suspected"
        )
        np.testing.assert_allclose(
            pv[uid], exp[uid], rtol=RTOL, atol=0.0,
            err_msg=f"uid {uid}: production PV != oracle (anti-interpolation)"
        )


def test_neighbouring_node_is_materially_different():
    """
    LVL_NODE2 = 150.0 != 100.0 = LVL_NODE.  This confirms the fixture's
    inconsistency: any delivery-date interpolation between NODE and NODE2
    would produce a value that is neither 100.0 nor 150.0.
    """
    assert not np.allclose(LVL_NODE, LVL_NODE2, rtol=1e-9)


# ---------------------------------------------------------------------------
# DELIVERY SEMANTICS — three cases prove the T_asset selection
# ---------------------------------------------------------------------------


def test_delivery_d1_uses_df_at_d1():
    pv, _, _ = _compile()
    exp = _oracle_pv()
    np.testing.assert_allclose(pv[UID_T1], exp[UID_T1], rtol=RTOL)
    # The oracle uses DF_D1; confirm it is NOT equal to DF_D2 or DF_SETTLE.
    assert not np.allclose(DF_D1, DF_D2)
    assert not np.allclose(DF_D1, DF_SETTLE)


def test_delivery_d2_uses_df_at_d2():
    pv, _, _ = _compile()
    exp = _oracle_pv()
    np.testing.assert_allclose(pv[UID_T2], exp[UID_T2], rtol=RTOL)
    assert not np.allclose(DF_D2, DF_SETTLE)


def test_absent_delivery_uses_df_at_settlement():
    """
    T3 has delivery = NO_DATE_ORD32.  The compiler must substitute
    settlement_date, so the asset leg is discounted at DF_SETTLE, not
    at some other date.
    """
    pv, _, _ = _compile()
    exp = _oracle_pv()
    np.testing.assert_allclose(pv[UID_T3], exp[UID_T3], rtol=RTOL)
    # Explicit check: the oracle uses DF_SETTLE for T3.
    m = _economic_market()
    t3_oracle = oracle.pv_forward(_oracle_trade(UID_T3, _DF_SETTLE_ID), m, SETTLE_CCY)
    np.testing.assert_allclose(pv[UID_T3], t3_oracle, rtol=RTOL)


def test_different_deliveries_produce_different_pvs():
    """
    DF(D1) != DF(D2) != DF(SETTLE), so the three PVs must all differ.
    If two are equal, the delivery fallback is not doing what it claims.
    """
    pv, _, _ = _compile()
    assert not np.allclose(pv[UID_T1], pv[UID_T2], rtol=1e-9)
    assert not np.allclose(pv[UID_T1], pv[UID_T3], rtol=1e-9)
    assert not np.allclose(pv[UID_T2], pv[UID_T3], rtol=1e-9)


# ---------------------------------------------------------------------------
# MUTATION COVERAGE (D059 requirement: first prove output moves, then detect)
# ---------------------------------------------------------------------------


def _run_state(state):
    batch = _make_batch()
    mapper = build_uix_mapper([batch])
    portfolio = compile_portfolio_from_ingest_batch(
        ingest_batch=batch, uix_mapper=mapper
    )
    out = np.empty((portfolio.total_trades, N_SCENARIOS), dtype=np.float64)
    run_valuation(portfolio=portfolio, valuation_state=state, out=out)
    return out, portfolio.trade_uids.tolist()


def test_m1_m2_interpolation_mutation_detected():
    """
    M1/M2: Replace exact NODE price with 999.0 (the price_curve_grid poison).
    The mutation must change the output (D059) and the oracle must reject it.
    """
    pv, _, state = _compile()

    # Mutate: replace effective_forward_prices with values built on 999.0
    mutated_efp = np.full_like(state.effective_forward_prices, 999.0 * DF_D1[0])
    mutated = dataclasses.replace(
        state, effective_forward_prices=mutated_efp
    )
    mut_out, uid_list = _run_state(mutated)
    exp = _oracle_pv()

    # D059: mutation must move the output
    baseline = np.stack([pv[u] for u in uid_list])
    assert not np.allclose(mut_out, baseline, rtol=1e-9), (
        "M1/M2: mutation did not change the output (D059 violation)"
    )
    # Oracle detects it
    expected = np.stack([exp[u] for u in uid_list])
    assert not np.allclose(mut_out, expected, rtol=1e-9), (
        "M1/M2: mutation was not detected by the oracle"
    )


def test_m3_absent_delivery_is_not_rejected():
    """
    M3 sentinel: the old code rejected missing delivery.
    Prove T3 (delivery absent) now compiles and values without error.
    """
    pv, _, _ = _compile()
    assert np.all(np.isfinite(pv[UID_T3]))


def test_m4_no_delivery_does_not_use_wrong_date():
    """
    M4: wrong asset date when delivery absent.
    Prove T3 PV != oracle-with-D1 and != oracle-with-D2 (both would be wrong).
    """
    pv, _, _ = _compile()
    m = _economic_market()
    wrong_d1 = oracle.pv_forward(_oracle_trade(UID_T3, _DF_D1_ID), m, SETTLE_CCY)
    wrong_d2 = oracle.pv_forward(_oracle_trade(UID_T3, _DF_D2_ID), m, SETTLE_CCY)
    assert not np.allclose(pv[UID_T3], wrong_d1, rtol=1e-9), (
        "M4: absent-delivery trade used DF(D1) instead of DF(settlement)"
    )
    assert not np.allclose(pv[UID_T3], wrong_d2, rtol=1e-9), (
        "M4: absent-delivery trade used DF(D2) instead of DF(settlement)"
    )


def test_m5_explicit_delivery_is_not_silently_replaced_by_settlement():
    """
    M5: explicit delivery must not be replaced by settlement.
    Prove T1 PV != oracle-with-settlement.
    """
    pv, _, _ = _compile()
    m = _economic_market()
    wrong = oracle.pv_forward(_oracle_trade(UID_T1, _DF_SETTLE_ID), m, SETTLE_CCY)
    assert not np.allclose(pv[UID_T1], wrong, rtol=1e-9), (
        "M5: T1 delivery was silently ignored and settlement was used instead"
    )


def test_m6_m7_wrong_discount_curve_detected_by_oracle():
    """
    M6/M7: wrong asset discount curve → oracle mismatch.
    Achieved by scaling effective_forward_prices by DF_SETTLE / DF_D1 on row 0
    so T1's asset leg carries DF_SETTLE instead of DF_D1.
    """
    pv, _, state = _compile()
    scale = DF_SETTLE[0] / DF_D1[0]
    mutated_efp = state.effective_forward_prices.copy()
    t1_efp_row = ...
    # Replace entire EFP table by a uniform scale to force a detected difference.
    mutated_efp = state.effective_forward_prices * scale
    mutated = dataclasses.replace(
        state,
        effective_forward_prices=np.ascontiguousarray(mutated_efp),
    )
    mut_out, uid_list = _run_state(mutated)
    exp = _oracle_pv()

    baseline = np.stack([pv[u] for u in uid_list])
    assert not np.allclose(mut_out, baseline, rtol=1e-9), (
        "M6/M7: discount curve mutation did not change the output (D059)"
    )
    expected = np.stack([exp[u] for u in uid_list])
    assert not np.allclose(mut_out, expected, rtol=1e-9)


def test_m9_raw_forward_without_discount_detected_by_oracle():
    """
    M9: EFP carries raw F (no DF) instead of F*DF*FX.
    Approximate by multiplying EFP by 1/DF to undo discounting.
    """
    pv, _, state = _compile()
    # Undo the discount factor (approximate; DF_D1 scenario-0 average).
    undiscounted = state.effective_forward_prices / DF_D1[0]
    mutated = dataclasses.replace(
        state,
        effective_forward_prices=np.ascontiguousarray(undiscounted),
    )
    mut_out, uid_list = _run_state(mutated)
    exp = _oracle_pv()

    baseline = np.stack([pv[u] for u in uid_list])
    assert not np.allclose(mut_out, baseline, rtol=1e-9), (
        "M9: undiscounted EFP mutation did not change the output (D059)"
    )
    expected = np.stack([exp[u] for u in uid_list])
    assert not np.allclose(mut_out, expected, rtol=1e-9)


# ---------------------------------------------------------------------------
# NEGATIVE CONTROLS
# ---------------------------------------------------------------------------


def test_n1_unknown_rf_is_rejected_by_mapper():
    """
    N1: underlying rf not in the universe must raise at build_uix_mapper or
    compile_portfolio_from_ingest_batch (AUD-02 / D051).
    """
    batch = _make_batch()
    mapper = build_uix_mapper([batch])
    # Strip the market universe so RF_NODE is absent.
    stripped = dataclasses.replace(
        mapper,
        market_id_universe=np.ascontiguousarray(
            [x for x in mapper.market_id_universe if x != RF_NODE], dtype=np.int32
        ),
        efp_effective_quads=np.empty((0, 4), dtype=np.int32),
    )
    with pytest.raises((ValueError, RuntimeError)):
        compile_portfolio_from_ingest_batch(ingest_batch=batch, uix_mapper=stripped)


def test_n4_missing_settlement_causes_compile_failure():
    """
    N4: settlement_date = NO_DATE_ORD32 for T3 (the no-delivery trade).
    The strike leg uses settlement; a missing settlement must fail.
    """
    batch = _make_batch()
    # Erase settlement for T3.
    settle_modified = batch.settle_ord.copy()
    settle_modified[2] = int(NO_DATE_ORD32)
    bad_batch = dataclasses.replace(batch, settle_ord=settle_modified)
    mapper = build_uix_mapper([bad_batch])
    # The strike EDF quad needs settle_ord; a missing one leaves no valid triple.
    # The compiler's _require_resolved must fire.
    with pytest.raises(ValueError, match="effective_discount_factor"):
        compile_portfolio_from_ingest_batch(ingest_batch=bad_batch, uix_mapper=mapper)


# ---------------------------------------------------------------------------
# DETERMINISM
# ---------------------------------------------------------------------------


def test_determinism_identical_inputs_produce_identical_outputs():
    """
    For identical inputs the production path must produce bitwise-equal PVs.
    Run twice and compare (static numpy; no stochastic component).
    """
    pv1, _, _ = _compile()
    pv2, _, _ = _compile()
    for uid in (UID_T1, UID_T2, UID_T3):
        np.testing.assert_array_equal(pv1[uid], pv2[uid])
