# tests/solutions/sensitivity_pnl/test_taylor_contract.py
"""
P1-A: Taylor representation integrity (D081 §4-§10).

Every expected value below comes from an independent analytical oracle -- a
scalar function with hand-derived derivatives -- never from the production
transformation or compiler. Each contract has at least one negative test that
proves a structurally valid but mathematically wrong representation is
REJECTED rather than priced.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import polars as pl
import pytest

from riskquant.sensitivities.result import BlockSensitivities
from riskquant.sensitivity_pnl.conventions import DisplacementConvention
from riskquant.solutions.sensitivity_pnl.api import (
    FactorAxis,
    SensitivityBookSpec,
    SensitivityPnlCalculator,
    build_sensitivity_book_spec,
    derive_taylor_declaration,
)
from riskquant.solutions.sensitivity_pnl.taylor_contract import (
    TaylorBaseBinding,
    TaylorContractError,
    TaylorDeclaration,
    TaylorOrder,
    base_levels_digest,
    base_settlement_values_digest,
    static_config_content_digest,
    valuation_contract_fingerprint,
    verify_base_binding,
)

X, Y = 11, 22  # factor ids
_EPS32 = 2.0**-24  # D081 §12: float32 displacement, gross-exposure error bound


# ----------------------------------------------------------------- helpers


def _own_axis_spec(*, sensitivities, **kwargs):
    """The book on its OWN referenced factors, declared explicitly (F-1): these
    P1-A tests price no production run axis."""
    return build_sensitivity_book_spec(
        sensitivities=sensitivities,
        required_factor_ids=sensitivities.active_factor_rf_ids,
        **kwargs,
    )


def _binding(**overrides) -> TaylorBaseBinding:
    fields = dict(
        as_of_ord32=20_000,
        base_levels_digest=base_levels_digest(
            factor_ids=np.array([X, Y], np.int32), base_levels=np.array([2.0, 0.0])
        ),
        discounting_mode="NONE",
        static_config_digest=static_config_content_digest({}),
        valuation_contract_fingerprint=valuation_contract_fingerprint(),
        base_settlement_values_digest=base_settlement_values_digest(
            position_uids=[1], values=np.array([0.0])
        ),
    )
    fields.update(overrides)
    return TaylorBaseBinding(**fields)


def _spec(rows, *, order, deps, cross=None, uids=(1,), zero=(), axis=(X, Y), conv=None):
    conv = conv or dict.fromkeys(axis, DisplacementConvention.ABSOLUTE)
    sens = pl.DataFrame(
        rows,
        schema=["position_id", "factor_id", "first_order", "second_order"],
        orient="row",
    )
    cross_df = (
        None
        if cross is None
        else pl.DataFrame(
            cross,
            schema=["position_id", "factor_id_i", "factor_id_j", "cross_second_order"],
            orient="row",
        )
    )
    meta = pl.DataFrame({
        "factor_id": list(axis),
        "displacement_convention_id": [int(conv[f]) for f in axis],
        "quote_base": [1.0] * len(axis),
    })
    return SensitivityBookSpec(
        sensitivities=sens,
        factor_meta=meta,
        factor_axis=FactorAxis.from_keys(list(axis)),
        position_uids=list(uids),
        cross_sensitivities=cross_df,
        taylor_declaration=TaylorDeclaration(
            order=order,
            dependencies=pl.DataFrame(
                deps, schema=["position_id", "factor_id"], orient="row"
            ),
            zero_exposure_position_uids=tuple(zero),
            base_binding=_binding(),
        ),
    )


def _pnl(calc, base, shocked):
    market = calc.axis.levels(
        base_levels=np.asarray(base, dtype=np.float64),
        shocked_levels=np.asarray(shocked, dtype=np.float64),
    )
    return np.asarray(calc.run(market).pnl, dtype=np.float64)


def _block(d1, d2, *, rf=X):
    return BlockSensitivities(
        position_uids=np.array([1], np.int64),
        factor_rf_ids=np.array([rf], np.int32),
        first_order=np.array([d1], np.float64),
        second_order=np.array([d2], np.float64),
        cross_position_uids=np.empty(0, np.int64),
        cross_factor_rf_ids_i=np.empty(0, np.int32),
        cross_factor_rf_ids_j=np.empty(0, np.int32),
        cross_second_order=np.empty(0, np.float64),
    )


def _lv(mapping):
    """Authoritative identity-keyed levels digest of a {factor: level} base."""
    return base_levels_digest(
        factor_ids=np.array(list(mapping), np.int32),
        base_levels=np.array(list(mapping.values()), np.float64),
    )


BOUND = {X: 2.0, Y: 0.0}  # the base `_binding()` is bound to


# Oracle for B/C/G: V(x) = x^3 + 2x^2 + 7x at x0 = 2 -> V_x = 27, V_xx = 16.
X0, VX, VXX = 2.0, 27.0, 16.0


# ================================================================ coordinates
def test_a_relabelling_absolute_as_log_is_refused():
    with pytest.raises(ValueError, match="relabelling"):
        _own_axis_spec(
            sensitivities=_block(VX, VXX),
            displacement_convention=DisplacementConvention.LOG,
        )


def test_b_relative_transformation_is_exact():
    spec = _own_axis_spec(
        sensitivities=_block(VX, VXX),
        displacement_convention=DisplacementConvention.RELATIVE,
        base_levels_by_factor=BOUND,
        base_binding=_binding(),
    )
    row = spec.sensitivities.row(0, named=True)
    assert row["first_order"] == pytest.approx(VX * X0)  # 54
    assert row["second_order"] == pytest.approx(VXX * X0**2)  # 64


def test_c_log_transformation_carries_the_first_derivative_term():
    spec = _own_axis_spec(
        sensitivities=_block(VX, VXX),
        displacement_convention=DisplacementConvention.LOG,
        base_levels_by_factor=BOUND,
        base_binding=_binding(),
    )
    row = spec.sensitivities.row(0, named=True)
    assert row["first_order"] == pytest.approx(54.0)
    assert row["second_order"] == pytest.approx(VXX * X0**2 + VX * X0)  # 118, not 64
    assert row["second_order"] != pytest.approx(VXX * X0**2)


def test_c_log_taylor_matches_the_exact_second_order_expansion_in_log_space():
    """ln-space second-order Taylor of V(x0*e^y) at a small y, via the real engine."""
    spec = _own_axis_spec(
        sensitivities=_block(VX, VXX),
        displacement_convention=DisplacementConvention.LOG,
        base_levels_by_factor=BOUND,
        base_binding=_binding(),
    )
    calc = SensitivityPnlCalculator.compile(
        SensitivityBookSpec(
            sensitivities=spec.sensitivities,
            factor_meta=spec.factor_meta,
            factor_axis=FactorAxis.from_keys(
                pl.Series("factor_id", [X], dtype=pl.Int32)
            ),
            position_uids=[1],
        )
    )
    y = 0.01
    produced = _pnl(calc, [X0], [[X0 * np.exp(y)]])[0, 0]
    oracle = 54.0 * y + 0.5 * 118.0 * y * y  # V_y y + 1/2 V_yy y^2
    assert produced == pytest.approx(oracle, rel=1e-6)


def test_d_relative_zero_base_fails_closed():
    with pytest.raises(ValueError, match="non-zero base"):
        _own_axis_spec(
            sensitivities=_block(VX, VXX),
            displacement_convention=DisplacementConvention.RELATIVE,
            base_levels_by_factor={X: 0.0, Y: 0.0},
            base_binding=_binding(base_levels_digest=_lv({X: 0.0, Y: 0.0})),
        )


@pytest.mark.parametrize("x0", [0.0, -1.5])
def test_e_log_non_positive_base_fails_closed(x0):
    with pytest.raises(ValueError, match="strictly positive"):
        _own_axis_spec(
            sensitivities=_block(VX, VXX),
            displacement_convention=DisplacementConvention.LOG,
            base_levels_by_factor={X: x0, Y: 0.0},
            base_binding=_binding(base_levels_digest=_lv({X: x0, Y: 0.0})),
        )


def test_f_log_non_positive_scenario_level_fails_closed():
    spec = _own_axis_spec(
        sensitivities=_block(VX, VXX),
        displacement_convention=DisplacementConvention.LOG,
        base_levels_by_factor=BOUND,
        base_binding=_binding(),
    )
    calc = SensitivityPnlCalculator.compile(
        SensitivityBookSpec(
            sensitivities=spec.sensitivities,
            factor_meta=spec.factor_meta,
            factor_axis=FactorAxis.from_keys(
                pl.Series("factor_id", [X], dtype=pl.Int32)
            ),
            position_uids=[1],
        )
    )
    with pytest.raises(Exception, match=r"(?i)positive"):
        _pnl(calc, [X0], [[-0.5]])


def test_g_quote_base_is_a_linear_rescale_verified_against_an_oracle():
    """Greeks per QUOTED unit (1/b natural units) with quote_base = b."""
    b, a, c, dx = 10_000.0, 3.0, 400.0, 0.002
    spec = _own_axis_spec(sensitivities=_block(a / b, c / b**2), quote_base=b)
    calc = SensitivityPnlCalculator.compile(spec)
    produced = _pnl(calc, [1.0], [[1.0 + dx]])[0, 0]
    exact = a * dx + 0.5 * c * dx * dx
    assert abs(produced - exact) <= _EPS32 * (
        abs(a * dx) + abs(c * dx * dx) + abs(exact)
    )
    # the rescale direction matters: the wrong one is off by b^2
    assert abs(produced - (a / b**2) * dx) > 1e-3


# ============================================================ base identity
@pytest.mark.parametrize(
    "field, value",
    [
        ("base_levels_digest", "0" * 64),  # H: wrong base levels
        ("as_of_ord32", 20_001),  # I: wrong as-of
        ("static_config_digest", "1" * 64),  # J: static config
        ("valuation_contract_fingerprint", "2" * 64),  # K: valuation contract
        ("discounting_mode", "CURVE"),
        ("base_settlement_values_digest", "3" * 64),  # wrong V_0
    ],
)
def test_hijk_any_base_identity_difference_fails_closed(field, value):
    with pytest.raises(TaylorContractError, match=field):
        verify_base_binding(
            representation=_binding(), expected=_binding(**{field: value})
        )


def test_j_static_config_digest_changes_with_one_semantic_field():
    from riskquant.config.static_config.compile import (
        compile_risk_static_config_from_dict_of_dfs,
    )
    from riskquant.ingest_fixtures.config.risk_static_config.canonical_risk_static_config import (
        CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS,
    )

    dfs = dict(CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS)
    a = static_config_content_digest(compile_risk_static_config_from_dict_of_dfs(dfs))
    cm = dfs["curve_meta"]
    dfs["curve_meta"] = cm.with_columns(
        pl
        .when(pl.col("curve_name") == "USD_SOFR_OIS")
        .then(pl.lit(1))
        .otherwise(pl.col("spot_lag_days"))
        .alias("spot_lag_days")
    )
    b = static_config_content_digest(compile_risk_static_config_from_dict_of_dfs(dfs))
    assert a != b


def test_k_valuation_contract_fingerprint_changes_with_a_constant(monkeypatch):
    from riskquant import constants

    before = valuation_contract_fingerprint()
    monkeypatch.setattr(constants, "TTE_FLOOR", constants.TTE_FLOOR * 2.0)
    assert valuation_contract_fingerprint() != before


def test_s_base_levels_digest_is_keyed_by_identity_not_position():
    ids, lv = np.array([X, Y], np.int32), np.array([2.0, 5.0])
    same = base_levels_digest(factor_ids=ids[::-1].copy(), base_levels=lv[::-1].copy())
    swapped = base_levels_digest(factor_ids=ids, base_levels=lv[::-1].copy())
    assert base_levels_digest(factor_ids=ids, base_levels=lv) == same
    assert base_levels_digest(factor_ids=ids, base_levels=lv) != swapped


# ============================================= F(p), completeness, sparsity
XY_ROWS = [(1, X, 0.0, 0.0), (1, Y, 2.0, 0.0)]  # V = x*y at x0 = 2, y0 = 0
XY_DEPS = [(1, X), (1, Y)]


def test_m_zero_first_derivative_does_not_remove_the_dependency():
    calc = SensitivityPnlCalculator.compile(
        _spec(
            XY_ROWS,
            order=TaylorOrder.SECOND_ORDER_FULL,
            deps=XY_DEPS,
            cross=[(1, X, Y, 1.0)],
        )
    )
    assert calc.taylor_declaration is not None


def test_n_missing_derivative_is_never_a_zero():
    with pytest.raises(TaylorContractError, match="MISSING"):
        SensitivityPnlCalculator.compile(
            _spec(XY_ROWS[1:], order=TaylorOrder.FIRST_ORDER, deps=XY_DEPS)
        )


def test_derivative_outside_f_of_p_is_a_different_model():
    with pytest.raises(TaylorContractError, match="outside"):
        SensitivityPnlCalculator.compile(
            _spec(XY_ROWS, order=TaylorOrder.SECOND_ORDER_TRUNCATED, deps=[(1, Y)])
        )


def test_o_truncated_accepts_absent_cross_only_by_declaration():
    SensitivityPnlCalculator.compile(
        _spec(XY_ROWS, order=TaylorOrder.SECOND_ORDER_TRUNCATED, deps=XY_DEPS)
    )
    with pytest.raises(TaylorContractError, match="cross derivatives MISSING"):
        SensitivityPnlCalculator.compile(
            _spec(XY_ROWS, order=TaylorOrder.SECOND_ORDER_FULL, deps=XY_DEPS)
        )
    with pytest.raises(TaylorContractError, match="omits cross terms by declaration"):
        SensitivityPnlCalculator.compile(
            _spec(
                XY_ROWS,
                order=TaylorOrder.SECOND_ORDER_TRUNCATED,
                deps=XY_DEPS,
                cross=[(1, X, Y, 1.0)],
            )
        )


def test_p_known_zero_cross_is_accepted_and_distinct_from_absent():
    # V = x + y: every second derivative is analytically zero
    rows = [(1, X, 1.0, 0.0), (1, Y, 1.0, 0.0)]
    SensitivityPnlCalculator.compile(
        _spec(
            rows,
            order=TaylorOrder.SECOND_ORDER_FULL,
            deps=XY_DEPS,
            cross=[(1, X, Y, 0.0)],
        )
    )


def test_q_cross_term_counts_exactly_once():
    calc = SensitivityPnlCalculator.compile(
        _spec(
            XY_ROWS,
            order=TaylorOrder.SECOND_ORDER_FULL,
            deps=XY_DEPS,
            cross=[(1, X, Y, 1.0)],
        )
    )
    dx, dy = 0.3, 0.05
    produced = _pnl(calc, [2.0, 0.0], [[2.0 + dx], [0.0 + dy]])[0, 0]
    exact = (2.0 + dx) * dy - 2.0 * 0.0  # V is bilinear: Taylor is exact
    gross = abs(2.0 * dy) + 2.0 * abs(dx * dy)
    assert abs(produced - exact) <= _EPS32 * (gross + abs(exact))
    assert produced != pytest.approx(0.0 * dx + 2.0 * dy + 2.0 * dx * dy)  # doubled
    assert produced != pytest.approx(0.0 * dx + 2.0 * dy + 0.5 * dx * dy)  # halved


def test_q_reversed_duplicate_cross_pair_fails_closed():
    with pytest.raises(TaylorContractError, match="duplicated pair"):
        SensitivityPnlCalculator.compile(
            _spec(
                XY_ROWS,
                order=TaylorOrder.SECOND_ORDER_FULL,
                deps=XY_DEPS,
                cross=[(1, X, Y, 1.0), (1, Y, X, 1.0)],
            )
        )


def test_first_order_with_second_order_content_fails_closed():
    with pytest.raises(TaylorContractError, match="FIRST_ORDER"):
        SensitivityPnlCalculator.compile(
            _spec([(1, X, 1.0, 3.0)], order=TaylorOrder.FIRST_ORDER, deps=[(1, X)])
        )


def test_zero_exposure_must_be_declared_explicitly():
    rows = [(1, X, 1.0, 0.0)]
    with pytest.raises(TaylorContractError, match="absence is not zero exposure"):
        SensitivityPnlCalculator.compile(
            _spec(rows, order=TaylorOrder.FIRST_ORDER, deps=[(1, X)], uids=(1, 2))
        )
    SensitivityPnlCalculator.compile(
        _spec(
            rows, order=TaylorOrder.FIRST_ORDER, deps=[(1, X)], uids=(1, 2), zero=(2,)
        )
    )


# ======================================================= invariance, DERIVED
def test_r_batch_invariance():
    calc = SensitivityPnlCalculator.compile(
        _spec(
            XY_ROWS,
            order=TaylorOrder.SECOND_ORDER_FULL,
            deps=XY_DEPS,
            cross=[(1, X, Y, 1.0)],
        )
    )
    shocked = np.array([[2.1, 1.9, 2.5, 2.0, 1.7], [0.1, -0.2, 0.05, 0.0, 0.3]])
    whole = _pnl(calc, [2.0, 0.0], shocked)
    parts = np.hstack([
        _pnl(calc, [2.0, 0.0], shocked[:, i : i + 2]) for i in range(0, 5, 2)
    ])
    np.testing.assert_array_equal(whole, parts)


def test_s_factor_order_invariance():
    a = SensitivityPnlCalculator.compile(
        _spec(
            XY_ROWS,
            order=TaylorOrder.SECOND_ORDER_FULL,
            deps=XY_DEPS,
            cross=[(1, X, Y, 1.0)],
        )
    )
    b = SensitivityPnlCalculator.compile(
        _spec(
            XY_ROWS[::-1],
            order=TaylorOrder.SECOND_ORDER_FULL,
            deps=XY_DEPS[::-1],
            cross=[(1, Y, X, 1.0)],
            axis=(Y, X),
        )
    )
    pa = _pnl(a, [2.0, 0.0], [[2.3], [0.1]])
    pb = _pnl(b, [0.0, 2.0], [[0.1], [2.3]])
    assert pa[0, 0] == pytest.approx(pb[0, 0], rel=1e-15)


def test_derived_declaration_keeps_zero_greek_dependencies():
    """A sensitizer row with a zero Greek still declares its factor in F(p)."""
    blk = BlockSensitivities(
        position_uids=np.array([1, 1], np.int64),
        factor_rf_ids=np.array([X, Y], np.int32),
        first_order=np.array([0.0, 2.0]),
        second_order=np.array([0.0, 0.0]),
        cross_position_uids=np.array([1], np.int64),
        cross_factor_rf_ids_i=np.array([X], np.int32),
        cross_factor_rf_ids_j=np.array([Y], np.int32),
        cross_second_order=np.array([1.0]),
    )
    decl = derive_taylor_declaration(
        sensitivities=blk, order=TaylorOrder.SECOND_ORDER_FULL, base_binding=_binding()
    )
    assert set(decl.dependencies.iter_rows()) == {(1, X), (1, Y)}
    spec = dataclasses.replace(
        _own_axis_spec(sensitivities=blk), taylor_declaration=decl
    )
    SensitivityPnlCalculator.compile(spec)


# ============================================ P1-A closure: transform base = bound base
@pytest.mark.parametrize(
    "convention", [DisplacementConvention.RELATIVE, DisplacementConvention.LOG]
)
def test_coordinate_transform_rejects_base_mismatch(convention):
    """
    Bound at base A (x0 = 2), transformed at base B (x0 = 3). A wrong-base LOG
    transform would give V_y = 81, V_yy = 225 -- plausible numbers -- so the
    OPERATION itself must be refused, before any coefficient is produced.
    """
    produced = None
    with pytest.raises(TaylorContractError, match="transformation base differs"):
        produced = _own_axis_spec(
            sensitivities=_block(VX, VXX),
            displacement_convention=convention,
            base_levels_by_factor={X: 3.0, Y: 0.0},
            base_binding=_binding(),
        )
    assert produced is None


def test_coordinate_transform_accepts_the_bound_base_with_oracle_values():
    spec = _own_axis_spec(
        sensitivities=_block(VX, VXX),
        displacement_convention=DisplacementConvention.LOG,
        base_levels_by_factor=BOUND,
        base_binding=_binding(),
    )
    row = spec.sensitivities.row(0, named=True)
    assert row["first_order"] == pytest.approx(VX * X0)
    assert row["second_order"] == pytest.approx(VXX * X0**2 + VX * X0)
    assert spec.coordinate_transform_base_digest == _binding().base_levels_digest


def test_coordinate_transform_base_is_identity_keyed_not_positional():
    reordered = {Y: 0.0, X: 2.0}  # same factor -> level associations
    _own_axis_spec(
        sensitivities=_block(VX, VXX),
        displacement_convention=DisplacementConvention.LOG,
        base_levels_by_factor=reordered,
        base_binding=_binding(),
    )
    swapped = {X: 0.0, Y: 2.0}  # levels exchanged between identities
    with pytest.raises(TaylorContractError, match="transformation base differs"):
        _own_axis_spec(
            sensitivities=_block(VX, VXX),
            displacement_convention=DisplacementConvention.RELATIVE,
            base_levels_by_factor=swapped,
            base_binding=_binding(),
        )


def test_transformed_book_cannot_be_rebound_to_another_base():
    """Transform at A, then attach a declaration bound at B: compile refuses."""
    spec = _own_axis_spec(
        sensitivities=_block(VX, VXX),
        displacement_convention=DisplacementConvention.LOG,
        base_levels_by_factor=BOUND,
        base_binding=_binding(),
    )
    at_b = _binding(base_levels_digest=_lv({X: 3.0, Y: 0.0}))
    rebound = dataclasses.replace(
        spec,
        factor_axis=FactorAxis.from_keys(pl.Series("factor_id", [X], dtype=pl.Int32)),
        position_uids=[1],
        taylor_declaration=TaylorDeclaration(
            order=TaylorOrder.SECOND_ORDER_TRUNCATED,
            dependencies=pl.DataFrame(
                {"position_id": [1], "factor_id": [X]},
                schema={"position_id": pl.Int64, "factor_id": pl.Int32},
            ),
            zero_exposure_position_uids=(),
            base_binding=at_b,
        ),
    )
    with pytest.raises(TaylorContractError, match="transformation base differs"):
        SensitivityPnlCalculator.compile(rebound)


def test_transform_without_a_binding_is_refused():
    with pytest.raises(TaylorContractError, match="must be bound"):
        _own_axis_spec(
            sensitivities=_block(VX, VXX),
            displacement_convention=DisplacementConvention.LOG,
            base_levels_by_factor=BOUND,
        )


def test_native_absolute_needs_no_transformation_base():
    """Rule D: ABSOLUTE is the sensitizer's native coordinate, so no
    base-dependent operation happens and no transformation base is recorded."""
    spec = _own_axis_spec(sensitivities=_block(VX, VXX))
    assert spec.coordinate_transform_base_digest is None
    row = spec.sensitivities.row(0, named=True)
    assert (row["first_order"], row["second_order"]) == (VX, VXX)
