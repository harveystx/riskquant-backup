# tests/solutions/risk_run/test_axis_input_contract.py
"""
PR-04 hardening: the run-axis input contract, proven adversarially.

Classification of each invariant (header of every test):

    EXACT          integer / order / identity / bitwise-float equality
    APPROXIMATE    floating point with a tolerance derived from the dtype
    STATISTICAL    a sampling property with a stated acceptance bound

Oracles never call `riskquant.reporting`, a scenario generator or a valuation
kernel to build an expected value. Repository contracts used as oracle inputs:
scenario shocks are float32 (D026 F7), the PnlGrid stores dV in float32, and
the parametric RNG draws all base normals of the run up front from
`numpy.random.default_rng(seed)` in C order (D021).
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_reporting_axis import (
    AS_OF,
    NAME,
    RF,
    RSC,
    World,
    capture,  # noqa: F401  (fixture)
    spot,
)

import riskquant.solutions.risk_run.api as api
from riskquant.constants import INVALID_UIX, NO_FX_RF_REQUIRED
from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.config.proxy_model.schema_objects import NodeProxyTerm
from riskquant.market_data.risk_factors.shift_types import SHIFT_LOG_RETURN_ID, ShiftType
from riskquant.simlab.scenarios.parametric_data import (
    FORECAST_PERIOD_UNIT_BUSINESS_DAY_ID,
    ParametricRiskData,
)
from riskquant.solutions.risk_run.api import (
    derive_run_factor_axis,
    run_scenario_var_es,
)
from riskquant.solutions.risk_run.schemas import (
    CompositionContractError,
    ReportingSpec,
    RunFactorAxis,
)
from riskquant.solutions.scenario_models.api import (
    parse_scenario_model_config,
)
from riskquant.solutions.sensitivity_pnl.api import (
    SensitivityPnlCalculator,
)

_AXIS_INPUTS = (
    "required_factor_ids",
    "required_shift_type_id",
    "base_levels",
)

# ---- worlds --------------------------------------------------------------------------

# USD asset settled in USD, reported in EUR: USDEUR is a reporting-only suffix.
SUFFIX_ROWS = [spot(1, "EQ_USD", "USD", 3.0), spot(2, "EQ_USD", "USD", -1.0)]
SUFFIX_BASE = {"EQ_USD": 100.0, "USDEUR": 0.90}
_R = np.random.default_rng(23)
SUFFIX_HIST = {
    "EQ_USD": 100.0 * np.exp(np.cumsum(np.r_[0.0, _R.normal(0.0, 0.02, 11)])),
    "USDEUR": 0.90 * np.exp(np.cumsum(np.r_[0.0, _R.normal(0.0, 0.01, 11)])),
}


def _suffix_world(base: dict[str, float] = SUFFIX_BASE) -> World:
    w = World(SUFFIX_ROWS, "EUR", base, SUFFIX_HIST)
    assert w.names == ["EQ_USD", "USDEUR"] and w.axis.num_valuation_factors == 1
    return w


def _matrix(hist: dict[str, Any]) -> RiskFactorMatrix:
    """Scenario data with exactly the named columns (ids ascending)."""
    ids = sorted(RF[n] for n in hist)
    values = np.column_stack([np.asarray(hist[NAME[i]], dtype=np.float64) for i in ids])
    t = values.shape[0]
    return RiskFactorMatrix(
        dates=np.arange(AS_OF - t + 1, AS_OF + 1, dtype=np.int32),
        factor_ids=np.array(ids, dtype=np.int32),
        values=np.asfortranarray(values),
    )


@pytest.fixture
def boundary(monkeypatch):
    """Every numerical stage after composition; entering any one is a defect."""
    reached: list[str] = []

    def stop(name):
        def _stop(*a: object, **k: object) -> None:
            reached.append(name)
            raise AssertionError(f"reached {name}")

        return _stop

    for name in (
        "build_run_manifest",
        "build_scenario_generator",
        "_value_positions",
        "apply_linear_terms_to_base_levels",
        "allocate_pnl_grid",
        "convert_scenario_pnl_to_reporting_currency",
        "compute_var_es",
    ):
        monkeypatch.setattr(api, name, stop(name))
    monkeypatch.setattr(
        SensitivityPnlCalculator, "run_into_grid", stop("run_into_grid")
    )
    return reached


def _refused(kw: dict[str, Any], match: str, reached: list[str]) -> None:
    with pytest.raises(CompositionContractError, match=match):
        run_scenario_var_es(**kw)
    assert reached == [], f"numerical stage entered before refusal: {reached}"


# ---- 1. exact dtype / dimensionality / layout of the five axis inputs (EXACT) --------


@pytest.mark.parametrize(
    "name, wrong",
    [
        ("required_factor_ids", np.int64),
        ("required_shift_type_id", np.int32),
        ("base_levels", np.float32),
    ],
)
def test_wrong_dtype_with_valid_values_is_refused_at_composition(
    boundary, name, wrong
) -> None:
    """EXACT. Values and shape are otherwise valid; only the dtype differs
    (dyadic base levels, so the float32 copy holds the same values)."""
    w = _suffix_world({"EQ_USD": 100.0, "USDEUR": 0.875})
    kw = w.inputs()
    kw[name] = kw[name].astype(wrong)
    assert np.array_equal(kw[name], w.inputs()[name])  # same values: dtype only
    _refused(kw, f"{name} must have dtype", boundary)


@pytest.mark.parametrize("name", _AXIS_INPUTS)
def test_two_dimensional_input_is_refused_at_composition(boundary, name) -> None:
    """EXACT. A (K, 1) column of the right dtype is not a (K,) axis vector."""
    w = _suffix_world()
    kw = w.inputs()
    kw[name] = np.ascontiguousarray(kw[name][:, None])
    _refused(kw, f"{name} must be 1-D", boundary)


@pytest.mark.parametrize("name", _AXIS_INPUTS)
def test_strided_input_is_refused_not_relaid(boundary, name) -> None:
    """EXACT. A strided view of the right values and dtype is not re-laid out."""
    w = _suffix_world()
    kw = w.inputs()
    doubled = np.repeat(kw[name], 2)
    kw[name] = doubled[::2]
    assert np.array_equal(kw[name], w.inputs()[name])
    assert not kw[name].flags.c_contiguous
    _refused(kw, f"{name} must be C-contiguous", boundary)


def test_non_array_input_is_refused(boundary) -> None:
    w = _suffix_world()
    kw = w.inputs()
    kw["base_levels"] = kw["base_levels"].tolist()
    _refused(kw, "base_levels must be a numpy.ndarray", boundary)


def test_run_takes_owned_copies_and_accepts_the_read_only_descriptor(
    monkeypatch,
) -> None:
    """EXACT. The read-only `RunFactorAxis` arrays go into the run as-is; the
    arrays reaching the shock kernel are the run's own writable copies."""
    w = _suffix_world()
    kw = w.inputs(
        required_factor_ids=w.axis.factor_ids,
        required_shift_type_id=w.axis.shift_type_ids,
    )
    assert not kw["required_shift_type_id"].flags.writeable
    seen: dict[str, Any] = {}
    real = api.apply_linear_terms_to_base_levels

    def spy(**k: Any) -> None:
        seen.setdefault("base", k["base_levels"])
        seen.setdefault("shift", k["target_shift_types"])
        real(**k)

    monkeypatch.setattr(api, "apply_linear_terms_to_base_levels", spy)
    run = run_scenario_var_es(**kw)
    assert np.isfinite(run.var).all()
    assert not np.shares_memory(seen["base"], kw["base_levels"])
    # kappa reaches the kernel from the run's own lowered operator (D088),
    # equal to, but never aliasing, the caller's read-only axis descriptor.
    assert not np.shares_memory(seen["shift"], w.axis.shift_type_ids)
    assert np.array_equal(seen["shift"], w.axis.shift_type_ids)


# ---- 2. RunFactorAxis ownership and immutability (EXACT) -----------------------------


def test_run_factor_axis_owns_read_only_exact_arrays() -> None:
    ids = np.array([7, 31], dtype=np.int32)
    shifts = np.array([1, 1], dtype=np.int8)
    axis = RunFactorAxis(factor_ids=ids, shift_type_ids=shifts, num_valuation_factors=1)
    # 1. ownership: no alias of the constructor arguments
    assert not np.shares_memory(axis.factor_ids, ids)
    assert not np.shares_memory(axis.shift_type_ids, shifts)
    # 2. mutating the sources afterwards cannot alter the axis
    ids[:] = 99
    shifts[:] = 5
    assert axis.factor_ids.tolist() == [7, 31]
    assert axis.shift_type_ids.tolist() == [1, 1]
    # 3./4. the stored arrays are not writable
    for arr in (axis.factor_ids, axis.shift_type_ids):
        assert not arr.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            arr[:] = 0
    # 5. frozen fields
    with pytest.raises(dataclasses.FrozenInstanceError):
        axis.factor_ids = np.array([1, 2], dtype=np.int32)  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        axis.num_valuation_factors = 2  # type: ignore[misc]
    # 6./7. layout and exact dtypes
    assert axis.factor_ids.flags.c_contiguous and axis.shift_type_ids.flags.c_contiguous
    assert axis.factor_ids.dtype == np.int32 and axis.shift_type_ids.dtype == np.int8
    assert axis.reporting_factor_ids.tolist() == [31]
    assert not axis.reporting_factor_ids.flags.writeable


def test_the_derived_axis_is_read_only_too() -> None:
    w = _suffix_world()
    assert not w.axis.factor_ids.flags.writeable
    assert not w.axis.shift_type_ids.flags.writeable


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        ({"factor_ids": np.array([7, 31], np.int64)}, TypeError, "int32"),
        ({"shift_type_ids": np.array([1, 1], np.int32)}, TypeError, "int8"),
        ({"factor_ids": np.array([[7, 31]], np.int32)}, TypeError, "1-D"),
        ({"num_valuation_factors": True}, TypeError, "int"),
        ({"num_valuation_factors": 3}, ValueError, r"\[0, K\]"),
        ({"factor_ids": np.array([7, 7], np.int32)}, ValueError, "repeat"),
        ({"shift_type_ids": np.array([1], np.int8)}, ValueError, "row-aligned"),
    ],
)
def test_invalid_run_factor_axis_is_refused(kwargs, error, match) -> None:
    base = {
        "factor_ids": np.array([7, 31], np.int32),
        "shift_type_ids": np.array([1, 1], np.int8),
        "num_valuation_factors": 1,
    }
    with pytest.raises(error, match=match):
        RunFactorAxis(**{**base, **kwargs})


# ---- 3. D088 authored proxy terms: the model selects the columns (EXACT) --------------

# EQ_GBP has NO history of its own: its scenario shock is an authored Proxy V1
# term (D088). Reported in EUR: GBP -> EUR reads EURGBP reciprocally, USD -> EUR
# reads USDEUR directly.
PROXY_ROWS = [spot(1, "EQ_USD", "USD", 5.0), spot(2, "EQ_GBP", "GBP", 4.0)]
PROXY_BASE = {"EQ_USD": 212.37, "EQ_GBP": 148.91, "USDEUR": 0.9259, "EURGBP": 0.8561}
_P = np.random.default_rng(31)
PROXY_HIST = {
    n: v * np.exp(np.cumsum(np.r_[0.0, _P.normal(0.0, 0.02, 15)]))
    for n, v in {**PROXY_BASE, "EQ_EUR": 53.12}.items()
    if n != "EQ_GBP"
}


# World() needs a column per axis factor for its default data; every run
# below replaces it with `_matrix(PROXY_HIST)`, which has NO EQ_GBP column.
_WORLD_HIST = {**PROXY_HIST, "EQ_GBP": PROXY_HIST["EQ_USD"]}


def _term(target: str, driver: str, beta: float = 1.0) -> NodeProxyTerm:
    return NodeProxyTerm(
        target_name=target,
        driver_name=driver,
        driver_chart=ShiftType.LOG_RETURN,
        loading=beta,
    )


def _proxy_world(proxy: str) -> tuple[World, dict[str, Any]]:
    w = World(PROXY_ROWS, "EUR", PROXY_BASE, _WORLD_HIST)
    kw = w.inputs(data=_matrix(PROXY_HIST))
    kw["proxy_terms"] = (_term("EQ_GBP", proxy),)
    return w, kw


def test_a_proxied_row_reads_its_proxy_shock(capture) -> None:
    """EXACT (float32 shock). Row EQ_GBP moves with EQ_EUR's historical return
    (authored term, beta = 1); every other row has its identity row."""
    w, kw = _proxy_world("EQ_EUR")
    run_scenario_var_es(**kw)
    shocked = np.hstack(capture["shocked"])
    got = np.log(shocked / w.base[:, None]).astype(np.float32)
    for row_name, src in (
        ("EQ_GBP", "EQ_EUR"),
        ("EQ_USD", "EQ_USD"),
        ("USDEUR", "USDEUR"),
    ):
        h = np.asarray(PROXY_HIST[src])
        expected = np.log(h[1:] / h[:-1]).astype(np.float32)
        assert np.array_equal(got[w.row(row_name)], expected), row_name


def test_a_different_authored_model_changes_result_and_identity() -> None:
    """EXACT. The proxy model is result-defining and identified by
    `shocks.proxy_model_digest` (D088 §7)."""
    _, kw_eur = _proxy_world("EQ_EUR")
    _, kw_usd = _proxy_world("EQ_USD")
    a, b = run_scenario_var_es(**kw_eur), run_scenario_var_es(**kw_usd)
    assert a.identity != b.identity
    assert a.manifest.shocks.proxy_model_digest != b.manifest.shocks.proxy_model_digest
    assert [v.hex() for v in a.var.tolist()] != [v.hex() for v in b.var.tolist()]


def test_data_availability_never_overrides_an_authored_model(capture) -> None:
    """EXACT. USDEUR has its own history, yet an authored model USDEUR <- 1.0
    (EQ_EUR, LOG) is used: the model selects columns, data only validates them
    (D088 §9; D085 clause 8 superseded)."""
    w, kw = _proxy_world("EQ_EUR")
    kw["proxy_terms"] = (*kw["proxy_terms"], _term("USDEUR", "EQ_EUR"))
    run_scenario_var_es(**kw)
    got = np.log(np.hstack(capture["shocked"]) / w.base[:, None]).astype(np.float32)
    h = np.asarray(PROXY_HIST["EQ_EUR"])
    assert np.array_equal(got[w.row("USDEUR")], np.log(h[1:] / h[:-1]).astype(np.float32))


def test_a_row_without_a_column_and_without_a_model_is_refused_before_numerics(
    boundary,
) -> None:
    """EXACT. With no authored term, EQ_GBP's identity row requests
    (EQ_GBP, LOG, h), which the world does not hold: no substitute, no tier."""
    w, kw = _proxy_world("EQ_EUR")
    kw["proxy_terms"] = ()
    with pytest.raises(ValueError, match="not in the matrix world"):
        run_scenario_var_es(**kw)
    assert boundary == [], f"numerical stage entered before refusal: {boundary}"


# ---- 4. prefix / suffix isolation under controlled poisoning -------------------------


def _poisoned_run(monkeypatch, kw: dict[str, Any], k_val: int, poison: float | None):
    """Run through production; after the real shock application, overwrite the
    reporting-only suffix rows of the ONE shocked-level buffer with `poison`."""
    seen: dict[str, Any] = {"shocked": []}
    real_apply = api.apply_linear_terms_to_base_levels
    real_convert = api.convert_scenario_pnl_to_reporting_currency

    def apply(**k: Any) -> None:
        real_apply(**k)
        if poison is not None:
            k["out"][k_val:, :] = poison
        seen["shocked"].append(k["out"].copy())

    def convert(**k: Any):
        seen["settlement"] = np.array(k["settlement_pnl"], copy=True)
        seen["v0"] = np.array(k["base_settlement_values"], copy=True)
        out = real_convert(**k)
        seen["reported"] = np.array(out.converted_pnl, copy=True)
        return out

    monkeypatch.setattr(api, "apply_linear_terms_to_base_levels", apply)
    monkeypatch.setattr(api, "convert_scenario_pnl_to_reporting_currency", convert)
    run = run_scenario_var_es(**kw)
    monkeypatch.undo()
    return run, seen


@pytest.mark.parametrize("derived", [(), (1, 2)], ids=["FULL_REVAL", "TAYLOR_DERIVED"])
def test_suffix_poison_leaves_valuation_bitwise_and_moves_reporting(
    monkeypatch, derived
) -> None:
    """EXACT (valuation) + EXACT D081 oracle (reporting).

    Test A: poisoning the reporting-only suffix changes nothing valuation owns
    (V_0, settlement dV, bitwise). Test B: reporting moves exactly as
    dW = dV q_s + V_0 (q_s - q_0), with q = USDEUR (USD -> EUR, direct),
    evaluated by hand from the poison level.
    """
    w = _suffix_world()
    k_val = w.axis.num_valuation_factors
    block = w.derived_block(list(derived)) if derived else None
    kw = w.inputs(derived=derived, block=block)
    clean, c = _poisoned_run(monkeypatch, kw, k_val, None)
    poison = 0.5
    kw2 = w.inputs(derived=derived, block=block)
    dirty, d = _poisoned_run(monkeypatch, kw2, k_val, poison)
    # Test A: valuation owns the prefix only
    prefix_dirty = np.hstack(d["shocked"])[:k_val]
    assert np.array_equal(prefix_dirty, np.hstack(c["shocked"])[:k_val])
    assert d["v0"].tobytes() == c["v0"].tobytes()
    assert d["settlement"].tobytes() == c["settlement"].tobytes()
    # Test B: reporting consumes the suffix, D081 by hand
    q0 = float(w.base[w.row("USDEUR")])
    expected = d["settlement"] * poison + d["v0"][:, None] * (poison - q0)
    assert d["reported"].tobytes() == expected.tobytes()
    assert not np.array_equal(d["reported"], c["reported"])
    assert dirty.identity == clean.identity  # the poison is not an input


def test_taylor_axis_is_the_valuation_prefix_only() -> None:
    """EXACT. The certified book's axis is `factor_ids[:K_val]`; a book compiled
    on the expanded axis is refused before numerics."""
    w = _suffix_world()
    k_val = w.axis.num_valuation_factors
    block = w.derived_block([1, 2])
    assert block.calculator.axis.keys.to_list() == w.axis.factor_ids[:k_val].tolist()
    assert RF["USDEUR"] not in block.calculator.axis.keys.to_list()
    wide = w.derived_block([1, 2], axis_ids=np.array(w.axis.factor_ids, copy=True))
    with pytest.raises(CompositionContractError, match="valuation prefix"):
        run_scenario_var_es(**w.inputs(derived=(1, 2), block=wide))


# ---- 5. identity: material reporting distinctions (EXACT) ----------------------------

TWO_CCY_ROWS = [spot(1, "EQ_USD", "USD", 2.0), spot(2, "EQ_EUR", "EUR", -3.0)]
TWO_CCY_BASE = {"EQ_USD": 212.37, "EQ_EUR": 53.12, "USDEUR": 0.9259, "USDCHF": 0.8812}
TWO_CCY_HIST = {
    n: v * np.exp(np.cumsum(np.r_[0.0, _P.normal(0.0, 0.015, 9)]))
    for n, v in TWO_CCY_BASE.items()
}


def _run_with_plan(monkeypatch, rows, ccy, base=None):
    w = World(rows, ccy, base or TWO_CCY_BASE, TWO_CCY_HIST)
    seen: dict[str, Any] = {}
    real = api.convert_scenario_pnl_to_reporting_currency

    def convert(**k: Any):
        seen["plan"] = k["plan"]
        return real(**k)

    monkeypatch.setattr(api, "convert_scenario_pnl_to_reporting_currency", convert)
    run = run_scenario_var_es(**w.inputs())
    monkeypatch.undo()
    return w, run, seen["plan"]


def test_reporting_currency_flips_direct_and_reciprocal_and_the_identity(
    monkeypatch,
) -> None:
    """USD reporting: EUR-settled reads USDEUR reciprocally (invert 1).
    EUR reporting: USD-settled reads USDEUR directly (invert 0). Same book,
    same suffix factor, different route: different identity and FX digest."""
    wu, ru, pu = _run_with_plan(monkeypatch, TWO_CCY_ROWS, "USD")
    we, re_, pe = _run_with_plan(monkeypatch, TWO_CCY_ROWS, "EUR")
    assert wu.axis.reporting_factor_ids.tolist() == [RF["USDEUR"]]
    assert we.axis.reporting_factor_ids.tolist() == [RF["USDEUR"]]
    assert pu.invert.tolist() == [0, 1] and pu.fx_row_index[0] == INVALID_UIX
    assert pe.invert.tolist() == [0, 0] and pe.fx_row_index[1] == INVALID_UIX
    assert ru.identity != re_.identity
    assert ru.manifest.reporting.fx_digest != re_.manifest.reporting.fx_digest


def test_reporting_fx_factor_and_its_base_level_are_in_the_identity(
    monkeypatch,
) -> None:
    rows = [spot(1, "EQ_USD", "USD", 2.0)]
    _, r_eur, _ = _run_with_plan(monkeypatch, rows, "EUR")
    w_chf, r_chf, _ = _run_with_plan(monkeypatch, rows, "CHF")
    assert w_chf.axis.reporting_factor_ids.tolist() == [RF["USDCHF"]]
    assert r_eur.identity != r_chf.identity  # different reporting FX factor
    moved = {**TWO_CCY_BASE, "USDEUR": TWO_CCY_BASE["USDEUR"] * 1.001}
    _, r_moved, _ = _run_with_plan(monkeypatch, rows, "EUR", base=moved)
    assert r_moved.identity != r_eur.identity  # suffix base level
    moved_digest = r_moved.manifest.pnl.base_levels_digest
    assert moved_digest != r_eur.manifest.pnl.base_levels_digest


def test_same_economics_rebuilt_from_scratch_gives_the_same_run(monkeypatch) -> None:
    """EXACT converse: identity, result_digest, VaR and ES bitwise."""
    _, a, _ = _run_with_plan(monkeypatch, TWO_CCY_ROWS, "EUR")
    _, b, _ = _run_with_plan(monkeypatch, TWO_CCY_ROWS, "EUR")
    assert a.identity == b.identity and a.result_digest == b.result_digest
    assert a.var.tobytes() == b.var.tobytes() and a.es.tobytes() == b.es.tobytes()


# ---- 6. one scenario world across prefix and suffix ----------------------------------

# Asymmetric markers: every daily return is distinct and strictly monotone in t,
# so "prefix at t, suffix at t + 1" and "both at t" are distinguishable.
_T = 12
MARKER_HIST = {
    "EQ_USD": 100.0 * np.exp(np.cumsum(np.r_[0.0, 0.004 * np.arange(1, _T)])),
    "USDEUR": 0.90 * np.exp(np.cumsum(np.r_[0.0, -0.0007 * np.arange(1, _T) ** 1.5])),
}


def _returns32(name: str) -> np.ndarray:
    h = np.asarray(MARKER_HIST[name])
    return np.log(h[1:] / h[:-1]).astype(np.float32)


def _shocks32(capture_seen, w: World) -> np.ndarray:
    levels = np.hstack(capture_seen["shocked"])
    return np.log(levels / w.base[:, None]).astype(np.float32)


def test_historical_prefix_and_suffix_read_the_same_date(capture) -> None:
    """EXACT (float32 shocks). Scenario s is the move t = s -> s + 1 on BOTH
    rows; the markers prove a one-step misalignment would be visible."""
    w = World(SUFFIX_ROWS, "EUR", SUFFIX_BASE, MARKER_HIST)
    run_scenario_var_es(**w.inputs(data=_matrix(MARKER_HIST)))
    got = _shocks32(capture, w)
    ru, rf_ = _returns32("EQ_USD"), _returns32("USDEUR")
    assert np.array_equal(got[w.row("EQ_USD")], ru)
    assert np.array_equal(got[w.row("USDEUR")], rf_)
    # the markers make a one-step misalignment (t + 1) visible
    assert not np.array_equal(got[w.row("USDEUR")][:-1], rf_[1:])


def test_bootstrap_prefix_and_suffix_share_every_time_index(capture) -> None:
    """EXACT. The date of each scenario is identified from the prefix row and,
    independently, from the suffix row; they agree in every scenario."""
    w = World(SUFFIX_ROWS, "EUR", SUFFIX_BASE, MARKER_HIST)
    cfg = parse_scenario_model_config({
        "model": "sbb", "horizon": 1, "total_scenarios": 300,
        "expected_block_size": 3.0, "seed": 17,
    })
    run_scenario_var_es(**w.inputs(scenario_config=cfg, data=_matrix(MARKER_HIST)))
    got = _shocks32(capture, w)

    def dates(row_values, returns):
        hit = row_values[:, None] == returns[None, :]
        assert (hit.sum(axis=1) == 1).all()  # every scenario is one exact date
        return hit.argmax(axis=1)

    t_prefix = dates(got[w.row("EQ_USD")], _returns32("EQ_USD"))
    t_suffix = dates(got[w.row("USDEUR")], _returns32("USDEUR"))
    assert np.array_equal(t_prefix, t_suffix)
    assert np.unique(t_prefix).size > 3  # the bootstrap really resampled


def test_parametric_full_axis_matches_an_independent_fixed_seed_oracle(capture) -> None:
    """EXACT (float32 shocks) against a hand construction: Z from
    `default_rng(seed).standard_normal((S, K))` (D021: all base normals, C
    order, before batching), shocks = (Z L^T) * vol per factor, rounded to
    float32. The production generator is not used to build the expectation."""
    w = _suffix_world()
    ids = np.array([RF["EQ_USD"], RF["USDEUR"]], dtype=np.int32)  # ascending
    rho, vols, seed, s = 0.6, np.array([0.02, 0.01]), 5, 8
    chol = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    data = ParametricRiskData(
        factor_ids=ids,
        cholesky_matrix=np.ascontiguousarray(chol),
        volatilities=np.ascontiguousarray(vols),
        effective_period_count=1,
        effective_period_unit=FORECAST_PERIOD_UNIT_BUSINESS_DAY_ID,
        chart_ids=np.full(ids.shape[0], SHIFT_LOG_RETURN_ID, dtype=np.int8),
    )
    cfg = parse_scenario_model_config({
        "model": "parametric", "total_scenarios": s,
        "forecast_horizon": {"count": 1, "unit": "BUSINESS_DAY"},
        "distribution": {"kind": "standard_normal"}, "seed": seed,
    })
    run_scenario_var_es(**w.inputs(scenario_config=cfg, data=data, batch_size=3))
    got = _shocks32(capture, w)
    z = np.random.default_rng(seed).standard_normal((s, 2))
    expected = {
        "EQ_USD": (z[:, 0] * chol[0, 0]) * vols[0],
        "USDEUR": (z[:, 0] * chol[1, 0] + z[:, 1] * chol[1, 1]) * vols[1],
    }
    for name, exp in expected.items():
        assert np.array_equal(got[w.row(name)], exp.astype(np.float32)), name


# ---- 7. route invariants (EXACT) -----------------------------------------------------


def test_route_shares_one_row_per_settlement_reporting_pair(capture) -> None:
    """Same settlement + reporting currency -> same rf_id -> one shared row.
    Suffix rows <= distinct non-identity settlement -> reporting pairs."""
    rows = [
        spot(1, "EQ_USD", "USD", 1.0),
        spot(2, "EQ_USD", "USD", 2.0),
        spot(3, "EQ_GBP", "GBP", 3.0),
        spot(4, "EQ_GBP", "GBP", 4.0),
        spot(5, "EQ_EUR", "EUR", 5.0),
    ]
    base = {
        "EQ_USD": 212.37,
        "EQ_GBP": 148.91,
        "EQ_EUR": 53.12,
        "USDEUR": 0.9259,
        "EURGBP": 0.8561,
    }
    hist = {
        n: v * np.exp(np.cumsum(np.r_[0.0, _P.normal(0.0, 0.01, 7)]))
        for n, v in base.items()
    }
    w = World(rows, "EUR", base, hist)
    run_scenario_var_es(**w.inputs())
    plan = capture["convert_in"]["plan"]
    pairs = {(c, "EUR") for c in w.book.settlement_ccy if c != "EUR"}
    assert w.axis.reporting_factor_ids.size <= len(pairs) == 2
    idx = plan.fx_row_index.tolist()
    assert idx[0] == idx[1] != INVALID_UIX and idx[2] == idx[3] != INVALID_UIX
    assert idx[4] == INVALID_UIX  # identity conversion: no FX row
    assert plan.invert.tolist() == [0, 0, 1, 1, 0]  # USDEUR direct, EURGBP reciprocal


def test_static_config_carries_one_orientation_per_pair() -> None:
    """D060 structural invariant the route relies on: a pair is configured in
    exactly one orientation, so direct-first-else-reciprocal is unambiguous.
    (Configuring the reciprocal too is refused at compile time:
    tests/market_data/remediation/test_contract_failure_paths.py.)"""
    m = np.asarray(RSC.fx_spot_matrix)
    set_ = m != int(NO_FX_RF_REQUIRED)
    assert not (set_ & set_.T).any()


def test_missing_route_is_refused_before_numerics(boundary) -> None:
    w = _suffix_world()
    kw = w.inputs(reporting=ReportingSpec(reporting_ccy="JPY"))
    _refused(kw, "no configured FX pair", boundary)


# ---- 8. fail fast: every axis misalignment stops before numerics (EXACT) -------------


@pytest.mark.parametrize(
    "case, match",
    [
        ("factor_ids_permuted", "canonical axis"),
        ("shift_order_swapped", "canonical axis"),
        ("suffix_omitted", "canonical axis"),
        ("K_mismatch_base_levels", "base_levels has shape"),
    ],
)
def test_axis_misalignment_fails_before_numerics(boundary, case, match) -> None:
    w, kw = _proxy_world("EQ_EUR")
    k_val = w.axis.num_valuation_factors
    if case == "factor_ids_permuted":
        for name in _AXIS_INPUTS:
            kw[name] = np.ascontiguousarray(kw[name][[1, 0, *range(2, kw[name].size)]])
    elif case == "shift_order_swapped":
        s = kw["required_shift_type_id"].copy()
        s[0] = np.int8(2) if s[0] != 2 else np.int8(1)
        kw["required_shift_type_id"] = s
    elif case == "suffix_omitted":
        for name in _AXIS_INPUTS:
            kw[name] = np.ascontiguousarray(kw[name][:k_val])
    else:  # K_mismatch_base_levels
        kw["base_levels"] = np.ascontiguousarray(kw["base_levels"][:-1])
    _refused(kw, match, boundary)


def test_derived_axis_equals_the_runs_own_derivation() -> None:
    """EXACT, single owner: the helper and the run derive one axis."""
    w = _suffix_world()
    again = derive_run_factor_axis(valuation_context=w.context, reporting_ccy="EUR")
    assert again.factor_ids.tobytes() == w.axis.factor_ids.tobytes()
    assert again.shift_type_ids.tobytes() == w.axis.shift_type_ids.tobytes()
    assert again.num_valuation_factors == w.axis.num_valuation_factors
