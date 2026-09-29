# tests/simlab/scenarios/test_boundary_contracts.py
"""
R2 strict boundary contracts.

F3 — discrete quantities (horizon, total_scenarios) accept int / np.integer,
     reject float and bool, require > 0, and persist a canonical Python int.
     expected_block_size is a CONTINUOUS mean block length (not discrete):
     real accepted, bool/non-finite rejected, >= 1.0.
F5 — a non-float64 historical matrix fails at the generator boundary (D017).
F7 — ScenarioSetBatch enforces its dtype / ndim / layout / scalar / mapping
     contract at construction.
"""
from __future__ import annotations

import numpy as np
import pytest

from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.simlab.scenarios.base import ScenarioSetBatch
from riskquant.simlab.scenarios.bootstrap_fhs import FilteredBootstrapGenerator
from riskquant.simlab.scenarios.bootstrap_sbb import StationaryBlockBootstrapGenerator
from riskquant.simlab.scenarios.historical import HistoricalGenerator
from riskquant.simlab.scenarios.historical_fhs import FilteredHistoricalGenerator
from riskquant.simlab.scenarios.parametric import ParametricGenerator


def _matrix(dtype=np.float64, t=12, k=2) -> RiskFactorMatrix:
    a = np.asfortranarray(
        np.array([[100.0 + i + 3 * j for i in range(t)] for j in range(k)], dtype=dtype).T
    )
    return RiskFactorMatrix(
        dates=np.arange(t, dtype=np.int32),
        factor_ids=np.arange(k, dtype=np.int32),
        values=a,
    )


# generator factories parametrised by the discrete quantity under test ---------


def _with_horizon(gen_key, horizon, matrix=None):
    m = matrix if matrix is not None else _matrix()
    if gen_key == "historical":
        return HistoricalGenerator(matrix=m, horizon=horizon)
    if gen_key == "fhs":
        return FilteredHistoricalGenerator(matrix=m, horizon=horizon)
    if gen_key == "sbb":
        return StationaryBlockBootstrapGenerator(
            matrix=m, total_scenarios=10, horizon=horizon
        )
    if gen_key == "filtboot":
        return FilteredBootstrapGenerator(matrix=m, total_scenarios=10, horizon=horizon)
    raise AssertionError(gen_key)


def _parametric(total_scenarios):
    return ParametricGenerator(
        available_factor_ids=np.array([0, 1], dtype=np.int32),
        cholesky_matrix=np.eye(2, dtype=np.float64),
        volatilities=np.array([0.1, 0.2], dtype=np.float64),
        total_scenarios=total_scenarios,
        distribution_kind="standard_normal",
    )


def _with_total(gen_key, total):
    m = _matrix()
    if gen_key == "sbb":
        return StationaryBlockBootstrapGenerator(
            matrix=m, total_scenarios=total, horizon=1
        )
    if gen_key == "filtboot":
        return FilteredBootstrapGenerator(matrix=m, total_scenarios=total, horizon=1)
    if gen_key == "parametric":
        return _parametric(total)
    raise AssertionError(gen_key)


# --------------------------------------------------------- F3 horizon ----------

_HORIZON_GENS = ["historical", "fhs", "sbb", "filtboot"]


@pytest.mark.parametrize("gen_key", _HORIZON_GENS)
@pytest.mark.parametrize("value", [1, np.int64(1), 3, np.int32(2)])
def test_horizon_accepts_integers(gen_key, value) -> None:
    gen = _with_horizon(gen_key, value)
    assert gen.horizon == int(value)
    assert type(gen.horizon) is int  # persisted as canonical Python int


@pytest.mark.parametrize("gen_key", _HORIZON_GENS)
@pytest.mark.parametrize("value", [1.0, 1.5, True])
def test_horizon_rejects_float_and_bool(gen_key, value) -> None:
    with pytest.raises(TypeError):
        _with_horizon(gen_key, value)


@pytest.mark.parametrize("gen_key", _HORIZON_GENS)
@pytest.mark.parametrize("value", [0, -1])
def test_horizon_rejects_non_positive(gen_key, value) -> None:
    with pytest.raises(ValueError):
        _with_horizon(gen_key, value)


# ---------------------------------------------------- F3 total_scenarios -------

_TOTAL_GENS = ["sbb", "filtboot", "parametric"]


@pytest.mark.parametrize("gen_key", _TOTAL_GENS)
@pytest.mark.parametrize("value", [100, np.int64(100), np.int32(50)])
def test_total_scenarios_accepts_integers(gen_key, value) -> None:
    gen = _with_total(gen_key, value)
    assert gen.total_scenarios == int(value)
    assert type(gen.total_scenarios) is int


@pytest.mark.parametrize("gen_key", _TOTAL_GENS)
@pytest.mark.parametrize("value", [100.0, 100.5, True])
def test_total_scenarios_rejects_float_and_bool(gen_key, value) -> None:
    with pytest.raises(TypeError):
        _with_total(gen_key, value)


@pytest.mark.parametrize("gen_key", _TOTAL_GENS)
@pytest.mark.parametrize("value", [0, -5])
def test_total_scenarios_rejects_non_positive(gen_key, value) -> None:
    with pytest.raises(ValueError):
        _with_total(gen_key, value)


# ------------------------------------------------- expected_block_size ---------
# Continuous mean block length (geometric): real accepted, bool/non-finite
# rejected, >= 1.0. NOT the discrete int-only policy (its default is 1.0).


@pytest.mark.parametrize("value", [1.0, 2.0, 3, np.float64(2.5), np.int64(2)])
def test_expected_block_size_accepts_reals(value) -> None:
    gen = StationaryBlockBootstrapGenerator(
        matrix=_matrix(), total_scenarios=10, expected_block_size=value, horizon=1
    )
    assert gen.expected_block_size == float(value)
    assert type(gen.expected_block_size) is float


def test_expected_block_size_rejects_bool() -> None:
    with pytest.raises(TypeError):
        StationaryBlockBootstrapGenerator(
            matrix=_matrix(), total_scenarios=10, expected_block_size=True, horizon=1
        )


@pytest.mark.parametrize("value", [0.5, 0.0, float("inf"), float("nan")])
def test_expected_block_size_rejects_out_of_domain(value) -> None:
    with pytest.raises(ValueError):
        StationaryBlockBootstrapGenerator(
            matrix=_matrix(), total_scenarios=10, expected_block_size=value, horizon=1
        )


# ------------------------------------------------------------- F5 dtype --------


@pytest.mark.parametrize("gen_key", _HORIZON_GENS)
def test_float32_matrix_rejected_at_boundary(gen_key) -> None:
    with pytest.raises(TypeError, match="must be float64"):
        _with_horizon(gen_key, 1, matrix=_matrix(dtype=np.float32))


def test_float64_matrix_accepted() -> None:
    # control: the same construction succeeds with a float64 matrix.
    _with_horizon("historical", 1, matrix=_matrix(dtype=np.float64))


# ----------------------------------------------------- F7 batch contract -------


def _good_batch_kwargs():
    return dict(
        shocks=np.zeros((3, 2), dtype=np.float32, order="F"),
        scenario_offset=0,
        scenario_count_total=3,
    )


def test_batch_valid_construction() -> None:
    ScenarioSetBatch(**_good_batch_kwargs())


def test_batch_scenario_count_none_ok() -> None:
    ScenarioSetBatch(**{**_good_batch_kwargs(), "scenario_count_total": None})


@pytest.mark.parametrize(
    "override,exc",
    [
        # wrong dtype
        ({"shocks": np.zeros((3, 2), np.float64, order="F")}, TypeError),
        # wrong ndim
        ({"shocks": np.zeros(6, np.float32)}, TypeError),
        # wrong layout where required
        ({"shocks": np.zeros((3, 2), np.float32, order="C")}, ValueError),
        # invalid integer scalars
        ({"scenario_offset": -1}, ValueError),
        ({"scenario_offset": True}, TypeError),
        ({"scenario_offset": 1.0}, TypeError),
        ({"scenario_count_total": 0}, ValueError),
        ({"scenario_count_total": True}, TypeError),
        ({"scenario_count_total": 2.0}, TypeError),
        # inconsistent mapping dimensions
    ],
)
def test_batch_malformed_rejected(override, exc) -> None:
    with pytest.raises(exc):
        ScenarioSetBatch(**{**_good_batch_kwargs(), **override})
