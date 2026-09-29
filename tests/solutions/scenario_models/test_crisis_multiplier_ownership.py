# tests/solutions/scenario_models/test_crisis_multiplier_ownership.py
"""
D092 — regime-switching crisis multiplier ownership and identity.

Expected values are literal or built from an independent re-derivation of the
D087 draw order; no expectation is produced by the code under test.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from riskquant.simlab.scenarios.parametric import ParametricGenerator
from riskquant.simlab.scenarios.parametric_data import ParametricRiskData
from riskquant.solutions.run_identity.hashing import hash_parametric_risk_data
from riskquant.solutions.scenario_models.api import (
    build_scenario_generator,
    parse_scenario_model_config,
    resolve_parametric_crisis_multipliers,
)

_K = 3
_S = 60
_SEED = 17
_SIGMA = np.array([0.10, 0.20, 0.15])


def _data(vector: Any = None, **over: Any) -> ParametricRiskData:
    kw: dict[str, Any] = {
        "factor_ids": np.array([4, 7, 9], dtype=np.int32),
        "cholesky_matrix": np.ascontiguousarray(np.eye(_K, dtype=np.float64)),
        "volatilities": np.ascontiguousarray(_SIGMA.copy()),
        "effective_period_count": 1,
        "effective_period_unit": np.int8(1),
        "chart_ids": np.ones(_K, dtype=np.int8),
        "crisis_vol_multipliers": vector,
    }
    kw.update(over)
    return ParametricRiskData(**kw)


def _raw(distribution: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": "parametric",
        "forecast_horizon": {"count": 1, "unit": "BUSINESS_DAY"},
        "total_scenarios": _S,
        "seed": _SEED,
        "distribution": distribution,
    }


def _regime(mult: Any, p: float = 0.3) -> dict[str, Any]:
    return _raw({"kind": "regime_switching_gmm", "prob_crisis": p, "crisis_vol_multipliers": mult})


def _shocks(gen: Any, batch_size: int = 16) -> np.ndarray:
    return np.vstack([
        np.array(b.shocks, copy=True)
        for b in gen.iter_batches(
            required_shift_type_id=np.ones(_K, dtype=np.int8),
            source_factor_ids=np.array([4, 7, 9], dtype=np.int32),
            shock_scale=np.ones(_K, dtype=np.float32),
            batch_size=batch_size,
        )
    ])


def _bits(a: np.ndarray) -> bytes:
    return np.ascontiguousarray(a).tobytes()


# ---- schema: exactly two admissible states ----------------------------------------


def test_scalar_form_is_accepted_as_a_float() -> None:
    got = parse_scenario_model_config(_regime(2.5)).distribution.crisis_vol_multipliers
    assert type(got) is float and got == 2.5


def test_per_factor_declaration_is_accepted() -> None:
    got = parse_scenario_model_config(_regime("per_factor")).distribution
    assert got.crisis_vol_multipliers == "per_factor"


@pytest.mark.parametrize(
    "mult",
    [[2.0, 3.0, 4.0], (2.0, 3.0, 4.0), np.array([2.0, 3.0, 4.0]), [], "PER_FACTOR",
     "uniform", True, 0.0, -1.0, float("nan"), float("inf"), float("-inf"), None],
)
def test_every_other_form_is_rejected(mult: Any) -> None:
    with pytest.raises(ValidationError):
        parse_scenario_model_config(_regime(mult))


def test_the_field_is_required_for_regime_switching() -> None:
    with pytest.raises(ValidationError):
        parse_scenario_model_config(
            _raw({"kind": "regime_switching_gmm", "prob_crisis": 0.3})
        )


@pytest.mark.parametrize(
    "distribution",
    [
        {"kind": "standard_normal"},
        {"kind": "laplace"},
        {"kind": "student_t", "df": 5.0},
        {"kind": "variance_gamma", "nu": 0.5},
    ],
)
@pytest.mark.parametrize("mult", ["per_factor", 2.0])
def test_non_regime_variants_cannot_carry_the_field(distribution, mult) -> None:
    with pytest.raises(ValidationError):
        parse_scenario_model_config(_raw({**distribution, "crisis_vol_multipliers": mult}))


# ---- data: optional, validated whenever present -------------------------------------


def test_the_vector_is_optional() -> None:
    assert _data().crisis_vol_multipliers is None


@pytest.mark.parametrize("vector", [[2.0, 1.5, 3.0], [1e-300, 1.0, 5e-324], [1e300, 2.0, 1.0]])
def test_finite_positive_vectors_are_accepted_unchanged(vector: list[float]) -> None:
    m = np.array(vector, dtype=np.float64)
    assert _data(m).crisis_vol_multipliers is m


@pytest.mark.parametrize(
    ("vector", "exc"),
    [
        (np.array([2.0, 1.5]), ValueError),  # length 2 != K = 3
        (np.array([2.0, 1.5, 3.0, 4.0]), ValueError),  # length 4 != K = 3
        (np.array([[2.0, 1.5, 3.0]]), TypeError),  # 2-D
        (np.array([2.0, 0.0, 3.0]), ValueError),
        (np.array([2.0, -1.0, 3.0]), ValueError),
        (np.array([2.0, np.nan, 3.0]), ValueError),
        (np.array([2.0, np.inf, 3.0]), ValueError),
        (np.array([2.0, -np.inf, 3.0]), ValueError),
        (np.array([2.0, 1.5, 3.0], dtype=np.float32), TypeError),
        (np.array([2, 1, 3], dtype=np.int64), TypeError),
        ([2.0, 1.5, 3.0], TypeError),
        (2.0, TypeError),
        (np.array([2.0, 9.0, 1.5, 9.0, 3.0, 9.0])[::2], ValueError),  # strided view
    ],
)
def test_invalid_vectors_are_rejected(vector: Any, exc: type) -> None:
    with pytest.raises(exc):
        _data(vector)


# ---- composition: the declared form must match the data ---------------------------------


def test_scalar_with_no_vector_yields_the_scalar() -> None:
    cfg = parse_scenario_model_config(_regime(2.0))
    assert resolve_parametric_crisis_multipliers(cfg, _data()) == 2.0


def test_per_factor_with_a_vector_yields_the_stored_vector() -> None:
    m = np.array([2.0, 1.5, 3.0])
    cfg = parse_scenario_model_config(_regime("per_factor"))
    assert resolve_parametric_crisis_multipliers(cfg, _data(m)) is m


def test_scalar_with_a_vector_fails() -> None:
    cfg = parse_scenario_model_config(_regime(2.0))
    with pytest.raises(ValueError, match="D092"):
        resolve_parametric_crisis_multipliers(cfg, _data(np.array([2.0, 1.5, 3.0])))


def test_per_factor_without_a_vector_fails() -> None:
    cfg = parse_scenario_model_config(_regime("per_factor"))
    with pytest.raises(ValueError, match="D092"):
        resolve_parametric_crisis_multipliers(cfg, _data())


@pytest.mark.parametrize(
    "distribution",
    [{"kind": "standard_normal"}, {"kind": "student_t", "df": 5.0}],
)
def test_a_non_regime_law_with_a_vector_fails(distribution) -> None:
    cfg = parse_scenario_model_config(_raw(distribution))
    assert resolve_parametric_crisis_multipliers(cfg, _data()) is None
    with pytest.raises(ValueError, match="D092"):
        resolve_parametric_crisis_multipliers(cfg, _data(np.array([2.0, 1.5, 3.0])))


def test_building_the_generator_applies_the_same_gate() -> None:
    vec = _data(np.array([2.0, 1.5, 3.0]))
    with pytest.raises(ValueError, match="D092"):
        build_scenario_generator(parse_scenario_model_config(_regime(2.0)), data=vec)
    with pytest.raises(ValueError, match="D092"):
        build_scenario_generator(parse_scenario_model_config(_regime("per_factor")), data=_data())
    with pytest.raises(ValueError, match="D092"):
        build_scenario_generator(
            parse_scenario_model_config(_raw({"kind": "laplace"})), data=vec
        )


# ---- identity ---------------------------------------------------------------------------------


def test_the_vector_enters_the_data_digest() -> None:
    m = np.array([2.0, 1.5, 3.0])
    absent, present = _data(), _data(m)
    assert hash_parametric_risk_data(present) == hash_parametric_risk_data(_data(m.copy()))
    assert hash_parametric_risk_data(absent) != hash_parametric_risk_data(present)
    changed = _data(np.array([2.0, 1.5, 3.5]))
    assert hash_parametric_risk_data(present) != hash_parametric_risk_data(changed)


def test_other_data_fields_still_change_the_digest_with_a_vector() -> None:
    m = np.array([2.0, 1.5, 3.0])
    base = hash_parametric_risk_data(_data(m))
    for over in (
        {"factor_ids": np.array([4, 7, 10], dtype=np.int32)},
        {"chart_ids": np.array([1, 3, 1], dtype=np.int8)},
        {"effective_period_count": 2},
    ):
        assert hash_parametric_risk_data(_data(m, **over)) != base


def test_p_scalar_and_declaration_are_configuration_identity() -> None:
    dump = parse_scenario_model_config(_regime("per_factor", p=0.3)).model_dump(mode="json")
    assert dump["distribution"] == {
        "kind": "regime_switching_gmm",
        "prob_crisis": 0.3,
        "crisis_vol_multipliers": "per_factor",
    }
    dumps = [
        parse_scenario_model_config(raw).model_dump(mode="json")
        for raw in (_regime(2.0, p=0.3), _regime(2.0, p=0.4), _regime(2.5, p=0.3),
                    _regime("per_factor", p=0.3))
    ]
    assert len({repr(d) for d in dumps}) == 4


# ---- numerical law, RNG and the scalar / uniform-vector equivalence ---------------------------


def test_a_scalar_and_an_equal_uniform_vector_are_bitwise_equal_shocks() -> None:
    scalar_cfg = parse_scenario_model_config(_regime(2.0))
    vector_cfg = parse_scenario_model_config(_regime("per_factor"))
    uniform = _data(np.array([2.0, 2.0, 2.0]))
    a = _shocks(build_scenario_generator(scalar_cfg, data=_data()))
    b = _shocks(build_scenario_generator(vector_cfg, data=uniform))
    assert _bits(a) == _bits(b)
    # ... and they remain distinct artifacts.
    assert scalar_cfg.model_dump(mode="json") != vector_cfg.model_dump(mode="json")
    assert hash_parametric_risk_data(_data()) != hash_parametric_risk_data(uniform)


def test_a_heterogeneous_vector_matches_an_independent_draw_order_oracle() -> None:
    """D087 draw order re-derived independently: all normals, then one uniform
    per scenario; crisis rows scale column i by m_i, then sigma_i; one float32
    rounding. L = I keeps the oracle exact."""
    m = np.array([2.0, 1.5, 3.0])
    p = 0.3
    rng = np.random.default_rng(_SEED)
    z = rng.standard_normal((_S, _K))
    u = rng.random(_S)
    crisis = (u < p)[:, None]
    expected = ((z * np.where(crisis, m, 1.0)) * _SIGMA).astype(np.float32)
    got = _shocks(build_scenario_generator(parse_scenario_model_config(_regime("per_factor", p)), data=_data(m)))
    assert _bits(got) == _bits(expected)


@pytest.mark.parametrize("batch_size", [1, 7, _S, _S + 13])
def test_the_per_factor_world_is_batch_invariant(batch_size: int) -> None:
    gen = build_scenario_generator(
        parse_scenario_model_config(_regime("per_factor")), data=_data(np.array([2.0, 1.5, 3.0]))
    )
    assert _bits(_shocks(gen, batch_size)) == _bits(_shocks(gen, 16))


# ---- the directly callable generator ----------------------------------------------------------


def _direct(kind: str, **kw: Any) -> ParametricGenerator:
    return ParametricGenerator(
        available_factor_ids=np.array([4, 7, 9], dtype=np.int32),
        cholesky_matrix=np.ascontiguousarray(np.eye(_K)),
        volatilities=np.ascontiguousarray(_SIGMA.copy()),
        total_scenarios=_S,
        distribution_kind=kind,
        seed=_SEED,
        **kw,
    )


@pytest.mark.parametrize("kind", ["standard_normal", "student_t", "laplace", "variance_gamma"])
@pytest.mark.parametrize("mult", [2.0, np.array([2.0, 1.5, 3.0]), np.array([-1.0, 0.0, np.nan])])
def test_the_generator_rejects_multipliers_for_non_regime_laws(kind: str, mult: Any) -> None:
    extra = {"student_t": {"student_t_df": 5.0}, "variance_gamma": {"vg_nu": 0.5}}.get(kind, {})
    with pytest.raises(ValueError, match="only valid for regime_switching_gmm"):
        _direct(kind, crisis_vol_multipliers=mult, **extra)


@pytest.mark.parametrize(
    "mult",
    [0.0, -2.0, float("nan"), np.array([2.0, 1.5]), np.array([2.0, np.inf, 3.0]),
     np.array([[2.0, 1.5, 3.0]]), "per_factor"],
)
def test_the_generator_validates_regime_multipliers(mult: Any) -> None:
    """Scalar and K-aligned 1-D vector are validated; the configuration string
    "per_factor" is not a numerical input and fails."""
    with pytest.raises((ValueError, TypeError)):
        _direct("regime_switching_gmm", prob_crisis=0.3, crisis_vol_multipliers=mult)


def test_dataclass_copies_keep_the_vector_aligned() -> None:
    m = np.array([2.0, 1.5, 3.0])
    copy = dataclasses.replace(_data(m))
    assert copy.crisis_vol_multipliers is m
    assert copy.factor_ids.tolist() == [4, 7, 9]
