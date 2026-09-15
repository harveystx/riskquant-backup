# tests/solutions/scenario_models/test_parametric_seam.py
"""
Parametric configuration-seam tests: the heterogeneous-construction-data proof
point. Covers the nested distribution discriminated union (invalid parameter
combinations unrepresentable), strict real-number typing, the typed
``ParametricRiskData`` contract, fail-fast config/data pairing, and
direct-vs-seam equivalence + D029 invariance.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.market_data.risk_factors.shift_types import SHIFT_LOG_RETURN_ID
from riskquant.simlab.scenarios.parametric import ParametricGenerator
from riskquant.simlab.scenarios.parametric_data import ParametricRiskData
from riskquant.solutions.scenario_models.api import (
    build_scenario_generator,
    parse_scenario_model_config,
)
from riskquant.solutions.scenario_models.schemas import (
    HistoricalGeneratorConfig,
    ParametricGeneratorConfig,
)

_K = 3


def _fids():
    return np.arange(_K, dtype=np.int32)


def _chol():
    return np.ascontiguousarray(np.eye(_K, dtype=np.float64))


def _vols():
    return np.ascontiguousarray(np.array([0.10, 0.20, 0.15], dtype=np.float64))


def _pdata():
    return ParametricRiskData(
        factor_ids=_fids(), cholesky_matrix=_chol(), volatilities=_vols()
    )


def _rf_matrix(t=12):
    return RiskFactorMatrix(
        dates=np.arange(t, dtype=np.int32),
        factor_ids=np.arange(2, dtype=np.int32),
        values=np.asfortranarray(
            np.array([[100.0 + i for i in range(t)]] * 2, np.float64).T
        ),
    )


def _drive(gen, batch_size=4):
    chunks = []
    for b in gen.iter_batches(
        required_shift_type_id=np.array([SHIFT_LOG_RETURN_ID] * _K, np.int8),
        source_factor_ids=np.arange(_K, dtype=np.int32),
        shock_scale=np.ones(_K, np.float32),
        batch_size=batch_size,
    ):
        chunks.append(np.array(b.shocks, copy=True))
    return np.vstack(chunks)


def _cfg(distribution, **over):
    base = {
        "model": "parametric",
        "total_scenarios": 10,
        "seed": 7,
        "distribution": distribution,
    }
    base.update(over)
    return base


# ---- ParametricRiskData contract (strict; no silent cast) ---------------------


def test_valid_parametric_risk_data() -> None:
    d = _pdata()
    assert d.factor_ids.dtype == np.int32 and d.factor_ids.shape == (_K,)
    assert d.cholesky_matrix.shape == (_K, _K)
    assert d.volatilities.shape == (_K,)


@pytest.mark.parametrize(
    "override,exc",
    [
        ({"factor_ids": np.arange(_K, dtype=np.int64)}, TypeError),  # dtype
        ({"factor_ids": np.array([2, 0, 1], np.int32)}, ValueError),  # unsorted
        ({"factor_ids": np.array([0, 0, 1], np.int32)}, ValueError),  # dup
        ({"factor_ids": np.zeros((_K, 1), np.int32)}, TypeError),  # ndim
        ({"cholesky_matrix": np.asfortranarray(np.eye(_K))}, ValueError),  # F layout
        (
            {"cholesky_matrix": np.ascontiguousarray(np.eye(_K, dtype=np.float32))},
            TypeError,
        ),  # dtype
        (
            {"cholesky_matrix": np.ascontiguousarray(np.ones((_K, _K + 1)))},
            ValueError,
        ),  # non-square
        (
            {"volatilities": np.array([0.1, -0.2, 0.1], np.float64)},
            ValueError,
        ),  # non-positive
        (
            {"volatilities": np.array([0.1, np.inf, 0.1], np.float64)},
            ValueError,
        ),  # non-finite
        ({"volatilities": np.array([0.1, 0.2], np.float64)}, ValueError),  # length != K
    ],
)
def test_parametric_risk_data_rejects_non_canonical(override, exc) -> None:
    base = {"factor_ids": _fids(), "cholesky_matrix": _chol(), "volatilities": _vols()}
    base.update(override)
    with pytest.raises(exc):
        ParametricRiskData(**base)


def test_empty_factor_ids_rejected() -> None:
    with pytest.raises(ValueError):
        ParametricRiskData(
            factor_ids=np.zeros(0, np.int32),
            cholesky_matrix=np.zeros((0, 0), np.float64),
            volatilities=np.zeros(0, np.float64),
        )


# ---- distribution union: invalid combinations unrepresentable ----------------


@pytest.mark.parametrize(
    "distribution",
    [
        {"kind": "standard_normal"},
        {"kind": "laplace"},
        {"kind": "student_t", "df": 5.0},
        {"kind": "variance_gamma", "nu": 0.5},
        {
            "kind": "regime_switching_gmm",
            "prob_crisis": 0.1,
            "crisis_vol_multipliers": 3.0,
        },
        {
            "kind": "regime_switching_gmm",
            "prob_crisis": 0.1,
            "crisis_vol_multipliers": [2.0, 3.0, 4.0],
        },
    ],
)
def test_valid_distributions(distribution) -> None:
    cfg = parse_scenario_model_config(_cfg(distribution))
    assert isinstance(cfg, ParametricGeneratorConfig)
    assert cfg.distribution.kind == distribution["kind"]


@pytest.mark.parametrize(
    "distribution",
    [
        {"kind": "standard_normal", "df": 5.0},  # normal can't carry df
        {"kind": "student_t"},  # missing df
        {"kind": "student_t", "df": 5.0, "nu": 0.5},  # foreign nu
        {"kind": "variance_gamma"},  # missing nu
        {"kind": "regime_switching_gmm", "prob_crisis": 0.1},  # missing multipliers
        {"kind": "regime_switching_gmm", "crisis_vol_multipliers": 2.0},  # missing prob
        {"kind": "garch"},  # unknown distribution
    ],
)
def test_invalid_distribution_combinations_rejected(distribution) -> None:
    with pytest.raises(ValidationError):
        parse_scenario_model_config(_cfg(distribution))


# ---- strict real typing on distribution parameters ---------------------------


@pytest.mark.parametrize("df", [2.0, 1.0, "5", True, float("inf"), float("nan")])
def test_student_t_df_invalid(df) -> None:
    with pytest.raises(ValidationError):
        parse_scenario_model_config(_cfg({"kind": "student_t", "df": df}))


@pytest.mark.parametrize("df", [3, 5.5])  # int and float accepted (>2)
def test_student_t_df_valid_real(df) -> None:
    cfg = parse_scenario_model_config(_cfg({"kind": "student_t", "df": df}))
    assert cfg.distribution.df == float(df)


@pytest.mark.parametrize("nu", [0.0, -1.0, "0.5", True, float("inf")])
def test_variance_gamma_nu_invalid(nu) -> None:
    with pytest.raises(ValidationError):
        parse_scenario_model_config(_cfg({"kind": "variance_gamma", "nu": nu}))


@pytest.mark.parametrize("p", [-0.1, 1.1, "0.5", True, float("nan")])
def test_prob_crisis_invalid(p) -> None:
    with pytest.raises(ValidationError):
        parse_scenario_model_config(
            _cfg({
                "kind": "regime_switching_gmm",
                "prob_crisis": p,
                "crisis_vol_multipliers": 2.0,
            })
        )


@pytest.mark.parametrize("m", [0.0, -1.0, "2", True, [2.0, -1.0], []])
def test_crisis_multipliers_invalid(m) -> None:
    with pytest.raises(ValidationError):
        parse_scenario_model_config(
            _cfg({
                "kind": "regime_switching_gmm",
                "prob_crisis": 0.1,
                "crisis_vol_multipliers": m,
            })
        )


# ---- config-level foreign params & unknown model -----------------------------


@pytest.mark.parametrize(
    "foreign",
    [
        {"horizon": 2},
        {"expected_block_size": 2.0},
        {"decay_factor": 0.9},
        {"max_vol_ratio": 8.0},
    ],
)
def test_foreign_model_parameters_rejected(foreign) -> None:
    with pytest.raises(ValidationError):
        parse_scenario_model_config(_cfg({"kind": "standard_normal"}, **foreign))


def test_total_scenarios_strict_and_seed_strict() -> None:
    with pytest.raises(ValidationError):
        parse_scenario_model_config(
            _cfg({"kind": "standard_normal"}, total_scenarios=10.0)
        )
    with pytest.raises(ValidationError):
        parse_scenario_model_config(_cfg({"kind": "standard_normal"}, seed=1.5))


# ---- construction, pairing, equivalence, D029 --------------------------------


def test_config_data_pairing_fail_fast() -> None:
    param_cfg = ParametricGeneratorConfig.model_validate(
        _cfg({"kind": "standard_normal"})
    )
    hist_cfg = HistoricalGeneratorConfig(model="historical", horizon=2)
    with pytest.raises(TypeError):  # parametric config + matrix data
        build_scenario_generator(param_cfg, data=_rf_matrix())
    with pytest.raises(TypeError):  # matrix config + parametric data
        build_scenario_generator(hist_cfg, data=_pdata())


def test_build_returns_parametric_generator() -> None:
    cfg = parse_scenario_model_config(_cfg({"kind": "student_t", "df": 5.0}))
    gen = build_scenario_generator(cfg, data=_pdata())
    assert isinstance(gen, ParametricGenerator)
    assert gen.distribution_kind == "student_t" and gen.student_t_df == 5.0
    assert gen.total_scenarios == 10 and gen.seed == 7


def test_build_runs_generator_own_validation() -> None:
    """A df <= 2 is caught at config; but a data/model mismatch the config can't
    see (e.g. per-factor multipliers of wrong length) still fails at
    construction — the generator remains authoritative."""
    cfg = parse_scenario_model_config(
        _cfg({
            "kind": "regime_switching_gmm",
            "prob_crisis": 0.1,
            "crisis_vol_multipliers": [2.0, 3.0],
        })  # length 2 != K=3
    )
    with pytest.raises(ValueError):
        build_scenario_generator(cfg, data=_pdata())


@pytest.mark.parametrize(
    "distribution",
    [
        {"kind": "standard_normal"},
        {"kind": "student_t", "df": 5.0},
        {"kind": "variance_gamma", "nu": 0.5},
        {
            "kind": "regime_switching_gmm",
            "prob_crisis": 0.2,
            "crisis_vol_multipliers": 3.0,
        },
    ],
)
def test_seam_equivalent_to_direct(distribution) -> None:
    cfg = parse_scenario_model_config(_cfg(distribution))
    seamed = build_scenario_generator(cfg, data=_pdata())

    kwargs = dict(
        available_factor_ids=_fids(),
        cholesky_matrix=_chol(),
        volatilities=_vols(),
        total_scenarios=10,
        distribution_kind=distribution["kind"],
        seed=7,
    )
    if distribution["kind"] == "student_t":
        kwargs["student_t_df"] = distribution["df"]
    elif distribution["kind"] == "variance_gamma":
        kwargs["vg_nu"] = distribution["nu"]
    elif distribution["kind"] == "regime_switching_gmm":
        kwargs["prob_crisis"] = distribution["prob_crisis"]
        kwargs["crisis_vol_multipliers"] = distribution["crisis_vol_multipliers"]
    direct = ParametricGenerator(**kwargs)

    rd, rs = _drive(direct), _drive(seamed)
    assert rd.shape == rs.shape
    assert rd.dtype == rs.dtype == np.float32
    assert rs.flags["F_CONTIGUOUS"] == rd.flags["F_CONTIGUOUS"]
    assert np.array_equal(rd, rs)


def test_seeded_seam_preserves_d029_invariance() -> None:
    cfg = parse_scenario_model_config(_cfg({"kind": "student_t", "df": 5.0}, seed=42))
    gen = build_scenario_generator(cfg, data=_pdata())
    assert np.array_equal(_drive(gen, 1), _drive(gen, 7))
    assert np.array_equal(_drive(gen, 1), _drive(gen, 10))
