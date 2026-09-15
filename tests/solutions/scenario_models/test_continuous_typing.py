# tests/solutions/scenario_models/test_continuous_typing.py
"""
D038 — canonical continuous-input typing for the scenario-model configuration
seam, proved uniformly across every continuous parameter of every model.

Contract: a continuous config input is a finite real scalar of the float family
(``int``, ``float``, ``np.integer``, ``np.floating``), in domain, stored as a
Python ``float``. ``bool``, ``str``, ``Decimal``, ``Fraction``, ``NaN`` and
``+/-inf`` are rejected at the solution boundary: representations of numbers are
parsed at EDGE, not coerced here.
"""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest
from pydantic import ValidationError

from riskquant.solutions.scenario_models.api import parse_scenario_model_config

# (label, config factory taking the value, attribute getter) for every
# continuous scalar parameter in the seam. The probe value must be valid in that
# parameter's domain, so only TYPING decides accept/reject.
_FIELDS = [
    (
        "sbb.expected_block_size",
        lambda v: {
            "model": "sbb",
            "horizon": 2,
            "total_scenarios": 5,
            "expected_block_size": v,
            "seed": 1,
        },
        lambda c: c.expected_block_size,
    ),
    (
        "filtered_historical.decay_factor",
        lambda v: {"model": "filtered_historical", "horizon": 2, "decay_factor": v},
        lambda c: c.decay_factor,
    ),
    (
        "filtered_historical.max_vol_ratio",
        lambda v: {"model": "filtered_historical", "horizon": 2, "max_vol_ratio": v},
        lambda c: c.max_vol_ratio,
    ),
    (
        "filtered_bootstrap.decay_factor",
        lambda v: {
            "model": "filtered_bootstrap",
            "horizon": 2,
            "total_scenarios": 5,
            "decay_factor": v,
        },
        lambda c: c.decay_factor,
    ),
    (
        "filtered_bootstrap.max_vol_ratio",
        lambda v: {
            "model": "filtered_bootstrap",
            "horizon": 2,
            "total_scenarios": 5,
            "max_vol_ratio": v,
        },
        lambda c: c.max_vol_ratio,
    ),
    (
        "parametric.df",
        lambda v: {
            "model": "parametric",
            "total_scenarios": 5,
            "distribution": {"kind": "student_t", "df": v},
        },
        lambda c: c.distribution.df,
    ),
    (
        "parametric.nu",
        lambda v: {
            "model": "parametric",
            "total_scenarios": 5,
            "distribution": {"kind": "variance_gamma", "nu": v},
        },
        lambda c: c.distribution.nu,
    ),
    (
        "parametric.prob_crisis",
        lambda v: {
            "model": "parametric",
            "total_scenarios": 5,
            "distribution": {
                "kind": "regime_switching_gmm",
                "prob_crisis": v,
                "crisis_vol_multipliers": 2.0,
            },
        },
        lambda c: c.distribution.prob_crisis,
    ),
    (
        "parametric.crisis_vol_multipliers",
        lambda v: {
            "model": "parametric",
            "total_scenarios": 5,
            "distribution": {
                "kind": "regime_switching_gmm",
                "prob_crisis": 0.1,
                "crisis_vol_multipliers": v,
            },
        },
        lambda c: c.distribution.crisis_vol_multipliers,
    ),
]
_IDS = [f[0] for f in _FIELDS]

# Valid in EVERY field's domain simultaneously: decay_factor needs (0,1),
# max_vol_ratio >= 1, df > 2, nu > 0, prob_crisis in [0,1], ebs >= 1.
# No single value satisfies all, so accepted-type probes use per-field values.
_VALID_PROBE = {
    "sbb.expected_block_size": (2.5, 3),
    "filtered_historical.decay_factor": (0.9, None),
    "filtered_historical.max_vol_ratio": (2.5, 8),
    "filtered_bootstrap.decay_factor": (0.9, None),
    "filtered_bootstrap.max_vol_ratio": (2.5, 8),
    "parametric.df": (2.5, 3),
    "parametric.nu": (0.5, 1),
    "parametric.prob_crisis": (0.5, 1),
    "parametric.crisis_vol_multipliers": (2.5, 3),
}


# ---- accepted types ----------------------------------------------------------


@pytest.mark.parametrize("label,make,get", _FIELDS, ids=_IDS)
def test_python_float_accepted_and_stored_as_float(label, make, get) -> None:
    fval, _ = _VALID_PROBE[label]
    got = get(parse_scenario_model_config(make(fval)))
    assert type(got) is float and got == fval


@pytest.mark.parametrize("label,make,get", _FIELDS, ids=_IDS)
def test_numpy_floating_accepted_and_canonicalized(label, make, get) -> None:
    """np.float64 and np.float32 are real scalars; stored as Python float."""
    fval, _ = _VALID_PROBE[label]
    for np_val in (np.float64(fval), np.float32(fval)):
        got = get(parse_scenario_model_config(make(np_val)))
        assert type(got) is float


@pytest.mark.parametrize("label,make,get", _FIELDS, ids=_IDS)
def test_integer_valued_reals_accepted(label, make, get) -> None:
    """Continuous parameters legitimately take integer-valued reals (D038 §G).
    Fields whose domain excludes every integer are skipped."""
    _, ival = _VALID_PROBE[label]
    if ival is None:
        pytest.skip("domain (0, 1) contains no valid integer")
    for candidate in (ival, np.int64(ival)):
        got = get(parse_scenario_model_config(make(candidate)))
        assert type(got) is float and got == float(ival)


# ---- rejected types ----------------------------------------------------------


@pytest.mark.parametrize("label,make,get", _FIELDS, ids=_IDS)
@pytest.mark.parametrize(
    "bad",
    [
        True,  # bool: subclass of int, but not a numeric value
        "2.5",  # numeric string: a representation, parsed at EDGE
        "not-a-number",
        Decimal("2.5"),  # not float-family
        Fraction(5, 2),  # not float-family
        None,
        2 + 3j,  # complex
    ],
    ids=["bool", "numeric_str", "str", "Decimal", "Fraction", "None", "complex"],
)
def test_non_real_types_rejected(label, make, get, bad) -> None:
    with pytest.raises(ValidationError):
        parse_scenario_model_config(make(bad))


@pytest.mark.parametrize("label,make,get", _FIELDS, ids=_IDS)
@pytest.mark.parametrize(
    "bad", [float("nan"), float("inf"), float("-inf")], ids=["nan", "posinf", "neginf"]
)
def test_non_finite_rejected(label, make, get, bad) -> None:
    with pytest.raises(ValidationError):
        parse_scenario_model_config(make(bad))


@pytest.mark.parametrize("label,make,get", _FIELDS, ids=_IDS)
@pytest.mark.parametrize(
    "bad", [np.float64("nan"), np.float64("inf")], ids=["np_nan", "np_inf"]
)
def test_numpy_non_finite_rejected(label, make, get, bad) -> None:
    """A NumPy scalar passes the type gate, so finiteness must still reject it."""
    with pytest.raises(ValidationError):
        parse_scenario_model_config(make(bad))


# ---- domain boundaries unchanged --------------------------------------------


@pytest.mark.parametrize(
    "raw,ok",
    [
        (
            {
                "model": "sbb",
                "horizon": 2,
                "total_scenarios": 5,
                "expected_block_size": 1.0,
                "seed": 1,
            },
            True,
        ),  # ge=1 closed
        (
            {
                "model": "sbb",
                "horizon": 2,
                "total_scenarios": 5,
                "expected_block_size": 0.999,
                "seed": 1,
            },
            False,
        ),
        ({"model": "filtered_historical", "horizon": 2, "decay_factor": 0.999}, True),
        (
            {"model": "filtered_historical", "horizon": 2, "decay_factor": 1.0},
            False,
        ),  # lt=1 open
        (
            {"model": "filtered_historical", "horizon": 2, "decay_factor": 0.0},
            False,
        ),  # gt=0 open
        (
            {
                "model": "filtered_bootstrap",
                "horizon": 2,
                "total_scenarios": 5,
                "max_vol_ratio": 1.0,
            },
            True,
        ),  # ge=1 closed
        (
            {
                "model": "filtered_bootstrap",
                "horizon": 2,
                "total_scenarios": 5,
                "max_vol_ratio": 0.999,
            },
            False,
        ),
        (
            {
                "model": "parametric",
                "total_scenarios": 5,
                "distribution": {"kind": "student_t", "df": 2.0},
            },
            False,
        ),  # gt=2 open
        (
            {
                "model": "parametric",
                "total_scenarios": 5,
                "distribution": {"kind": "student_t", "df": 2.001},
            },
            True,
        ),
        (
            {
                "model": "parametric",
                "total_scenarios": 5,
                "distribution": {
                    "kind": "regime_switching_gmm",
                    "prob_crisis": 0.0,
                    "crisis_vol_multipliers": 2.0,
                },
            },
            True,
        ),  # closed [0,1]
        (
            {
                "model": "parametric",
                "total_scenarios": 5,
                "distribution": {
                    "kind": "regime_switching_gmm",
                    "prob_crisis": 1.0,
                    "crisis_vol_multipliers": 2.0,
                },
            },
            True,
        ),
        (
            {
                "model": "parametric",
                "total_scenarios": 5,
                "distribution": {
                    "kind": "regime_switching_gmm",
                    "prob_crisis": 1.0001,
                    "crisis_vol_multipliers": 2.0,
                },
            },
            False,
        ),
    ],
)
def test_domain_boundaries_unchanged(raw, ok) -> None:
    if ok:
        parse_scenario_model_config(raw)
    else:
        with pytest.raises(ValidationError):
            parse_scenario_model_config(raw)


# ---- vector semantics of crisis_vol_multipliers preserved --------------------


def _regime(mult):
    return {
        "model": "parametric",
        "total_scenarios": 5,
        "distribution": {
            "kind": "regime_switching_gmm",
            "prob_crisis": 0.1,
            "crisis_vol_multipliers": mult,
        },
    }


@pytest.mark.parametrize(
    "mult,expected",
    [
        (3.0, 3.0),
        (3, 3.0),
        (np.float64(3), 3.0),
        (np.int64(3), 3.0),
    ],
)
def test_multipliers_scalar_form(mult, expected) -> None:
    got = parse_scenario_model_config(_regime(mult)).distribution.crisis_vol_multipliers
    assert type(got) is float and got == expected


@pytest.mark.parametrize(
    "mult",
    [[2.0, 3.0], [2, 3], [np.float64(2), np.int64(3)]],
    ids=["floats", "ints", "numpy_scalars"],
)
def test_multipliers_vector_form_leaves_canonicalized(mult) -> None:
    """The sequence form stays a sequence; D038 typing applies to its leaves."""
    got = parse_scenario_model_config(_regime(mult)).distribution.crisis_vol_multipliers
    assert isinstance(got, list) and len(got) == 2
    assert all(type(x) is float for x in got)
    assert got == [2.0, 3.0]


@pytest.mark.parametrize(
    "mult",
    [
        [2.0, "3.0"],
        [2.0, True],
        [2.0, Decimal("3.0")],
        [2.0, float("nan")],
        [2.0, float("inf")],
        [2.0, -1.0],
        [2.0, 0.0],
        [],
    ],
    ids=[
        "str_leaf",
        "bool_leaf",
        "decimal_leaf",
        "nan_leaf",
        "inf_leaf",
        "negative_leaf",
        "zero_leaf",
        "empty",
    ],
)
def test_multipliers_vector_invalid_leaves_rejected(mult) -> None:
    with pytest.raises(ValidationError):
        parse_scenario_model_config(_regime(mult))
