# tests/solutions/credit_var/test_audit_fixes.py
"""
WS_CREDIT_VAR_API S7: pre-release audit fixes (C.18, C.33 to C.37).

One test group per audit finding:

* S7.T1 (R-001, C.33): UL attribution is 0.0 on a zero-variance book, and
  0.0 for a constant obligor in a varying book.
* S7.T2 (R-001, C.34, Revision 8): UL attribution foots to ``loss_std_dev``
  within ``1e-9 x loss_std_dev + sd(delta)``, with delta the rounding error of
  the summed portfolio loss; the unit values equal a ``math.fsum`` oracle on
  centred rows.
* S7.T3 (R-009, C.35): `obligor_ul_contribution` makes no BLAS call, and the
  result bytes do not depend on the BLAS and OpenMP thread settings.
* S7.T4 (R-002, C.36): ``n_scenarios = 1`` gives a result for every catalog
  key through both entry points; a scalar EAD that is not float64 raises
  and is not cast.
* S7.T5 (R-003, C.18): a float field admits a float or an int only.
* S7.T6 (R-008, C.37): ``provenance.params`` is in ascending key order.

The oracles are independent of the code under test: the expected values are
written out here from A_CONTRACT and the C clauses, the UL oracle uses
``math.fsum`` over centred rows, and the input-hash oracle recomputes the
A_CONTRACT section 4.2 bytes.
"""

from __future__ import annotations

import ast
import decimal
import fractions
import hashlib
import importlib
import inspect
import json
import math
import os
import subprocess
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from pydantic import ValidationError

import riskquant
from riskquant.credit.exposure.obligor_ead.compiled import CompiledObligorEad
from riskquant.credit.jtd.engine import run_jtd_for_obligor_ead
from riskquant.solutions.credit_var import _attribution
from riskquant.solutions.credit_var._attribution import obligor_ul_contribution
from riskquant.solutions.credit_var.api import (
    MODEL_CATALOG,
    CreditVarRequest,
    CreditVarResult,
    run_credit_var,
    simulate_portfolio_losses,
    supported_model_keys,
)

_EXCLUDE_GENERATED_AT: dict[str, Any] = {"provenance": {"generated_at"}}
_FOOT_TOLERANCE = 1e-9  # C.34
_ORACLE_TOLERANCE = 1e-12  # S7.T2 unit oracle


def _obligor(
    uid: int, ead: float, pd: float, lgd: float, *, forced: bool = False
) -> dict[str, Any]:
    row: dict[str, Any] = {"obligor_uid": uid, "ead": ead, "pd": pd, "lgd": lgd}
    if forced:
        row["force_default"] = True
    return row


def _catalog_book(model: str, *, n_scenarios: int) -> dict[str, Any]:
    """A valid book for any catalog key: 3 obligors, catalog default params.

    PDs 0.02, 0.04, 0.06 keep marshall_olkin valid with its default
    systemic_pd. Grouped models get two groups with the catalog defaults;
    multi-factor models load each obligor on one factor.
    """
    entry = MODEL_CATALOG[model]
    obligors = [_obligor(1 + i, 1.0 + i, 0.02 * (i + 1), 0.45) for i in range(3)]
    book: dict[str, Any] = {
        "model": model,
        "obligors": obligors,
        "n_scenarios": n_scenarios,
        "alphas": [0.95, 0.99],
        "seed": 7,
    }
    if entry.requires_groups:
        names = ("G1", "G2")
        book["groups"] = [
            {"name": name, "params": {p.key: p.default for p in entry.params}}
            for name in names
        ]
        for i, row in enumerate(obligors):
            row["group"] = names[i % 2]
    if entry.requires_loadings:
        book["loadings"] = [
            {"obligor_uid": row["obligor_uid"], "factor": "F1", "loading": 0.3}
            for row in obligors
        ]
    return book


def _repro_json(result: CreditVarResult) -> str:
    return result.model_dump_json(exclude=_EXCLUDE_GENERATED_AT)


# ---------------------------------------------------------------------------
# S7.T1 (R-001, C.33): zero variance
# ---------------------------------------------------------------------------

_ZERO_VARIANCE_BOOKS: dict[str, dict[str, Any]] = {
    "forced_default one obligor": {
        "model": "forced_default",
        "obligors": [_obligor(1, 450000.3, 0.02, 1.0)],
        "n_scenarios": 2000,
        "alphas": [0.95, 0.99],
        "seed": 7,
    },
    "forced_default three obligors": {
        "model": "forced_default",
        "obligors": [
            _obligor(1, 1.0, 0.02, 0.45),
            _obligor(2, 2.0, 0.04, 0.45),
            _obligor(3, 3.0, 0.06, 0.45),
        ],
        "n_scenarios": 2000,
        "alphas": [0.95, 0.99],
        "seed": 7,
    },
    "sf_gaussian PD-0 plus forced": {
        "model": "sf_gaussian",
        "obligors": [
            _obligor(1, 0.3, 0.0, 1.0),
            _obligor(2, 0.1, 0.02, 1.0, forced=True),
        ],
        "n_scenarios": 2000,
        "alphas": [0.95, 0.99],
        "seed": 7,
    },
}


@pytest.mark.parametrize(
    "book", list(_ZERO_VARIANCE_BOOKS.values()), ids=list(_ZERO_VARIANCE_BOOKS)
)
def test_s7_t1_zero_variance_book_has_zero_ul(book: dict[str, Any]) -> None:
    result = run_credit_var(book)
    assert result.summary.loss_min == result.summary.loss_max
    for contribution in result.obligor_contributions:
        assert contribution.ul_contribution == 0.0
        assert contribution.ul_pct == 0.0


def test_s7_t1_constant_portfolio_of_varying_rows_has_zero_ul() -> None:
    """Two offsetting rows: each row varies, the portfolio is constant 0.3.

    The ddof = 1 std of 2000 copies of 0.3 is 1.1e-16 from rounding, so only
    the exact min == max test gives 0.0 here; the constant-row rule does not
    apply, because no row is constant.
    """
    first = np.tile(np.array([0.3, 0.0]), 1000)
    second = np.tile(np.array([0.0, 0.3]), 1000)
    obligor_loss = np.ascontiguousarray(np.vstack([first, second]))
    portfolio = obligor_loss.sum(axis=0)
    assert portfolio.min() == portfolio.max()
    contribution, total = obligor_ul_contribution(obligor_loss, portfolio)
    assert total == 0.0
    assert contribution.tolist() == [0.0, 0.0]


def test_s7_t1_constant_row_gives_zero_total() -> None:
    """One constant row 450000.3: its ddof = 1 std is 1.2e-10 from rounding."""
    obligor_loss = np.full((1, 2000), 450000.3, dtype=np.float64)
    contribution, total = obligor_ul_contribution(
        obligor_loss, obligor_loss.sum(axis=0)
    )
    assert total == 0.0
    assert contribution.tolist() == [0.0]


_CONSTANT_LOSS = 450000.3  # forced default, lgd 1.0: loss 450000.3 in each scenario
_CONSTANT_UID = 1


def test_s7_t1_constant_obligor_in_varying_book_has_zero_ul() -> None:
    """C.33 sentence 2 through run_credit_var: loss_min != loss_max.

    The forced obligor has the constant loss 450000.3 in each of 2000
    scenarios. The float64 mean of that row is not 450000.3, so the centred
    row is not exactly zero and its covariance with L is rounding noise. Only
    the constant-row rule gives 0.0. The precondition below checks the inexact
    mean with numpy, independent of the code under test.
    """
    scenarios = 2000
    row = np.full(scenarios, _CONSTANT_LOSS, dtype=np.float64)
    assert float(row.mean()) != _CONSTANT_LOSS  # the centred row is not zero
    book = {
        "model": "sf_gaussian",
        "obligors": [
            _obligor(_CONSTANT_UID, _CONSTANT_LOSS, 0.02, 1.0, forced=True),
            *[_obligor(2 + i, 1.0 + i, 0.02, 0.45) for i in range(5)],
        ],
        "n_scenarios": scenarios,
        "alphas": [0.95, 0.99],
        "seed": 7,
    }
    result = run_credit_var(book)
    assert result.summary.loss_min != result.summary.loss_max
    by_uid = {c.obligor_uid: c for c in result.obligor_contributions}
    constant = by_uid[_CONSTANT_UID]
    assert constant.ul_contribution == 0.0
    assert constant.ul_pct == 0.0


# ---------------------------------------------------------------------------
# S7.T2 (R-001, C.34): footing and the centred-row oracle
# ---------------------------------------------------------------------------

_FOOTING_BOOKS: dict[str, dict[str, Any]] = {
    # The modelled lgd is 1.0, so 1e9 + 0.3 + 1.0 rounds exactly and the
    # summed loss carries no rounding error (owner-accepted deviation).
    "independent_bernoulli forced 1e9+0.3": {
        "model": "independent_bernoulli",
        "obligors": [
            _obligor(1, 1e9 + 0.3, 0.02, 1.0, forced=True),
            _obligor(2, 1.0, 0.02, 1.0),
        ],
        "n_scenarios": 20000,
        "alphas": [0.99],
        "seed": 3,
    },
    "sf_gaussian forced 5e8": {
        "model": "sf_gaussian",
        "obligors": [
            _obligor(1, 5e8, 0.02, 0.45, forced=True),
            *[_obligor(2 + i, 1e6, 0.02, 0.45) for i in range(50)],
        ],
        "n_scenarios": 20000,
        "alphas": [0.99],
        "seed": 3,
    },
}


@pytest.mark.parametrize(
    "book", list(_FOOTING_BOOKS.values()), ids=list(_FOOTING_BOOKS)
)
def test_s7_t2_ul_contributions_foot_to_loss_std_dev(book: dict[str, Any]) -> None:
    """API books, bound 1e-9 x loss_std_dev alone.

    C.34 (Revision 8) allows ``1e-9 x loss_std_dev + sd(delta)``. For these two
    books sd(delta) is negligible (0.0 in the S7 sandbox run): in the first,
    1e9 + 0.3 plus 0.0 or 1.0 rounds exactly; in the second, every obligor
    loss (2.25e8 and multiples of 450000.0) is an integer below 2**53, so
    every sum is exact. The stricter bound therefore holds and the test keeps
    it.
    """
    result = run_credit_var(book)
    summary = result.summary
    assert summary.loss_min != summary.loss_max
    ul_sum = math.fsum(c.ul_contribution for c in result.obligor_contributions)
    assert abs(ul_sum - summary.loss_std_dev) <= _FOOT_TOLERANCE * summary.loss_std_dev


def _rounding_error_sd(obligor_loss: npt.NDArray[np.float64]) -> float:
    """sd(delta), ddof = 1, of the rounding error of the summed loss (C.34).

    ``delta_s = L_s - (exact sum of the obligor losses of scenario s)`` with
    ``L = obligor_loss.sum(axis=0)``, the summation of the engine. Each
    ``delta_s`` is ``math.fsum([L_s, -x_1s, ..., -x_Ns])``: fsum rounds the
    exact value of that sum once, so delta_s is the exact error, correctly
    rounded. ``L_s - math.fsum([x_1s, ..., x_Ns])`` would lose it: fsum rounds
    the exact sum to the float grid of L_s, and for two rows that rounding
    equals L_s, so the difference is 0.0.
    """
    portfolio = obligor_loss.sum(axis=0)
    columns = obligor_loss.T.tolist()
    delta = [
        math.fsum([float(total), *(-value for value in column)])
        for total, column in zip(portfolio.tolist(), columns, strict=True)
    ]
    mean = math.fsum(delta) / len(delta)
    return math.sqrt(math.fsum((d - mean) ** 2 for d in delta) / (len(delta) - 1))


# (constant obligor loss, modelled lgd): the two books of the S7 report at
# max|constant| / loss_std_dev of about 8.2e8 and 6.7e9.
_ROUNDING_BOOKS: dict[str, tuple[float, float]] = {
    "constant 1e8+0.3, lgd 0.9": (1e8 + 0.3, 0.9),
    "constant 3e8+0.1, lgd 0.33": (3e8 + 0.1, 0.33),
}


@pytest.mark.parametrize(
    ("constant", "lgd"), list(_ROUNDING_BOOKS.values()), ids=list(_ROUNDING_BOOKS)
)
def test_s7_t2_footing_needs_the_rounding_term(constant: float, lgd: float) -> None:
    """Synthetic book: one constant row and one Bernoulli row (pd 0.02, ead 1).

    The rounding of L_s = constant + x_s depends on x_s, so it correlates with
    L. loss_std_dev contains it; no centred obligor row does. The footing then
    fails 1e-9 x loss_std_dev alone and holds with the sd(delta) term.
    """
    scenarios = 20000
    rng = np.random.default_rng(1)
    modelled = lgd * (rng.random(scenarios) < 0.02).astype(np.float64)
    obligor_loss = np.ascontiguousarray(
        np.vstack([np.full(scenarios, constant, dtype=np.float64), modelled])
    )
    portfolio = obligor_loss.sum(axis=0)
    assert portfolio.min() != portfolio.max()

    contribution, loss_std_dev = obligor_ul_contribution(obligor_loss, portfolio)
    assert loss_std_dev == float(portfolio.std(ddof=1))  # the RiskSummary call
    assert 1e8 < constant / loss_std_dev < 1e10  # the reported ratio range
    error = abs(math.fsum(contribution.tolist()) - loss_std_dev)
    delta_sd = _rounding_error_sd(obligor_loss)

    assert error > _FOOT_TOLERANCE * loss_std_dev  # the old bound fails
    assert error <= _FOOT_TOLERANCE * loss_std_dev + delta_sd  # C.34 holds


def _fsum_ul_oracle(
    obligor_loss: npt.NDArray[np.float64], portfolio: npt.NDArray[np.float64]
) -> tuple[list[float], float]:
    """Cov((L_i - mean L_i), (L - mean L)) / sigma_L with math.fsum sums."""
    scenarios = portfolio.shape[0]
    pf = [float(v) for v in portfolio]
    pf_mean = math.fsum(pf) / scenarios
    pf_centred = [v - pf_mean for v in pf]
    sigma = math.sqrt(math.fsum(v * v for v in pf_centred) / (scenarios - 1))
    out: list[float] = []
    for row in obligor_loss:
        values = [float(v) for v in row]
        if min(values) == max(values):
            out.append(0.0)
            continue
        mean = math.fsum(values) / scenarios
        cov = math.fsum(
            (v - mean) * p for v, p in zip(values, pf_centred, strict=True)
        ) / (scenarios - 1)
        out.append(cov / sigma)
    return out, sigma


def test_s7_t2_unit_values_equal_centred_fsum_oracle() -> None:
    rng = np.random.default_rng(20261104)
    scenarios = 5000
    modelled = rng.lognormal(0.0, 0.5, size=(5, scenarios)) * (
        rng.random((5, scenarios)) < 0.3
    )
    constant = np.full((1, scenarios), 1e9, dtype=np.float64)
    obligor_loss = np.ascontiguousarray(np.vstack([modelled, constant]))
    portfolio = obligor_loss.sum(axis=0)

    got, total = obligor_ul_contribution(obligor_loss, portfolio)
    want, sigma = _fsum_ul_oracle(obligor_loss, portfolio)

    assert total == float(portfolio.std(ddof=1))  # the RiskSummary call
    assert total == pytest.approx(sigma, rel=_ORACLE_TOLERANCE)
    assert got[-1] == 0.0
    for value, expected in zip(got[:-1], want[:-1], strict=True):
        assert value == pytest.approx(expected, rel=_ORACLE_TOLERANCE)


# ---------------------------------------------------------------------------
# S7.T3 (R-009, C.35): no BLAS call; thread settings do not change the bytes
# ---------------------------------------------------------------------------

_BLAS_NAMES = frozenset({"dot", "matmul", "inner", "vdot", "tensordot", "einsum"})
_BLAS_ENV = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")


def test_s7_t3_obligor_ul_contribution_makes_no_blas_call() -> None:
    source = textwrap.dedent(inspect.getsource(obligor_ul_contribution))
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.BinOp, ast.AugAssign)) and isinstance(
            node.op, ast.MatMult
        ):
            found.append(f"@ at line {node.lineno}")
        if isinstance(node, ast.Call):
            func = node.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else func.id
                if isinstance(func, ast.Name)
                else None
            )
            if name in _BLAS_NAMES:
                found.append(f"{name} at line {node.lineno}")
    assert not found, f"{_attribution.__name__}.obligor_ul_contribution: {found}"


_BLAS_CHILD = """
import json, sys
from riskquant.solutions.credit_var.api import run_credit_var
book = json.loads(sys.stdin.buffer.read().decode("utf-8"))
result = run_credit_var(book)
text = result.model_dump_json(exclude={"provenance": {"generated_at"}})
sys.stdout.buffer.write(text.encode("utf-8"))
"""


def _d2_book() -> dict[str, Any]:
    """Probe D2 book: independent_bernoulli, N = 8, S = 20000, one forced and
    two stressed obligors (``standard_book`` of _gates/audit_credit_var.py)."""
    obligors = [
        _obligor(1 + i, 1.0 + i, 0.02 + 0.02 * (i % 5), 0.45, forced=i == 0)
        for i in range(8)
    ]
    for i in (1, 2):
        obligors[i]["in_stress_test"] = True
    return {
        "model": "independent_bernoulli",
        "obligors": obligors,
        "n_scenarios": 20000,
        "alphas": [0.999, 0.95, 0.99],
        "seed": 11,
    }


def _blas_child(book: dict[str, Any], *, one_thread: bool) -> str:
    assert riskquant.__file__ is not None
    source_root = str(Path(riskquant.__file__).parent.parent)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([
        source_root,
        *filter(None, [env.get("PYTHONPATH")]),
    ])
    for name in _BLAS_ENV:
        env.pop(name, None)
        if one_thread:
            env[name] = "1"
    child = subprocess.run(
        [sys.executable, "-c", _BLAS_CHILD],
        input=json.dumps(book).encode("utf-8"),
        capture_output=True,
        env=env,
        timeout=900,
        check=False,
    )
    assert child.returncode == 0, child.stderr.decode("utf-8", errors="replace")
    return child.stdout.decode("utf-8")


def test_s7_t3_result_bytes_do_not_depend_on_blas_threads() -> None:
    book = _d2_book()
    default = _blas_child(book, one_thread=False)
    single = _blas_child(book, one_thread=True)
    assert default == single


# ---------------------------------------------------------------------------
# S7.T4 (R-002, C.36): n_scenarios = 1
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", supported_model_keys())
def test_s7_t4_one_scenario_gives_a_result(model: str) -> None:
    book = _catalog_book(model, n_scenarios=1)
    result = run_credit_var(book)
    summary = result.summary
    assert summary.loss_min == summary.loss_max
    assert sum(result.loss_histogram.counts) == 1
    for contribution in result.obligor_contributions:
        assert contribution.ul_contribution == 0.0
        assert contribution.ul_pct == 0.0
    losses = simulate_portfolio_losses(book)
    assert losses.shape == (1,)
    assert losses.dtype == np.float64


_RUN_MODULE = importlib.import_module("riskquant.solutions.credit_var.run")


@pytest.mark.parametrize("dtype", [np.float32, np.longdouble, np.int64])
def test_s7_t4_scalar_ead_that_is_not_float64_raises(
    dtype: type[np.generic], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R-002: an (N,) EAD with another dtype raises; no cast to float64 occurs.

    The engine inputs are the real ones of a run_credit_var call at
    n_scenarios = 1, taken at the run_jtd_for_obligor_ead call. Only the EAD
    dtype changes; uid and values stay the same.
    """
    captured: dict[str, Any] = {}

    def capture(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return run_jtd_for_obligor_ead(**kwargs)

    monkeypatch.setattr(_RUN_MODULE, "run_jtd_for_obligor_ead", capture)
    run_credit_var(_catalog_book("sf_gaussian", n_scenarios=1))

    obligor_ead: CompiledObligorEad = captured["obligor_ead"]
    assert obligor_ead.ead.ndim == 1
    assert obligor_ead.ead.dtype == np.float64  # the api caller gives float64
    narrowed = CompiledObligorEad(
        uid=obligor_ead.uid, ead=obligor_ead.ead.astype(dtype)
    )
    with pytest.raises(ValueError, match=r"expected float64\. No cast occurs"):
        run_jtd_for_obligor_ead(
            jtd_portfolio=captured["jtd_portfolio"],
            obligor_ead=narrowed,
            credit_factors=captured["credit_factors"],
        )


# ---------------------------------------------------------------------------
# S7.T5 (R-003, C.18): float fields admit float and int only
# ---------------------------------------------------------------------------


class _FloatLike:
    """An object with ``__float__`` and no other numeric protocol."""

    def __init__(self, value: float) -> None:
        self._value = value

    def __float__(self) -> float:
        return self._value


_REJECTED_FLOAT_INPUTS: dict[str, Callable[[float], object]] = {
    "decimal.Decimal": lambda v: decimal.Decimal(str(v)),
    "fractions.Fraction": lambda v: fractions.Fraction(str(v)),
    "numpy.float16": np.float16,
    "numpy.float32": np.float32,
    "numpy.longdouble": np.longdouble,
    "numpy.int64": lambda v: np.int64(1),
    "numpy.uint8": lambda v: np.uint8(1),
    "numpy.bool_": lambda v: np.bool_(True),
    "0-d numpy array": np.array,
    "object with __float__": _FloatLike,
}


def _sf_book() -> dict[str, Any]:
    obligors = [
        _obligor(1, 1.0, 0.02, 0.45),
        _obligor(2, 2.0, 0.04, 0.45),
    ]
    obligors[0]["rho"] = 0.25
    return {
        "model": "sf_gaussian",
        "obligors": obligors,
        "params": {"rho": 0.15},
        "n_scenarios": 200,
        "alphas": [0.95, 0.99],
        "seed": 7,
    }


def _mf_book() -> dict[str, Any]:
    book = _catalog_book("mf_gaussian", n_scenarios=200)
    book["loadings"][0]["loading"] = 0.5
    return book


def _grouped_book() -> dict[str, Any]:
    book = _catalog_book("beta_binomial_grouped", n_scenarios=200)
    book["groups"][0]["params"]["alpha"] = 1.0
    return book


# (field, book builder, path, canonical float value, valid int value or None)
_FLOAT_FIELDS: list[
    tuple[str, Callable[[], dict[str, Any]], tuple[str | int, ...], float, int | None]
] = [
    ("ObligorRow.ead", _sf_book, ("obligors", 0, "ead"), 1.0, 1),
    ("ObligorRow.pd", _sf_book, ("obligors", 0, "pd"), 0.02, 0),
    ("ObligorRow.lgd", _sf_book, ("obligors", 0, "lgd"), 0.45, 1),
    ("ObligorRow.rho", _sf_book, ("obligors", 0, "rho"), 0.25, 0),
    ("LoadingRow.loading", _mf_book, ("loadings", 0, "loading"), 0.5, 0),
    (
        "GroupRow.params value",
        _grouped_book,
        ("groups", 0, "params", "alpha"),
        1.0,
        1,
    ),
    ("CreditVarRequest.params value", _sf_book, ("params", "rho"), 0.15, None),
    # No int lies in (0, 1).
    ("CreditVarRequest.alphas element", _sf_book, ("alphas", 0), 0.95, None),
]


def _set(book: dict[str, Any], path: tuple[str | int, ...], value: object) -> None:
    target: Any = book
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def _get(obj: Any, path: tuple[str | int, ...]) -> Any:
    for key in path:
        obj = obj[key]
    return obj


@pytest.mark.parametrize(
    ("field", "build", "path", "canonical"),
    [(f, b, p, c) for f, b, p, c, _ in _FLOAT_FIELDS],
    ids=[f for f, *_ in _FLOAT_FIELDS],
)
@pytest.mark.parametrize(
    "make_input",
    list(_REJECTED_FLOAT_INPUTS.values()),
    ids=list(_REJECTED_FLOAT_INPUTS),
)
def test_s7_t5_float_field_rejects_other_types(
    field: str,
    build: Callable[[], dict[str, Any]],
    path: tuple[str | int, ...],
    canonical: float,
    make_input: Callable[[float], object],
) -> None:
    book = build()
    _set(book, path, make_input(canonical))
    with pytest.raises(ValidationError) as excinfo:
        CreditVarRequest.model_validate(book)
    errors = excinfo.value.errors()
    assert len(errors) == 1, f"{field}: {errors}"
    assert errors[0]["loc"] == path
    assert errors[0]["type"] == "float_type"


@pytest.mark.parametrize(
    ("field", "build", "path", "canonical", "as_int"),
    _FLOAT_FIELDS,
    ids=[f for f, *_ in _FLOAT_FIELDS],
)
def test_s7_t5_float_field_accepts_float_int_and_float_subclass(
    field: str,
    build: Callable[[], dict[str, Any]],
    path: tuple[str | int, ...],
    canonical: float,
    as_int: int | None,
) -> None:
    controls: list[object] = [canonical, np.float64(canonical)]
    if as_int is not None:
        controls.append(as_int)
    for value in controls:
        book = build()
        _set(book, path, value)
        request = CreditVarRequest.model_validate(book)
        stored = _get(request.model_dump(mode="python"), path)
        assert type(stored) is float, f"{field}: {type(value).__name__}"
        assert stored == float(value)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# S7.T6 (R-008, C.37): params key order
# ---------------------------------------------------------------------------


def _canonical(value: Any) -> Any:
    """A_CONTRACT section 4.2 step 2: float -> {"__f64__": hex}, tuple -> list."""
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return {"__f64__": value.hex()}
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


def _contract_input_hash(request: CreditVarRequest) -> str:
    data = json.dumps(
        _canonical(request.model_dump(mode="python")),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def test_s7_t6_params_key_order_does_not_change_the_result() -> None:
    def book(params: dict[str, float]) -> dict[str, Any]:
        return {
            "model": "sf_students_t",
            "obligors": [
                _obligor(1, 1.0, 0.02, 0.45),
                _obligor(2, 2.0, 0.04, 0.45),
                _obligor(3, 3.0, 0.06, 0.45),
            ],
            "params": params,
            "n_scenarios": 500,
            "alphas": [0.95, 0.99],
            "seed": 7,
        }

    rho_first = book({"rho": 0.2, "nu": 5.0})
    nu_first = book({"nu": 5.0, "rho": 0.2})
    first = run_credit_var(rho_first)
    second = run_credit_var(nu_first)

    expected_hash = _contract_input_hash(CreditVarRequest.model_validate(rho_first))
    assert first.provenance.input_hash == expected_hash
    assert second.provenance.input_hash == expected_hash
    assert list(first.provenance.params) == ["nu", "rho"]
    assert list(second.provenance.params) == ["nu", "rho"]
    assert _repro_json(first) == _repro_json(second)
