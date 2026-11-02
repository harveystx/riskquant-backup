# tests/solutions/credit_var/test_oracles.py
"""
WS_CREDIT_VAR_API S4: one independent oracle per catalog key (C.26, C.27).

S4.T1 to S4.T13 run each MODEL_CATALOG key through
`riskquant.solutions.credit_var.api` and compare EL, VaR and ES with the loss
distribution of A_ORACLES.md section 2. The distributions come from
`_oracle_pmf`, which uses numpy, scipy and closed forms only. S4.T14 checks
the imports of both files by AST (C.27).

Order in each test (A_ORACLES rules 4, 6 and 7):
1. build the oracle distribution;
2. rule 6 self-check: total mass 1 within 1e-10, EL equal to the closed-form
   mean within 1e-8 (R2-02: this runs before any K12 assertion);
3. rule 4 at each level of section 3, and at least 2 levels;
4. one engine run;
5. rule 5 assertions: VaR exact, EL and ES within the stated bounds.

EAD = 1 and LGD = 1 per obligor unless a key states other values (rule 2), so
every loss atom is an exact float.
"""

from __future__ import annotations

import ast
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

import tests.solutions.credit_var._oracle_pmf as oracle
from riskquant.solutions.credit_var.api import CreditVarResult, run_credit_var

#: A_ORACLES rule 7.
_SEED = 20261029
_S = 100_000
_CANDIDATES: tuple[float, ...] = (0.95, 0.975, 0.99, 0.995, 0.999)

#: A_ORACLES rule 6.
_MASS_TOL = 1e-10
_EL_SELF_TOL = 1e-8

#: A_ORACLES K1: relative tolerance of the constant-loss case.
_K1_REL = 1e-12


# ---------------------------------------------------------------------------
# Request rows
# ---------------------------------------------------------------------------


def _obligor(
    uid: int, pd: float, *, ead: float = 1.0, lgd: float = 1.0, group: str | None = None
) -> dict[str, Any]:
    row: dict[str, Any] = {"obligor_uid": uid, "ead": ead, "pd": pd, "lgd": lgd}
    if group is not None:
        row["group"] = group
    return row


def _homogeneous(n: int, pd: float) -> list[dict[str, Any]]:
    return [_obligor(uid, pd) for uid in range(1, n + 1)]


def _request(model: str, alphas: tuple[float, ...], **parts: Any) -> dict[str, Any]:
    return {
        "model": model,
        "n_scenarios": _S,
        "alphas": list(alphas),
        "seed": _SEED,
        **parts,
    }


def _two_sector_loadings() -> list[dict[str, Any]]:
    """K11, K12: sector A (uids 1 to 10) loads F1 = 0.5; sector B (uids 11 to
    20) loads F1 = 0.3 and F2 = 0.4."""
    rows = [
        {"obligor_uid": uid, "factor": "F1", "loading": 0.5} for uid in range(1, 11)
    ]
    for uid in range(11, 21):
        rows.append({"obligor_uid": uid, "factor": "F1", "loading": 0.3})
        rows.append({"obligor_uid": uid, "factor": "F2", "loading": 0.4})
    return rows


# ---------------------------------------------------------------------------
# Cases K2 to K13 (A_ORACLES section 2, levels of section 3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Case:
    test_id: str
    model: str
    levels: tuple[float, ...]
    pmf: Callable[[], oracle.LossPmf]
    closed_form_el: float
    parts: dict[str, Any] = field(default_factory=dict)


def _k2() -> _Case:
    ead, pd = (1, 2, 4), (0.05, 0.10, 0.20)
    return _Case(
        "S4.T2",
        "independent_bernoulli",
        (0.95, 0.975, 0.99, 0.995),
        lambda: oracle.independent_bernoulli(ead, pd),
        math.fsum(e * p for e, p in zip(ead, pd, strict=True)),
        {
            "obligors": [
                _obligor(uid, p, ead=float(e))
                for uid, (e, p) in enumerate(zip(ead, pd, strict=True), start=1)
            ]
        },
    )


def _k3() -> _Case:
    n, a, b = 20, 2.0, 38.0
    return _Case(
        "S4.T3",
        "beta_binomial",
        (0.95, 0.975, 0.99),
        lambda: oracle.beta_binomial(n, a, b),
        n * a / (a + b),
        # Master PD 0.30 is ignored by contract (D-C18 adjudication, C-18).
        {"obligors": _homogeneous(n, 0.30), "params": {"alpha": a, "beta": b}},
    )


def _k4() -> _Case:
    n, (a_a, b_a), (a_b, b_b) = 10, (1.0, 99.0), (4.0, 16.0)
    obligors = [_obligor(uid, 0.30, group="A") for uid in range(1, 11)]
    obligors += [_obligor(uid, 0.30, ead=2.0, group="B") for uid in range(11, 21)]
    return _Case(
        "S4.T4",
        "beta_binomial_grouped",
        (0.95, 0.99),
        lambda: oracle.weighted_sum(
            oracle.beta_binomial_count(n, a_a, b_a),
            1,
            oracle.beta_binomial_count(n, a_b, b_b),
            2,
        ),
        n * a_a / (a_a + b_a) + 2 * n * a_b / (a_b + b_b),
        {
            "obligors": obligors,
            "groups": [
                {"name": "A", "params": {"alpha": a_a, "beta": b_a}},
                {"name": "B", "params": {"alpha": a_b, "beta": b_b}},
            ],
        },
    )


def _stress_mean(n: int, pd: float, q: float, m: float) -> float:
    return n * ((1.0 - q) * pd + q * min(1.0, m * pd))


def _k5() -> _Case:
    n, pd, q, m = 20, 0.02, 0.10, 3.0
    return _Case(
        "S4.T5",
        "stress_regime_bernoulli",
        (0.95, 0.975, 0.99, 0.995),
        lambda: oracle.stress_regime_bernoulli(n, pd, q, m),
        _stress_mean(n, pd, q, m),
        {
            "obligors": _homogeneous(n, pd),
            "params": {"stress_prob": q, "stress_pd_multiplier": m},
        },
    )


def _k6() -> _Case:
    n = 10
    pd_a, q_a, m_a = 0.02, 0.10, 3.0
    pd_b, q_b, m_b = 0.05, 0.20, 4.0
    obligors = [_obligor(uid, pd_a, group="A") for uid in range(1, 11)]
    obligors += [_obligor(uid, pd_b, ead=2.0, group="B") for uid in range(11, 21)]
    return _Case(
        "S4.T6",
        "stress_regime_bernoulli_grouped",
        (0.95, 0.995),
        lambda: oracle.weighted_sum(
            oracle.stress_regime_count(n, pd_a, q_a, m_a),
            1,
            oracle.stress_regime_count(n, pd_b, q_b, m_b),
            2,
        ),
        _stress_mean(n, pd_a, q_a, m_a) + 2 * _stress_mean(n, pd_b, q_b, m_b),
        {
            "obligors": obligors,
            "groups": [
                {
                    "name": "A",
                    "params": {"stress_prob": q_a, "stress_pd_multiplier": m_a},
                },
                {
                    "name": "B",
                    "params": {"stress_prob": q_b, "stress_pd_multiplier": m_b},
                },
            ],
        },
    )


def _k7() -> _Case:
    n, pd, rho = 20, 0.02, 0.15
    return _Case(
        "S4.T7",
        "sf_gaussian",
        (0.95, 0.99, 0.995),
        lambda: oracle.sf_gaussian(n, pd, rho),
        n * pd,
        {"obligors": _homogeneous(n, pd), "params": {"rho": rho}},
    )


def _k8() -> _Case:
    n, pd = 20, 0.02
    return _Case(
        "S4.T8",
        "sf_gaussian_basel",
        (0.95, 0.995),
        lambda: oracle.sf_gaussian(n, pd, oracle.basel_corporate_rho(pd)),
        n * pd,
        {"obligors": _homogeneous(n, pd)},
    )


def _k9() -> _Case:
    n, pd, rho, nu = 20, 0.02, 0.15, 7.0
    return _Case(
        "S4.T9",
        "sf_students_t",
        (0.95, 0.99),
        lambda: oracle.sf_students_t(n, pd, rho, nu),
        n * pd,
        {"obligors": _homogeneous(n, pd), "params": {"rho": rho, "nu": nu}},
    )


def _k10() -> _Case:
    n, pd, nu = 20, 0.02, 7.0
    return _Case(
        "S4.T10",
        "sf_students_t_basel",
        (0.95, 0.99),
        lambda: oracle.sf_students_t(n, pd, oracle.basel_corporate_rho(pd), nu),
        n * pd,
        {"obligors": _homogeneous(n, pd), "params": {"nu": nu}},
    )


def _k11() -> _Case:
    pd = 0.02
    return _Case(
        "S4.T11",
        "mf_gaussian",
        (0.95, 0.975),
        lambda: oracle.mf_gaussian(10, (0.5, 0.0), 10, (0.3, 0.4), pd),
        20 * pd,
        {"obligors": _homogeneous(20, pd), "loadings": _two_sector_loadings()},
    )


def _k12() -> _Case:
    pd, nu = 0.015, 7.0
    return _Case(
        "S4.T12",
        "mf_students_t",
        (0.95, 0.975),
        lambda: oracle.mf_students_t(10, (0.5, 0.0), 10, (0.3, 0.4), pd, nu),
        20 * pd,
        {
            "obligors": _homogeneous(20, pd),
            "loadings": _two_sector_loadings(),
            "params": {"nu": nu},
        },
    )


def _k13() -> _Case:
    n, pd, s = 20, 0.05, 0.02
    return _Case(
        "S4.T13",
        "marshall_olkin",
        (0.95, 0.99, 0.995, 0.999),
        lambda: oracle.marshall_olkin(n, pd, s),
        n * pd,
        {"obligors": _homogeneous(n, pd), "params": {"systemic_pd": s}},
    )


_CASES: tuple[_Case, ...] = tuple(
    build()
    for build in (_k2, _k3, _k4, _k5, _k6, _k7, _k8, _k9, _k10, _k11, _k12, _k13)
)


# ---------------------------------------------------------------------------
# Rule checks
# ---------------------------------------------------------------------------


def _assert_rule6(case: _Case, pmf: oracle.LossPmf) -> None:
    mass_error = pmf.mass - 1.0
    el_error = pmf.mean() - case.closed_form_el
    assert abs(mass_error) <= _MASS_TOL, f"{case.model}: mass - 1 = {mass_error!r}"
    assert abs(el_error) <= _EL_SELF_TOL, (
        f"{case.model}: oracle EL - closed form = {el_error!r}"
    )


def _assert_rule4(case: _Case, pmf: oracle.LossPmf) -> None:
    assert set(case.levels) <= set(_CANDIDATES), case.levels
    assert len(case.levels) >= 2, f"{case.model}: fewer than 2 levels"
    for alpha in case.levels:
        gap = oracle.atom_gap(pmf, alpha, _S)
        assert gap.holds, f"{case.model}: rule 4 fails: {gap}"


# ---------------------------------------------------------------------------
# S4.T1: forced_default (A_ORACLES K1; rules 4 and 5 do not apply)
# ---------------------------------------------------------------------------


def test_s4_t1_forced_default() -> None:
    ead = (1.0e6, 2.5e6, 3.0e5, 7.0e5)
    lgd = (0.45, 0.6, 1.0, 0.25)
    obligors = [
        _obligor(uid, 0.01, ead=e, lgd=g)
        for uid, (e, g) in enumerate(zip(ead, lgd, strict=True), start=1)
    ]
    loss = math.fsum(e * g for e, g in zip(ead, lgd, strict=True))

    result = run_credit_var(_request("forced_default", _CANDIDATES, obligors=obligors))

    summary = result.summary
    assert math.isclose(summary.loss_mean, loss, rel_tol=_K1_REL, abs_tol=0.0)
    assert summary.loss_std_dev == 0.0
    for alpha in _CANDIDATES:
        record = summary.at(alpha)
        assert math.isclose(record.var, loss, rel_tol=_K1_REL, abs_tol=0.0), alpha
        assert math.isclose(record.es, loss, rel_tol=_K1_REL, abs_tol=0.0), alpha
    contributions = {c.obligor_uid: c for c in result.obligor_contributions}
    assert sorted(contributions) == [1, 2, 3, 4]
    for uid, (e, g) in enumerate(zip(ead, lgd, strict=True), start=1):
        assert math.isclose(
            contributions[uid].es_contribution, e * g, rel_tol=_K1_REL, abs_tol=0.0
        ), uid
    assert math.isclose(
        summary.expected_loss_by_model["FORCED_DEFAULT"],
        loss,
        rel_tol=_K1_REL,
        abs_tol=0.0,
    )


# ---------------------------------------------------------------------------
# S4.T2 to S4.T13
# ---------------------------------------------------------------------------


def _assert_rule5(case: _Case, pmf: oracle.LossPmf, result: CreditVarResult) -> None:
    summary = result.summary
    el_error = summary.loss_mean - pmf.mean()
    el_bound = oracle.el_bound(pmf, _S)
    assert abs(el_error) <= el_bound, (
        f"{case.model}: EL_mc {summary.loss_mean!r} oracle {pmf.mean()!r} "
        f"error {el_error!r} bound {el_bound!r}"
    )
    for alpha in case.levels:
        record = summary.at(alpha)
        expected_var = float(pmf.var(alpha))
        assert record.var == expected_var, (
            f"{case.model} a={alpha}: VaR engine {record.var!r} oracle {expected_var!r}"
        )
        expected_es = pmf.es(alpha)
        es_bound = oracle.es_bound(pmf, alpha, _S)
        assert abs(record.es - expected_es) <= es_bound, (
            f"{case.model} a={alpha}: ES engine {record.es!r} oracle "
            f"{expected_es!r} error {record.es - expected_es!r} bound {es_bound!r}"
        )


@pytest.mark.parametrize("case", _CASES, ids=[f"{c.test_id}-{c.model}" for c in _CASES])
def test_s4_t2_to_t13_oracle(case: _Case) -> None:
    pmf = case.pmf()
    _assert_rule6(case, pmf)
    _assert_rule4(case, pmf)
    result = run_credit_var(_request(case.model, case.levels, **case.parts))
    assert result.model == case.model
    _assert_rule5(case, pmf, result)


# ---------------------------------------------------------------------------
# S4.T14 (C.27): import check by AST
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_EXTERNAL = frozenset({"numpy", "scipy"})


def _imported_modules(path: Path) -> list[str]:
    """Each module that an import statement in `path` names. A relative import
    is returned with its leading dots, so it never matches an allowed name."""
    modules: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append("." * node.level + (node.module or ""))
    return modules


def _dynamic_imports(path: Path) -> list[str]:
    """Calls that import by name at runtime: `__import__` and any attribute
    `import_module`."""
    found: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "__import__":
                found.append("__import__")
            elif isinstance(func, ast.Attribute) and func.attr == "import_module":
                found.append("import_module")
    return found


def _is_stdlib(module: str) -> bool:
    return (
        module.split(".", maxsplit=1)[0] in sys.stdlib_module_names
        or module == "__future__"
    )


def _is_external(module: str) -> bool:
    return module.split(".", maxsplit=1)[0] in _EXTERNAL


#: C.27: the non-stdlib, non-numpy, non-scipy modules each file may import.
_ALLOWED: dict[str, frozenset[str]] = {
    "test_oracles.py": frozenset({
        "riskquant.solutions.credit_var.api",
        "tests.solutions.credit_var._oracle_pmf",
        "pytest",
    }),
    "_oracle_pmf.py": frozenset(),
}


def test_s4_t14_imports_are_limited() -> None:
    """C.27: each import of test_oracles.py and _oracle_pmf.py names the
    standard library, numpy, scipy or a module of `_ALLOWED`, and no call
    imports a module by name at runtime."""
    for file_name, allowed in _ALLOWED.items():
        path = _HERE / file_name
        modules = _imported_modules(path)
        assert modules, f"{file_name}: no import found"
        rejected = sorted({
            m for m in modules if not (_is_stdlib(m) or _is_external(m) or m in allowed)
        })
        assert rejected == [], f"{file_name}: imports outside C.27: {rejected}"
        assert _dynamic_imports(path) == [], f"{file_name}: dynamic import"
