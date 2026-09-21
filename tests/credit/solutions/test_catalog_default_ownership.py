# tests/credit/solutions/test_catalog_default_ownership.py
"""
P2-09: one owner per model-parameter default, across the whole catalog.

`MODEL_CATALOG` is exported from `solutions.credit_var.api`, so
`ParamSpec.default` is a PUBLISHED value an author or front-end reads. It was
declared as a literal in `catalog.py` while the value an omitted param is
actually modelled with lives in `recipes.py` (sourced from `constants.py`). Six
of the seven pairs happened to agree; `rho` had already drifted — catalog 0.20
against an effective 0.15 — so the catalog advertised a model the engine would
not run.

These tests assert single ownership, not equality of two literals:

  * structural — every ParamSpec default is a NAME bound to the shared constant,
    so a future edit cannot re-fork the value even at the same number;
  * operational — for every non-grouped parameterised model, the value observed
    at model construction is compared at runtime against the published catalog
    default, never against a number written here.

Grouped models are deliberately excluded from the operational check: their
schema validator requires every group to supply every param, so no runtime
default is ever resolved for them.
"""

from __future__ import annotations

import ast
import inspect
import pathlib

import pandas as pd
import pytest

from riskquant.solutions.credit_var import catalog, recipes
from riskquant.solutions.credit_var.catalog import MODEL_CATALOG
from riskquant.solutions.credit_var.recipes import assemble_inputs

# param key -> the name recipes.py binds its runtime default to
RUNTIME_BINDING = {
    "rho": "_DEFAULT_RHO",
    "nu": "_DEFAULT_NU",
    "alpha": "_DEFAULT_BETA_ALPHA",
    "beta": "_DEFAULT_BETA_BETA",
    "stress_prob": "_DEFAULT_STRESS_PROB",
    "stress_pd_multiplier": "_DEFAULT_STRESS_MULT",
    "systemic_pd": "_DEFAULT_MO_SYSTEMIC_PD",
}


def _published_defaults() -> dict[str, float]:
    out: dict[str, float] = {}
    for entry in MODEL_CATALOG.values():
        for spec in entry.params:
            out.setdefault(spec.key, spec.default)
    return out


def _table(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame({
        "uid": list(range(1, n + 1)),
        "ead": [1_000_000.0] * n,
        "pd": [0.30] * n,  # high enough for every model's own constraints
        "lgd": [0.5] * n,
    })


# ==========================================================================
# 1. STRUCTURAL — no ParamSpec may carry its own literal
# ==========================================================================
def test_every_param_spec_default_is_a_name_not_a_literal() -> None:
    """The drift guard for the whole catalog. Re-introducing
    `default=0.20` fails here even if it matches the engine that day."""
    tree = ast.parse(pathlib.Path(inspect.getsourcefile(catalog)).read_text())
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and getattr(node.value.func, "id", "") == "ParamSpec"
        ):
            continue
        name = node.targets[0].id
        default_kw = next(k for k in node.value.keywords if k.arg == "default")
        if not isinstance(default_kw.value, ast.Name):
            offenders.append(f"{name}: default={ast.unparse(default_kw.value)}")
    assert not offenders, (
        "ParamSpec defaults must be bound to the shared constant in "
        f"constants.py, not restated as literals: {offenders}"
    )


def test_every_param_spec_default_names_a_constants_module_symbol() -> None:
    from riskquant.solutions.credit_var import constants

    tree = ast.parse(pathlib.Path(inspect.getsourcefile(catalog)).read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and getattr(node.value.func, "id", "") == "ParamSpec"
        ):
            default_kw = next(k for k in node.value.keywords if k.arg == "default")
            assert hasattr(constants, default_kw.value.id), default_kw.value.id


@pytest.mark.parametrize("key,binding", sorted(RUNTIME_BINDING.items()))
def test_published_default_is_the_same_object_as_the_runtime_default(
    key, binding
) -> None:
    """Identity, not equality: a re-forked literal is a different object even at
    the same value."""
    published = _published_defaults()[key]
    runtime = getattr(recipes, binding)
    assert published is runtime, f"{key}: catalog and recipe no longer share one owner"


def test_the_catalog_covers_every_runtime_default() -> None:
    """A recipe default with no published counterpart would be an unadvertised
    knob; a published param with no runtime binding would be unreachable."""
    assert set(_published_defaults()) == set(RUNTIME_BINDING)


# ==========================================================================
# 2. OPERATIONAL — the published default is the one the model is built with
# ==========================================================================
def _assembled_universe(model_key: str, params: dict[str, float]):
    assembled = assemble_inputs(
        model_key=model_key, obligor_table=_table(), params=params
    )
    return assembled


@pytest.mark.parametrize(
    "model_key,key,reader",
    [
        ("sf_students_t", "nu", lambda u, b: float(u.chi_nu_vector[0])),
        ("beta_binomial", "alpha", lambda u, b: float(u.beta_alpha[0])),
        ("beta_binomial", "beta", lambda u, b: float(u.beta_beta[0])),
        (
            "marshall_olkin",
            "systemic_pd",
            lambda u, b: float(b.marshall_olkin_df["systemic_pd"].iloc[0]),
        ),
    ],
)
def test_omitted_param_is_modelled_with_the_published_default(
    model_key, key, reader
) -> None:
    assembled = _assembled_universe(model_key, {})
    effective = reader(assembled.masters.factor_universe, assembled.bundle)
    assert effective == _published_defaults()[key]


def test_omitted_rho_is_modelled_with_the_published_default() -> None:
    """rho reaches the block as the loading sqrt(rho), so recover it by squaring.
    This is the parameter that had actually drifted."""
    assembled = _assembled_universe("sf_gaussian", {})
    frame = assembled.bundle.sf_gaussian_copula_df
    assert frame is not None
    loading = float(frame["factor_loading"].iloc[0])
    assert loading**2 == pytest.approx(_published_defaults()["rho"], rel=1e-12)


@pytest.mark.parametrize("explicit", [0.05, 0.33, 0.75])
def test_explicit_rho_is_not_replaced_by_any_default(explicit) -> None:
    assembled = _assembled_universe("sf_gaussian", {"rho": explicit})
    frame = assembled.bundle.sf_gaussian_copula_df
    loading = float(frame["factor_loading"].iloc[0])
    assert loading**2 == pytest.approx(explicit, rel=1e-12)
    assert loading**2 != pytest.approx(_published_defaults()["rho"], rel=1e-9)


@pytest.mark.parametrize("explicit", [3.0, 12.5, 40.0])
def test_explicit_nu_is_not_replaced_by_any_default(explicit) -> None:
    assembled = _assembled_universe("sf_students_t", {"rho": 0.2, "nu": explicit})
    assert float(assembled.masters.factor_universe.chi_nu_vector[0]) == explicit


# ==========================================================================
# 3. GROUPED MODELS — no runtime default exists, by contract
# ==========================================================================
@pytest.mark.parametrize(
    "model_key", ["beta_binomial_grouped", "stress_regime_bernoulli_grouped"]
)
def test_grouped_models_resolve_no_default_at_all(model_key) -> None:
    """Their assemblers index `gp["<param>"]` directly, so an omitted per-group
    param is a KeyError, never a silent default. The published ParamSpec default
    is authoring metadata for these models only."""
    source = pathlib.Path(inspect.getsourcefile(recipes)).read_text()
    tree = ast.parse(source)
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == f"_assemble_{model_key}"
    )
    calls = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and getattr(n.func, "attr", None) == "get"
        and len(n.args) >= 2
    ]
    assert not calls, f"{model_key} must not resolve a per-group param default"


def test_grouped_request_requires_every_param_per_group() -> None:
    from pydantic import ValidationError

    from riskquant.solutions.credit_var.schemas import CreditVarRequest

    book = pd.DataFrame({
        "obligor_uid": [1, 2],
        "ead": [1e6, 1e6],
        "pd": [0.3, 0.3],
        "lgd": [0.5, 0.5],
        "group": ["A", "A"],
    })
    with pytest.raises(ValidationError, match="missing required param"):
        CreditVarRequest.from_dataframe(
            obligor_table=book,
            model="beta_binomial_grouped",
            group_params={"A": {"alpha": 2.0}},  # 'beta' omitted
            n_scenarios=1_000,
            alphas=[0.99],
            seed=1,
        )


# ==========================================================================
# 4. THE REGRESSION THAT MOTIVATED P2-09
# ==========================================================================
def test_rho_no_longer_advertises_a_value_the_engine_will_not_use() -> None:
    """Before this tranche: catalog 0.20, engine 0.15. A request omitting rho was
    validated and displayed against one correlation and modelled with another."""
    published = _published_defaults()["rho"]
    assembled = _assembled_universe("sf_gaussian", {})
    loading = float(assembled.bundle.sf_gaussian_copula_df["factor_loading"].iloc[0])
    assert loading**2 == pytest.approx(published, rel=1e-12)
    assert published == pytest.approx(recipes._DEFAULT_RHO, rel=1e-12)
