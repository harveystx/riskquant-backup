# tests/credit/solutions/test_marshall_olkin_default_ownership.py
"""
P2-08: exactly one authoritative default for the Marshall-Olkin shock probability.

After P2-07 there were two independently maintained 0.02 literals on the same
path: `catalog._SYSTEMIC_PD.default`, which the request validator resolves an
omitted param against, and `recipes._DEFAULT_MO_SYSTEMIC_PD`, which is what
actually reaches the builder (`run.py` forwards `request.params` verbatim, so
the recipe — not the request — fills the omission). Had the two drifted, a
request could validate against one shock probability and then be modelled with
another.

These tests do NOT assert that two constants are equal; that would preserve the
duplicated ownership. They assert single ownership:

  * structurally — the recipe default is a NAME, not a numeric literal, so the
    value cannot be re-forked by editing one file;
  * operationally — the value observed at the validation boundary and the value
    observed at the model-construction boundary are read at runtime and compared
    to each other, never to a number written in this file.
"""

from __future__ import annotations

import ast
import inspect
import pathlib

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from riskquant.solutions.credit_var import catalog, recipes
from riskquant.solutions.credit_var.catalog import model_entry
from riskquant.solutions.credit_var.constants import SF_MO_SYSTEMIC_PD
from riskquant.solutions.credit_var.recipes import assemble_inputs
from riskquant.solutions.credit_var.schemas import CreditVarRequest

MODEL = "marshall_olkin"


def _spec():
    """The ParamSpec the request validator resolves an omitted param against."""
    return next(p for p in model_entry(MODEL).params if p.key == "systemic_pd")


def _recipe_table(pds: list[float]) -> pd.DataFrame:
    """The recipe's own column contract (`uid`), not the request's."""
    return pd.DataFrame({
        "uid": list(range(1, len(pds) + 1)),
        "ead": [1_000_000.0] * len(pds),
        "pd": pds,
        "lgd": [0.5] * len(pds),
    })


def _effective_systemic_pd_at_construction(params: dict[str, float]) -> float:
    """The value the builder will actually see, read out of the assembled bundle."""
    assembled = assemble_inputs(
        model_key=MODEL, obligor_table=_recipe_table([0.5, 0.5]), params=params
    )
    frame = assembled.bundle.marshall_olkin_df
    assert frame is not None
    column = frame["systemic_pd"].to_numpy(dtype=np.float64)
    assert np.all(column == column[0])  # one shock, one probability
    return float(column[0])


# ==========================================================================
# 1. STRUCTURAL — the recipe cannot carry its own literal
# ==========================================================================
def test_recipe_default_is_a_name_not_a_numeric_literal() -> None:
    """The drift guard. A future edit that re-introduces
    `_DEFAULT_MO_SYSTEMIC_PD = 0.02` fails here, even if the number happens to
    still agree with the catalog."""
    tree = ast.parse(pathlib.Path(inspect.getsourcefile(recipes)).read_text())
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "_DEFAULT_MO_SYSTEMIC_PD"
            for t in node.targets
        )
    ]
    assert len(assignments) == 1, "expected exactly one definition"
    value = assignments[0].value
    assert isinstance(value, ast.Name), (
        "the Marshall-Olkin default must be bound to the shared constant, not "
        f"restated as a literal (found {ast.dump(value)[:60]})"
    )
    assert value.id == "SF_MO_SYSTEMIC_PD"


def test_catalog_param_spec_default_is_a_name_not_a_numeric_literal() -> None:
    """The same guard on the other side of the fork."""
    source = pathlib.Path(inspect.getsourcefile(catalog)).read_text()
    tree = ast.parse(source)
    spec_assign = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "_SYSTEMIC_PD" for t in node.targets
        )
    )
    default_kw = next(kw for kw in spec_assign.value.keywords if kw.arg == "default")
    assert isinstance(default_kw.value, ast.Name)
    assert default_kw.value.id == "SF_MO_SYSTEMIC_PD"


def test_all_three_bindings_are_the_same_object() -> None:
    """Identity, not equality: a re-forked literal would be a different object
    even at the same value."""
    assert _spec().default is SF_MO_SYSTEMIC_PD
    assert recipes._DEFAULT_MO_SYSTEMIC_PD is SF_MO_SYSTEMIC_PD


# ==========================================================================
# 2. OPERATIONAL — both boundaries observe the same value at runtime
# ==========================================================================
def test_omitted_param_resolves_identically_at_both_boundaries() -> None:
    """Neither side is compared to a number written in this test."""
    at_validation = _spec().default
    at_construction = _effective_systemic_pd_at_construction({})
    assert at_construction == at_validation


@pytest.mark.parametrize("explicit", [0.0, 0.001, 0.05, 0.25, 0.9999])
def test_explicit_value_propagates_unchanged_to_model_construction(explicit) -> None:
    """An authored value must arrive at the builder bit-for-bit, with no layer
    substituting its own default."""
    assert _effective_systemic_pd_at_construction({"systemic_pd": explicit}) == explicit


def test_explicit_value_is_not_overridden_by_the_default() -> None:
    other = float(np.nextafter(SF_MO_SYSTEMIC_PD, 1.0))
    assert other != SF_MO_SYSTEMIC_PD
    assert _effective_systemic_pd_at_construction({"systemic_pd": other}) == other


# ==========================================================================
# 3. BOUNDARY CONSISTENCY — validation and construction agree exactly
# ==========================================================================
def _request(pds: list[float], **kwargs) -> CreditVarRequest:
    return CreditVarRequest.from_dataframe(
        obligor_table=pd.DataFrame({
            "obligor_uid": list(range(1, len(pds) + 1)),
            "ead": [1_000_000.0] * len(pds),
            "pd": pds,
            "lgd": [0.5] * len(pds),
        }),
        model=MODEL,
        n_scenarios=1_000,
        alphas=[0.99],
        seed=1,
        **kwargs,
    )


def test_validator_threshold_is_exactly_the_constructed_shock_probability() -> None:
    """The request gate and the model must not disagree by one ULP."""
    effective = _effective_systemic_pd_at_construction({})

    # PD exactly at the effective shock probability: legal (p_idio = 0).
    accepted = _request([effective, 0.5])
    assert accepted.model == MODEL

    # One ULP below it: rejected, by the same number.
    with pytest.raises(ValidationError, match="below systemic_pd"):
        _request([float(np.nextafter(effective, 0.0)), 0.5])


def test_validator_threshold_follows_an_explicit_param_not_the_default() -> None:
    explicit = 0.25
    assert _effective_systemic_pd_at_construction({"systemic_pd": explicit}) == explicit

    accepted = _request([explicit, 0.5], systemic_pd=explicit)
    assert accepted.params["systemic_pd"] == explicit

    with pytest.raises(ValidationError, match="below systemic_pd"):
        _request([float(np.nextafter(explicit, 0.0)), 0.5], systemic_pd=explicit)


def test_the_request_path_never_reaches_the_recipe_with_an_omitted_param() -> None:
    """`run.py` forwards `request.params` verbatim, so an omitted param is still
    filled by the recipe. That is exactly why the two defaults had to be unified
    rather than merely kept equal."""
    omitted = _request([0.5, 0.5])
    assert "systemic_pd" not in omitted.params
    assert (
        _effective_systemic_pd_at_construction(dict(omitted.params)) == _spec().default
    )
