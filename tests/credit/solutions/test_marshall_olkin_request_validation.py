# tests/credit/solutions/test_marshall_olkin_request_validation.py
"""
P2-07: the Marshall-Olkin common-shock probability is validated on the request.

Before this change a request carrying an obligor whose total PD sat below
`systemic_pd` was accepted, and failed only after factor-universe and master
construction, inside `build_marshall_olkin_block`. The mathematics is unchanged
and the builder guard is untouched — this only moves the FIRST rejection to the
earliest boundary that holds both the obligor PDs and the shock probability.

The constraint is Marshall-Olkin's alone: the decoupling

    p_idio_i = 1 - (1 - PD_i) / (1 - systemic_pd)

is a probability only when PD_i >= systemic_pd. Equality is legal and must stay
accepted — it means the obligor defaults through the shared shock alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest
from pydantic import ValidationError

from riskquant.credit.jtd.ingest.builders.marshall_olkin import (
    build_marshall_olkin_block,
)
from riskquant.credit.masters.obligors.compiled import CompiledObligorCredit
from riskquant.solutions.credit_var.catalog import model_entry
from riskquant.solutions.credit_var.schemas import CreditVarRequest

# The catalog default the recipe also uses when the author omits the param.
DEFAULT_SYSTEMIC_PD = next(
    p for p in model_entry("marshall_olkin").params if p.key == "systemic_pd"
).default


def _table(pds: list[float], groups: list[str] | None = None) -> pd.DataFrame:
    frame = pd.DataFrame({
        "obligor_uid": list(range(1, len(pds) + 1)),
        "ead": [1_000_000.0] * len(pds),
        "pd": pds,
        "lgd": [0.5] * len(pds),
    })
    if groups is not None:
        frame["group"] = groups
    return frame


def _request(pds: list[float], **kwargs) -> CreditVarRequest:
    return CreditVarRequest.from_dataframe(
        obligor_table=_table(pds),
        model="marshall_olkin",
        n_scenarios=1_000,
        alphas=[0.99],
        seed=1,
        **kwargs,
    )


# ==========================================================================
# VALID
# ==========================================================================
def test_every_pd_above_the_default_shock_is_accepted() -> None:
    req = _request([0.05, 0.10, 0.30])
    assert req.model == "marshall_olkin"
    assert req.n_obligors == 3


def test_explicit_systemic_pd_below_every_pd_is_accepted() -> None:
    req = _request([0.05, 0.10], systemic_pd=0.01)
    assert req.params["systemic_pd"] == 0.01


def test_zero_systemic_pd_accepts_any_pd() -> None:
    """systemic_pd = 0 degenerates to independent Bernoulli: nothing to violate."""
    req = _request([0.0, 0.001, 0.9], systemic_pd=0.0)
    assert req.params["systemic_pd"] == 0.0


# ==========================================================================
# BOUNDARY — equality must remain legal
# ==========================================================================
def test_pd_exactly_equal_to_systemic_pd_is_accepted() -> None:
    """p_idio = 1 - (1-PD)/(1-systemic_pd) = 0 at equality: a valid probability.
    The obligor defaults through the shared shock alone."""
    req = _request([0.02, 0.50])  # 0.02 == the catalog default
    assert req.obligors[0].pd == pytest.approx(DEFAULT_SYSTEMIC_PD)

    req2 = _request([0.07, 0.07], systemic_pd=0.07)
    assert req2.params["systemic_pd"] == 0.07


def test_pd_one_ulp_below_systemic_pd_is_rejected() -> None:
    just_below = float(np.nextafter(0.07, 0.0))
    with pytest.raises(ValidationError, match="below systemic_pd"):
        _request([just_below, 0.50], systemic_pd=0.07)


# ==========================================================================
# INVALID
# ==========================================================================
def test_pd_below_systemic_pd_is_rejected_at_request_construction() -> None:
    with pytest.raises(ValidationError) as excinfo:
        _request([0.01, 0.05, 0.10])
    message = str(excinfo.value)
    assert "marshall_olkin" in message
    assert "below systemic_pd" in message
    assert "0.02" in message  # the shock probability
    assert "uid 1" in message  # the offending obligor
    assert "PD=0.01" in message  # its PD
    assert "decoupling" in message  # the mathematical reason


def test_error_names_the_offender_count_and_bounds_the_preview() -> None:
    with pytest.raises(ValidationError) as excinfo:
        _request([0.001] * 25, systemic_pd=0.10)
    message = str(excinfo.value)
    assert "25 obligor(s)" in message
    assert message.count("uid ") <= 10  # bounded preview


# ==========================================================================
# SCOPE — no other model may be affected
# ==========================================================================
@pytest.mark.parametrize(
    "model,kwargs",
    [
        ("independent_bernoulli", {}),
        ("sf_gaussian", {}),
        ("beta_binomial", {}),
        ("forced_default", {}),
    ],
)
def test_low_pd_does_not_break_unrelated_models(model, kwargs) -> None:
    """A PD of 0.001 is far below the Marshall-Olkin default shock, and must be
    irrelevant to every model that does not carry the constraint."""
    req = CreditVarRequest.from_dataframe(
        obligor_table=_table([0.001, 0.002, 0.30]),
        model=model,
        n_scenarios=1_000,
        alphas=[0.99],
        seed=1,
        **kwargs,
    )
    assert req.model == model


# ==========================================================================
# DEFENCE IN DEPTH — the builder guard is untouched
# ==========================================================================
def test_builder_guard_still_rejects_when_called_directly() -> None:
    """The request validator is an earlier gate, not a replacement. Calling the
    builder with the same violation must still fail."""
    frame = pl.DataFrame({
        "uid": [1, 2],
        "uniform_factor": ["U_001", "U_002"],
        "systemic_uniform_factor": ["SYSTEMIC_SHOCK", "SYSTEMIC_SHOCK"],
        "systemic_pd": [0.10, 0.10],
    })
    master = CompiledObligorCredit(
        uid=np.array([1, 2], dtype=np.int64),
        pd=np.array([0.01, 0.50]),  # obligor 1 violates PD >= systemic_pd
        lgd=np.array([0.5, 0.5]),
        rating_id=np.array([0, 0], dtype=np.int32),
        sector_id=np.array([0, 0], dtype=np.int32),
        country_id=np.array([0, 0], dtype=np.int32),
        parent_uid=np.array([1, 2], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="systemic"):
        build_marshall_olkin_block(
            frame,
            master,
            {"U_001": 0, "U_002": 1, "SYSTEMIC_SHOCK": 2},
        )
