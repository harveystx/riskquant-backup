# tests/credit/test_dataclass_equality_contract.py
"""
C-02: equality semantics for array-bearing credit dataclasses.

THE CONTRACT (D045 / VAL-09 / MD-13, the estate's locked position, now applied
to credit): an array-bearing container has NO domain-level equality. Two
distinct objects are different objects, full stop, whatever their arrays hold.
`eq=False` gives exactly that — identity `__eq__` and identity `__hash__`
inherited from `object`.

Why identity and not content equality. These are runtime containers, not value
objects:

  * the arrays are large — `(N, S)` losses, `(F, S)` factors, CSR loading
    values — so a content `==` would silently cost O(N*S) wherever Python
    reaches for it, including `x in [...]`;
  * `frozen=True` does NOT freeze the arrays. Nothing is copied or marked
    read-only at construction, so an object's "value" can change under a
    content-based `__eq__` while the object itself is unchanged. Content
    equality on a mutable payload is not a well-defined relation;
  * no production or test call site anywhere in `riskquant.credit` or
    `riskquant.solutions.credit_var` compares two of these objects. There is no
    domain requirement to serve.

What the generated `__eq__` did instead: `x == x` returned True (tuple
comparison short-circuits on identity), while `x == equivalent_copy` RAISED
`ValueError: The truth value of an array with more than one element is
ambiguous`. Shape-dependent, too — a one-element or zero-dim array compared
fine, so a fixture with a single scenario passed and the same test with two
scenarios blew up. `hash()` raised `TypeError`, so these objects could not go in
a set or a dict at all.
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import polars as pl
import pytest

from riskquant.credit.aggregate.loss_metrics import PortfolioLossDistribution
from riskquant.credit.aggregate.summary import JtdRiskSummary
from riskquant.credit.exposure.results import NettingSetExposurePaths
from riskquant.credit.factors.state import CreditFactorMatrix
from riskquant.credit.jtd.blocks.forced_default import ForcedDefaultJtdBlock
from riskquant.credit.jtd.ingest.bundle import JtdIngestBundle
from riskquant.credit.jtd.results import JtdLossResult
from riskquant.credit.masters.netting_set.compiled import CompiledNettingSetMaster
from riskquant.solutions.credit_var.catalog import ParamSpec, model_entry

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "riskquant"

# ==========================================================================
# 1. STRUCTURAL GUARD — every array-bearing dataclass must declare eq=False
# ==========================================================================
_ARRAY_HINTS = ("ndarray", "NDArray")
_ARRAY_ALIASES = {"_F64_1D", "_F64"}  # module-local NDArray aliases
_INDIRECT = {  # hold array-bearing payloads
    "JtdIngestBundle",  # polars DataFrames: `==` is elementwise too
    "CompiledCreditMasters",
    "AssembledInputs",
    "AssembledExposureInputs",
}
_ARRAY_BASES = {
    "_JtdBlock",
    "_MultiFactorCopulaBlock",
    "_SingleFactorCopulaBlock",
    "_NettingSetView",
}
# Pure-scalar value objects: generated equality is valid AND useful. They must
# NOT be swept into eq=False by a future mechanical pass.
_VALUE_OBJECTS = {
    "JtdRiskSummary",
    "JtdModelContract",
    "MfLoadingsContract",
    "JtdModelSpec",
    "ParamSpec",
    "ModelEntry",
    "ExposureModelEntry",
    "_FactorRow",
}

_ROOTS = ("credit", "solutions/credit_var", "solutions/credit_exposure")


def _dataclasses():
    """(qualified name, class node, decorator source) for every credit dataclass."""
    for root in _ROOTS:
        for path in sorted((SRC / root).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                deco = next(
                    (
                        ast.unparse(d)
                        for d in node.decorator_list
                        if "dataclass" in ast.unparse(d)
                    ),
                    None,
                )
                if deco is None:
                    continue
                yield f"{path.relative_to(SRC)}::{node.name}", node, deco


def _is_array_bearing(node: ast.ClassDef) -> bool:
    anns = [
        ast.unparse(s.annotation) for s in node.body if isinstance(s, ast.AnnAssign)
    ]
    if any(any(h in a for h in _ARRAY_HINTS) or a in _ARRAY_ALIASES for a in anns):
        return True
    if node.name in _INDIRECT:
        return True
    return bool({ast.unparse(b) for b in node.bases} & _ARRAY_BASES)


def test_every_array_bearing_dataclass_declares_eq_false() -> None:
    """The drift guard. A new SoA container added without `eq=False` inherits
    the ambiguous-truth behaviour and fails here rather than in production."""
    offenders = [
        qual
        for qual, node, deco in _dataclasses()
        if _is_array_bearing(node)
        and node.name not in _VALUE_OBJECTS
        and "eq=False" not in deco
    ]
    assert not offenders, (
        "array-bearing dataclasses without eq=False (D045/VAL-09/MD-13):\n  "
        + "\n  ".join(offenders)
    )


def test_the_audit_actually_found_the_population_it_claims_to_guard() -> None:
    """A classifier that matches nothing would pass the guard above vacuously."""
    guarded = [q for q, n, _ in _dataclasses() if _is_array_bearing(n)]
    assert len(guarded) >= 40, len(guarded)


def test_pure_scalar_value_objects_keep_generated_equality() -> None:
    """eq=False is not a style rule. Scalar value objects have valid, useful
    equality and must not be swept up by a mechanical pass."""
    offenders = [
        qual
        for qual, node, deco in _dataclasses()
        if node.name in _VALUE_OBJECTS and "eq=False" in deco
    ]
    assert not offenders, offenders


def test_every_class_in_an_array_bearing_hierarchy_declares_it_itself() -> None:
    """`eq=False` does NOT inherit: a dataclass subclass regenerates `__eq__`
    over inherited fields unless it opts out too. Base and subclass must both
    carry it."""
    by_name = {node.name: deco for _, node, deco in _dataclasses()}
    for base in _ARRAY_BASES:
        assert "eq=False" in by_name[base], base
    for qual, node, deco in _dataclasses():
        if {ast.unparse(b) for b in node.bases} & _ARRAY_BASES:
            assert "eq=False" in deco, qual


# ==========================================================================
# 2. BEHAVIOURAL — identity semantics, across the shapes that used to break
# ==========================================================================
def _loss(values):
    return JtdLossResult(
        uid=np.arange(len(values), dtype=np.int64), loss=np.asarray(values)
    )


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda: _loss([[1.0, 2.0], [3.0, 4.0]]), id="multi_element"),
        pytest.param(lambda: _loss([[1.0]]), id="one_element"),
        pytest.param(
            lambda: PortfolioLossDistribution(portfolio_loss=np.empty(0)), id="empty"
        ),
        pytest.param(
            lambda: PortfolioLossDistribution(portfolio_loss=np.array(1.0)),
            id="zero_dim",
        ),
        pytest.param(
            lambda: CreditFactorMatrix(
                latent_factors=np.ascontiguousarray(np.ones((3, 5)))
            ),
            id="factor_matrix",
        ),
        pytest.param(
            lambda: NettingSetExposurePaths(
                uid=np.array([1, 2], dtype=np.int64),
                exposure=np.ascontiguousarray(np.ones((2, 3))),
            ),
            id="exposure_paths",
        ),
        pytest.param(
            lambda: CompiledNettingSetMaster(
                uid=np.array([1, 2], dtype=np.int64),
                obligor_uid=np.array([9, 9], dtype=np.int64),
                net_collateral=np.zeros(2),
            ),
            id="netting_set_master",
        ),
        pytest.param(
            lambda: ForcedDefaultJtdBlock(
                obligor_uids=np.array([1, 2], dtype=np.int64), lgd=np.full(2, 0.5)
            ),
            id="block_with_inherited_arrays",
        ),
        pytest.param(
            lambda: JtdIngestBundle(
                independent_bernoulli_df=pl.DataFrame({"uid": [1, 2]})
            ),
            id="bundle_of_dataframes",
        ),
    ],
)
def test_identity_equality_holds_whatever_the_arrays_contain(make) -> None:
    """The §17 adversarial case: two DISTINCT objects built from IDENTICAL
    inputs. Under this contract they are not equal, because equality is
    identity — the objects are different runtime containers, and their arrays
    are mutable payloads that were never copied or frozen.

    Every one of these shapes previously raised on `x == y`; the one-element and
    zero-dim cases previously returned True, which is the same defect wearing a
    friendlier face."""
    x, y = make(), make()

    # Both dunders are exercised on purpose: ruff's simplification would drop
    # the __ne__ half of the contract, which is the half that used to raise.
    assert x == x  # reflexive
    assert not (x != x)  # noqa: SIM202
    assert x != y  # distinct objects, identical contents
    assert not (x == y)  # noqa: SIM201
    assert x is not y

    # Container operations must not raise, and must follow identity.
    assert [x] == [x]
    assert [x] != [y]
    assert x in [x]
    assert x not in [y]
    assert (x,) != (y,)

    # eq=False leaves object.__hash__ in place: usable in sets and dicts.
    assert isinstance(hash(x), int)
    assert hash(x) == hash(x)
    assert len({x, y}) == 2
    assert {x: "a"}[x] == "a"


def test_different_shapes_compare_false_rather_than_raising() -> None:
    """Previously a broadcast ValueError, which no caller could reasonably
    anticipate from an `==`."""
    a = PortfolioLossDistribution(portfolio_loss=np.ones(5))
    b = PortfolioLossDistribution(portfolio_loss=np.ones(3))
    assert a != b
    assert not (a == b)  # noqa: SIM201  - __eq__ and __ne__ both checked


def test_broadcastable_but_different_shapes_compare_false() -> None:
    """(3,1) against (1,3) used to broadcast to a (3,3) mask and then raise."""
    a = PortfolioLossDistribution(portfolio_loss=np.ones((3, 1)))
    b = PortfolioLossDistribution(portfolio_loss=np.ones((1, 3)))
    assert a != b


def test_equality_never_touches_the_arrays() -> None:
    """Identity equality must be O(1). A sentinel array that raises on any
    element access proves the arrays are not read."""

    class Tripwire(np.ndarray):
        def __eq__(self, other):  # pragma: no cover - must never be reached
            raise AssertionError("equality touched the array payload")

        __hash__ = None

    payload = np.ones((4, 4)).view(Tripwire)
    x = CreditFactorMatrix(latent_factors=np.ascontiguousarray(np.ones((4, 4))))
    y = CreditFactorMatrix(latent_factors=payload)
    assert x != y  # must not invoke Tripwire.__eq__
    assert y != x
    assert y == y


def test_a_mutated_array_does_not_change_an_object_s_identity() -> None:
    """frozen=True does not freeze the payload. Under content equality an
    object's "value" would drift as its arrays are written; under identity it
    cannot. This is why content equality was rejected."""
    x = PortfolioLossDistribution(portfolio_loss=np.ones(4))
    before = hash(x)
    x.portfolio_loss[0] = 99.0  # frozen dataclass, mutable array
    assert hash(x) == before
    assert x == x


# ==========================================================================
# 3. VALUE OBJECTS — unchanged, and proven to still work
# ==========================================================================
def test_param_spec_remains_a_comparable_hashable_value_object() -> None:
    a = ParamSpec(key="rho", label="L", default=0.15, min_value=0.0, max_value=1.0)
    b = ParamSpec(key="rho", label="L", default=0.15, min_value=0.0, max_value=1.0)
    c = ParamSpec(key="rho", label="L", default=0.20, min_value=0.0, max_value=1.0)
    assert a == b and a is not b  # value equality, deliberately kept
    assert a != c
    assert hash(a) == hash(b)
    assert len({a, b, c}) == 2


def test_catalog_entries_still_compare_by_value() -> None:
    assert model_entry("sf_gaussian") == model_entry("sf_gaussian")
    assert model_entry("sf_gaussian") != model_entry("sf_students_t")


def test_jtd_risk_summary_keeps_scalar_value_equality() -> None:
    """Scalars and dicts only — no ambiguous truth, so generated equality is
    correct and worth keeping."""
    kwargs = dict(
        n_scenarios=4,
        expected_loss=1.0,
        var={0.99: 2.0},
        es={0.99: 3.0},
        expected_loss_by_model={"m": 1.0},
    )
    assert JtdRiskSummary(**kwargs) == JtdRiskSummary(**kwargs)
    assert JtdRiskSummary(**{**kwargs, "expected_loss": 2.0}) != JtdRiskSummary(
        **kwargs
    )
