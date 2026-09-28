# tests/config/proxy_model/test_proxy_model_identity.py
"""
Identity of the compiled Proxy V1 operator (D088 §7).

Rule under test: ``hash_proxy_model`` is a function of the canonical resolved
operator only. Two authored configurations share a digest if and only if they
compile to bit-identical canonical arrays.
"""

from __future__ import annotations

import hashlib
from datetime import date

import numpy as np
import pytest

from riskquant.config.proxy_model.compile import compile_proxy_model
from riskquant.config.proxy_model.compiled import CompiledProxyModel
from riskquant.config.proxy_model.schema_objects import (
    ContainerProxyTerm,
    NodeProxyTerm,
)
from riskquant.market_data.risk_factors.shift_types import ShiftType
from riskquant.solutions.run_identity.hashing import hash_proxy_model

LOG = ShiftType.LOG_RETURN
DIFF = ShiftType.DIFFERENCE
AS_OF = date(2026, 9, 25)
_FIELDS = (
    "row_target_rf_ids",
    "row_term_offsets",
    "term_driver_rf_ids",
    "term_chart_ids",
    "term_loadings",
)


def _node(target: str, driver: str, chart: ShiftType, loading: float) -> NodeProxyTerm:
    return NodeProxyTerm(
        target_name=target, driver_name=driver, driver_chart=chart, loading=loading
    )


def _axis(cfg, *names: str) -> np.ndarray:
    return np.array([int(cfg.get_rf_id(n)) for n in names], dtype=np.int32)


def _compile(cfg, terms, axis, as_of=AS_OF) -> CompiledProxyModel:
    return compile_proxy_model(
        authored_terms=terms, static_config=cfg, as_of_date=as_of, run_axis_rf_ids=axis
    )


def _bitwise_equal(a: CompiledProxyModel, b: CompiledProxyModel) -> bool:
    for f in _FIELDS:
        x, y = getattr(a, f), getattr(b, f)
        if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
            return False
    return True


def test_digest_encoding_is_pinned():
    """
    Independent re-derivation of the digest bytes, so a silent change of name,
    field order, dtype or encoding cannot pass unnoticed.
    """
    model = CompiledProxyModel(
        row_target_rf_ids=np.array([7], dtype=np.int32),
        row_term_offsets=np.array([0, 2], dtype=np.int32),
        term_driver_rf_ids=np.array([3, 3], dtype=np.int32),
        term_chart_ids=np.array([1, 3], dtype=np.int8),
        term_loadings=np.array([0.95, -0.04], dtype=np.float64),
    )
    outer = hashlib.sha256()
    for name in _FIELDS:
        arr = np.ascontiguousarray(getattr(model, name))
        inner = hashlib.sha256()
        inner.update(arr.dtype.str.encode())
        inner.update(b"|")
        inner.update(str(arr.shape).encode())
        inner.update(b"|")
        inner.update(arr.tobytes(order="C"))
        outer.update(name.encode() + b"=" + inner.hexdigest().encode() + b";")
    assert hash_proxy_model(model) == outer.hexdigest()


# --------------------------------------------------------------------------- #
# Digest equality  <=>  canonical binary equality                              #
# --------------------------------------------------------------------------- #


def _variants(cfg):
    axis = _axis(cfg, "WTI_CRUDE_FWD[M+0]", "USDEUR")
    wti, brent, henry = (
        "WTI_CRUDE_FWD[M+0]",
        "BRENT_CRUDE_FWD[M+0]",
        "HENRY_HUB_GAS_FWD[M+0]",
    )
    one_ulp = float(np.nextafter(0.95, np.inf))
    tiny, tiny_ulp = 1e-17, float(np.nextafter(1e-17, np.inf))
    return axis, {
        "base": [_node(wti, brent, LOG, 0.95)],
        "base_ulp": [_node(wti, brent, LOG, one_ulp)],
        # 1e-17 and its successor are both absorbed by 1.0 in binary64.
        "absorbed": [_node(wti, brent, LOG, 1.0), _node(wti, brent, LOG, tiny)],
        "absorbed_ulp": [_node(wti, brent, LOG, 1.0), _node(wti, brent, LOG, tiny_ulp)],
        "plain_one": [_node(wti, brent, LOG, 1.0)],
        # An exactly cancelling pair vanishes whatever its (perturbed) value.
        "cancel": [
            _node(wti, brent, LOG, 0.95),
            _node(wti, henry, LOG, 0.3),
            _node(wti, henry, LOG, -0.3),
        ],
        "cancel_ulp": [
            _node(wti, brent, LOG, 0.95),
            _node(wti, henry, LOG, float(np.nextafter(0.3, 1.0))),
            _node(wti, henry, LOG, -float(np.nextafter(0.3, 1.0))),
        ],
        # 0.45 + 0.5 is not exact: it rounds (half-ULP tie, round-to-even) to
        # the binary64 value of the literal 0.95.
        "split_rounds_to_0_95": [
            _node(wti, brent, LOG, 0.5),
            _node(wti, brent, LOG, 0.45),
        ],
        # 0.3 + 0.2 is exact: the rational sum of the two doubles is 0.5.
        "split_exact": [_node(wti, brent, LOG, 0.3), _node(wti, brent, LOG, 0.2)],
        "authored_0_5": [_node(wti, brent, LOG, 0.5)],
        "split_inexact": [_node(wti, brent, LOG, 0.1), _node(wti, brent, LOG, 0.2)],
        "authored_0_3": [_node(wti, brent, LOG, 0.3)],
        "other_chart": [_node(wti, brent, DIFF, 0.95)],
        "other_driver": [_node(wti, henry, LOG, 0.95)],
        "other_target": [_node("USDEUR", "USDCHF", LOG, 0.95)],
    }


def test_digest_equality_is_exactly_canonical_binary_equality(extended_static_config):
    cfg = extended_static_config
    axis, variants = _variants(cfg)
    models = {k: _compile(cfg, v, axis) for k, v in variants.items()}
    for (ka, a), (kb, b) in [(x, y) for x in models.items() for y in models.items()]:
        same_digest = hash_proxy_model(a) == hash_proxy_model(b)
        assert same_digest == _bitwise_equal(a, b), (ka, kb)


def test_one_ulp_changes_identity_iff_it_survives_canonicalisation(
    extended_static_config,
):
    cfg = extended_static_config
    axis, variants = _variants(cfg)
    h = {k: hash_proxy_model(_compile(cfg, v, axis)) for k, v in variants.items()}
    # Survives: a sole loading moved by one ULP is new canonical content.
    assert h["base"] != h["base_ulp"]
    # Absorbed: 1.0 + 1e-17 == 1.0 + nextafter(1e-17) == 1.0 in binary64.
    assert (1.0 + 1e-17) == (1.0 + float(np.nextafter(1e-17, np.inf))) == 1.0
    assert h["absorbed"] == h["absorbed_ulp"] == h["plain_one"]
    # Cancelled: the perturbed pair still cancels exactly and vanishes.
    assert h["cancel"] == h["cancel_ulp"] == h["base"]


def test_real_equivalence_is_not_identity(extended_static_config):
    cfg = extended_static_config
    axis, variants = _variants(cfg)
    h = {k: hash_proxy_model(_compile(cfg, v, axis)) for k, v in variants.items()}
    # Rounded, not exact, but the binary64 result equals the literal 0.95.
    assert 0.5 + 0.45 == 0.95
    assert h["split_rounds_to_0_95"] == h["base"]
    # Exact: the rational sum of the doubles 0.3 and 0.2 is 0.5.
    assert 0.3 + 0.2 == 0.5
    assert h["split_exact"] == h["authored_0_5"]
    assert 0.1 + 0.2 != 0.3  # one ULP apart in binary64
    assert h["split_inexact"] != h["authored_0_3"]


def test_semantic_changes_change_identity(extended_static_config):
    cfg = extended_static_config
    axis, variants = _variants(cfg)
    h = {k: hash_proxy_model(_compile(cfg, v, axis)) for k, v in variants.items()}
    for other in ("other_chart", "other_driver", "other_target"):
        assert h[other] != h["base"], other
    # Row order is run-axis order, a run input.
    reordered = _compile(cfg, variants["base"], axis[::-1].copy())
    assert hash_proxy_model(reordered) != h["base"]


def test_non_semantic_syntax_does_not_change_identity(extended_static_config):
    cfg = extended_static_config
    tokens = ("M+0", "M+1", "M+2")
    axis = _axis(cfg, *[f"WTI_CRUDE_FWD[{t}]" for t in tokens])
    explicit = [
        _node(f"WTI_CRUDE_FWD[{t}]", f"BRENT_CRUDE_FWD[{t}]", LOG, 0.95) for t in tokens
    ]
    forms = [
        explicit,
        list(reversed(explicit)),
        [
            ContainerProxyTerm(
                target_container_name="WTI_CRUDE_FWD",
                driver_container_name="BRENT_CRUDE_FWD",
                driver_chart=LOG,
                loading=0.95,
            )
        ],
        [
            _node("WTI_CRUDE_FWD_202610", "BRENT_CRUDE_FWD_202610", LOG, 0.95),
            _node("WTI_CRUDE_FWD_202611", "BRENT_CRUDE_FWD_202611", LOG, 0.95),
            _node("WTI_CRUDE_FWD_202612", "BRENT_CRUDE_FWD_202612", LOG, 0.95),
        ],
    ]
    digests = {hash_proxy_model(_compile(cfg, f, axis)) for f in forms}
    assert len(digests) == 1


def test_identity_default_is_part_of_the_operator_identity(extended_static_config):
    cfg = extended_static_config
    axis = _axis(cfg, "WTI_CRUDE_FWD[M+0]", "USDEUR")
    omitted = _compile(
        cfg, [_node("WTI_CRUDE_FWD[M+0]", "BRENT_CRUDE_FWD[M+0]", LOG, 0.95)], axis
    )
    explicit = _compile(
        cfg,
        [
            _node("WTI_CRUDE_FWD[M+0]", "BRENT_CRUDE_FWD[M+0]", LOG, 0.95),
            _node("USDEUR", "USDEUR", LOG, 1.0),
        ],
        axis,
    )
    assert hash_proxy_model(omitted) == hash_proxy_model(explicit)
    # Adding a V1 node to the axis adds its materialised identity row.
    wider = _compile(
        cfg,
        [_node("WTI_CRUDE_FWD[M+0]", "BRENT_CRUDE_FWD[M+0]", LOG, 0.95)],
        _axis(cfg, "WTI_CRUDE_FWD[M+0]", "USDEUR", "USDCHF"),
    )
    assert hash_proxy_model(wider) != hash_proxy_model(omitted)


def test_as_of_is_upstream_of_identity(extended_static_config):
    cfg = extended_static_config
    term = ContainerProxyTerm(
        target_container_name="TEST_ANNUAL_CAL_FWD",
        driver_container_name="CO2_EUA_ANNUAL_FWD",
        driver_chart=LOG,
        loading=1.2,
    )
    axis = _axis(cfg, "TEST_ANNUAL_CAL_FWD[Y+0]", "TEST_ANNUAL_CAL_FWD[Y+1]")
    before = _compile(cfg, [term], axis, as_of=date(2026, 2, 15))
    after = _compile(cfg, [term], axis, as_of=date(2026, 9, 25))
    assert hash_proxy_model(before) != hash_proxy_model(after)
    # A NODE pairing does not depend on the as-of: same identity on both dates.
    node = [_node("TEST_ANNUAL_CAL_FWD[Y+0]", "CO2_EUA_ANNUAL_FWD[Y+0]", LOG, 1.2)]
    axis0 = _axis(cfg, "TEST_ANNUAL_CAL_FWD[Y+0]")
    assert hash_proxy_model(_compile(cfg, node, axis0, as_of=date(2026, 2, 15))) == (
        hash_proxy_model(_compile(cfg, node, axis0, as_of=date(2026, 9, 25)))
    )


@pytest.mark.parametrize("wti_shift", ["LOG_RETURN", "SIMPLE_RETURN", "DIFFERENCE"])
def test_target_chart_is_not_part_of_the_proxy_identity(
    static_config_builder, wti_shift
):
    """kappa_A is owned by the factor definition (D088 §3): for an authored row
    it changes reconstruction, not the proxy operator's identity."""
    cfg = static_config_builder(wti_shift_type=wti_shift)
    reference = static_config_builder()
    axis_cfg = _axis(cfg, "WTI_CRUDE_FWD[M+0]")
    terms = [_node("WTI_CRUDE_FWD[M+0]", "BRENT_CRUDE_FWD[M+0]", LOG, 0.95)]
    assert hash_proxy_model(_compile(cfg, terms, axis_cfg)) == hash_proxy_model(
        _compile(reference, terms, _axis(reference, "WTI_CRUDE_FWD[M+0]"))
    )
