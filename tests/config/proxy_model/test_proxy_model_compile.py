# tests/config/proxy_model/test_proxy_model_compile.py
"""
Proxy V1 compiler semantics (D088, Slice 1).

Every expected value below is derived by hand from D088, never by re-running
the compiler's own logic: expected rows come from the run axis, expected
drivers from hand-read absolute coordinates, expected loadings from explicit
binary64 arithmetic written out in the test.
"""

from __future__ import annotations

import inspect
import itertools
import math
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pytest

import riskquant.config.proxy_model as proxy_model_pkg
from riskquant.config.proxy_model.compile import compile_proxy_model
from riskquant.config.proxy_model.compiled import CompiledProxyModel
from riskquant.config.proxy_model.schema_objects import (
    ContainerProxyTerm,
    NodeProxyTerm,
)
from riskquant.exceptions import ConfigurationError
from riskquant.market_data.risk_factors.shift_types import ShiftType

LOG = ShiftType.LOG_RETURN
SIMPLE = ShiftType.SIMPLE_RETURN
DIFF = ShiftType.DIFFERENCE
AS_OF = date(2026, 9, 25)


def _node(target: str, driver: str, chart: ShiftType, loading: float) -> NodeProxyTerm:
    return NodeProxyTerm(
        target_name=target, driver_name=driver, driver_chart=chart, loading=loading
    )


def _axis(static_config, *names: str) -> np.ndarray:
    return np.array([int(static_config.get_rf_id(n)) for n in names], dtype=np.int32)


def _compile(static_config, terms, axis, as_of=AS_OF) -> CompiledProxyModel:
    return compile_proxy_model(
        authored_terms=terms,
        static_config=static_config,
        as_of_date=as_of,
        run_axis_rf_ids=axis,
    )


def _rows(
    static_config, model: CompiledProxyModel
) -> dict[int, list[tuple[int, int, float]]]:
    out: dict[int, list[tuple[int, int, float]]] = {}
    off = model.row_term_offsets
    for r, target in enumerate(model.row_target_rf_ids.tolist()):
        out[target] = [
            (
                int(model.term_driver_rf_ids[t]),
                int(model.term_chart_ids[t]),
                float(model.term_loadings[t]),
            )
            for t in range(int(off[r]), int(off[r + 1]))
        ]
    return out


def _same_arrays(a: CompiledProxyModel, b: CompiledProxyModel) -> bool:
    return all(
        np.array_equal(getattr(a, f), getattr(b, f))
        and getattr(a, f).dtype == getattr(b, f).dtype
        for f in (
            "row_target_rf_ids",
            "row_term_offsets",
            "term_driver_rf_ids",
            "term_chart_ids",
        )
    ) and np.array_equal(
        a.term_loadings.view(np.uint64), b.term_loadings.view(np.uint64)
    )


# --------------------------------------------------------------------------- #
# A-B. Explicit terms: exact canonical arrays                                  #
# --------------------------------------------------------------------------- #


def test_single_explicit_term_produces_exact_canonical_arrays(extended_static_config):
    cfg = extended_static_config
    wti, brent = (
        cfg.get_rf_id("WTI_CRUDE_FWD[M+0]"),
        cfg.get_rf_id("BRENT_CRUDE_FWD[M+0]"),
    )
    model = _compile(
        cfg,
        [_node("WTI_CRUDE_FWD[M+0]", "BRENT_CRUDE_FWD[M+0]", LOG, 0.95)],
        _axis(cfg, "WTI_CRUDE_FWD[M+0]", "BRENT_CRUDE_FWD[M+0]"),
    )
    # Row order = run-axis order; BRENT has no authored term -> identity row.
    np.testing.assert_array_equal(model.row_target_rf_ids, [wti, brent])
    np.testing.assert_array_equal(model.row_term_offsets, [0, 1, 2])
    np.testing.assert_array_equal(model.term_driver_rf_ids, [brent, brent])
    np.testing.assert_array_equal(model.term_chart_ids, [int(LOG), int(LOG)])
    assert model.term_loadings.tolist() == [0.95, 1.0]
    assert model.row_target_rf_ids.dtype == np.int32
    assert model.row_term_offsets.dtype == np.int32
    assert model.term_driver_rf_ids.dtype == np.int32
    assert model.term_chart_ids.dtype == np.int8
    assert model.term_loadings.dtype == np.float64
    for f in (
        "row_target_rf_ids",
        "row_term_offsets",
        "term_driver_rf_ids",
        "term_chart_ids",
        "term_loadings",
    ):
        assert not getattr(model, f).flags.writeable


def test_multiple_terms_are_sorted_by_driver_then_chart(extended_static_config):
    cfg = extended_static_config
    brent, henry = (
        cfg.get_rf_id("BRENT_CRUDE_FWD[M+0]"),
        cfg.get_rf_id("HENRY_HUB_GAS_FWD[M+0]"),
    )
    assert brent < henry
    model = _compile(
        cfg,
        [
            _node("WTI_CRUDE_FWD[M+0]", "HENRY_HUB_GAS_FWD[M+0]", LOG, 0.3),
            _node("WTI_CRUDE_FWD[M+0]", "BRENT_CRUDE_FWD[M+0]", LOG, 0.7),
        ],
        _axis(cfg, "WTI_CRUDE_FWD[M+0]"),
    )
    assert _rows(cfg, model) == {
        int(cfg.get_rf_id("WTI_CRUDE_FWD[M+0]")): [
            (int(brent), int(LOG), 0.7),
            (int(henry), int(LOG), 0.3),
        ]
    }


# --------------------------------------------------------------------------- #
# C-F. Bit-exact canonicalisation                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("perm", list(itertools.permutations((0.1, 0.2, 0.3))))
def test_duplicate_terms_merge_in_bit_pattern_order(extended_static_config, perm):
    cfg = extended_static_config
    terms = [_node("WTI_CRUDE_FWD[M+0]", "BRENT_CRUDE_FWD[M+0]", LOG, x) for x in perm]
    model = _compile(cfg, terms, _axis(cfg, "WTI_CRUDE_FWD[M+0]"))
    # Bit-pattern order of positive doubles is numeric order: 0.1, 0.2, 0.3.
    expected = (0.1 + 0.2) + 0.3
    assert expected.hex() == "0x1.3333333333334p-1"
    assert model.term_loadings.tolist() == [expected]
    assert model.term_loadings[0] != 0.6  # not symbolically normalised


def test_exact_cancellation_removes_the_term(extended_static_config):
    cfg = extended_static_config
    henry = int(cfg.get_rf_id("HENRY_HUB_GAS_FWD[M+0]"))
    model = _compile(
        cfg,
        [
            _node("WTI_CRUDE_FWD[M+0]", "BRENT_CRUDE_FWD[M+0]", LOG, 0.3),
            _node("WTI_CRUDE_FWD[M+0]", "BRENT_CRUDE_FWD[M+0]", LOG, -0.3),
            _node("WTI_CRUDE_FWD[M+0]", "HENRY_HUB_GAS_FWD[M+0]", LOG, 0.5),
        ],
        _axis(cfg, "WTI_CRUDE_FWD[M+0]"),
    )
    assert _rows(cfg, model)[int(cfg.get_rf_id("WTI_CRUDE_FWD[M+0]"))] == [
        (henry, int(LOG), 0.5)
    ]


@pytest.mark.parametrize("perm", list(itertools.permutations((0.1, 0.2, -0.3))))
def test_inexact_cancellation_survives_with_its_binary64_residue(
    extended_static_config, perm
):
    cfg = extended_static_config
    terms = [_node("WTI_CRUDE_FWD[M+0]", "BRENT_CRUDE_FWD[M+0]", LOG, x) for x in perm]
    model = _compile(cfg, terms, _axis(cfg, "WTI_CRUDE_FWD[M+0]"))
    # Unsigned bit order puts positives before negatives: (0.1 + 0.2) + (-0.3).
    expected = (0.1 + 0.2) + (-0.3)
    assert expected == 2.0**-54
    assert model.term_loadings.tolist() == [expected]


def test_signed_zero_authored_terms_disappear(extended_static_config):
    cfg = extended_static_config
    brent = int(cfg.get_rf_id("BRENT_CRUDE_FWD[M+0]"))
    base = [_node("WTI_CRUDE_FWD[M+0]", "BRENT_CRUDE_FWD[M+0]", LOG, 0.95)]
    zeros = [
        _node("WTI_CRUDE_FWD[M+0]", "HENRY_HUB_GAS_FWD[M+0]", LOG, 0.0),
        _node("WTI_CRUDE_FWD[M+0]", "HENRY_HUB_GAS_FWD[M+0]", DIFF, -0.0),
    ]
    axis = _axis(cfg, "WTI_CRUDE_FWD[M+0]")
    with_zeros = _compile(cfg, base + zeros, axis)
    without = _compile(cfg, base, axis)
    assert _same_arrays(with_zeros, without)
    assert _rows(cfg, with_zeros)[int(axis[0])] == [(brent, int(LOG), 0.95)]
    assert not np.signbit(with_zeros.term_loadings).any()


def test_authored_target_never_falls_back_to_identity(extended_static_config):
    cfg = extended_static_config
    axis = _axis(cfg, "WTI_CRUDE_FWD[M+0]")
    for terms in (
        [_node("WTI_CRUDE_FWD[M+0]", "BRENT_CRUDE_FWD[M+0]", LOG, -0.0)],
        [
            _node("WTI_CRUDE_FWD[M+0]", "BRENT_CRUDE_FWD[M+0]", LOG, 0.3),
            _node("WTI_CRUDE_FWD[M+0]", "BRENT_CRUDE_FWD[M+0]", LOG, -0.3),
        ],
    ):
        with pytest.raises(ConfigurationError, match="no term after canonicalisation"):
            _compile(cfg, terms, axis)


def test_merged_loading_overflow_is_rejected(extended_static_config):
    cfg = extended_static_config
    terms = [_node("WTI_CRUDE_FWD[M+0]", "BRENT_CRUDE_FWD[M+0]", LOG, 1e308)] * 2
    with pytest.raises(ConfigurationError, match="non-finite"):
        _compile(cfg, terms, _axis(cfg, "WTI_CRUDE_FWD[M+0]"))


# --------------------------------------------------------------------------- #
# G-I. Identity closure, permutation, aliases                                  #
# --------------------------------------------------------------------------- #


def test_identity_closure_equals_explicit_identity(extended_static_config):
    cfg = extended_static_config
    axis = _axis(cfg, "WTI_CRUDE_FWD[M+0]", "USD_LIBOR_3M[10Y]")
    omitted = _compile(cfg, [], axis)
    explicit = _compile(
        cfg,
        [
            _node("WTI_CRUDE_FWD[M+0]", "WTI_CRUDE_FWD[M+0]", LOG, 1.0),
            _node("USD_LIBOR_3M[10Y]", "USD_LIBOR_3M[10Y]", DIFF, 1.0),
        ],
        axis,
    )
    assert _same_arrays(omitted, explicit)
    # The identity term's chart is the node's configured chart: LOG and DIFF.
    np.testing.assert_array_equal(omitted.term_chart_ids, [int(LOG), int(DIFF)])
    # An "identity" authored under the wrong chart is a different operator.
    wrong = _compile(
        cfg, [_node("USD_LIBOR_3M[10Y]", "USD_LIBOR_3M[10Y]", LOG, 1.0)], axis
    )
    assert not _same_arrays(omitted, wrong)


def test_authoring_permutation_is_invariant(extended_static_config):
    cfg = extended_static_config
    terms = [
        _node("WTI_CRUDE_FWD[M+0]", "BRENT_CRUDE_FWD[M+0]", LOG, 0.7),
        _node("WTI_CRUDE_FWD[M+0]", "HENRY_HUB_GAS_FWD[M+0]", LOG, 0.3),
        _node("WTI_CRUDE_FWD[M+0]", "BRENT_CRUDE_FWD[M+0]", LOG, 0.1),
        _node("EURCHF", "USDCHF", LOG, 1.0),
        _node("EURCHF", "USDEUR", LOG, -1.0),
    ]
    axis = _axis(cfg, "EURCHF", "WTI_CRUDE_FWD[M+0]")
    reference = _compile(cfg, terms, axis)
    for perm in itertools.permutations(terms):
        assert _same_arrays(reference, _compile(cfg, list(perm), axis))


def test_aliases_resolving_to_one_node_are_identical(extended_static_config):
    cfg = extended_static_config
    axis = _axis(cfg, "WTI_CRUDE_FWD[M+0]")
    relative = _compile(
        cfg, [_node("WTI_CRUDE_FWD[M+0]", "BRENT_CRUDE_FWD[M+1]", LOG, 0.95)], axis
    )
    absolute = _compile(
        cfg, [_node("WTI_CRUDE_FWD_202610", "BRENT_CRUDE_FWD_202611", LOG, 0.95)], axis
    )
    assert _same_arrays(relative, absolute)


# --------------------------------------------------------------------------- #
# J. As-of is upstream of identity (EXACT)                                     #
# --------------------------------------------------------------------------- #


def test_exact_resolution_follows_the_as_of(extended_static_config):
    cfg = extended_static_config
    term = ContainerProxyTerm(
        target_container_name="TEST_ANNUAL_CAL_FWD",
        driver_container_name="CO2_EUA_ANNUAL_FWD",
        driver_chart=LOG,
        loading=1.2,
    )
    axis = _axis(
        cfg,
        "TEST_ANNUAL_CAL_FWD[Y+0]",
        "TEST_ANNUAL_CAL_FWD[Y+1]",
        "TEST_ANNUAL_CAL_FWD[Y+2]",
    )
    # Hand-read coordinates. TEST (calendar, anchor 0): Y+k -> 2026+k on both
    # dates. CO2 (roll 04-01): Y+0 -> 2025 on 2026-02-15, 2026 on 2026-09-25.
    before_roll = _compile(cfg, [term], axis, as_of=date(2026, 2, 15))
    after_roll = _compile(cfg, [term], axis, as_of=date(2026, 9, 25))
    co2 = [int(cfg.get_rf_id(f"CO2_EUA_ANNUAL_FWD[Y+{k}]")) for k in range(4)]
    np.testing.assert_array_equal(before_roll.term_driver_rf_ids, co2[1:4])
    np.testing.assert_array_equal(after_roll.term_driver_rf_ids, co2[0:3])
    assert not _same_arrays(before_roll, after_roll)


def test_exact_pairs_by_resolved_coordinate_not_token(extended_static_config):
    cfg = extended_static_config
    term = ContainerProxyTerm(
        target_container_name="TEST_ANCHOR2_FWD",
        driver_container_name="BRENT_CRUDE_FWD",
        driver_chart=LOG,
        loading=0.95,
    )
    # 2026-09-25: TEST_ANCHOR2[M+k] -> 2026-11 + k; BRENT[M+k] -> 2026-10 + k.
    axis = _axis(
        cfg, "TEST_ANCHOR2_FWD[M+0]", "TEST_ANCHOR2_FWD[M+1]", "TEST_ANCHOR2_FWD[M+2]"
    )
    model = _compile(cfg, [term], axis)
    np.testing.assert_array_equal(
        model.term_driver_rf_ids,
        [int(cfg.get_rf_id(f"BRENT_CRUDE_FWD[M+{k}]")) for k in (1, 2, 3)],
    )
    # TEST_ANCHOR2[M+3] -> 2027-02 has no BRENT pillar (BRENT[M+4] is absent):
    # the covered pillar has no term and never falls back to identity.
    with pytest.raises(ConfigurationError, match="no term after canonicalisation"):
        _compile(cfg, [term], _axis(cfg, "TEST_ANCHOR2_FWD[M+3]"))


def test_container_shorthand_equals_explicit_expansion(extended_static_config):
    cfg = extended_static_config
    tokens = ("M+0", "M+1", "M+2", "M+3", "M+6", "M+12")
    axis = _axis(cfg, *[f"WTI_CRUDE_FWD[{t}]" for t in tokens])
    container = _compile(
        cfg,
        [
            ContainerProxyTerm(
                target_container_name="WTI_CRUDE_FWD",
                driver_container_name="BRENT_CRUDE_FWD",
                driver_chart=LOG,
                loading=0.95,
            )
        ],
        axis,
    )
    explicit = _compile(
        cfg,
        [
            _node(f"WTI_CRUDE_FWD[{t}]", f"BRENT_CRUDE_FWD[{t}]", LOG, 0.95)
            for t in tokens
        ],
        axis,
    )
    assert _same_arrays(container, explicit)


def test_container_and_node_terms_on_one_pillar_are_summed(extended_static_config):
    cfg = extended_static_config
    axis = _axis(cfg, "WTI_CRUDE_FWD[M+0]")
    model = _compile(
        cfg,
        [
            ContainerProxyTerm(
                target_container_name="WTI_CRUDE_FWD",
                driver_container_name="BRENT_CRUDE_FWD",
                driver_chart=LOG,
                loading=0.5,
            ),
            _node("WTI_CRUDE_FWD[M+0]", "BRENT_CRUDE_FWD[M+0]", LOG, 0.25),
        ],
        axis,
    )
    assert model.term_loadings.tolist() == [0.5 + 0.25]


# --------------------------------------------------------------------------- #
# K-L. Driver chart tau is independent of target chart kappa                   #
# --------------------------------------------------------------------------- #


def _forward(chart: int, v0: float, v1: float) -> float:
    return {1: math.log(v1 / v0), 2: v1 / v0 - 1.0, 3: v1 - v0}[chart]


def _inverse(chart: int, a0: float, x: float) -> float:
    return {1: a0 * math.exp(x), 2: a0 * (1.0 + x), 3: a0 + x}[chart]


def test_driver_chart_is_independent_of_target_chart(extended_static_config):
    cfg = extended_static_config
    wti = int(cfg.get_rf_id("WTI_CRUDE_FWD[M+0]"))
    spread = int(cfg.get_rf_id("TEST_BRENT_WTI_SPREAD_FWD[M+0]"))
    model = _compile(
        cfg,
        [_node("WTI_CRUDE_FWD[M+0]", "TEST_BRENT_WTI_SPREAD_FWD[M+0]", DIFF, 0.04)],
        _axis(cfg, "WTI_CRUDE_FWD[M+0]"),
    )
    kappa = int(cfg.rf_id_to_shift_type_id[wti])
    assert kappa == int(LOG)
    [(driver, tau, beta)] = _rows(cfg, model)[wti]
    assert (driver, tau, beta) == (spread, int(DIFF), 0.04)

    # Independent closed-form oracle on the compiled term (not a kernel):
    # X = beta * T_tau(2.00, 2.50); A_s = T^-1_kappa(70, X).
    a_s = _inverse(kappa, 70.0, beta * _forward(tau, 2.00, 2.50))
    assert a_s == pytest.approx(70.0 * math.exp(0.02), rel=0, abs=1e-12)
    assert round(a_s, 4) == 71.4141
    # Routing the target chart into the driver column would give another value.
    misrouted = _inverse(kappa, 70.0, beta * _forward(kappa, 2.00, 2.50))
    assert abs(misrouted - a_s) > 0.5


def test_driver_configured_shift_type_does_not_select_the_chart(extended_static_config):
    cfg = extended_static_config
    brent = int(cfg.get_rf_id("BRENT_CRUDE_FWD[M+0]"))
    libor = int(cfg.get_rf_id("USD_LIBOR_3M[10Y]"))
    model = _compile(
        cfg,
        [
            # LOG-configured driver read under DIFF by a LOG target.
            _node("WTI_CRUDE_FWD[M+0]", "BRENT_CRUDE_FWD[M+0]", DIFF, 0.01),
            # DIFF target driven by a LOG column of a LOG-configured price.
            _node("USD_LIBOR_3M[10Y]", "BRENT_CRUDE_FWD[M+0]", LOG, 0.002),
        ],
        _axis(cfg, "WTI_CRUDE_FWD[M+0]", "USD_LIBOR_3M[10Y]"),
    )
    rows = _rows(cfg, model)
    assert rows[int(cfg.get_rf_id("WTI_CRUDE_FWD[M+0]"))] == [(brent, int(DIFF), 0.01)]
    assert rows[libor] == [(brent, int(LOG), 0.002)]


def test_one_driver_under_two_charts_is_two_terms(extended_static_config):
    cfg = extended_static_config
    brent = int(cfg.get_rf_id("BRENT_CRUDE_FWD[M+0]"))
    model = _compile(
        cfg,
        [
            _node("WTI_CRUDE_FWD[M+0]", "BRENT_CRUDE_FWD[M+0]", DIFF, 0.02),
            _node("WTI_CRUDE_FWD[M+0]", "BRENT_CRUDE_FWD[M+0]", LOG, 0.5),
        ],
        _axis(cfg, "WTI_CRUDE_FWD[M+0]"),
    )
    assert _rows(cfg, model)[int(cfg.get_rf_id("WTI_CRUDE_FWD[M+0]"))] == [
        (brent, int(LOG), 0.5),
        (brent, int(DIFF), 0.02),
    ]


# --------------------------------------------------------------------------- #
# Native rows (D025) and the V1 universe                                       #
# --------------------------------------------------------------------------- #


def test_native_axis_nodes_are_not_rows(extended_static_config):
    cfg = extended_static_config
    assert int(cfg.rf_id_to_shift_type_id[cfg.get_rf_id("AAPL_EQ")]) == int(
        ShiftType.ABSOLUTE_BASE_RETURN
    )
    assert int(cfg.rf_id_to_shift_type_id[cfg.get_rf_id("AAPL_ATM_VOL")]) == int(
        ShiftType.NO_SHIFT
    )
    with_native = _compile(cfg, [], _axis(cfg, "AAPL_EQ", "USDEUR", "AAPL_ATM_VOL"))
    without = _compile(cfg, [], _axis(cfg, "USDEUR"))
    np.testing.assert_array_equal(
        with_native.row_target_rf_ids, [int(cfg.get_rf_id("USDEUR"))]
    )
    assert _same_arrays(with_native, without)


@pytest.mark.parametrize("target", ["AAPL_EQ", "AAPL_ATM_VOL"])
def test_native_targets_are_rejected(extended_static_config, target):
    cfg = extended_static_config
    with pytest.raises(ConfigurationError, match="outside the V1 target universe"):
        _compile(cfg, [_node(target, "USDEUR", LOG, 1.0)], _axis(cfg, "USDEUR"))


@pytest.mark.parametrize("driver", ["AAPL_EQ", "AAPL_ATM_VOL"])
def test_native_nodes_may_drive_under_a_v1_chart(extended_static_config, driver):
    cfg = extended_static_config
    model = _compile(cfg, [_node("USDEUR", driver, LOG, 0.5)], _axis(cfg, "USDEUR"))
    np.testing.assert_array_equal(
        model.term_driver_rf_ids, [int(cfg.get_rf_id(driver))]
    )
    np.testing.assert_array_equal(model.term_chart_ids, [int(LOG)])


def test_off_axis_terms_are_validated_but_produce_no_row(extended_static_config):
    cfg = extended_static_config
    axis = _axis(cfg, "USDEUR")
    base = _compile(cfg, [], axis)
    off_axis = _compile(cfg, [_node("EURCHF", "USDCHF", LOG, 1.0)], axis)
    assert _same_arrays(base, off_axis)
    with pytest.raises(ConfigurationError, match="driver-chart catalogue"):
        _compile(cfg, [_node("EURCHF", "USDCHF", ShiftType.NO_SHIFT, 1.0)], axis)


# --------------------------------------------------------------------------- #
# Fail fast                                                                    #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "chart", [ShiftType.NO_SHIFT, ShiftType.ABSOLUTE_BASE_RETURN, 1, "LOG_RETURN"]
)
def test_invalid_driver_charts_are_rejected(extended_static_config, chart):
    cfg = extended_static_config
    with pytest.raises(ConfigurationError, match="driver_chart"):
        _compile(cfg, [_node("USDEUR", "USDCHF", chart, 1.0)], _axis(cfg, "USDEUR"))


@pytest.mark.parametrize(
    "loading", [math.nan, math.inf, -math.inf, 1, True, np.float32(0.5), "0.5"]
)
def test_invalid_loadings_are_rejected(extended_static_config, loading):
    cfg = extended_static_config
    with pytest.raises(ConfigurationError, match="loading"):
        _compile(cfg, [_node("USDEUR", "USDCHF", LOG, loading)], _axis(cfg, "USDEUR"))


@pytest.mark.parametrize(
    ("target", "driver", "field"),
    [
        ("NOT_A_FACTOR", "USDCHF", "target_name"),
        ("USDEUR", "NOT_A_FACTOR", "driver_name"),
        ("", "USDCHF", "target_name"),
        ("WTI_CRUDE_FWD", "USDCHF", "target_name"),
    ],
)
def test_unresolved_node_names_are_rejected(
    extended_static_config, target, driver, field
):
    cfg = extended_static_config
    with pytest.raises(ConfigurationError, match=field):
        _compile(cfg, [_node(target, driver, LOG, 1.0)], _axis(cfg, "USDEUR"))


def test_unresolved_container_names_are_rejected(extended_static_config):
    cfg = extended_static_config
    for target, driver in (("NOPE", "BRENT_CRUDE_FWD"), ("WTI_CRUDE_FWD", "USDEUR")):
        with pytest.raises(ConfigurationError, match="not a curve or surface"):
            _compile(
                cfg,
                [
                    ContainerProxyTerm(
                        target_container_name=target,
                        driver_container_name=driver,
                        driver_chart=LOG,
                        loading=1.0,
                    )
                ],
                _axis(cfg, "USDEUR"),
            )


def test_exact_rejects_unequal_coordinate_kinds(canonical_static_config):
    # The legacy canonical row POWER_CH_BASELOAD_HY <- CO2_EUA_ANNUAL_FWD pairs
    # OFFSET_HY with OFFSET_Y: no pillar can match under EXACT (D088 §8).
    cfg = canonical_static_config
    with pytest.raises(ConfigurationError, match="coordinate kinds"):
        _compile(
            cfg,
            [
                ContainerProxyTerm(
                    target_container_name="POWER_CH_BASELOAD_HY",
                    driver_container_name="CO2_EUA_ANNUAL_FWD",
                    driver_chart=LOG,
                    loading=1.2,
                )
            ],
            _axis(cfg, "POWER_CH_BASELOAD_HY[HY+0]"),
        )


def test_malformed_terms_and_inputs_are_rejected(extended_static_config):
    cfg = extended_static_config
    axis = _axis(cfg, "USDEUR")
    with pytest.raises(TypeError, match="NodeProxyTerm or ContainerProxyTerm"):
        _compile(cfg, [("USDEUR", "USDCHF", LOG, 1.0)], axis)
    with pytest.raises(TypeError, match="as_of_date"):
        _compile(cfg, [], axis, as_of=datetime(2026, 9, 25, tzinfo=timezone.utc))
    with pytest.raises(TypeError, match="as_of_date"):
        _compile(cfg, [], axis, as_of="2026-09-25")
    with pytest.raises(TypeError, match="int32"):
        _compile(cfg, [], axis.astype(np.int64))
    with pytest.raises(TypeError, match="int32"):
        _compile(cfg, [], axis.reshape(1, 1))
    with pytest.raises(ValueError, match="unique"):
        _compile(cfg, [], np.concatenate([axis, axis]))
    with pytest.raises(ConfigurationError, match="outside"):
        _compile(cfg, [], np.array([cfg.max_rf_id + 1], dtype=np.int32))
    with pytest.raises(ConfigurationError, match="unoccupied"):
        _compile(cfg, [], np.array([500], dtype=np.int32))


# --------------------------------------------------------------------------- #
# Boundary: no world, no runtime, no legacy source selection                   #
# --------------------------------------------------------------------------- #


def test_compiler_inputs_are_exactly_the_four_semantic_inputs():
    params = inspect.signature(compile_proxy_model).parameters
    assert list(params) == [
        "authored_terms",
        "static_config",
        "as_of_date",
        "run_axis_rf_ids",
    ]


def test_compiler_imports_no_world_runtime_or_legacy_module():
    forbidden = (
        "riskquant.simlab",
        "riskquant.valuation",
        "riskquant.sensitivity_pnl",
        "riskquant.pnl",
        "riskquant.reporting",
        "riskquant.scenario_metrics",
        "riskquant.portfolio",
        "riskquant.proxy_resolver",
        "riskquant.config.proxy_config",
        "riskquant.solutions.risk_run",
        "riskquant.solutions.scenario_models",
    )
    code = (
        "import sys\n"
        "import riskquant.config.proxy_model.compile\n"
        "import riskquant.solutions.run_identity.hashing\n"
        f"bad = sorted(m for m in sys.modules if m.startswith({forbidden!r}))\n"
        "print('\\n'.join(bad))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "", out.stdout


def test_legacy_source_selection_vocabulary_is_absent():
    tokens = (
        "source_factor",
        "shock_scale",
        "ID_PROXY",
        "CONTAINER_PROXY",
        "TENOR_FALLBACK",
        "DIRECT",
    )
    offenders = []
    for path in sorted(Path(proxy_model_pkg.__path__[0]).glob("*.py")):
        text = path.read_text(encoding="utf-8")
        offenders += [f"{path.name}: {t}" for t in tokens if t in text]
    assert not offenders, offenders


def test_compiled_model_rejects_non_canonical_construction():
    good = {
        "row_target_rf_ids": np.array([5, 1], dtype=np.int32),
        "row_term_offsets": np.array([0, 2, 3], dtype=np.int32),
        "term_driver_rf_ids": np.array([1, 1, 1], dtype=np.int32),
        "term_chart_ids": np.array([1, 3, 1], dtype=np.int8),
        "term_loadings": np.array([0.5, 0.25, 1.0], dtype=np.float64),
    }
    CompiledProxyModel(**good)
    cases = [
        ("term_chart_ids", np.array([3, 1, 1], dtype=np.int8), "strictly increasing"),
        ("term_chart_ids", np.array([1, 4, 1], dtype=np.int8), "catalogue"),
        ("term_loadings", np.array([0.5, 0.0, 1.0]), "non-zero"),
        ("term_loadings", np.array([0.5, np.nan, 1.0]), "finite"),
        ("row_term_offsets", np.array([0, 3, 3], dtype=np.int32), "at least one term"),
        ("row_target_rf_ids", np.array([5, 5], dtype=np.int32), "unique"),
        ("row_target_rf_ids", np.array([5, 1], dtype=np.int64), "dtype"),
    ]
    for field, value, message in cases:
        with pytest.raises((TypeError, ValueError), match=message):
            CompiledProxyModel(**{**good, field: value})
