# tests/simlab/shocks/test_linear_term_application.py
"""
Proxy V1 Slice 2: linear term aggregation and target reconstruction.

Expected values come from closed-form formulas or from `_reference`, a plain
Python loop written independently of the kernel. Comparisons against the
reference use the D088 §13.6 policy (rel 1e-12, abs 1e-15); determinism,
batch and thread checks compare kernel output with kernel output bitwise.
"""

from __future__ import annotations

import itertools
import math
from datetime import date

import numba
import numpy as np
import pytest

from riskquant.simlab.shocks.column_binding import bind_term_columns
from riskquant.simlab.shocks.linear_term_apply import apply_linear_terms_to_base_levels

LOG, SIMPLE, DIFF, ABS, NO_SHIFT = 1, 2, 3, 4, -1
RTOL, ATOL = 1e-12, 1e-15


# --------------------------------------------------------------------------- #
# Independent reference and helpers                                            #
# --------------------------------------------------------------------------- #


def _reconstruct(kappa: int, base: float, x: float) -> float:
    if kappa == LOG:
        return base * math.exp(x)
    if kappa == SIMPLE:
        return base * (1.0 + x)
    if kappa == DIFF:
        return base + x
    if kappa == ABS:
        return base + x * abs(base)
    if kappa == NO_SHIFT:
        return base
    raise AssertionError(kappa)


def _reference(base, offsets, cols, loadings, kappas, shocks) -> np.ndarray:
    """Row, scenario, canonical term; float64 arithmetic; then reconstruct."""
    n_rows, n_scn = len(base), shocks.shape[0]
    out = np.empty((n_rows, n_scn), dtype=np.float64)
    for n in range(n_rows):
        for b in range(n_scn):
            x = 0.0
            for t in range(int(offsets[n]), int(offsets[n + 1])):
                x += float(loadings[t]) * float(shocks[b, int(cols[t])])
            out[n, b] = _reconstruct(int(kappas[n]), float(base[n]), x)
    return out


def _arrays(base, row_terms, kappas, shocks):
    """row_terms: list per row of (column, loading) in canonical order."""
    offsets = np.zeros(len(row_terms) + 1, dtype=np.int32)
    offsets[1:] = np.cumsum([len(r) for r in row_terms])
    cols = np.array([c for r in row_terms for c, _ in r], dtype=np.int32)
    loads = np.array([w for r in row_terms for _, w in r], dtype=np.float64)
    return (
        np.array(base, dtype=np.float64),
        offsets,
        cols,
        loads,
        np.array(kappas, dtype=np.int8),
        np.asfortranarray(np.asarray(shocks, dtype=np.float32)),
    )


def _run(base, offsets, cols, loads, kappas, shocks) -> np.ndarray:
    out = np.empty((base.shape[0], shocks.shape[0]), dtype=np.float64)
    apply_linear_terms_to_base_levels(
        base_levels=base,
        row_term_offsets=offsets,
        term_column_uix=cols,
        term_loadings=loads,
        target_shift_types=kappas,
        shocks=shocks,
        out=out,
    )
    return out


def _f32(x: float) -> float:
    return float(np.float32(x))


def _random_case(seed: int, n_rows: int = 9, n_scn: int = 37, n_cols: int = 11):
    rng = np.random.default_rng(seed)
    kappa_cycle = [LOG, SIMPLE, DIFF, ABS, NO_SHIFT]
    kappas, row_terms, base = [], [], []
    for n in range(n_rows):
        k = kappa_cycle[n % len(kappa_cycle)]
        kappas.append(k)
        base.append(
            float(rng.uniform(-50.0, 50.0))
            if k in (DIFF, ABS)
            else float(rng.uniform(1.0, 100.0))
        )
        m = 0 if k == NO_SHIFT else 1 + n % 4
        row_terms.append([
            (int(rng.integers(0, n_cols)), float(rng.uniform(-2.0, 2.0)))
            for _ in range(m)
        ])
    # Asymmetric markers: every (scenario, column) value is distinct.
    shocks = (
        rng.standard_normal((n_scn, n_cols)) * 0.01
        + np.arange(n_scn)[:, None] * 1e-4
        + np.arange(n_cols)[None, :] * 1e-6
    )
    return _arrays(base, row_terms, kappas, shocks)


# --------------------------------------------------------------------------- #
# Mathematical contract                                                        #
# --------------------------------------------------------------------------- #


def test_single_log_term_matches_closed_form():
    y = np.float32(math.log(82 / 80))
    out = _run(*_arrays([70.0], [[(0, 0.95)]], [LOG], [[y]]))
    expected = 70.0 * math.exp(0.95 * float(y))
    np.testing.assert_allclose(out, [[expected]], rtol=RTOL, atol=ATOL)
    assert abs(out[0, 0] - 71.66146991455523) < 1e-5  # D088 §11, float32 storage


def test_multi_term_aggregation_then_one_reconstruction():
    y1, y2 = np.float32(math.log(1.025)), np.float32(math.log(0.98))
    out = _run(*_arrays([70.0], [[(0, 0.7), (1, 0.3)]], [LOG], [[y1, y2]]))
    expected = 70.0 * math.exp(0.7 * float(y1) + 0.3 * float(y2))
    np.testing.assert_allclose(out, [[expected]], rtol=RTOL, atol=ATOL)
    assert abs(out[0, 0] - 70.79010695536556) < 1e-5


def test_log_target_diff_driver_uses_the_driver_chart_column():
    # World: driver 9000 under LOG (column 0) and under DIFF (column 1), for a
    # spread path 2.00 -> 2.50. The term asks for (9000, DIFF, h).
    world_d = np.array([9000, 9000], dtype=np.int32)
    world_tau = np.array([LOG, DIFF], dtype=np.int8)
    shocks = np.asfortranarray(
        np.array([[math.log(2.50 / 2.00), 0.50]], dtype=np.float32)
    )
    cols = bind_term_columns(
        term_driver_rf_ids=np.array([9000], dtype=np.int32),
        term_chart_ids=np.array([DIFF], dtype=np.int8),
        requested_horizon=1,
        world_column_driver_rf_ids=world_d,
        world_column_chart_ids=world_tau,
        world_horizon=1,
    )
    assert cols.tolist() == [1]
    base, offsets, _, loads, kappas, _ = _arrays([70.0], [[(0, 0.04)]], [LOG], [[0.0]])
    out = _run(base, offsets, cols, loads, kappas, shocks)
    expected = 70.0 * math.exp(0.04 * 0.50)  # 0.5 is exact in float32
    np.testing.assert_allclose(out, [[expected]], rtol=RTOL, atol=ATOL)
    assert round(float(out[0, 0]), 8) == 71.41409380
    # Routing the target chart (LOG) into the column lookup reads column 0.
    misrouted = _run(
        base, offsets, np.array([0], dtype=np.int32), loads, kappas, shocks
    )
    assert abs(float(misrouted[0, 0]) - float(out[0, 0])) > 0.5
    assert abs(float(misrouted[0, 0]) - 70.0 * math.exp(0.04 * math.log(1.25))) < 1e-5


def test_same_driver_under_two_charts_consumes_both_columns():
    world_d = np.array([9000, 9000, 777], dtype=np.int32)
    world_tau = np.array([DIFF, LOG, LOG], dtype=np.int8)
    cols = bind_term_columns(
        term_driver_rf_ids=np.array([9000, 9000], dtype=np.int32),
        term_chart_ids=np.array([LOG, DIFF], dtype=np.int8),
        requested_horizon=1,
        world_column_driver_rf_ids=world_d,
        world_column_chart_ids=world_tau,
        world_horizon=1,
    )
    assert cols.tolist() == [1, 0]
    shocks = np.asfortranarray(
        np.array([[2.0, math.log(82 / 80), 9.0]], dtype=np.float32)
    )
    base, offsets, _, loads, kappas, _ = _arrays(
        [70.0], [[(0, 0.5), (0, 0.02)]], [LOG], [[0.0]]
    )
    out = _run(base, offsets, cols, loads, kappas, shocks)
    expected = 70.0 * math.exp(0.5 * _f32(math.log(82 / 80)) + 0.02 * 2.0)
    np.testing.assert_allclose(out, [[expected]], rtol=RTOL, atol=ATOL)
    for c in (0, 1):  # perturb each bound column: the output must move
        bumped = shocks.copy(order="F")
        bumped[0, c] += np.float32(0.25)
        assert _run(base, offsets, cols, loads, kappas, bumped)[0, 0] != out[0, 0]
    unrelated = shocks.copy(order="F")
    unrelated[0, 2] = np.float32(1e30)
    assert _run(base, offsets, cols, loads, kappas, unrelated)[0, 0] == out[0, 0]


def test_signed_loadings_are_not_made_positive():
    y1, y2 = np.float32(0.02), np.float32(-0.01)
    out = _run(*_arrays([70.0], [[(0, 0.7), (1, -0.3)]], [LOG], [[y1, y2]]))
    correct = 70.0 * math.exp(0.7 * float(y1) - 0.3 * float(y2))
    absolute = 70.0 * math.exp(0.7 * float(y1) + 0.3 * float(y2))
    np.testing.assert_allclose(out, [[correct]], rtol=RTOL, atol=ATOL)
    assert abs(out[0, 0] - absolute) > 1e-3


def test_terms_are_summed_in_ascending_canonical_order_bitwise():
    # Products are exact (loading 1.0, shocks exact in float32), so FMA
    # contraction cannot change them; only the addition order matters.
    # Ascending: ((0 + 1) + 2^-53) + 2^-53 = 1.0 (each tie rounds to even).
    # Descending would give (2^-53 + 2^-53) + 1 = 1 + 2^-52.
    tiny = 2.0**-53
    shocks = [[1.0, tiny, tiny]]
    ascending = _run(*_arrays([0.0], [[(0, 1.0), (1, 1.0), (2, 1.0)]], [DIFF], shocks))
    assert ascending[0, 0] == 1.0
    reordered = _run(*_arrays([0.0], [[(1, 1.0), (2, 1.0), (0, 1.0)]], [DIFF], shocks))
    assert reordered[0, 0] == 1.0 + 2.0**-52


def test_aggregation_happens_before_reconstruction():
    y1, y2 = np.float32(0.2), np.float32(0.3)
    out = _run(*_arrays([70.0], [[(0, 1.0), (1, 1.0)]], [LOG], [[y1, y2]]))
    aggregated = 70.0 * math.exp(float(y1) + float(y2))
    per_term_levels = 70.0 * math.exp(float(y1)) + 70.0 * math.exp(float(y2)) - 70.0
    np.testing.assert_allclose(out, [[aggregated]], rtol=RTOL, atol=ATOL)
    assert abs(out[0, 0] - per_term_levels) > 1.0


def test_no_shift_row_needs_no_column_and_returns_the_base():
    base, offsets, cols, loads, kappas, _ = _arrays(
        [42.5, -3.0], [[], []], [NO_SHIFT, NO_SHIFT], [[]]
    )
    shocks = np.zeros((5, 0), dtype=np.float32, order="F")  # no shock column at all
    out = _run(base, offsets, cols, loads, kappas, shocks)
    assert out.tolist() == [[42.5] * 5, [-3.0] * 5]


def test_abs_row_uses_base_plus_x_times_abs_base():
    y = np.float32(0.25)
    out = _run(*_arrays([-2.0], [[(0, 1.0)]], [ABS], [[y]]))
    assert out[0, 0] == -2.0 + float(y) * 2.0  # -1.5; SIMPLE would give -2.5
    simple = _run(*_arrays([-2.0], [[(0, 1.0)]], [SIMPLE], [[y]]))
    assert simple[0, 0] == -2.5


def test_float32_storage_is_widened_not_reinterpreted():
    out = _run(*_arrays([0.0], [[(0, 1.0)]], [DIFF], [[0.1]]))
    assert out[0, 0] == float(np.float32(0.1))  # 0.10000000149011612
    assert out[0, 0] != 0.1


def test_scenario_results_are_independent():
    case = list(_random_case(3))
    base = _run(*case)
    shocks = case[5].copy(order="F")
    shocks[17, :] += np.float32(0.5)
    changed = _run(*case[:5], shocks)
    others = [b for b in range(shocks.shape[0]) if b != 17]
    assert np.array_equal(changed[:, others], base[:, others])
    assert not np.array_equal(changed[:, 17], base[:, 17])


def test_general_case_matches_the_independent_reference():
    for seed in range(5):
        case = _random_case(seed)
        out = _run(*case)
        np.testing.assert_allclose(out, _reference(*case), rtol=RTOL, atol=ATOL)


def test_duplicate_terms_one_term_rows_and_extreme_finite_values():
    shocks = [[3.0e38, 1.0, 700.0], [-3.0e38, -1.0, -700.0]]
    row_terms = [
        [(1, 0.5), (1, 0.25), (1, -0.125)],  # duplicate (column, chart) terms
        [(0, 1e-30)],  # one-term row, extreme stored shock
        [(2, 1.0)],  # LOG with |X| = 700: finite
        [(0, 1.0), (0, -1.0)],  # large cancellation in DIFF space
    ]
    case = _arrays([10.0, 5.0, 1.0, 4.0], row_terms, [DIFF, DIFF, LOG, DIFF], shocks)
    out = _run(*case)
    np.testing.assert_allclose(out, _reference(*case), rtol=RTOL, atol=ATOL)
    assert np.isfinite(out).all()


# --------------------------------------------------------------------------- #
# Determinism, batches, threads, inputs                                        #
# --------------------------------------------------------------------------- #


def test_batch_partitions_are_bitwise_invariant():
    case = _random_case(11, n_scn=37)
    base, offsets, cols, loads, kappas, shocks = case
    full = _run(*case)
    for sizes in ([37], [10, 10, 10, 7], [1, 36], [36, 1], [5] * 7 + [2], [1] * 37):
        parts, start = [], 0
        for size in sizes:
            batch = np.asfortranarray(shocks[start : start + size])
            parts.append(_run(base, offsets, cols, loads, kappas, batch))
            start += size
        assert np.array_equal(np.concatenate(parts, axis=1), full), sizes


def test_repetition_is_bitwise_invariant():
    case = _random_case(19, n_rows=40, n_scn=64)
    first = _run(*case)
    for _ in range(3):
        assert np.array_equal(_run(*case), first)


def test_thread_counts_are_bitwise_invariant():
    # Evidence must not be overstated: with a single available Numba thread no
    # comparison between distinct thread counts exists, so the test skips.
    available = numba.config.NUMBA_NUM_THREADS
    if available < 2:
        pytest.skip(
            f"NUMBA_NUM_THREADS={available}: only one thread count is available, "
            "so a comparison between distinct thread counts cannot be executed."
        )
    case = _random_case(19, n_rows=40, n_scn=64)
    previous = numba.get_num_threads()
    try:
        results = {}
        for threads in (1, available):
            numba.set_num_threads(threads)
            results[threads] = _run(*case)
    finally:
        numba.set_num_threads(previous)
    assert np.array_equal(results[available], results[1])


def test_caller_inputs_are_not_mutated_and_frozen_inputs_are_accepted():
    case = list(_random_case(5))
    before = [a.copy() for a in case]
    _run(*case)
    for a, b in zip(case, before):
        assert np.array_equal(a, b) and a.flags.writeable
    for a in case:
        a.setflags(write=False)
    frozen = _run(*case)
    np.testing.assert_allclose(frozen, _reference(*before), rtol=RTOL, atol=ATOL)


def test_chain_from_compiled_model_is_invariant_to_authoring_order():
    from riskquant.config.proxy_model.compile import compile_proxy_model
    from riskquant.config.proxy_model.schema_objects import NodeProxyTerm
    from riskquant.config.static_config.compile import (
        compile_risk_static_config_from_dict_of_dfs,
    )
    from riskquant.ingest_fixtures.config.risk_static_config.canonical_risk_static_config import (
        CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS,
    )
    from riskquant.market_data.risk_factors.shift_types import ShiftType

    cfg = compile_risk_static_config_from_dict_of_dfs(
        dict(CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS)
    )
    wti, brent, henry = (
        int(cfg.get_rf_id(n))
        for n in (
            "WTI_CRUDE_FWD[M+0]",
            "BRENT_CRUDE_FWD[M+0]",
            "HENRY_HUB_GAS_FWD[M+0]",
        )
    )
    terms = [
        NodeProxyTerm(
            target_name="WTI_CRUDE_FWD[M+0]",
            driver_name="HENRY_HUB_GAS_FWD[M+0]",
            driver_chart=ShiftType.LOG_RETURN,
            loading=-0.3,
        ),
        NodeProxyTerm(
            target_name="WTI_CRUDE_FWD[M+0]",
            driver_name="BRENT_CRUDE_FWD[M+0]",
            driver_chart=ShiftType.DIFFERENCE,
            loading=0.04,
        ),
        NodeProxyTerm(
            target_name="WTI_CRUDE_FWD[M+0]",
            driver_name="BRENT_CRUDE_FWD[M+0]",
            driver_chart=ShiftType.LOG_RETURN,
            loading=0.7,
        ),
    ]
    axis = np.array([wti, brent], dtype=np.int32)
    world_d = np.array([henry, brent, brent], dtype=np.int32)
    world_tau = np.array([LOG, DIFF, LOG], dtype=np.int8)
    rng = np.random.default_rng(7)
    shocks = np.asfortranarray((rng.standard_normal((16, 3)) * 0.02).astype(np.float32))
    kappas = cfg.rf_id_to_shift_type_id[axis].astype(np.int8)
    base = np.array([70.0, 82.0])
    outputs = []
    for perm in itertools.permutations(terms):
        model = compile_proxy_model(
            authored_terms=list(perm),
            static_config=cfg,
            as_of_date=date(2026, 9, 25),
            run_axis_rf_ids=axis,
        )
        cols = bind_term_columns(
            term_driver_rf_ids=model.term_driver_rf_ids,
            term_chart_ids=model.term_chart_ids,
            requested_horizon=1,
            world_column_driver_rf_ids=world_d,
            world_column_chart_ids=world_tau,
            world_horizon=1,
        )
        out = _run(
            base, model.row_term_offsets, cols, model.term_loadings, kappas, shocks
        )
        np.testing.assert_allclose(
            out,
            _reference(
                base, model.row_term_offsets, cols, model.term_loadings, kappas, shocks
            ),
            rtol=RTOL,
            atol=ATOL,
        )
        outputs.append(out)
    for out in outputs[1:]:
        assert np.array_equal(out, outputs[0])


# --------------------------------------------------------------------------- #
# Fail fast                                                                    #
# --------------------------------------------------------------------------- #


def _valid():
    return dict(
        zip(
            (
                "base_levels",
                "row_term_offsets",
                "term_column_uix",
                "term_loadings",
                "target_shift_types",
                "shocks",
            ),
            _arrays(
                [70.0, 5.0, 1.0],
                [[(0, 0.5), (1, 0.5)], [], [(1, 1.0)]],
                [LOG, NO_SHIFT, DIFF],
                [[0.01, 0.02], [0.03, 0.04]],
            ),
        )
    )


def _call(args, out=None):
    if out is None:
        out = np.empty((np.shape(args["base_levels"])[0], np.shape(args["shocks"])[0]))
    apply_linear_terms_to_base_levels(**args, out=out)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("row_term_offsets", np.array([1, 2, 2, 3], dtype=np.int32), "must be 0"),
        ("row_term_offsets", np.array([0, 2, 1, 3], dtype=np.int32), "non-decreasing"),
        ("row_term_offsets", np.array([0, 2, 2, 4], dtype=np.int32), "term count"),
        ("row_term_offsets", np.array([0, 2, 3], dtype=np.int32), "N \\+ 1"),
        ("row_term_offsets", np.array([0, 2, 2, 3], dtype=np.int64), "dtype"),
        ("term_column_uix", np.array([0, 2, 1], dtype=np.int32), r"\[0, K\)"),
        ("term_column_uix", np.array([0, -1, 1], dtype=np.int32), r"\[0, K\)"),
        ("term_loadings", np.array([0.5, np.nan, 1.0]), "finite"),
        ("term_loadings", np.array([0.5, 0.5, 1.0], dtype=np.float32), "dtype"),
        ("target_shift_types", np.array([LOG, 0, DIFF], dtype=np.int8), "outside"),
        ("target_shift_types", np.array([LOG, NO_SHIFT, 5], dtype=np.int8), "outside"),
        ("target_shift_types", np.array([LOG, LOG, DIFF], dtype=np.int8), "zero terms"),
        (
            "target_shift_types",
            np.array([NO_SHIFT, NO_SHIFT, DIFF], dtype=np.int8),
            "carry terms",
        ),
        (
            "target_shift_types",
            np.array([LOG, NO_SHIFT, DIFF], dtype=np.int32),
            "dtype",
        ),
        (
            "shocks",
            np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float64, order="F"),
            "dtype",
        ),
        (
            "shocks",
            np.ascontiguousarray([[0.01, 0.02], [0.03, 0.04]], dtype=np.float32),
            "F-contiguous",
        ),
        ("base_levels", np.array([70.0, 5.0]), "N \\+ 1"),
        ("base_levels", np.array([70.0, 0.0, 5.0, 0.0, 1.0, 0.0])[::2], "C-contiguous"),
        ("base_levels", [70.0, 5.0, 1.0], "numpy.ndarray"),
    ],
)
def test_contract_violations_are_rejected(field, value, match):
    args = _valid()
    args[field] = value
    with pytest.raises((ValueError, TypeError), match=match):
        _call(args)


@pytest.mark.parametrize("victim", ["base_levels", "term_loadings"])
def test_output_overlapping_an_input_is_rejected(victim):
    # `out` and one float64 input are views of one buffer: writing `out` would
    # overwrite the input while the kernel still reads it.
    args = _valid()
    n_rows, n_scn = args["base_levels"].shape[0], args["shocks"].shape[0]
    buffer = np.zeros(n_rows * n_scn)
    out = buffer.reshape(n_rows, n_scn)
    shared = buffer[: args[victim].shape[0]]
    shared[:] = args[victim]
    args[victim] = shared
    assert np.shares_memory(out, args[victim])
    assert out.flags.c_contiguous and out.flags.writeable
    assert args[victim].flags.c_contiguous
    with pytest.raises(ValueError, match=f"out must not overlap {victim}"):
        _call(args, out)
    # Control: the same values in separate memory are accepted.
    args[victim] = shared.copy()
    _call(args, np.empty((n_rows, n_scn)))


def test_output_buffer_violations_are_rejected():
    args = _valid()
    for out, match in (
        (np.empty((3, 1)), "out shape"),
        (np.empty((3, 2), order="F"), "C-contiguous"),
        (np.empty((3, 2), dtype=np.float32), "dtype"),
    ):
        with pytest.raises(ValueError, match=match):
            _call(args, out)
    frozen = np.empty((3, 2))
    frozen.setflags(write=False)
    with pytest.raises(ValueError, match="writable"):
        _call(args, frozen)
