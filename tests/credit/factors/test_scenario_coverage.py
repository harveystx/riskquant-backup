# tests/credit/factors/test_scenario_coverage.py
"""
Regression tests for audit C-04: a batched producer that does not partition
``[0, S)`` exactly must fail loudly, not ship uninitialised memory.

`build_credit_factor_matrix` and `run_jtd_streaming` both allocate with
`np.empty` and fill one scenario span per batch. A truncated generator, a gap,
an overlap or an overrun previously produced plausible risk numbers from heap
residue — and because the residue is often a previous run's data, the corruption
is not even reliably visible. The `poisoned` test below pins that down.

Coverage truth here is written out by hand as spans; nothing is re-derived from
the implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from riskquant.credit.factors.simulate.base import (
    CreditFactorBatch,
    build_credit_factor_matrix,
)
from riskquant.credit.jtd.blocks.forced_default import ForcedDefaultJtdBlock
from riskquant.credit.jtd.compiled import CompiledJtdPortfolio
from riskquant.credit.jtd.streaming import constant_ead_provider, run_jtd_streaming

S = 8
N_FACTORS = 3

# (label, spans, expected error fragment) — None means "must be accepted".
SPAN_CASES = [
    ("complete_two_batches", [(0, 4), (4, 4)], None),
    ("complete_uneven", [(0, 5), (5, 3)], None),
    ("complete_single", [(0, 8)], None),
    ("truncated", [(0, 4)], "never written"),
    ("empty_producer", [], "never written"),
    ("gap", [(0, 2), (4, 4)], "never written"),
    ("overlap", [(0, 4), (2, 4), (4, 4)], "overlaps an earlier batch"),
    ("duplicate_batch", [(0, 4), (0, 4), (4, 4)], "overlaps an earlier batch"),
    ("overrun", [(0, 4), (6, 4)], "only \\[0, 8\\) exist"),
    ("negative_offset", [(-2, 4)], "negative"),
    ("empty_batch", [(0, 0), (0, 8)], "non-positive size"),
]


def _factor_batches(spans):
    return [
        CreditFactorBatch(
            latent_factors=np.full((n, N_FACTORS), float(o + 1)),
            scenario_offset=o,
            scenario_count_total=S,
        )
        for o, n in spans
    ]


def _portfolio() -> CompiledJtdPortfolio:
    return CompiledJtdPortfolio(
        forced_default_jtd_block=ForcedDefaultJtdBlock(
            obligor_uids=np.array([1, 2], dtype=np.int64), lgd=np.ones(2)
        )
    )


@pytest.mark.parametrize("label,spans,err", SPAN_CASES, ids=[c[0] for c in SPAN_CASES])
def test_factor_matrix_coverage(label, spans, err) -> None:
    if err is None:
        matrix = build_credit_factor_matrix(
            _factor_batches(spans), num_factors=N_FACTORS, num_scenarios=S
        )
        # Each span was filled with (offset + 1), so the assembled matrix is
        # fully determined by the span list — independent of the assembler.
        expected = np.empty((N_FACTORS, S))
        for o, n in spans:
            expected[:, o : o + n] = float(o + 1)
        np.testing.assert_array_equal(matrix.latent_factors, expected)
    else:
        with pytest.raises(ValueError, match=err):
            build_credit_factor_matrix(
                _factor_batches(spans), num_factors=N_FACTORS, num_scenarios=S
            )


@pytest.mark.parametrize("label,spans,err", SPAN_CASES, ids=[c[0] for c in SPAN_CASES])
def test_streaming_coverage(label, spans, err) -> None:
    ead = np.array([100.0, 200.0])
    kwargs = dict(
        jtd_portfolio=_portfolio(),
        ead_provider=constant_ead_provider(ead),
        total_scenarios=S,
    )
    if err is None:
        result = run_jtd_streaming(factor_batches=_factor_batches(spans), **kwargs)
        # Forced default: every scenario's portfolio loss is sum(EAD * LGD).
        np.testing.assert_array_equal(result.portfolio_loss, np.full(S, 300.0))
    else:
        with pytest.raises(ValueError, match=err):
            run_jtd_streaming(factor_batches=_factor_batches(spans), **kwargs)


def test_factor_axis_mismatch_is_named_explicitly() -> None:
    batches = [
        CreditFactorBatch(
            latent_factors=np.zeros((S, N_FACTORS + 1)),
            scenario_offset=0,
            scenario_count_total=S,
        )
    ]
    with pytest.raises(ValueError, match="factor column"):
        build_credit_factor_matrix(batches, num_factors=N_FACTORS, num_scenarios=S)


def test_zero_total_scenarios_is_rejected() -> None:
    with pytest.raises(ValueError, match="total_scenarios must be positive"):
        build_credit_factor_matrix([], num_factors=N_FACTORS, num_scenarios=0)


def test_truncated_producer_fails_even_when_the_heap_looks_plausible() -> None:
    """The failure mode this guards is NOT 'obviously garbage appears'.

    `np.empty` frequently hands back a block a previous run just released, so a
    truncated producer can return numbers that look entirely correct. Poison a
    same-shaped block, release it, then truncate: before the coverage proof this
    returned the poison as if it were risk output.
    """
    poison = np.full(S, -7777.0)
    del poison  # the allocator may hand this exact block back

    with pytest.raises(ValueError, match="never written"):
        run_jtd_streaming(
            jtd_portfolio=_portfolio(),
            factor_batches=_factor_batches([(0, 4)]),
            ead_provider=constant_ead_provider(np.array([100.0, 200.0])),
            total_scenarios=S,
        )
