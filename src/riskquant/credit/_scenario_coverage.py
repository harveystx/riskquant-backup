# src/riskquant/credit/_scenario_coverage.py
"""
Scenario-coverage proof for batched producers.

Both batched credit consumers — `factors.simulate.base.build_credit_factor_matrix`
and `jtd.streaming.run_jtd_streaming` — allocate an accumulator with `np.empty`
and fill it one scenario span at a time. That is correct only if the spans
partition ``[0, total_scenarios)`` exactly. A short generator, an exhausted
iterator, a duplicated batch or an off-by-one offset otherwise leaves
uninitialised heap in the scenario axis, and the run produces plausible risk
numbers with no error (audit C-04).

The invariant is PROVED here, not inferred from a length tally: one bool per
scenario records which scenarios have been written. Overlap and out-of-range
fail at the offending batch; gaps fail at the end. Cost is one byte per
scenario against the eight bytes per scenario the portfolio-loss accumulator
already carries, and O(S) marking against the O(N*S) kernel work it guards —
negligible in both dimensions.

Python-side only. Nothing here crosses a JIT boundary.
"""

from __future__ import annotations

__all__ = [
    "assert_scenario_coverage_complete",
    "new_scenario_coverage",
    "record_scenario_span",
]

import numpy as np
import numpy.typing as npt

_PREVIEW = 10


def new_scenario_coverage(
    total_scenarios: int,
    *,
    ctx: str,
) -> npt.NDArray[np.bool_]:
    """Allocate the per-scenario written-flag array for ``total_scenarios``.

    Parameters
    ----------
    total_scenarios
        Scenario count the producer must cover exactly. Must be positive.
    ctx
        Caller tag used to prefix every error raised against this coverage.

    Returns
    -------
    npt.NDArray[np.bool_]
        ``(total_scenarios,)`` all-False write flags.
    """
    if total_scenarios <= 0:
        raise ValueError(
            f"{ctx}: total_scenarios must be positive, got {total_scenarios}."
        )
    return np.zeros(total_scenarios, dtype=np.bool_)


def record_scenario_span(
    coverage: npt.NDArray[np.bool_],
    *,
    offset: int,
    count: int,
    ctx: str,
) -> None:
    """Record one batch's scenario span, rejecting empty, out-of-range or repeat writes.

    Fails at the offending batch rather than at the end, so the error names the
    batch that broke the partition.

    Raises
    ------
    ValueError
        If ``count`` is not positive, the span leaves ``[0, S)``, or any
        scenario in the span was already written by an earlier batch.
    """
    total = int(coverage.shape[0])

    if count <= 0:
        raise ValueError(
            f"{ctx}: batch at scenario offset {offset} has non-positive size "
            f"{count}; every batch must cover at least one scenario."
        )
    if offset < 0:
        raise ValueError(f"{ctx}: batch scenario offset {offset} is negative.")
    if offset + count > total:
        raise ValueError(
            f"{ctx}: batch covers scenarios [{offset}, {offset + count}) but "
            f"only [0, {total}) exist; the producer overruns the declared "
            f"scenario count."
        )

    span = coverage[offset : offset + count]
    if span.any():
        first = offset + int(np.flatnonzero(span)[0])
        raise ValueError(
            f"{ctx}: batch covering scenarios [{offset}, {offset + count}) "
            f"overlaps an earlier batch; scenario {first} would be written "
            f"twice. Batches must partition [0, {total})."
        )
    span[:] = True


def assert_scenario_coverage_complete(
    coverage: npt.NDArray[np.bool_],
    *,
    ctx: str,
) -> None:
    """Assert every scenario was written exactly once.

    Raises
    ------
    ValueError
        If any scenario is unwritten. The message names how many are missing
        and the first few indices, so a truncated generator and a gapped one
        are distinguishable from the error alone.
    """
    if bool(coverage.all()):
        return

    missing = np.flatnonzero(~coverage)
    total = int(coverage.shape[0])
    raise ValueError(
        f"{ctx}: the producer covered {total - missing.size} of {total} "
        f"scenarios; {missing.size} were never written. First unwritten "
        f"scenario index(es): {missing[:_PREVIEW].tolist()}. The accumulator "
        f"would otherwise carry uninitialised memory for those scenarios."
    )
