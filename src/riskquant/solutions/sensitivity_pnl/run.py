# src/riskquant/solutions/sensitivity_pnl/run.py
"""
The one-shot entry point: a request in, an `(M, S)` PnL block out.

Unchanged in signature and behaviour, and now a two-line composition of the
cached-book path — compile once, run once. Keeping it as a wrapper rather than a
parallel implementation means there is exactly one compile path, one validation
path, and one streaming path to keep correct.

Reach for `SensitivityPnlCalculator` when the same book is priced against many
scenario sets; this remains the right call when it is priced against one.

Nothing about the compute topology crosses this boundary. The caller does not
choose a batch size, does not see a `PnlGrid`, is never told a scratch arena
exists, and cannot express a misaligned factor axis. What comes back is the same
`PnlMatrix` the full-reval historical-simulation model produces, so the
substitution is invisible to everything downstream.

Batch sizing is derived, not exposed
------------------------------------
`SensitivityPnlEngine` holds a `(Ka, B)` float32 displacement scratch, so batch
size is a memory decision — and one that depends on `Ka`, an output of
compilation and therefore unknowable to the caller before the run. Asking a desk
user for a batch size is asking them to guess at an array nobody has told them
about. It is derived from a fixed scratch budget and reported in provenance,
where it belongs: as an explanation of a run rather than a knob on it.
"""

from __future__ import annotations

__all__ = [
    "run_sensitivity_pnl",
]

from riskquant.solutions.sensitivity_pnl.calculator import SensitivityPnlCalculator
from riskquant.solutions.sensitivity_pnl.schemas import (
    SensitivityPnlRequest,
    SensitivityPnlResult,
)


def run_sensitivity_pnl(request: SensitivityPnlRequest) -> SensitivityPnlResult:
    """
    Compile a sensitivity book and evaluate it across one scenario set.

    Returns
    -------
    SensitivityPnlResult
        `.pnl_matrix` is a `PnlMatrix`: `(M,)` int64 position UIDs ascending, and
        an `(M, S)` C-contiguous float32 block whose column order matches the
        market's scenario order.

    Raises
    ------
    SensitivityPnlInputError
        Listing every defect found in the request, before anything is compiled
        or allocated.
    ValueError
        From the compiler or engine, for the contracts this layer deliberately
        does not duplicate — duplicate `(position, factor)` keys, a cross term
        with `i == j`, a non-positive base level under a ratio convention. Their
        messages are already written in the caller's terms, and
        `SensitivityPnlInputError` is itself a `ValueError`, so one `except`
        catches both.
    """
    calculator = SensitivityPnlCalculator.compile(request.book_spec())
    return calculator.run(request.market, book_reused=False)
