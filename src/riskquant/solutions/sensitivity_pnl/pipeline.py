# src/riskquant/solutions/sensitivity_pnl/pipeline.py
"""
The stages every entry point shares, in one place.

The one-shot runner, the calculator, and the grid handoff differ only in how
often they compile and where the `(M, S)` block lives. Everything between the
inputs and the kernel is identical, so it is written once here rather than
imported across module privacy.

Factor-metadata scoping
-----------------------
`factor_meta` is scoped to the axis before the compiler sees it, which is what
lets a caller pass the house-wide factor table — twenty thousand rows — against a
forty-factor axis.

That is not a convenience. Under SLOT encoding the compiler would resolve
`factor_meta.factor_id` through the same lookup as the sensitivity frames, so a
metadata row for a factor outside the axis is not ignored: it raises. Scoping
first makes "extra rows are ignored" true under both encodings, which is the kind
of asymmetry that otherwise survives review and surfaces a quarter later.

Slot relabelling
----------------
Under SLOT encoding the factor-key columns are rewritten to row positions HERE,
in the facade, and the compiler is handed `factor_id_lookup=None` on every path.
Measured at 792k entries the two are within 2 ms of each other, and doing it here
keeps the whole label-to-index compaction inside the perimeter that owns it —
`riskquant.sensitivity_pnl.compile` needs no change to support string factor
keys, and no signature there has to widen to admit non-string labels.
"""

from __future__ import annotations

__all__ = [
    "ENGINE_VERSION",
    "build_provenance",
    "compile_book",
    "make_engine",
    "resolve_batch_size",
    "scope_factor_meta",
    "stream_scenarios",
]

import numpy as np
import numpy.typing as npt
import polars as pl

from riskquant.sensitivity_pnl.compile import (
    COL_FACTOR_ID,
    COL_FACTOR_ID_I,
    COL_FACTOR_ID_J,
    compile_sensitivity_book,
)
from riskquant.sensitivity_pnl.contracts import CompiledSensitivityBook
from riskquant.sensitivity_pnl.engine import SensitivityPnlEngine
from riskquant.solutions.sensitivity_pnl.market import (
    FactorAxis,
    MarketScenarios,
    ScenarioLevels,
    ScenarioShocks,
)
from riskquant.solutions.sensitivity_pnl.schemas import (
    SensitivityBookSpec,
    SensitivityPnlProvenance,
)

ENGINE_VERSION = "riskquant-sensitivity-pnl/1.1.0"

#: Ceiling on the engine's `(Ka, B)` float32 displacement scratch. Sized so the
#: arena stays small enough to stay resident for typical `Ka`, and so a
#: pathological `Ka` cannot turn a large scenario count into a multi-gigabyte
#: allocation the caller never asked for.
SCRATCH_BUDGET_BYTES: int = 16 * 1024 * 1024

#: Batches wider than this stop paying: per-batch Python overhead is already
#: amortised to nothing, while the per-thread float64 accumulator keeps growing.
MAX_BATCH_SCENARIOS: int = 4096

_DIAGONAL_FACTOR_COLUMNS = (COL_FACTOR_ID,)
_CROSS_FACTOR_COLUMNS = (COL_FACTOR_ID_I, COL_FACTOR_ID_J)


def scope_factor_meta(spec: SensitivityBookSpec) -> pl.DataFrame:
    """
    Restrict `factor_meta` to factors the axis carries.

    Returns the frame untouched when the key column is absent or in the wrong key
    space — validation reports that and raises, and filtering on a mismatched
    dtype would only replace one error with a worse one.
    """
    frame = spec.factor_meta
    keys = spec.factor_axis.keys
    if COL_FACTOR_ID not in frame.columns or frame.schema[COL_FACTOR_ID] != keys.dtype:
        return frame
    return frame.filter(pl.col(COL_FACTOR_ID).is_in(keys))


def compile_book(
    spec: SensitivityBookSpec, *, factor_meta: pl.DataFrame
) -> CompiledSensitivityBook:
    """
    Resolve the book against the axis, relabelling factor keys where needed.

    `level_universe_factor_ids` comes FROM the axis rather than being accepted
    alongside it. That is the whole reason an unaligned scenario matrix is
    unrepresentable here rather than merely discouraged.
    """
    axis = spec.factor_axis
    cross = spec.cross_sensitivities
    return compile_sensitivity_book(
        sensitivities=axis.relabel(spec.sensitivities, _DIAGONAL_FACTOR_COLUMNS),
        factor_meta=axis.relabel(factor_meta, _DIAGONAL_FACTOR_COLUMNS),
        level_universe_factor_ids=axis.universe_ids,
        cross_sensitivities=(
            None if cross is None else axis.relabel(cross, _CROSS_FACTOR_COLUMNS)
        ),
        extra_position_uids=_declared_uids(spec),
        duplicate_policy=spec.duplicate_policy,
    )


def _declared_uids(spec: SensitivityBookSpec) -> npt.NDArray[np.int64] | None:
    if spec.position_uids is None:
        return None
    return np.ascontiguousarray(spec.position_uids, dtype=np.int64)


def resolve_batch_size(*, num_active_factors: int, num_scenarios: int) -> int:
    """Widest batch whose displacement scratch fits the budget; never zero."""
    if num_active_factors == 0:
        return min(num_scenarios, MAX_BATCH_SCENARIOS)
    affordable = SCRATCH_BUDGET_BYTES // (num_active_factors * 4)
    return max(1, min(affordable, MAX_BATCH_SCENARIOS, num_scenarios))


def make_engine(
    *,
    book: CompiledSensitivityBook,
    market: MarketScenarios,
    out: npt.NDArray[np.float32],
    batch_size: int,
) -> SensitivityPnlEngine:
    """
    Bind the book to a caller-supplied `(M, S)` block.

    Built fresh for every scenario set even when the book is cached. The engine
    freezes `out` at construction, so reusing one would mean every result handed
    back aliased the same buffer and silently changed under the caller on the
    next tick. Measured at 0.4 ms against a 2,200 ms kernel: 0.02%, for an
    aliasing class of bug that is close to undebuggable in a live risk loop.
    """
    return SensitivityPnlEngine(
        book=book,
        out=out,
        batch_size=batch_size,
        base_levels=(
            market.base_levels
            if isinstance(market, ScenarioLevels)
            else np.empty(0, dtype=np.float64)
        ),
    )


def stream_scenarios(
    *,
    market: MarketScenarios,
    engine: SensitivityPnlEngine,
    num_scenarios: int,
) -> None:
    """
    Walk the scenario axis one batch at a time.

    The full market matrix is handed to the engine every batch and sliced by
    offset inside it — a column slice would not be C-contiguous, so passing one
    would mean copying each batch out.

    The two market contracts drive the identical Taylor kernel and differ only in
    how the displacement scratch is filled, so the split is a two-arm dispatch
    here rather than an abstraction either side has to carry.
    """
    batch_size = engine.batch_size
    for offset in range(0, num_scenarios, batch_size):
        length = min(batch_size, num_scenarios - offset)
        if isinstance(market, ScenarioShocks):
            engine.absorb_displacement(
                displacement=market.displacement,
                batch_len=length,
                scenario_offset=offset,
                source_offset=offset,
            )
        else:
            engine.absorb_levels(
                shocked_levels=market.shocked_levels,
                batch_len=length,
                scenario_offset=offset,
                levels_offset=offset,
            )


def build_provenance(
    *,
    market: MarketScenarios,
    axis: FactorAxis,
    book: CompiledSensitivityBook,
    batch_size: int,
    book_reused: bool,
) -> SensitivityPnlProvenance:
    diagonal_rows = np.diff(book.position_offsets)
    cross_rows = np.diff(book.cross_offsets)
    silent = int(np.count_nonzero((diagonal_rows == 0) & (cross_rows == 0)))
    return SensitivityPnlProvenance(
        engine_version=ENGINE_VERSION,
        market_contract=(
            "displacement" if isinstance(market, ScenarioShocks) else "levels"
        ),
        book_reused=book_reused,
        factor_key_encoding="direct" if axis.slot_map is None else "slot",
        num_positions=book.num_positions,
        num_scenarios=market.num_scenarios,
        num_market_factors=axis.num_factors,
        num_active_factors=book.num_active_factors,
        num_first_order_entries=book.nnz,
        num_cross_entries=book.nnz_cross,
        num_positions_without_sensitivities=silent,
        batch_size=batch_size,
    )
