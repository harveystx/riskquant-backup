# src/riskquant/solutions/sensitivity_pnl/market.py
"""
Scenario inputs, with the factor axis welded to the data.

The failure this module exists to prevent
-----------------------------------------
`SensitivityPnlEngine` consumes a `(K, S)` matrix whose ROW ORDER must match the
factor universe the book was compiled against. It checks that the matrix is tall
enough and nothing else — it cannot check the ordering, because a permutation of
float rows is indistinguishable from the correct one. Measured on the raw
engine: a three-factor book given its shock rows in reverse order returned 123
instead of 321, with no warning, no error, and no way for a caller to tell.

So the public API never accepts a bare matrix plus a promise about its rows. A
matrix and the factor keys naming those rows are constructed together, and the
compiler is handed the universe FROM that object. Misalignment stops being a
thing a caller can express.

The universe is the market, always
----------------------------------
Row `i` of the matrix defines factor key `keys[i]`, and the sensitivity book is
resolved INTO that space. A sensitivity referencing a factor with no shock row
therefore raises rather than silently contributing zero risk in every scenario —
the failure mode `compile_sensitivity_book` calls the most dangerous in the
module, closed at the perimeter instead of relied upon downstream.

Row order is free; the copy that used to enforce it is not
----------------------------------------------------------
An earlier version required integer factor keys to arrive sorted ascending and
offered `sort_factors=True` to permute the `(K, S)` matrix into place — 32 MB at
K=2000/S=4000, 2 GB at K=50k/S=10k, and paid again for every scenario set once a
book is reused across ticks.

That copy was never necessary. `kernel_gather_displacement` already reads row
`active_level_uix[k]` of the caller's matrix, and `kernel_build_displacement`
does the same on the levels path, so both kernels perform an arbitrary indexed
row gather at zero marginal cost. The permutation belongs in the INDEX, not the
DATA: keys in any order are relabelled to their own row position at compile
time, and the matrix is never touched. Measured, that relabel costs 47 ms once
on a 792k-entry book, against a 9.6 ms copy repeated per scenario set.

Two key encodings remain, and they differ only in diagnostics:

  DIRECT  Integer keys already sorted ascending and unique are handed to the
          compiler as the universe itself. Compiler-side messages then name real
          factor ids, which is worth keeping for the common case.
  SLOT    Anything else — unsorted integers, or string labels — is relabelled to
          its row position here, at the perimeter, where the caller's keys are
          still available to name in an error.

Both are zero-copy. The branch is an optimisation of error text, not a fork in
correctness, and `validate.py` reports every factor defect in the caller's own
keys under either.

Two market contracts, matching the engine's two absorb paths:

    ScenarioShocks   `(K, S)` f32 displacements already in each factor's declared
                     convention. Nothing derives them; no base levels needed.
    ScenarioLevels   `(K,)` f64 base + `(K, S)` f64 shocked levels, exactly the
                     buffer the full-reval runner fills, so both PnL blocks are
                     driven by one market state.

Position identifiers get the opposite treatment to factor keys: they must
already be integers. They travel out in `PnlMatrix.trade_uids` and are joined
downstream, so this layer has no licence to forge them.
"""

from __future__ import annotations

__all__ = [
    "FactorAxis",
    "MarketScenarios",
    "ScenarioLevels",
    "ScenarioShocks",
]

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt
import polars as pl

from riskquant.solutions.sensitivity_pnl.errors import MAX_REPORTED_ROWS, DefectLog

FactorKeys: TypeAlias = "Sequence[str] | Sequence[int] | npt.NDArray[Any] | pl.Series"


@dataclass(frozen=True, slots=True, kw_only=True)
class FactorAxis:
    """
    The `K` factor keys naming the rows of a scenario matrix, in row order.

    Build one with `from_keys`, then use it as the factory for every scenario set
    priced against it. Reusing the axis is what makes a cached book safe: a
    market built through `axis.shocks(...)` cannot disagree with the book
    compiled against that axis, because there is no second place for the row
    order to come from.
    """

    keys: pl.Series
    universe_ids: npt.NDArray[np.int32]
    """The `level_universe_factor_ids` the compiler is given: the keys themselves
    under DIRECT encoding, `arange(K)` under SLOT."""

    slot_map: Mapping[Any, int] | None
    """`{key: row}` under SLOT encoding; `None` when the keys index themselves."""

    @classmethod
    def from_keys(cls, keys: FactorKeys) -> FactorAxis:
        """
        Normalise caller keys into an axis. Never reorders, never copies a matrix.

        Raises
        ------
        SensitivityPnlInputError
            On an empty, duplicated, null, or float-dtyped key set. A float key
            would truncate and could silently merge two factors into one row.
        """
        series = keys if isinstance(keys, pl.Series) else pl.Series("factor_id", keys)
        _validate_keys(series)

        if series.dtype.is_integer() and _is_sorted_unique(series):
            return cls(
                keys=series,
                universe_ids=np.ascontiguousarray(series.to_numpy(), dtype=np.int32),
                slot_map=None,
            )
        return cls(
            keys=series,
            universe_ids=np.arange(series.len(), dtype=np.int32),
            slot_map={key: row for row, key in enumerate(series.to_list())},
        )

    @property
    def num_factors(self) -> int:
        return int(self.keys.len())

    def matches(self, other: FactorAxis) -> bool:
        """
        True when `other` names the same factors in the same row order.

        Compared by value, not identity: a caller who rebuilds the axis from the
        same key column each tick is doing nothing wrong.
        """
        return bool(self.keys.equals(other.keys))

    def relabel(self, frame: pl.DataFrame, columns: tuple[str, ...]) -> pl.DataFrame:
        """
        Rewrite factor-key columns to row positions, or return `frame` untouched.

        Polars is columnar, so the result shares every buffer with the input
        except the rewritten key column — an `O(nnz)` int32 column, not a copy of
        the frame.
        """
        if self.slot_map is None:
            return frame
        present = [column for column in columns if column in frame.columns]
        if not present:
            return frame
        return frame.with_columns(
            pl.col(column)
            .replace_strict(self.slot_map, return_dtype=pl.Int32)
            .alias(column)
            for column in present
        )

    def shocks(self, displacement: npt.NDArray[Any]) -> ScenarioShocks:
        """Bind a `(K, S)` displacement matrix to this axis."""
        return ScenarioShocks(
            axis=self,
            displacement=np.ascontiguousarray(displacement, dtype=np.float32),
        )

    def levels(
        self,
        *,
        base_levels: npt.NDArray[Any],
        shocked_levels: npt.NDArray[Any],
    ) -> ScenarioLevels:
        """Bind a `(K,)` base vector and a `(K, S)` shocked-level matrix to this axis."""
        return ScenarioLevels(
            axis=self,
            base_levels=np.ascontiguousarray(base_levels, dtype=np.float64),
            shocked_levels=np.ascontiguousarray(shocked_levels, dtype=np.float64),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ScenarioShocks:
    """`(K, S)` float32 displacements, already in each factor's declared convention."""

    axis: FactorAxis
    displacement: npt.NDArray[np.float32]

    def __post_init__(self) -> None:
        _validate_matrix(
            matrix=self.displacement,
            dtype=np.float32,
            num_factors=self.axis.num_factors,
            source="market.displacement",
        )

    @property
    def num_scenarios(self) -> int:
        return int(self.displacement.shape[1])

    @classmethod
    def from_matrix(
        cls, *, factor_keys: FactorKeys, displacement: npt.NDArray[Any]
    ) -> ScenarioShocks:
        """
        One-shot construction: build the axis and bind the matrix in one step.

        A `(K, S)` array that is already C-contiguous float32 is adopted by
        reference; anything else is converted once, here, at the edge. For
        repeated pricing against one book, build the axis once with
        `FactorAxis.from_keys` and call `axis.shocks(...)` per scenario set.
        """
        return FactorAxis.from_keys(factor_keys).shocks(displacement)

    @classmethod
    def from_frame(
        cls, frame: pl.DataFrame, *, factor_key_column: str = "factor_id"
    ) -> ScenarioShocks:
        """
        Ergonomic path: one row per factor, one column per scenario.

        Scenario columns are every column except `factor_key_column`, in frame
        order — that order becomes the output matrix's column order. Costs one
        `(K, S)` materialisation, because a columnar frame cannot be viewed as a
        row-major matrix; use `from_matrix` when the caller already holds one.
        """
        keys, values = _split_scenario_frame(frame, factor_key_column, "shocks")
        return cls.from_matrix(factor_keys=keys, displacement=values)


@dataclass(frozen=True, slots=True, kw_only=True)
class ScenarioLevels:
    """`(K,)` base levels and `(K, S)` shocked levels, both float64."""

    axis: FactorAxis
    base_levels: npt.NDArray[np.float64]
    shocked_levels: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        num_factors = self.axis.num_factors
        _validate_matrix(
            matrix=self.shocked_levels,
            dtype=np.float64,
            num_factors=num_factors,
            source="market.shocked_levels",
        )
        log = DefectLog()
        if self.base_levels.ndim != 1 or self.base_levels.shape[0] != num_factors:
            log.add(
                source="market.base_levels",
                problem=(
                    f"shape {self.base_levels.shape} against a "
                    f"{num_factors}-factor axis."
                ),
                remedy=(
                    "supply exactly one base level per factor key, in the same order."
                ),
            )
            log.raise_if_any()
        _require_finite(array=self.base_levels, source="market.base_levels", log=log)
        log.raise_if_any()

    @property
    def num_scenarios(self) -> int:
        return int(self.shocked_levels.shape[1])

    @classmethod
    def from_matrix(
        cls,
        *,
        factor_keys: FactorKeys,
        base_levels: npt.NDArray[Any],
        shocked_levels: npt.NDArray[Any],
    ) -> ScenarioLevels:
        return FactorAxis.from_keys(factor_keys).levels(
            base_levels=base_levels, shocked_levels=shocked_levels
        )

    @classmethod
    def from_frame(
        cls,
        frame: pl.DataFrame,
        *,
        factor_key_column: str = "factor_id",
        base_level_column: str = "base_level",
    ) -> ScenarioLevels:
        """One row per factor: a key column, a base-level column, then the scenarios."""
        log = DefectLog()
        if base_level_column not in frame.columns:
            log.add(
                source="market.levels_frame",
                problem=f"no column named '{base_level_column}'.",
                remedy=(
                    "the base level is the t0 market state each shocked level is "
                    "measured against; name the column or pass base_level_column."
                ),
            )
            log.raise_if_any()
        base = np.ascontiguousarray(
            frame.get_column(base_level_column).to_numpy(), dtype=np.float64
        )
        keys, values = _split_scenario_frame(
            frame.drop(base_level_column), factor_key_column, "levels"
        )
        return cls.from_matrix(
            factor_keys=keys, base_levels=base, shocked_levels=values
        )


MarketScenarios: TypeAlias = ScenarioShocks | ScenarioLevels


# =========================================================================
# HELPERS
# =========================================================================


def _is_sorted_unique(series: pl.Series) -> bool:
    if series.len() < 2:
        return True
    return bool(np.all(np.diff(series.to_numpy()) > 0))


def _validate_keys(series: pl.Series) -> None:
    log = DefectLog()
    if series.len() == 0:
        log.add(
            source="market.factor_keys",
            problem="the factor axis is empty.",
            remedy="supply one key per row of the scenario matrix.",
        )
        log.raise_if_any()

    nulls = series.is_null()
    if bool(nulls.any()):
        log.add(
            source="market.factor_keys",
            problem=f"{int(nulls.sum())} null factor key(s).",
            remedy=(
                "a scenario row with no factor key cannot be matched to a sensitivity."
            ),
            row_indices=tuple(int(i) for i in nulls.arg_true().head(MAX_REPORTED_ROWS)),
            total_rows=int(nulls.sum()),
        )
    duplicated = series.is_duplicated()
    if bool(duplicated.any()):
        log.add(
            source="market.factor_keys",
            problem=f"{int(duplicated.sum())} duplicate factor key(s).",
            remedy=(
                "each row of the scenario matrix must name a distinct factor; "
                "aggregate or drop the repeats upstream."
            ),
            row_indices=tuple(
                int(i) for i in duplicated.arg_true().head(MAX_REPORTED_ROWS)
            ),
            total_rows=int(duplicated.sum()),
        )
    if not (series.dtype.is_integer() or series.dtype == pl.String):
        log.add(
            source="market.factor_keys",
            problem=f"factor keys have dtype {series.dtype}.",
            remedy=(
                "use an integer dtype for numeric factor ids, or String for "
                "labels. A float key would be truncated and could silently merge "
                "two factors into one row."
            ),
        )
    log.raise_if_any()


def _split_scenario_frame(
    frame: pl.DataFrame, factor_key_column: str, kind: str
) -> tuple[pl.Series, npt.NDArray[Any]]:
    log = DefectLog()
    if factor_key_column not in frame.columns:
        log.add(
            source=f"market.{kind}_frame",
            problem=f"no column named '{factor_key_column}'.",
            remedy=(
                f"the frame needs one column of factor keys; got "
                f"{frame.columns[:8]}. Pass factor_key_column if it is named "
                "differently."
            ),
        )
        log.raise_if_any()
    scenario_columns = [c for c in frame.columns if c != factor_key_column]
    if not scenario_columns:
        log.add(
            source=f"market.{kind}_frame",
            problem="no scenario columns.",
            remedy=(
                "every column other than the factor key is one scenario; the "
                "frame must carry at least one."
            ),
        )
        log.raise_if_any()
    return frame.get_column(factor_key_column), frame.select(
        scenario_columns
    ).to_numpy()


def _validate_matrix(
    *,
    matrix: npt.NDArray[Any],
    dtype: type[np.floating[Any]],
    num_factors: int,
    source: str,
) -> None:
    log = DefectLog()
    if matrix.ndim != 2:
        log.add(
            source=source,
            problem=(
                f"{matrix.ndim}-D array; a scenario matrix is 2-D "
                "(factors x scenarios)."
            ),
            remedy="reshape to (K, S), one row per factor key.",
        )
        log.raise_if_any()
    if matrix.shape[0] != num_factors:
        log.add(
            source=source,
            problem=(
                f"{matrix.shape[0]} rows against a {num_factors}-factor axis. "
                "The engine cannot detect this once the matrix is tall enough — a "
                "mismatched axis returns a plausible, wrong number."
            ),
            remedy="supply exactly one matrix row per factor key, in the same order.",
        )
        log.raise_if_any()
    if matrix.shape[1] == 0:
        log.add(
            source=source,
            problem="zero scenarios.",
            remedy=(
                "a PnL matrix with no columns has nothing to aggregate; supply at "
                "least one scenario."
            ),
        )
        log.raise_if_any()
    if matrix.dtype != dtype or not matrix.flags.c_contiguous:
        log.add(
            source=source,
            problem=(
                f"dtype {matrix.dtype}, c_contiguous={matrix.flags.c_contiguous}; "
                f"expected {np.dtype(dtype).name} and C-contiguous."
            ),
            remedy=(
                "construct via FactorAxis.shocks / .levels or the from_matrix / "
                "from_frame constructors, which convert once at the edge."
            ),
        )
        log.raise_if_any()
    _require_finite(array=matrix, source=source, log=log)
    log.raise_if_any()


def _require_finite(*, array: npt.NDArray[Any], source: str, log: DefectLog) -> None:
    """
    Single-pass, zero-allocation finiteness test.

    `np.isfinite(a).all()` materialises a bool array — 500 MB against a 2 GB f32
    scenario matrix, on the happy path, for a check that almost always passes. A
    float64 accumulation is one streaming pass with no allocation.

    Why it cannot be fooled
    -----------------------
    Under IEEE-754 addition both NaN and infinity are ABSORBING: `NaN + x` is NaN
    for every `x`, `inf + x` is inf for every finite `x`, and `inf + -inf` is NaN
    rather than a cancellation to zero. Every partial sum touching a non-finite
    element is therefore itself non-finite, and since NumPy's pairwise summation
    only ever adds partials together, non-finiteness propagates to the root
    regardless of blocking. Swept over both dtypes, every arrangement of up to
    three poison values, and inf/-inf pairs placed at each pairwise-block
    boundary from n=7 to n=100,000: zero false negatives.

    `dtype=np.float64` is load-bearing, not decorative. Ten million elements at
    f32 max sum to 3.4e45, which overflows an f32 accumulator to inf and would
    REJECT a perfectly valid matrix; in float64 that sum is finite. Overflowing
    the f64 accumulator would need on the order of 1e270 elements, and would err
    toward rejection rather than acceptance in any case.

    `errstate` is not optional. A house that sets `np.seterr(all="raise")` — and
    they do — turns the `inf + -inf` case into a `FloatingPointError` raised from
    inside the validator, so the check written to produce a good error message
    would instead produce a worse one than no check at all. Verified, not assumed.

    Row localisation is deferred to the failure branch, where its cost is free.
    """
    if array.size == 0:
        return
    with np.errstate(all="ignore"):
        if bool(np.isfinite(array.sum(dtype=np.float64))):
            return
        per_row = array.reshape(array.shape[0], -1).sum(axis=1, dtype=np.float64)
    bad = np.flatnonzero(~np.isfinite(per_row))
    log.add(
        source=source,
        problem=f"non-finite value(s) on {bad.shape[0]} factor row(s).",
        remedy=(
            "a NaN or inf shock propagates into every scenario of every position "
            "sensitive to that factor. Repair or drop the factor upstream."
        ),
        row_indices=tuple(int(i) for i in bad[:MAX_REPORTED_ROWS]),
        total_rows=int(bad.shape[0]),
    )
