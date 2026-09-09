# src/riskquant/sensitivity_pnl/validity.py
"""
Taylor-validity diagnostics: quantify, without a full-reval reference, how
much of a book's modelled PnL is riding on the part of the Taylor expansion
that is least trustworthy.

    engine = TaylorValidityEngine(book=..., linear_out=..., quadratic_out=...)
    for batch in ...:
        engine.absorb_levels(...) / engine.absorb_displacement(...)
    grid = TaylorValidityGrid(position_uids=..., linear_term=linear_out,
                              quadratic_term=quadratic_out)
    fraction = grid.nonlinearity_fraction()          # (M, S) in [0, 1]
    summary = summarize_taylor_validity(grid, threshold=0.20)

This module has exactly one question in scope: is the second-order Taylor
expansion `kernel_taylor_pnl` evaluates behaving like a converging series, or
has it started to extrapolate? See `kernels/taylor_validity.py` for the
precise, honest statement of what the resulting metric can and cannot tell
you — it is a necessary-condition heuristic built from the coefficients
already in the book, not an error bound, and it requires no full revaluation,
no third-order coefficient, and no change to the compiled book at all.

Two complementary diagnostics
------------------------------
`TaylorValidityEngine` / `TaylorValidityGrid`   SCENARIO-DEPENDENT. Needs a
    market (levels or displacement) and answers "in the scenarios I actually
    ran, how much did curvature matter?" Symmetric with `SensitivityPnlEngine`:
    same book, same two entry points, same bounds contract.

`characteristic_displacement`                    SCENARIO-FREE. Needs only the
    compiled book and answers "at what displacement WOULD curvature start to
    matter for this entry?" before a single scenario is drawn — an early-
    warning scan over the book's own coefficients, cheap enough to run on
    every compile.

Neither one performs, or requires, a full revaluation.
"""

from __future__ import annotations

__all__ = [
    "TaylorValidityEngine",
    "TaylorValidityGrid",
    "TaylorValiditySummary",
    "characteristic_displacement",
    "compute_taylor_validity_from_displacement",
    "compute_taylor_validity_from_levels",
    "summarize_taylor_validity",
]

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from riskquant.sensitivity_pnl._batch_contract import (
    check_2d_batch_source,
    derive_min_source_rows,
    derive_positive_level_rows,
    require_positive_levels_for_log_factors,
    validate_base_levels_for_ratio_factors,
)
from riskquant.sensitivity_pnl.contracts import CompiledSensitivityBook
from riskquant.sensitivity_pnl.engine import (
    accumulator_row_width,
    numba_thread_ceiling,
)
from riskquant.sensitivity_pnl.kernels.displacement import kernel_build_displacement
from riskquant.sensitivity_pnl.kernels.gather import kernel_gather_displacement
from riskquant.sensitivity_pnl.kernels.taylor_validity import (
    kernel_taylor_validity_terms,
)
from riskquant.sensitivity_pnl.scheduling import (
    DEFAULT_CHUNKS_PER_THREAD,
    balanced_chunk_offsets,
    default_num_chunks,
)

# =========================================================================
# GRID
# =========================================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class TaylorValidityGrid:
    """
    A `(M, S)` split of Taylor PnL into its linear and quadratic components.

    `linear_term + quadratic_term` recovers exactly what `kernel_taylor_pnl`
    would have reported (to float32 precision — the two kernels round
    intermediate sums slightly differently; the golden master pins this
    against the real PnL engine as a cross-check, not an assumption).
    """

    position_uids: npt.NDArray[np.int64]
    linear_term: npt.NDArray[np.float32]
    quadratic_term: npt.NDArray[np.float32]

    def __post_init__(self) -> None:
        if self.position_uids.dtype != np.int64 or self.position_uids.ndim != 1:
            raise ValueError("position_uids must be 1-D int64")
        for name, array in (
            ("linear_term", self.linear_term),
            ("quadratic_term", self.quadratic_term),
        ):
            if array.dtype != np.float32 or array.ndim != 2:
                raise ValueError(f"{name} must be a 2-D float32 array")
            if not array.flags.c_contiguous:
                raise ValueError(f"{name} must be C-contiguous")
            if array.shape[0] != self.position_uids.shape[0]:
                raise ValueError(
                    f"{name} has {array.shape[0]} rows; position_uids has "
                    f"{self.position_uids.shape[0]}"
                )
        if self.linear_term.shape != self.quadratic_term.shape:
            raise ValueError(
                f"linear_term shape {self.linear_term.shape} != quadratic_term "
                f"shape {self.quadratic_term.shape}"
            )

    @property
    def num_positions(self) -> int:
        return int(self.position_uids.shape[0])

    @property
    def num_scenarios(self) -> int:
        return int(self.linear_term.shape[1])

    def total_pnl(self) -> npt.NDArray[np.float64]:
        """`linear + quadratic`, i.e. what the Taylor PnL kernel reports."""
        return self.linear_term.astype(np.float64) + self.quadratic_term.astype(
            np.float64
        )

    def nonlinearity_fraction(self) -> npt.NDArray[np.float64]:
        """
        `|quadratic| / (|linear| + |quadratic|)`, per cell, always in `[0, 1]`.

        Bounded and 0/0-safe by construction: the denominator is a SUM of
        absolute values, never a difference, so it can only be smaller than
        the numerator by rounding, never zero unless both terms are exactly
        zero (a position with no exposure to anything that moved). Two
        alternative denominators were rejected for being numerically unstable
        in exactly the regime this diagnostic needs to be reliable in:

            |quadratic| / |linear|          blows up whenever linear ~ 0, even
                                             for a harmless quadratic term.
            |quadratic| / |linear+quadratic| blows up whenever the two terms
                                             happen to nearly cancel, which is
                                             not the same thing as "safe".

        0 means the Taylor expansion is exact-linear for this cell (a pure
        delta position, or a position at a scenario with a factor it has zero
        gamma/cross exposure to). 1 means the entire reported PnL is coming
        from curvature. Neither extreme requires the position to be nonlinear
        in any absolute sense — this is a fraction of what THIS book reported,
        not a fraction of some external truth.
        """
        linear = np.abs(self.linear_term.astype(np.float64))
        quadratic = np.abs(self.quadratic_term.astype(np.float64))
        denominator = linear + quadratic
        fraction: npt.NDArray[np.float64] = np.divide(
            quadratic,
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0.0,
        )
        return fraction


# =========================================================================
# ENGINE
# =========================================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class TaylorValidityEngine:
    """
    Streaming linear/quadratic split, symmetric with `SensitivityPnlEngine`.

    Owns no output — writes into `linear_out` / `quadratic_out`, two `(M, S)`
    float32 C-contiguous views injected by the caller, exactly like
    `SensitivityPnlEngine.out`. Scratch is bounded by `batch_size`: a `(Ka, B)`
    f32 displacement buffer (shared shape with the PnL engine's) and TWO
    `(n_threads, Bp)` f64 accumulators — one per output stream, rather than a
    single array with an extra axis, to keep indexing unambiguous.

    Shares `SensitivityPnlEngine`'s bounds gate via `_batch_contract` rather
    than reimplementing it, so a malformed batch is rejected identically by
    both engines — see `_batch_contract`'s module docstring.
    """

    book: CompiledSensitivityBook
    linear_out: npt.NDArray[np.float32]
    quadratic_out: npt.NDArray[np.float32]
    batch_size: int = 500
    chunks_per_thread: int = DEFAULT_CHUNKS_PER_THREAD
    chunk_offsets: npt.NDArray[np.int32] | None = None
    base_levels: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )

    _chunk_offsets: npt.NDArray[np.int32] = field(init=False, repr=False)
    _displacement: npt.NDArray[np.float32] = field(init=False, repr=False)
    _linear_accumulator: npt.NDArray[np.float64] = field(init=False, repr=False)
    _quadratic_accumulator: npt.NDArray[np.float64] = field(init=False, repr=False)
    _min_source_rows: int = field(init=False, repr=False)
    _positive_level_uix: npt.NDArray[np.int32] = field(init=False, repr=False)
    _positive_level_factor_ids: npt.NDArray[np.int32] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        for name, out in (
            ("linear_out", self.linear_out),
            ("quadratic_out", self.quadratic_out),
        ):
            if out.dtype != np.float32 or out.ndim != 2:
                raise ValueError(f"{name} must be a 2-D float32 view")
            if not out.flags.c_contiguous:
                raise ValueError(
                    f"{name} must be C-contiguous. Row slices of a C-contiguous "
                    "grid are C-contiguous; column slices are not."
                )
            if out.shape[0] != self.book.num_positions:
                raise ValueError(
                    f"{name} has {out.shape[0]} rows; book has "
                    f"{self.book.num_positions} positions"
                )
        if self.linear_out.shape != self.quadratic_out.shape:
            raise ValueError(
                f"linear_out shape {self.linear_out.shape} != quadratic_out "
                f"shape {self.quadratic_out.shape}"
            )

        object.__setattr__(self, "_min_source_rows", derive_min_source_rows(self.book))
        positive_level_uix, positive_level_factor_ids = derive_positive_level_rows(
            self.book
        )
        object.__setattr__(self, "_positive_level_uix", positive_level_uix)
        object.__setattr__(
            self, "_positive_level_factor_ids", positive_level_factor_ids
        )

        if self.base_levels.size:
            validate_base_levels_for_ratio_factors(
                book=self.book,
                base_levels=self.base_levels,
                min_source_rows=self._min_source_rows,
            )

        object.__setattr__(self, "_chunk_offsets", self._resolve_chunk_offsets())

        object.__setattr__(
            self,
            "_displacement",
            np.empty(
                (self.book.num_active_factors, self.batch_size),
                dtype=np.float32,
                order="C",
            ),
        )
        accumulator_shape = (
            numba_thread_ceiling(),
            accumulator_row_width(self.batch_size),
        )
        object.__setattr__(
            self,
            "_linear_accumulator",
            np.empty(accumulator_shape, dtype=np.float64, order="C"),
        )
        object.__setattr__(
            self,
            "_quadratic_accumulator",
            np.empty(accumulator_shape, dtype=np.float64, order="C"),
        )

    @property
    def num_chunks(self) -> int:
        return int(self._chunk_offsets.shape[0]) - 1

    def _resolve_chunk_offsets(self) -> npt.NDArray[np.int32]:
        """See `SensitivityPnlEngine._resolve_chunk_offsets`: same reasoning,
        same function, same host-shaped partition. A book compiled once and
        diagnosed on many machines should not carry a schedule pinned to
        whichever machine happened to compile it."""
        if self.chunk_offsets is None:
            return balanced_chunk_offsets(
                book=self.book,
                num_chunks=default_num_chunks(
                    num_positions=self.book.num_positions,
                    thread_ceiling=numba_thread_ceiling(),
                    chunks_per_thread=self.chunks_per_thread,
                ),
            )
        offsets = self.chunk_offsets
        if offsets.dtype != np.int32 or offsets.ndim != 1 or offsets.shape[0] < 1:
            raise ValueError("chunk_offsets must be a non-empty 1-D int32 array")
        if not offsets.flags.c_contiguous:
            raise ValueError("chunk_offsets must be C-contiguous")
        if int(offsets[0]) != 0:
            raise ValueError("chunk_offsets must start at 0")
        if offsets.shape[0] > 1 and not bool(np.all(np.diff(offsets) > 0)):
            raise ValueError(
                "chunk_offsets must be strictly increasing — an empty chunk "
                "consumes a prange slot for no work"
            )
        if int(offsets[-1]) != self.book.num_positions:
            raise ValueError(
                f"chunk_offsets terminate at {int(offsets[-1])}, expected "
                f"M={self.book.num_positions}"
            )
        return offsets

    # ------------------------------------------------------------------

    def absorb_levels(
        self,
        *,
        shocked_levels: npt.NDArray[np.float64],
        batch_len: int,
        scenario_offset: int,
        levels_offset: int = 0,
    ) -> None:
        """Integrated path: mirrors `SensitivityPnlEngine.absorb_levels` exactly."""
        if self.base_levels.size == 0:
            raise ValueError(
                "absorb_levels requires base_levels; construct the engine with "
                "it, or use absorb_displacement for the standalone path."
            )
        check_2d_batch_source(
            source=shocked_levels,
            dtype=np.float64,
            name="shocked_levels",
            min_source_rows=self._min_source_rows,
            batch_len=batch_len,
            batch_size=self.batch_size,
            source_offset=levels_offset,
            scenario_offset=scenario_offset,
            out_num_scenarios=self.linear_out.shape[1],
        )
        if shocked_levels.shape[0] != self.base_levels.shape[0]:
            raise ValueError(
                f"shocked_levels has {shocked_levels.shape[0]} factor rows; "
                f"base_levels has {self.base_levels.shape[0]}"
            )
        require_positive_levels_for_log_factors(
            positive_level_uix=self._positive_level_uix,
            positive_level_factor_ids=self._positive_level_factor_ids,
            shocked_levels=shocked_levels,
            batch_len=batch_len,
            levels_offset=levels_offset,
        )
        kernel_build_displacement(
            self.base_levels,
            shocked_levels,
            self.book.active_level_uix,
            self.book.displacement_convention_ids,
            batch_len,
            levels_offset,
            self._displacement,
        )
        self._run_kernel(batch_len=batch_len, scenario_offset=scenario_offset)

    def absorb_displacement(
        self,
        *,
        displacement: npt.NDArray[np.float32],
        batch_len: int,
        scenario_offset: int,
        source_offset: int = 0,
    ) -> None:
        """Standalone path: mirrors `SensitivityPnlEngine.absorb_displacement`."""
        check_2d_batch_source(
            source=displacement,
            dtype=np.float32,
            name="displacement",
            min_source_rows=self._min_source_rows,
            batch_len=batch_len,
            batch_size=self.batch_size,
            source_offset=source_offset,
            scenario_offset=scenario_offset,
            out_num_scenarios=self.linear_out.shape[1],
        )
        kernel_gather_displacement(
            displacement,
            self.book.active_level_uix,
            batch_len,
            source_offset,
            self._displacement,
        )
        self._run_kernel(batch_len=batch_len, scenario_offset=scenario_offset)

    def _run_kernel(self, *, batch_len: int, scenario_offset: int) -> None:
        kernel_taylor_validity_terms(
            self._chunk_offsets,
            self.book.position_offsets,
            self.book.factor_uix,
            self.book.first_order,
            self.book.second_order,
            self.book.cross_offsets,
            self.book.cross_factor_uix_i,
            self.book.cross_factor_uix_j,
            self.book.cross_second_order,
            self._displacement,
            self._linear_accumulator,
            self._quadratic_accumulator,
            batch_len,
            scenario_offset,
            self.linear_out,
            self.quadratic_out,
        )


# =========================================================================
# ONE-SHOT ENTRY POINTS
# =========================================================================


def _allocate_pair(
    *, num_positions: int, num_scenarios: int
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    return (
        np.empty((num_positions, num_scenarios), dtype=np.float32, order="C"),
        np.empty((num_positions, num_scenarios), dtype=np.float32, order="C"),
    )


def compute_taylor_validity_from_levels(
    *,
    book: CompiledSensitivityBook,
    base_levels: npt.NDArray[np.float64],
    shocked_levels: npt.NDArray[np.float64],
    batch_size: int = 500,
) -> TaylorValidityGrid:
    """Integrated one-shot: `(N_req, S)` f64 shocked levels -> `TaylorValidityGrid`."""
    num_scenarios = shocked_levels.shape[1]
    linear_out, quadratic_out = _allocate_pair(
        num_positions=book.num_positions, num_scenarios=num_scenarios
    )
    engine = TaylorValidityEngine(
        book=book,
        linear_out=linear_out,
        quadratic_out=quadratic_out,
        base_levels=base_levels,
        batch_size=min(batch_size, num_scenarios),
    )
    for offset, length in _batches(num_scenarios, engine.batch_size):
        engine.absorb_levels(
            shocked_levels=shocked_levels,
            batch_len=length,
            scenario_offset=offset,
            levels_offset=offset,
        )
    return TaylorValidityGrid(
        position_uids=book.position_uids,
        linear_term=linear_out,
        quadratic_term=quadratic_out,
    )


def compute_taylor_validity_from_displacement(
    *,
    book: CompiledSensitivityBook,
    displacement: npt.NDArray[np.float32],
    batch_size: int = 500,
) -> TaylorValidityGrid:
    """Standalone one-shot: `(K, S)` f32 displacements -> `TaylorValidityGrid`."""
    num_scenarios = displacement.shape[1]
    linear_out, quadratic_out = _allocate_pair(
        num_positions=book.num_positions, num_scenarios=num_scenarios
    )
    engine = TaylorValidityEngine(
        book=book,
        linear_out=linear_out,
        quadratic_out=quadratic_out,
        batch_size=min(batch_size, num_scenarios),
    )
    for offset, length in _batches(num_scenarios, engine.batch_size):
        engine.absorb_displacement(
            displacement=displacement,
            batch_len=length,
            scenario_offset=offset,
            source_offset=offset,
        )
    return TaylorValidityGrid(
        position_uids=book.position_uids,
        linear_term=linear_out,
        quadratic_term=quadratic_out,
    )


def _batches(total: int, batch_size: int) -> list[tuple[int, int]]:
    return [
        (offset, min(batch_size, total - offset))
        for offset in range(0, total, batch_size)
    ]


# =========================================================================
# STATIC, SCENARIO-FREE DIAGNOSTIC
# =========================================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class CharacteristicDisplacement:
    """
    Per diagonal CSR entry: the displacement `x*` at which that entry's
    quadratic term equals its linear term, i.e. `0.5*d2*x*^2 = d1*x*` solved
    for `x* = 2*d1/d2`. Below `|x*|`, the linear term dominates; at `|x*|`,
    the two are equal; a realistic scenario set whose typical moves are a
    large fraction of `|x*|` is a book that will register a large
    `nonlinearity_fraction` before you have simulated a single scenario.

    `x_star` is `+inf` (not an error) wherever `second_order == 0`: a pure-delta
    entry has no curvature at any displacement, so it has no breakdown scale.
    Cross entries are OUT OF SCOPE: a cross term's size depends on the product
    of TWO displacements, so there is no single scalar threshold for one entry
    the way there is for a diagonal one. Cross-term strain still shows up in
    the scenario-dependent `nonlinearity_fraction` above; it is simply not
    part of this scenario-free scan.

    `active_factor_slot` carries the raw compiled active-factor-axis index
    (`book.factor_uix`) alongside `factor_ids` (the compiled factor_id VALUE)
    so a caller-facing layer can translate an entry back to the caller's own
    factor key via `active_level_uix[active_factor_slot]` — the row-index
    form that is valid under BOTH the direct and slot key encodings, unlike
    `factor_ids`, whose VALUE means something different under each. This
    module has no notion of a caller's key space; that translation belongs
    one tier up.
    """

    position_uids: npt.NDArray[np.int64]  # (NNZ,) — the position each entry belongs to
    factor_ids: npt.NDArray[np.int32]  # (NNZ,) — the compiled factor_id
    active_factor_slot: npt.NDArray[
        np.int32
    ]  # (NNZ,) — index into the active-factor axis
    x_star: npt.NDArray[np.float64]  # (NNZ,) — breakdown displacement, +inf if none


def characteristic_displacement(
    book: CompiledSensitivityBook,
) -> CharacteristicDisplacement:
    """
    Scenario-free early-warning scan over the compiled book's own coefficients.

    O(NNZ), pure NumPy — cheap enough to run on every compile, unlike the
    scenario-dependent grid above which needs a market. Complements it rather
    than replacing it: this catches a dangerously gamma-heavy entry before a
    single scenario is drawn; the grid catches actual realised strain.
    """
    row_of_entry = np.repeat(
        np.arange(book.num_positions, dtype=np.int64), np.diff(book.position_offsets)
    )
    position_uids = book.position_uids[row_of_entry]
    factor_ids = book.active_factor_ids[book.factor_uix]
    with np.errstate(divide="ignore", invalid="ignore"):
        x_star = np.where(
            book.second_order != 0.0,
            np.abs(2.0 * book.first_order / book.second_order),
            np.inf,
        )
    return CharacteristicDisplacement(
        position_uids=np.ascontiguousarray(position_uids, dtype=np.int64),
        factor_ids=np.ascontiguousarray(factor_ids, dtype=np.int32),
        active_factor_slot=np.ascontiguousarray(book.factor_uix, dtype=np.int32),
        x_star=np.ascontiguousarray(x_star, dtype=np.float64),
    )


# =========================================================================
# SUMMARY / REPORTING
# =========================================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class TaylorValiditySummary:
    """
    Per-position rollup of a `TaylorValidityGrid`, for a desk-facing "which
    positions are riding on curvature" report rather than a raw (M, S) grid.
    """

    position_uids: npt.NDArray[np.int64]
    worst_nonlinearity_fraction: npt.NDArray[np.float64]  # (M,) max over scenarios
    worst_scenario_index: npt.NDArray[np.int64]  # (M,) which scenario, for traceability
    exceedance_fraction: npt.NDArray[
        np.float64
    ]  # (M,) share of scenarios over threshold
    threshold: float

    @property
    def num_positions_exceeding(self) -> int:
        return int(np.count_nonzero(self.exceedance_fraction > 0.0))


def summarize_taylor_validity(
    grid: TaylorValidityGrid, *, threshold: float = 0.20
) -> TaylorValiditySummary:
    """
    Reduce a `(M, S)` grid to one row per position: worst-case strain, WHERE
    it occurred (a scenario index, so a reader can go look), and what fraction
    of scenarios breached `threshold`.

    `threshold` has no universally correct value — see the module docstring on
    what the metric can and cannot prove — so it is a caller-supplied
    parameter, not a constant baked into the library.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must lie in (0, 1]")
    fraction = grid.nonlinearity_fraction()
    return TaylorValiditySummary(
        position_uids=grid.position_uids,
        worst_nonlinearity_fraction=fraction.max(axis=1),
        worst_scenario_index=fraction.argmax(axis=1).astype(np.int64),
        exceedance_fraction=(fraction > threshold).mean(axis=1),
        threshold=threshold,
    )
