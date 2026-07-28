# src/riskquant/sensitivity_pnl/contracts.py
"""
Compiled, kernel-facing sensitivity book.

Layout
------
Sensitivity books are extremely sparse: a position touches O(1..50) factors out
of a universe of O(10k). A dense `(M, K)` greek matrix is not representable, so
the book is stored CSR-style — one flat entry array per SoA column, row-sliced
by `position_offsets`.

    position_offsets   (M+1,) int32  CSR row pointers into the entry arrays
    factor_uix         (NNZ,) int32  active-factor slot, ascending within a row
    first_order        (NNZ,) f64    dV/dx_j          (natural units)
    second_order       (NNZ,) f64    d2V/dx_j2        (natural units)

Cross terms carry their own CSR block over unordered factor pairs:

    cross_offsets      (M+1,) int32
    cross_factor_uix_i (NNZX,) int32  strictly less than ..._j
    cross_factor_uix_j (NNZX,) int32
    cross_second_order (NNZX,) f64    d2V/dx_i dx_j   (natural units)

`first_order` and `second_order` stay f64 even though the displacement matrix is
f32: they are read ONCE PER ENTRY, not once per entry-scenario, so their traffic
is `NNZ * 8` against the displacement's `NNZ * B * 4`. Narrowing them would save
nothing measurable and would throw away precision that `quote_base ** 2` folding
can push to 1e8.

Factor compaction
-----------------
`factor_uix` indexes the book's *active* factor set, not the caller's level
universe. Dense-by-universe-row arrays would force every displacement row to be
computed for factors no position is sensitive to. `active_level_uix` maps each
active slot back to its row in the caller's levels matrix.

Validation is a memory-safety boundary, not a courtesy
------------------------------------------------------
The kernels run under `boundscheck=False`. Every index they dereference is
range-checked HERE, or it is not checked at all and a malformed book is a silent
out-of-bounds read into adjacent heap. That makes `__post_init__` the last line
of defence, so it validates:

  - both CSR offset arrays start at 0, are non-decreasing, and terminate at NNZ
    (a descending pair yields `hi > NNZ` and walks off the entry arrays);
  - EVERY factor index in BOTH blocks is in `[0, Ka)` — the cross block included,
    which a `nnz == 0` short-circuit previously skipped entirely;
  - `active_level_uix` is non-negative (its upper bound depends on the caller's
    levels matrix and is therefore checked in the engine).

Unit contract (pinned)
----------------------
- All coefficients are in NATURAL units: the `quote_base` normalisation is
  folded in at compile time.
- Each unordered cross pair appears EXACTLY ONCE and contributes
  `c_ij * x_i * x_j` — the coefficient is the full mixed partial, NOT halved,
  and the kernel does not double it.
- Values are finite. Missing sensitivities are absent CSR entries, never NaN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from riskquant.sensitivity_pnl.conventions import DisplacementConvention

_VALID_CONVENTION_IDS: frozenset[int] = frozenset(
    int(c) for c in DisplacementConvention
)


@dataclass(frozen=True, slots=True, kw_only=True)
class CompiledSensitivityBook:
    """Frozen SoA sensitivity book bound to a caller-supplied level universe."""

    # --- Row identity (length M) ---
    position_uids: npt.NDArray[np.int64]

    # --- Active-factor axis (length Ka) ---
    active_factor_ids: npt.NDArray[np.int32]
    active_level_uix: npt.NDArray[np.int32]
    displacement_convention_ids: npt.NDArray[np.int8]

    # --- Diagonal CSR block ---
    position_offsets: npt.NDArray[np.int32]
    factor_uix: npt.NDArray[np.int32]
    first_order: npt.NDArray[np.float64]
    second_order: npt.NDArray[np.float64]

    # --- Cross CSR block ---
    cross_offsets: npt.NDArray[np.int32]
    cross_factor_uix_i: npt.NDArray[np.int32]
    cross_factor_uix_j: npt.NDArray[np.int32]
    cross_second_order: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        m = self.position_uids.shape[0]
        ka = self.active_factor_ids.shape[0]
        nnz = self.factor_uix.shape[0]
        nnzx = self.cross_factor_uix_i.shape[0]

        # ---- dtypes ----
        _require(self.position_uids.dtype == np.int64, "position_uids must be int64")
        _require(
            self.active_factor_ids.dtype == np.int32
            and self.active_level_uix.dtype == np.int32,
            "active-factor axis must be int32",
        )
        _require(
            self.displacement_convention_ids.dtype == np.int8,
            "displacement_convention_ids must be int8",
        )
        _require(
            self.position_offsets.dtype == np.int32
            and self.cross_offsets.dtype == np.int32,
            "CSR offsets must be int32",
        )
        _require(
            self.factor_uix.dtype == np.int32
            and self.cross_factor_uix_i.dtype == np.int32
            and self.cross_factor_uix_j.dtype == np.int32,
            "factor index arrays must be int32",
        )
        _require(
            self.first_order.dtype == np.float64
            and self.second_order.dtype == np.float64
            and self.cross_second_order.dtype == np.float64,
            "coefficient arrays must be float64",
        )

        # ---- shapes ----
        _require(
            self.active_level_uix.shape[0] == ka
            and self.displacement_convention_ids.shape[0] == ka,
            "active-factor axis arrays are misaligned",
        )
        _require(
            self.first_order.shape[0] == nnz and self.second_order.shape[0] == nnz,
            "diagonal entry arrays are misaligned",
        )
        _require(
            self.cross_factor_uix_j.shape[0] == nnzx
            and self.cross_second_order.shape[0] == nnzx,
            "cross entry arrays are misaligned",
        )

        # ---- contiguity ----
        for name in (
            "active_factor_ids",
            "active_level_uix",
            "displacement_convention_ids",
            "position_offsets",
            "factor_uix",
            "first_order",
            "second_order",
            "cross_offsets",
            "cross_factor_uix_i",
            "cross_factor_uix_j",
            "cross_second_order",
        ):
            arr: npt.NDArray[Any] = getattr(self, name)
            _require(arr.flags.c_contiguous, f"{name} must be C-contiguous")

        # ---- CSR structure: the memory-safety boundary ----
        _validate_csr(
            offsets=self.position_offsets, num_rows=m, nnz=nnz, block="diagonal"
        )
        _validate_csr(offsets=self.cross_offsets, num_rows=m, nnz=nnzx, block="cross")

        _require_index_range(self.factor_uix, ka, "factor_uix")
        _require_index_range(self.cross_factor_uix_i, ka, "cross_factor_uix_i")
        _require_index_range(self.cross_factor_uix_j, ka, "cross_factor_uix_j")
        _require(
            ka == 0 or int(self.active_level_uix.min()) >= 0,
            "active_level_uix must be non-negative",
        )

        # ---- semantics ----
        _require(
            _VALID_CONVENTION_IDS.issuperset(
                np.unique(self.displacement_convention_ids).tolist()
            ),
            "unknown displacement_convention_id",
        )
        _require(
            nnzx == 0
            or bool(np.all(self.cross_factor_uix_i < self.cross_factor_uix_j)),
            "cross pairs must satisfy i < j (each unordered pair exactly once)",
        )

    @property
    def num_positions(self) -> int:
        return int(self.position_uids.shape[0])

    @property
    def num_active_factors(self) -> int:
        return int(self.active_factor_ids.shape[0])

    @property
    def nnz(self) -> int:
        return int(self.factor_uix.shape[0])

    @property
    def nnz_cross(self) -> int:
        return int(self.cross_factor_uix_i.shape[0])

    @property
    def max_level_uix(self) -> int:
        """Highest levels-matrix row the book dereferences; -1 if it touches none."""
        return int(self.active_level_uix.max()) if self.num_active_factors else -1

    def print(self, *, max_items: int = 10) -> None:
        print(
            f"\nCompiledSensitivityBook: {self.num_positions} positions, "
            f"{self.num_active_factors} active factors, "
            f"nnz={self.nnz}, nnz_cross={self.nnz_cross}"
        )
        header = f"{'Position UID':>16} | {'Diag':>6} | {'Cross':>6}"
        print(header)
        print("-" * len(header))
        for i in range(min(self.num_positions, max_items)):
            d = int(self.position_offsets[i + 1]) - int(self.position_offsets[i])
            x = int(self.cross_offsets[i + 1]) - int(self.cross_offsets[i])
            print(f"{int(self.position_uids[i]):>16_d} | {d:>6} | {x:>6}")
        if self.num_positions > max_items:
            print(f"... ({self.num_positions - max_items} more positions)")


def _validate_csr(
    *, offsets: npt.NDArray[np.int32], num_rows: int, nnz: int, block: str
) -> None:
    """
    A CSR offset array is safe iff it starts at 0, never decreases, and ends at
    NNZ. Checking only the terminator lets `[0, 5, 0]` through, and the kernel
    then walks `range(0, 5)` over a zero-length entry array.
    """
    _require(
        offsets.shape == (num_rows + 1,),
        f"{block} offsets must have length M+1",
    )
    if num_rows == 0:
        _require(nnz == 0, f"{block} block has entries but no rows")
        return
    _require(int(offsets[0]) == 0, f"{block} offsets must start at 0")
    _require(
        bool(np.all(np.diff(offsets) >= 0)),
        f"{block} offsets must be non-decreasing",
    )
    _require(
        int(offsets[-1]) == nnz,
        f"{block} offsets terminate at {int(offsets[-1])}, expected NNZ={nnz}",
    )


def _require_index_range(arr: npt.NDArray[np.int32], bound: int, name: str) -> None:
    """Every value must land in `[0, bound)`; the kernels do not re-check."""
    if arr.shape[0] == 0:
        return
    _require(
        int(arr.min()) >= 0 and int(arr.max()) < bound,
        f"{name} out of range [0, {bound}) — would read out of bounds under "
        "boundscheck=False",
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"CompiledSensitivityBook: {message}")
