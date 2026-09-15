# src/riskquant/sensitivities/result.py
"""
Contiguous SoA container for computed block sensitivities.

This is the terminal representation of the numerical sensitivity core. It is
the sensitivity-side counterpart of `valuation.result.ValuationResult`: flat,
typed, C-contiguous NumPy arrays keyed by integer execution coordinates, with
no Polars, no DataFrames, and no risk-factor names.

Conversion into the validated edge contract
(`solutions.sensitivity_pnl.SensitivityBookSpec`) is deliberately NOT done
here -- that is the job of `sensitivities.book_adapter`, so that Polars and
Pydantic never reach a kernel. See `sensitivities.engine` for the full flow.

Layout
------
Two parallel groups, mirroring the two row-shapes the downstream
sensitivity-P&L book contract consumes:

    diagonal entries  (one row per (position, factor))
        position_uids  (E,)  int64
        factor_rf_ids  (E,)  int32
        first_order    (E,)  float64
        second_order   (E,)  float64

    cross entries     (one row per (position, factor_i, factor_j))
        cross_position_uids  (X,)  int64
        cross_factor_rf_ids_i (X,) int32
        cross_factor_rf_ids_j (X,) int32
        cross_second_order    (X,) float64

`factor_rf_ids` are GLOBAL `rf_id`s (the same identity `RiskStaticConfig`
issues and `UixMapper.market_id_universe` holds), not block-local uixs. The
sensitizer resolves uix -> rf_id once, at the end of the numerical stage, so
that downstream consumers never need the portfolio's mapper to interpret a
sensitivity row.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


@dataclass(frozen=True, slots=True, kw_only=True)
class BlockSensitivities:
    """Frozen SoA sensitivity output for one or more compiled blocks."""

    # --- Diagonal (per position, per factor) entries ---
    position_uids: npt.NDArray[np.int64]
    factor_rf_ids: npt.NDArray[np.int32]
    first_order: npt.NDArray[np.float64]
    second_order: npt.NDArray[np.float64]

    # --- Cross (per position, per factor pair) entries ---
    cross_position_uids: npt.NDArray[np.int64]
    cross_factor_rf_ids_i: npt.NDArray[np.int32]
    cross_factor_rf_ids_j: npt.NDArray[np.int32]
    cross_second_order: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        n = self.position_uids.shape[0]
        x = self.cross_position_uids.shape[0]

        # ---- dtypes (mirrors the downstream compiled-book contract) ----
        _require(self.position_uids.dtype == np.int64, "position_uids must be int64")
        _require(self.factor_rf_ids.dtype == np.int32, "factor_rf_ids must be int32")
        _require(
            self.first_order.dtype == np.float64
            and self.second_order.dtype == np.float64,
            "diagonal coefficient arrays must be float64",
        )
        _require(
            self.cross_position_uids.dtype == np.int64,
            "cross_position_uids must be int64",
        )
        _require(
            self.cross_factor_rf_ids_i.dtype == np.int32
            and self.cross_factor_rf_ids_j.dtype == np.int32,
            "cross factor id arrays must be int32",
        )
        _require(
            self.cross_second_order.dtype == np.float64,
            "cross_second_order must be float64",
        )

        # ---- shapes ----
        _require(
            self.factor_rf_ids.shape[0] == n
            and self.first_order.shape[0] == n
            and self.second_order.shape[0] == n,
            "diagonal entry arrays are misaligned",
        )
        _require(
            self.cross_factor_rf_ids_i.shape[0] == x
            and self.cross_factor_rf_ids_j.shape[0] == x
            and self.cross_second_order.shape[0] == x,
            "cross entry arrays are misaligned",
        )

        # ---- contiguity ----
        for name in (
            "position_uids",
            "factor_rf_ids",
            "first_order",
            "second_order",
            "cross_position_uids",
            "cross_factor_rf_ids_i",
            "cross_factor_rf_ids_j",
            "cross_second_order",
        ):
            arr: npt.NDArray[np.generic] = getattr(self, name)
            _require(arr.flags["C_CONTIGUOUS"], f"{name} must be C-contiguous")

        # ---- finiteness: a NaN/inf sensitivity is a silent-corruption bug ----
        _require(
            bool(np.isfinite(self.first_order).all())
            and bool(np.isfinite(self.second_order).all())
            and bool(np.isfinite(self.cross_second_order).all()),
            "sensitivity coefficients must all be finite",
        )

    @property
    def num_entries(self) -> int:
        return int(self.position_uids.shape[0])

    @property
    def num_cross_entries(self) -> int:
        return int(self.cross_position_uids.shape[0])

    @property
    def active_factor_rf_ids(self) -> npt.NDArray[np.int32]:
        """Sorted-unique global rf_ids referenced by any entry."""
        return np.unique(
            np.concatenate(
                (
                    self.factor_rf_ids,
                    self.cross_factor_rf_ids_i,
                    self.cross_factor_rf_ids_j,
                )
            )
        ).astype(np.int32, copy=False)
