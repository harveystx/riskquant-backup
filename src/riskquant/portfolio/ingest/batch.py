# src/riskquant/portfolio/ingest/batch.py
"""
Struct-of-Arrays (SoA) container for validated trade ingestion.

Design goals:
- contiguous homogeneous arrays
- strict dtype enforcement
- cache-friendly SoA layout
- immutable batch semantics
- zero object arrays in engine paths
- explicit debug-only reconstruction utilities

Dtype change in this revision
-----------------------------
`fx_u2s_inv` and `fx_k2s_inv` are now int8 (was int32). Flags are 1 byte per row,
matching the project flag convention. The block compilers convert int8 -> bool
via .view() (same itemsize, no allocation) for the block-layer signatures.

Architecture update
-------------------
- Purged all UIX coordinates. The batch acts purely as a global ID ledger.
- Semantic Collapsing: Sparse curve/node arrays fused into dense `underlying_ids`
  and `volatility_ids`.
- `vol_surface_y_axis_type_ids` is now resolved by `compile_ingest_soa_payload_
  from_df` (sourced from `static_config.surface_meta_dict[id].y_axis_kind`).
  `from_compiled_ingest_soa` is a pure copy with no synthetic sentinels.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import numpy as np
import numpy.typing as npt
import polars as pl

from riskquant.constants import (
    NO_DATE_ORD32,
    NO_FX_RF_REQUIRED,
    NO_ID,
    NOT_OPTION_MULT,
)
from riskquant.portfolio.ingest.compile import CompiledIngestSoA
from riskquant.utils.dates import ordinal32_to_datetime64D

# ==============================================================================
# MODULE-LEVEL DTYPE / LENGTH CONTRACT
# ==============================================================================
# Built once at module load. Each IngestBatch construction reads from these
# rather than rebuilding the spec dict inline.

_EXPECTED_DTYPES: Final = MappingProxyType({
    # CORE
    "instrument_ids": np.dtype(np.int32),
    "uids": np.dtype(np.int64),
    # MARKET RISK
    "qty": np.dtype(np.float64),
    "strike_price": np.dtype(np.float64),
    "cp_mult": np.dtype(np.int8),
    # DATES (Ordinals Only)
    "settle_ord": np.dtype(np.int32),
    "delivery_ord": np.dtype(np.int32),
    "expiry_ord": np.dtype(np.int32),
    # MARKET LINKAGES (Dense)
    "underlying_ids": np.dtype(np.int32),
    "volatility_ids": np.dtype(np.int32),
    # FX
    "fx_u2s_ids": np.dtype(np.int32),
    "fx_u2s_inv": np.dtype(np.int8),
    "fx_k2s_ids": np.dtype(np.int32),
    "fx_k2s_inv": np.dtype(np.int8),
    # CURVES & SURFACES
    "under_discount_curve_ids": np.dtype(np.int32),
    "strike_discount_curve_ids": np.dtype(np.int32),
    "vol_surface_y_axis_type_ids": np.dtype(np.int8),
    # NETTING SET
    "netting_set_id": np.dtype(np.int64),
})

_VARIABLE_LENGTH_FIELDS: Final[tuple[str, ...]] = tuple(_EXPECTED_DTYPES.keys())


# ==============================================================================
# INTERNAL HELPERS
# ==============================================================================


def _assert_c_contiguous(arr: npt.NDArray, *, name: str) -> None:
    if not arr.flags.c_contiguous:
        raise ValueError(f"IngestBatch field '{name}' must be C-contiguous.")


def _assert_dtype(
    arr: npt.NDArray,
    expected_dtype: np.dtype,
    *,
    name: str,
) -> None:
    if arr.dtype != expected_dtype:
        raise TypeError(
            f"IngestBatch field '{name}' has dtype={arr.dtype}, "
            f"expected={expected_dtype}."
        )


def _debug_column(
    name: str,
    arr: npt.NDArray[np.generic],
    *,
    dtype: pl.DataType,
    sentinel: float | None,
    copy: bool,
) -> pl.Series:
    """
    Debug-only helper: one typed Polars column of the debug frame.

    The SoA sentinel becomes null, so the column keeps one dtype. The v0.2.0
    pandas adapter mixed int values with the text "NONE" (object dtype) or
    put NaN in float columns. ``copy=False`` lets Polars share the SoA buffer
    where the dtype allows it; ``copy=True`` gives an independent buffer.
    NEVER use in pricing/risk kernels.
    """
    values = np.array(arr, copy=True) if copy else arr
    column = pl.Series(name, values, dtype=dtype)
    if sentinel is None:
        return column
    is_sentinel = pl.Series(arr == sentinel)
    return pl.select(
        pl.when(is_sentinel).then(None).otherwise(column).alias(name)
    ).to_series()


def _debug_date_column(
    name: str,
    ord_array: npt.NDArray[np.int32],
    *,
    sentinel: int,
) -> pl.Series:
    """Debug-only helper: ordinal32 dates as ``pl.Date`` with null at the sentinel."""
    dates = pl.Series(name, ordinal32_to_datetime64D(ord_array), dtype=pl.Date)
    is_sentinel = pl.Series(ord_array == sentinel)
    return pl.select(
        pl.when(is_sentinel).then(None).otherwise(dates).alias(name)
    ).to_series()


# ==============================================================================
# INGEST BATCH
# ==============================================================================


@dataclass(frozen=True, slots=True, kw_only=True)
class IngestBatch:
    """
    Immutable Struct-of-Arrays ingestion container.

    All arrays:
    - must be contiguous
    - must be strongly typed (see _EXPECTED_DTYPES)
    - must have length == size (see _VARIABLE_LENGTH_FIELDS)
    """

    size: int

    # ROUTING / DISPATCH
    instrument_ids: npt.NDArray[np.int32]

    # ECONOMICS
    uids: npt.NDArray[np.int64]
    qty: npt.NDArray[np.float64]
    strike_price: npt.NDArray[np.float64]
    cp_mult: npt.NDArray[np.int8]

    # DATES (ORDINAL32)
    settle_ord: npt.NDArray[np.int32]
    delivery_ord: npt.NDArray[np.int32]
    expiry_ord: npt.NDArray[np.int32]

    # MARKET LINKAGES
    underlying_ids: npt.NDArray[np.int32]
    volatility_ids: npt.NDArray[np.int32]

    # FX
    fx_u2s_ids: npt.NDArray[np.int32]
    fx_u2s_inv: npt.NDArray[np.int8]
    fx_k2s_ids: npt.NDArray[np.int32]
    fx_k2s_inv: npt.NDArray[np.int8]

    # DISCOUNT CURVES & SURFACES
    under_discount_curve_ids: npt.NDArray[np.int32]
    strike_discount_curve_ids: npt.NDArray[np.int32]
    vol_surface_y_axis_type_ids: npt.NDArray[np.int8]

    # NETTING SET LINKAGE
    netting_set_id: npt.NDArray[np.int64]

    # RUN AS-OF (D067). The as-of at which relative forward names were resolved
    # to dated contracts. NO_DATE_ORD32 means the batch resolved nothing
    # as-of-dependent. `build_uix_mapper` derives the run as-of from it.
    as_of_ord32: int = int(NO_DATE_ORD32)

    # ==========================================================================
    # POST INIT VALIDATION
    # ==========================================================================

    def __post_init__(self) -> None:
        """Enforce dtype, contiguity, and length consistency."""
        for field_name, expected_dtype in _EXPECTED_DTYPES.items():
            arr = getattr(self, field_name)
            if not isinstance(arr, np.ndarray):
                raise TypeError(
                    f"IngestBatch field '{field_name}' is not a numpy ndarray."
                )
            _assert_dtype(arr, expected_dtype, name=field_name)
            _assert_c_contiguous(arr, name=field_name)

        for field_name in _VARIABLE_LENGTH_FIELDS:
            arr = getattr(self, field_name)
            if arr.shape[0] != self.size:
                raise ValueError(
                    f"IngestBatch field '{field_name}' has "
                    f"length={arr.shape[0]}, expected={self.size}."
                )

    # ==========================================================================
    # DEBUG RECONSTRUCTION
    # ==========================================================================

    def to_df(self, *, copy: bool = True) -> pl.DataFrame:
        """
        DEBUG / AUDIT ONLY.

        Reconstruct a ``pl.DataFrame`` from the SoA container. Alloc-heavy.
        Not engine-safe. The columns and their order are the v0.2.0 columns.

        Each column has one dtype. A SoA sentinel is null: ``NO_ID`` in the
        id columns, ``NO_FX_RF_REQUIRED`` in the FX id columns, strike ``0.0``
        in ``k``, ``NOT_OPTION_MULT`` in ``cp_mult`` and ``NO_DATE_ORD32`` in
        the date columns.

        Parameters
        ----------
        copy
            ``True`` gives columns with independent buffers. ``False`` lets
            Polars share the SoA buffers (``.df``); the batch arrays must not
            change while the frame is in use.

        Returns
        -------
        pl.DataFrame
            ``uid`` Int64, ``inst_id`` Int32, ``qty`` Float64, ``k`` and
            ``cp_mult`` Float64, the id columns Int64, ``fx_*_inv`` and
            ``vol_surface_y_axis_type_id`` Int8, the date columns Date.
        """
        no_id = float(NO_ID)
        no_fx = float(NO_FX_RF_REQUIRED)
        columns = [
            # CORE
            _debug_column("uid", self.uids, dtype=pl.Int64(), sentinel=None, copy=copy),
            _debug_column(
                "inst_id",
                self.instrument_ids,
                dtype=pl.Int32(),
                sentinel=None,
                copy=copy,
            ),
            # ECONOMICS
            _debug_column(
                "qty", self.qty, dtype=pl.Float64(), sentinel=None, copy=copy
            ),
            _debug_column(
                "k", self.strike_price, dtype=pl.Float64(), sentinel=0.0, copy=copy
            ),
            _debug_column(
                "cp_mult",
                self.cp_mult,
                dtype=pl.Float64(),
                sentinel=float(NOT_OPTION_MULT),
                copy=copy,
            ),
            # MARKET LINKAGES
            _debug_column(
                "und_ids",
                self.underlying_ids,
                dtype=pl.Int64(),
                sentinel=no_id,
                copy=copy,
            ),
            _debug_column(
                "vol_ids",
                self.volatility_ids,
                dtype=pl.Int64(),
                sentinel=no_id,
                copy=copy,
            ),
            # FX IDs
            _debug_column(
                "fx_u2s_ids",
                self.fx_u2s_ids,
                dtype=pl.Int64(),
                sentinel=no_fx,
                copy=copy,
            ),
            _debug_column(
                "fx_u2s_inv",
                self.fx_u2s_inv,
                dtype=pl.Int8(),
                sentinel=None,
                copy=copy,
            ),
            _debug_column(
                "fx_k2s_ids",
                self.fx_k2s_ids,
                dtype=pl.Int64(),
                sentinel=no_fx,
                copy=copy,
            ),
            _debug_column(
                "fx_k2s_inv",
                self.fx_k2s_inv,
                dtype=pl.Int8(),
                sentinel=None,
                copy=copy,
            ),
            # CURVES
            _debug_column(
                "disc_und_id",
                self.under_discount_curve_ids,
                dtype=pl.Int64(),
                sentinel=no_id,
                copy=copy,
            ),
            _debug_column(
                "disc_k_id",
                self.strike_discount_curve_ids,
                dtype=pl.Int64(),
                sentinel=no_id,
                copy=copy,
            ),
            # DATES
            _debug_date_column(
                "settle_date", self.settle_ord, sentinel=int(NO_DATE_ORD32)
            ),
            _debug_date_column(
                "delivery_date", self.delivery_ord, sentinel=int(NO_DATE_ORD32)
            ),
            _debug_date_column(
                "expiry_date", self.expiry_ord, sentinel=int(NO_DATE_ORD32)
            ),
            # VOL
            _debug_column(
                "vol_surface_y_axis_type_id",
                self.vol_surface_y_axis_type_ids,
                dtype=pl.Int8(),
                sentinel=None,
                copy=copy,
            ),
            # NETTING
            _debug_column(
                "netting_set_id",
                self.netting_set_id,
                dtype=pl.Int64(),
                sentinel=no_id,
                copy=copy,
            ),
        ]
        return pl.DataFrame(columns)

    @property
    def df(self) -> pl.DataFrame:
        """Convenience debug alias for to_df(copy=False)."""
        return self.to_df(copy=False)

    # ==========================================================================
    # REORDERING
    # ==========================================================================

    def reorder(self, order: np.ndarray, *, copy: bool = False) -> IngestBatch:
        """Reorder batch arrays consistently."""
        if order.dtype.kind not in ("i", "u"):
            raise TypeError("order must contain integer indices.")
        if order.ndim != 1:
            raise ValueError("order must be 1-dimensional.")
        if order.shape[0] != self.size:
            raise ValueError(
                f"order length={order.shape[0]} does not match batch size={self.size}."
            )

        def _reorder(arr: np.ndarray) -> np.ndarray:
            return np.take(arr, order) if copy else arr[order]

        return IngestBatch(
            size=self.size,
            instrument_ids=_reorder(self.instrument_ids),
            uids=_reorder(self.uids),
            qty=_reorder(self.qty),
            strike_price=_reorder(self.strike_price),
            cp_mult=_reorder(self.cp_mult),
            settle_ord=_reorder(self.settle_ord),
            delivery_ord=_reorder(self.delivery_ord),
            expiry_ord=_reorder(self.expiry_ord),
            underlying_ids=_reorder(self.underlying_ids),
            volatility_ids=_reorder(self.volatility_ids),
            fx_u2s_ids=_reorder(self.fx_u2s_ids),
            fx_u2s_inv=_reorder(self.fx_u2s_inv),
            fx_k2s_ids=_reorder(self.fx_k2s_ids),
            fx_k2s_inv=_reorder(self.fx_k2s_inv),
            under_discount_curve_ids=_reorder(self.under_discount_curve_ids),
            strike_discount_curve_ids=_reorder(self.strike_discount_curve_ids),
            vol_surface_y_axis_type_ids=_reorder(self.vol_surface_y_axis_type_ids),
            netting_set_id=_reorder(self.netting_set_id),
            as_of_ord32=self.as_of_ord32,
        )

    @classmethod
    def from_compiled_ingest_soa(cls, c: CompiledIngestSoA) -> IngestBatch:
        # Pure copy. `vol_surface_y_axis_type_ids` is now produced by the
        # compiler against `static_config.surface_meta_dict`; no synthetic
        # sentinel fill happens here.
        return cls(
            size=c.size,
            instrument_ids=c.instrument_ids,
            uids=c.uids,
            qty=c.qty,
            strike_price=c.strike,
            cp_mult=c.cp_mult,
            settle_ord=c.settle_ord,
            delivery_ord=c.delivery_ord,
            expiry_ord=c.expiry_ord,
            underlying_ids=c.underlying_ids,
            volatility_ids=c.volatility_ids,
            fx_u2s_ids=c.fx_u2s_ids,
            fx_u2s_inv=c.fx_u2s_inv,
            fx_k2s_ids=c.fx_k2s_ids,
            fx_k2s_inv=c.fx_k2s_inv,
            under_discount_curve_ids=c.under_discount_curve_ids,
            strike_discount_curve_ids=c.strike_discount_curve_ids,
            vol_surface_y_axis_type_ids=c.vol_surface_y_axis_type_ids,
            netting_set_id=c.netting_set_id,
            as_of_ord32=c.as_of_ord32,
        )
