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
from typing import TYPE_CHECKING, Final

import numpy as np
import numpy.typing as npt

from riskquant.constants import (
    NO_DATE_ORD32,
    NO_FX_RF_REQUIRED,
    NO_ID,
    NOT_OPTION_MULT,
)
from riskquant.portfolio.ingest.compile import CompiledIngestSoA
from riskquant.utils.dates import ordinal32_to_datetime64D

if TYPE_CHECKING:
    from pandas import DataFrame

# ==============================================================================
# MODULE-LEVEL DTYPE / LENGTH CONTRACT
# ==============================================================================
# Built once at module load. Each IngestBatch construction reads from these
# rather than rebuilding the spec dict inline.

_EXPECTED_DTYPES: Final = MappingProxyType(
    {
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
    }
)

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


def _masked_float_copy(
    arr: npt.NDArray,
    *,
    mask_value: float | int,
    replace_with: float = np.nan,
    copy: bool,
) -> npt.NDArray[np.float64]:
    """Create float view/copy and replace sentinel-like values with NaN."""
    out = arr.copy() if copy else arr.astype(arr.dtype, copy=False)

    mask = out == mask_value

    if np.any(mask):
        out = out.copy()
        out[mask] = replace_with

    return out


def _decode_dates(
    ord_array: npt.NDArray[np.int32],
    *,
    sentinel: int,
    copy: bool,
) -> npt.NDArray[np.datetime64]:
    """Decode ordinal32 dates to datetime64[D] with NaT masking."""
    dts = ordinal32_to_datetime64D(ord_array)

    mask = ord_array == sentinel
    if copy or np.any(mask):
        dts = dts.copy()
    if np.any(mask):
        dts[mask] = np.datetime64("NaT")

    return dts


def _debug_mask_int_sentinel(
    arr: npt.NDArray,
    *,
    sentinel: int,
    replacement: str,
) -> npt.NDArray[np.object_]:
    """
    Debug-only helper.

    Converts integer arrays into object arrays for human-readable auditing.
    NEVER use in pricing/risk kernels.
    """
    out = arr.astype(object, copy=True)
    mask = arr == sentinel
    if np.any(mask):
        out[mask] = replacement
    return out


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

    def to_df(self, *, copy: bool = True) -> DataFrame:
        """
        DEBUG / AUDIT ONLY.

        Reconstruct a DataFrame from the SoA container. Alloc-heavy,
        produces object arrays. Not engine-safe.
        """
        from pandas import DataFrame

        strike_price = _masked_float_copy(
            self.strike_price,
            mask_value=0.0,
            copy=copy,
        )
        cp_mult = _masked_float_copy(
            self.cp_mult.astype(np.float64, copy=False),
            mask_value=int(NOT_OPTION_MULT),
            copy=copy,
        )

        return DataFrame(
            {
                # CORE
                "uid": self.uids,
                "inst_id": self.instrument_ids,
                # ECONOMICS
                "qty": self.qty,
                "k": strike_price,
                "cp_mult": cp_mult,
                # MARKET LINKAGES
                "und_ids": _debug_mask_int_sentinel(
                    self.underlying_ids,
                    sentinel=int(NO_ID),
                    replacement="NONE",
                ),
                "vol_ids": _debug_mask_int_sentinel(
                    self.volatility_ids,
                    sentinel=int(NO_ID),
                    replacement="NONE",
                ),
                # FX IDs
                "fx_u2s_ids": _debug_mask_int_sentinel(
                    self.fx_u2s_ids,
                    sentinel=int(NO_FX_RF_REQUIRED),
                    replacement="NONE",
                ),
                "fx_u2s_inv": self.fx_u2s_inv,
                "fx_k2s_ids": _debug_mask_int_sentinel(
                    self.fx_k2s_ids,
                    sentinel=int(NO_FX_RF_REQUIRED),
                    replacement="NONE",
                ),
                "fx_k2s_inv": self.fx_k2s_inv,
                # CURVES
                "disc_und_id": _debug_mask_int_sentinel(
                    self.under_discount_curve_ids,
                    sentinel=int(NO_ID),
                    replacement="NONE",
                ),
                "disc_k_id": _debug_mask_int_sentinel(
                    self.strike_discount_curve_ids,
                    sentinel=int(NO_ID),
                    replacement="NONE",
                ),
                # DATES
                "settle_date": _decode_dates(
                    self.settle_ord,
                    sentinel=int(NO_DATE_ORD32),
                    copy=copy,
                ),
                "delivery_date": _decode_dates(
                    self.delivery_ord,
                    sentinel=int(NO_DATE_ORD32),
                    copy=copy,
                ),
                "expiry_date": _decode_dates(
                    self.expiry_ord,
                    sentinel=int(NO_DATE_ORD32),
                    copy=copy,
                ),
                # VOL
                "vol_surface_y_axis_type_id": self.vol_surface_y_axis_type_ids,
                # NETTING
                "netting_set_id": _debug_mask_int_sentinel(
                    self.netting_set_id,
                    sentinel=int(NO_ID),
                    replacement="NONE",
                ),
            },
            copy=copy,
        )

    @property
    def df(self) -> DataFrame:
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
        )
