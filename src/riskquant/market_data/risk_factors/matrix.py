# riskquant/market_data/risk_factors/matrix.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from riskquant.market_data.risk_factors.snapshot import RiskFactorLevelSnapshot
from riskquant.utils.dates import ordinal32_to_datetime64D


@dataclass(frozen=True, slots=True)
class RiskFactorMatrix:
    """
    Canonical RiskFactorMatrix with God-Tier memory layout for time-series operations.

    Invariants:
      - dates: 1D, sorted ascending, unique (int32 ordinal or datetime64[D])
      - factor_ids: 1D int32, sorted ascending, unique
      - values: 2D float32/64, shape (T, N), F-contiguous by default for column-major access
      - len(dates)=T, len(factor_ids)=N
    """

    # datetime64[D] or int32 ordinal
    dates: np.ndarray

    # int32
    factor_ids: np.ndarray

    # float32/64, shape (T, N)
    values: np.ndarray

    # Default memory layout (F-contiguous for time-series)
    fortran_order: bool = True

    def __post_init__(self) -> None:
        # ---- Basic checks ----
        T, N = self.values.shape
        if self.dates.ndim != 1:
            raise ValueError("dates must be 1D")
        if self.factor_ids.ndim != 1:
            raise ValueError("factor_ids must be 1D")
        if self.values.ndim != 2:
            raise ValueError("values must be 2D")

        # ---- dtype checks ----
        if not np.issubdtype(self.values.dtype, np.floating):
            raise TypeError(
                f"values must be float dtype, got {self.values.dtype}"
            )

        if np.issubdtype(self.dates.dtype, np.integer):
            if self.dates.dtype != np.int32:
                raise TypeError(
                    f"int dates must be int32, got {self.dates.dtype}"
                )
        elif self.dates.dtype != np.dtype("datetime64[D]"):
            raise TypeError(
                f"datetime dates must be datetime64[D], got {self.dates.dtype}"
            )

        if self.factor_ids.dtype != np.int32:
            raise TypeError(
                f"factor_ids must be int32, got {self.factor_ids.dtype}"
            )

        # ---- shape checks ----
        if self.dates.size != T:
            raise ValueError(
                f"len(dates) ({self.dates.size}) != values.shape[0] ({T})"
            )
        if self.factor_ids.size != N:
            raise ValueError(
                f"len(factor_ids) ({self.factor_ids.size}) != values.shape[1] ({N})"
            )

        # ---- memory layout ----
        if self.fortran_order:
            if not self.values.flags["F_CONTIGUOUS"]:
                vals = np.asfortranarray(self.values)
                object.__setattr__(self, "values", vals)
        else:
            if not self.values.flags["C_CONTIGUOUS"]:
                vals = np.ascontiguousarray(self.values)
                object.__setattr__(self, "values", vals)

        # ---- dates sorted + unique ----
        if T > 1:
            if np.any(self.dates[1:] < self.dates[:-1]):
                raise ValueError("dates must be sorted ascending")
            if np.any(self.dates[1:] == self.dates[:-1]):
                raise ValueError("dates must be unique")

        # ---- factor_ids sorted + unique ----
        if N > 1 and np.any(self.factor_ids[1:] < self.factor_ids[:-1]):
            perm = np.argsort(self.factor_ids, kind="mergesort")
            sorted_ids = self.factor_ids[perm]
            if np.any(sorted_ids[1:] == sorted_ids[:-1]):
                raise ValueError("factor_ids must be unique")
            sorted_vals = (
                np.asfortranarray(self.values[:, perm])
                if self.fortran_order
                else np.ascontiguousarray(self.values[:, perm])
            )
            object.__setattr__(self, "factor_ids", sorted_ids)
            object.__setattr__(self, "values", sorted_vals)

    # ---------------------------- Convenience Methods ----------------------------
    def to_df(self) -> pd.DataFrame:
        index = pd.DatetimeIndex(
            ordinal32_to_datetime64D(self.dates)
            if np.issubdtype(self.dates.dtype, np.integer)
            else self.dates
        )
        df = pd.DataFrame(
            data=self.values, index=index, columns=self.factor_ids
        )
        df.index.name = "date"
        df.columns.name = "factor_id"
        return df

    def head(self, n: int = 5) -> pd.DataFrame:
        return self.to_df().head(n=n)

    def tail(self, n: int = 5) -> pd.DataFrame:
        return self.to_df().tail(n=n)

    def plot(self, **kwargs):
        return self.to_df().plot(**kwargs)

    # ---------------------------- Properties ----------------------------
    @property
    def index_dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(
            ordinal32_to_datetime64D(self.dates)
            if np.issubdtype(self.dates.dtype, np.integer)
            else self.dates
        )

    @property
    def index_factors(self) -> pd.Index:
        return pd.Index(self.factor_ids, name="factor_id")

    @property
    def df(self) -> pd.DataFrame:
        return self.to_df()

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    @property
    def num_dates(self) -> int:
        return self.values.shape[0]

    @property
    def num_factors(self) -> int:
        return self.values.shape[1]

    # ---------------------------- Slicing Methods ----------------------------
    def slice_dates(
        self, start: Optional[int] = None, end: Optional[int] = None
    ) -> "RiskFactorMatrix":
        start = 0 if start is None else start
        end = self.num_dates if end is None else end
        vals = self.values[start:end, :]
        if self.fortran_order and not vals.flags["F_CONTIGUOUS"]:
            vals = np.asfortranarray(vals)
        elif not self.fortran_order and not vals.flags["C_CONTIGUOUS"]:
            vals = np.ascontiguousarray(vals)
        return RiskFactorMatrix(
            dates=self.dates[start:end],
            factor_ids=self.factor_ids,
            values=vals,
            fortran_order=self.fortran_order,
        )

    def slice_factors(self, factor_indices: np.ndarray) -> "RiskFactorMatrix":
        vals = self.values[:, factor_indices]
        if self.fortran_order and not vals.flags["F_CONTIGUOUS"]:
            vals = np.asfortranarray(vals)
        elif not self.fortran_order and not vals.flags["C_CONTIGUOUS"]:
            vals = np.ascontiguousarray(vals)
        return RiskFactorMatrix(
            dates=self.dates,
            factor_ids=self.factor_ids[factor_indices],
            values=vals,
            fortran_order=self.fortran_order,
        )

    # ---------------------------- Time-Series Optimization ----------------------------
    def to_time_series_optimized_array(self) -> np.ndarray:
        """Return F-contiguous array for column-major operations (Numba/Cython)."""
        return (
            np.asfortranarray(self.values)
            if not self.fortran_order
            else self.values
        )

    def get_factor_time_series(self, factor_id: int) -> np.ndarray:
        idx = np.searchsorted(self.factor_ids, factor_id)
        if idx >= self.factor_ids.size or self.factor_ids[idx] != factor_id:
            raise KeyError(f"factor_id {factor_id} not found")
        return np.ascontiguousarray(self.values[:, idx])

    # ---------------------------- Snapshot ----------------------------
    def get_snapshot(self, target_date: date) -> RiskFactorLevelSnapshot:
        from riskquant.utils.dates import date_to_ordinal32

        target_val = (
            date_to_ordinal32(target_date)
            if np.issubdtype(self.dates.dtype, np.integer)
            else np.datetime64(target_date, "D")
        )
        idx = np.searchsorted(self.dates, target_val)
        if idx >= self.dates.size or self.dates[idx] != target_val:
            raise KeyError(f"Date {target_date} not found in matrix")

        levels = self.values[idx, :]
        valid_mask = ~np.isnan(levels)
        return RiskFactorLevelSnapshot(
            as_of_date=target_date,
            factor_ids=self.factor_ids[valid_mask],
            factor_levels=levels[valid_mask],
        )
