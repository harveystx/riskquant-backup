# src/riskquant/market_data/risk_factors/matrix.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from riskquant.market_data.risk_factors.snapshot import RiskFactorLevelSnapshot
from riskquant.utils.dates import ordinal32_to_datetime64D


@dataclass(frozen=True, slots=True)
class RiskFactorMatrix:
    """
    Canonical RiskFactorMatrix with God-Tier memory layout for time-series operations.

    Unit contract (D069 / AUD-06): `values` are CANONICAL base units. Source-unit
    histories enter only through `build_risk_factor_matrix_from_df` / `_xlsx`,
    which normalise once; this container, `get_snapshot` and `from_snapshot`
    never rescale.

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

    # float32/64, shape (T, N) --> (observations, factors)
    values: np.ndarray

    # Default memory layout (F-contiguous for time-series)
    fortran_order: bool = True

    def __post_init__(self) -> None:
        # ---- Basic checks ----
        t_obs, n_factors = self.values.shape
        if self.dates.ndim != 1:
            raise ValueError("dates must be 1D")
        if self.factor_ids.ndim != 1:
            raise ValueError("factor_ids must be 1D")
        if self.values.ndim != 2:
            raise ValueError("values must be 2D")

        # ---- dtype checks ----
        if not np.issubdtype(self.values.dtype, np.floating):
            raise TypeError(f"values must be float dtype, got {self.values.dtype}")

        if np.issubdtype(self.dates.dtype, np.integer):
            if self.dates.dtype != np.int32:
                raise TypeError(f"int dates must be int32, got {self.dates.dtype}")
        elif self.dates.dtype != np.dtype("datetime64[D]"):
            raise TypeError(
                f"datetime dates must be datetime64[D], got {self.dates.dtype}"
            )

        if self.factor_ids.dtype != np.int32:
            raise TypeError(f"factor_ids must be int32, got {self.factor_ids.dtype}")

        # ---- shape checks ----
        if self.dates.size != t_obs:
            raise ValueError(
                f"len(dates) ({self.dates.size}) != values.shape[0] ({t_obs})"
            )
        if self.factor_ids.size != n_factors:
            raise ValueError(
                f"len(factor_ids) ({self.factor_ids.size}) != values.shape[1] ({n_factors})"
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
        if t_obs > 1:
            if np.any(self.dates[1:] < self.dates[:-1]):
                raise ValueError("dates must be sorted ascending")
            if np.any(self.dates[1:] == self.dates[:-1]):
                raise ValueError("dates must be unique")

        # ---- factor_ids sorted + unique ----
        # Mirror the dates check above: always run the adjacent-equality
        # check, then sort only if needed. The previous form only entered
        # the duplicate guard on the unsorted path, so an already-sorted
        # input like [1, 2, 2, 3] would bypass uniqueness entirely.
        if n_factors > 1:
            is_unsorted = bool(np.any(self.factor_ids[1:] < self.factor_ids[:-1]))

            if is_unsorted:
                perm = np.argsort(self.factor_ids, kind="mergesort")
                sorted_ids = self.factor_ids[perm]
            else:
                perm = None
                sorted_ids = self.factor_ids

            if np.any(sorted_ids[1:] == sorted_ids[:-1]):
                raise ValueError("factor_ids must be unique")

            if perm is not None:
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
        df = pd.DataFrame(data=self.values, index=index, columns=self.factor_ids)
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
        self, start: int | None = None, end: int | None = None
    ) -> RiskFactorMatrix:
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

    def slice_factors(self, factor_indices: np.ndarray) -> RiskFactorMatrix:
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
        return np.asfortranarray(self.values) if not self.fortran_order else self.values

    def get_factor_time_series(self, factor_id: int) -> np.ndarray:
        idx = np.searchsorted(self.factor_ids, factor_id)
        if idx >= self.factor_ids.size or self.factor_ids[idx] != factor_id:
            raise KeyError(f"factor_id {factor_id} not found")
        return np.ascontiguousarray(self.values[:, idx])

    # ---------------------------- Snapshot ----------------------------
    def get_snapshot(self, target_date: date) -> RiskFactorLevelSnapshot:
        from riskquant.utils.dates import date_to_ord32

        target_val = (
            date_to_ord32(target_date)
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

    @classmethod
    def from_snapshot(
        cls,
        snapshot: RiskFactorLevelSnapshot,
        *,
        fortran_order: bool = True,
    ) -> RiskFactorMatrix:
        """
        Elevate this snapshot into a 1-row RiskFactorMatrix.

        S-Tier mechanics:
        - factor_ids: Passed by reference (zero-copy).
        - values: Reshaped via np.newaxis (zero-copy view).
        - dates: Minimal 1-element array allocation aligned to datetime64[D].
        """

        # 1. Align dates invariant: 1D array of datetime64[D]
        dates_arr = np.array([snapshot.as_of_date], dtype="datetime64[D]")

        # 2. Align values invariant: 2D array (T, N) -> (1, N)
        # Using np.newaxis creates a fast, zero-copy view of the 1D C-contiguous array.
        values_2d = snapshot.factor_levels[np.newaxis, :]

        return RiskFactorMatrix(
            dates=dates_arr,
            factor_ids=snapshot.factor_ids,  # Already validated, sorted, and unique
            values=values_2d,
            fortran_order=fortran_order,
        )
