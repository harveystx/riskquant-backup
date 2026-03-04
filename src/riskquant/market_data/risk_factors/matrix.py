# riskquant/market_data/risk_factors/matrix.py

# ----------------------------
# 1) Imports: remove unused + add dtype helper
# ----------------------------
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from riskquant.market_data.risk_factors.snapshot import RiskFactorLevelSnapshot
from riskquant.utils.dates import (
    ordinal32_to_datetime64D,
)


@dataclass(frozen=True, slots=True)
class RiskFactorMatrix:
    """
    Canonical invariants (enforced):
      - dates: 1D, sorted ascending, unique
      - factor_ids: 1D int32, sorted ascending, unique
      - values: 2D floating, shape (T, N), C-contiguous
      - shapes aligned: len(dates)=T, len(factor_ids)=N

    Canonicalization:
      - factor axis is automatically sorted (and values columns permuted) if needed.
      - dates are validated as sorted+unique (not auto-sorted) to avoid an O(T*N) row copy.
    """

    dates: np.ndarray  # datetime64[D] or int32 ordinal
    factor_ids: np.ndarray  # int32, shape (N,)
    values: np.ndarray  # float32/float64, shape (T, N), contiguous

    def __post_init__(self) -> None:
        """Validate and canonicalize a newly-created RiskFactorMatrix.

        Responsibilities:
          - verify simple shape/dtype/contiguity invariants cheaply (no heavy copies)
          - enforce that date encoding is either int32 ordinal or datetime64[D]
          - ensure `factor_ids` are int32 and unique; *automatically* sort + permute columns
            if the provided factor_ids are not sorted (keeps values aligned)

        The method raises clear, early exceptions to avoid subtle bugs later in
        numerical code that assumes the canonical invariants described on the
        class docstring.
        """
        # ---- Basic dimensionality checks (cheap) ----
        if self.dates.ndim != 1:
            raise ValueError("dates must be 1D!")
        if self.factor_ids.ndim != 1:
            raise ValueError("factor_ids must be 1D!")
        if self.values.ndim != 2:
            raise ValueError("values must be 2D!")

        # ---- dtype invariants ----
        # `values` must be a floating-point array (float32/float64 allowed).
        if not np.issubdtype(self.values.dtype, np.floating):
            raise TypeError(
                f"values must be floating dtype; got {self.values.dtype}"
            )

        # Dates may be encoded as int32 ordinals or as datetime64[D].
        # If integer-encoded, require int32 to avoid accidental upcasting.
        if np.issubdtype(self.dates.dtype, np.integer):
            if self.dates.dtype != np.int32:
                raise TypeError(
                    f"int date encoding must be int32 days since DATE_EPOCH; got {self.dates.dtype}"
                )
        else:
            if self.dates.dtype != np.dtype("datetime64[D]"):
                raise TypeError(
                    f"datetime date encoding must be datetime64[D]; got {self.dates.dtype}"
                )

        # Factor IDs must be 32-bit integers for compact storage and stable hashing.
        if self.factor_ids.dtype != np.int32:
            raise TypeError(
                f"factor_ids must be int32; got {self.factor_ids.dtype}"
            )

        # ---- shape invariants ----
        T, N = self.values.shape
        if self.dates.size != T:
            raise ValueError(
                f"dates length ({self.dates.size}) must match values.shape[0] ({T})"
            )
        if self.factor_ids.size != N:
            raise ValueError(
                f"factor_ids length ({self.factor_ids.size}) must match values.shape[1] ({N})"
            )

        # ---- contiguity invariant for values ----
        # Many numerical libraries expect C-contiguous 2D arrays; detect accidental
        # non-contiguous views early and ask callers to construct canonical arrays.
        if not self.values.flags["C_CONTIGUOUS"]:
            raise ValueError(
                "values must be C-contiguous (use np.ascontiguousarray at construction). If you sliced, "
                "call np.ascontiguousarray(...) or use the provided slice_* helpers."
            )

        # ---- dates: require sorted + unique (cheap, alloc-free checks) ----
        # We *do not* auto-sort dates because that would require an O(T*N) row copy.
        if T > 1:
            if np.any(self.dates[1:] < self.dates[:-1]):
                raise ValueError("dates must be sorted ascending.")
            if np.any(self.dates[1:] == self.dates[:-1]):
                raise ValueError("dates must be unique!")

        # ---- factor_ids: canonicalize to sorted + unique with minimal copying ----
        if N > 1:
            # Fast path: if already non-decreasing, just check uniqueness and return.
            if not np.any(self.factor_ids[1:] < self.factor_ids[:-1]):
                if np.any(self.factor_ids[1:] == self.factor_ids[:-1]):
                    raise ValueError("factor_ids must be unique!")
                return

            # Slow path: factor_ids are unsorted -> build a stable permutation and
            # reorder both `factor_ids` and the corresponding columns of `values`.
            # We use `mergesort` for determinism; `np.argsort` here allocates O(N).
            perm = np.argsort(self.factor_ids, kind="mergesort")
            sorted_ids = self.factor_ids[perm]

            # Check uniqueness after sorting (adjacent duplicates are easy to detect).
            if np.any(sorted_ids[1:] == sorted_ids[:-1]):
                raise ValueError("factor_ids must be unique!")

            # Permuting columns via fancy indexing produces a new array; ensure it's
            # C-contiguous so downstream code's contiguity assumptions hold.
            sorted_vals = np.ascontiguousarray(self.values[:, perm])

            # Bypass dataclass immutability to install canonicalized arrays.
            object.__setattr__(self, "factor_ids", sorted_ids)
            object.__setattr__(self, "values", sorted_vals)

    # -------------------- Convenience --------------------
    def to_df(self) -> pd.DataFrame:
        """Convert the matrix into a wide pandas DataFrame.

        Returns:
          DataFrame with `date` as the index (DatetimeIndex) and integer
          `factor_id` values as columns. The function preserves the internal
          canonical ordering of factors.
        """
        if np.issubdtype(self.dates.dtype, np.integer):
            dts = ordinal32_to_datetime64D(self.dates)
            index = pd.DatetimeIndex(dts)
        else:
            index = pd.DatetimeIndex(self.dates)

        df = pd.DataFrame(
            data=self.values, index=index, columns=self.factor_ids
        )
        df.index.name = "date"
        df.columns.name = "factor_id"
        return df

    def head(self, n: int = 5) -> pd.DataFrame:
        """Return the first `n` rows as a DataFrame (convenience wrapper)."""
        return self.to_df().head(n=n)

    def tail(self, n: int = 5) -> pd.DataFrame:
        """Return the last `n` rows as a DataFrame (convenience wrapper)."""
        return self.to_df().tail(n=n)

    def plot(self, **kwargs):
        """Plot all factors over time using pandas plotting backend.

        Any keyword args are forwarded to `DataFrame.plot`.
        """
        return self.to_df().plot(**kwargs)

    # ----------------------------
    # 1) PROPERTIES
    # ----------------------------

    @property
    def index_dates(self) -> pd.DatetimeIndex:
        """Return a pandas DatetimeIndex view of the stored dates.

        - If dates are stored as int32 ordinals, convert them to datetime64[D].
        - Does not copy underlying date storage when possible (uses views).
        """
        if np.issubdtype(self.dates.dtype, np.integer):
            dts = ordinal32_to_datetime64D(self.dates)
            return pd.DatetimeIndex(dts)
        else:
            return pd.DatetimeIndex(self.dates)

    @property
    def index_factors(self) -> pd.Index:
        """Return a pandas Index for `factor_ids` (named `factor_id`)."""
        return pd.Index(self.factor_ids, name="factor_id")

    @property
    def df(self) -> pd.DataFrame:
        """Alias for `to_df()` to present the matrix as a DataFrame."""
        return self.to_df()

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the (T, N) shape of the underlying values array."""
        return self.values.shape

    @property
    def num_dates(self) -> int:
        """Number of dates (time steps) in the matrix."""
        return self.values.shape[0]

    @property
    def num_factors(self) -> int:
        """Number of risk factors (columns) in the matrix."""
        return self.values.shape[1]

    # ----------------------------
    # 2) slice_dates: enforce C-contiguous values to satisfy invariant
    # ----------------------------
    def slice_dates(
        self,
        start: int | None = None,
        end: int | None = None,
    ) -> "RiskFactorMatrix":
        """Return a time-sliced view of the matrix as a new canonical object.

        Notes:
          - `start`/`end` use Python slice semantics (end is exclusive).
          - The returned `RiskFactorMatrix` always satisfies the class invariants
            (including C-contiguity of `values`).
        """
        if start is None:
            start = 0
        if end is None:
            end = self.num_dates

        vals = self.values[start:end, :]
        # Slicing a NumPy array can produce non-contiguous views; downstream
        # code depends on C-contiguity so we eagerly copy only when required.
        if not vals.flags["C_CONTIGUOUS"]:
            vals = np.ascontiguousarray(vals)

        return RiskFactorMatrix(
            dates=self.dates[start:end],
            factor_ids=self.factor_ids,
            values=vals,
        )

    # ----------------------------
    # 3) slice_factors: same contiguity enforcement
    # ----------------------------
    def slice_factors(self, factor_indices: np.ndarray) -> "RiskFactorMatrix":
        """Return a factor-sliced matrix.

        Parameters:
          factor_indices: integer indices (NumPy indexing) selecting columns from
            the canonical `values` array.

        Returns:
          New `RiskFactorMatrix` containing the selected factor columns. Note
          that `factor_ids` passed to the constructor may be unsorted; the
          constructor's canonicalization step will re-sort and permute values
          if necessary.
        """
        vals = self.values[:, factor_indices]
        # Ensure C-contiguity after fancy indexing / boolean masking.
        if not vals.flags["C_CONTIGUOUS"]:
            vals = np.ascontiguousarray(vals)

        return RiskFactorMatrix(
            dates=self.dates,
            factor_ids=self.factor_ids[factor_indices],
            values=vals,
        )

    # ----------------------------
    # 4) Time-Series Optimization
    # ----------------------------
    def to_time_series_optimized_array(self) -> np.ndarray:
        """
        Returns an F-contiguous (Fortran/column-major) copy of the values array.

        Use this when passing the matrix to a Cython/Numba kernel or perform heavy
        time-series math (e.g., historical volatility, EWMA) where the access
        pattern is factor-by-factor (column-by-column) across all dates.

        Returns:
            np.ndarray: A (T, N) array where columns are contiguous in memory.
                        Note: This forces a memory allocation/copy because the
                        source array is strictly C-contiguous.
        """
        # np.asfortranarray creates a column-major copy of the C-contiguous data.
        # This means arr[:, i] becomes an unbroken block of physical memory.
        return np.asfortranarray(self.values)

    def get_factor_time_series(self, factor_id: int) -> np.ndarray:
        """
        Extract a strictly C-contiguous 1D time-series array for a single factor.
        Useful for running math on exactly one instrument without copying the whole matrix.
        """
        # Find the column index for the requested factor_id
        idx = np.searchsorted(self.factor_ids, factor_id)
        if idx >= self.factor_ids.size or self.factor_ids[idx] != factor_id:
            raise KeyError(f"factor_id {factor_id} not found in matrix.")

        # Slicing a column from a C-contiguous 2D array creates a strided view.
        # We force it back to a contiguous 1D array for optimal cache performance.
        return np.ascontiguousarray(self.values[:, idx])

    # --- GET A SNAPSHOT FOR A TARGET DATE

    def get_snapshot(self, target_date: date) -> RiskFactorLevelSnapshot:
        """
        Extract a RiskFactorLevelSnapshot for a specific historical date.

        Filters out any risk factors that have NaN values for the given date.
        """
        from riskquant.utils.dates import date_to_ordinal32

        # 1. Convert target_date to the internal representation
        if np.issubdtype(self.dates.dtype, np.integer):
            target_val = date_to_ordinal32(target_date)
        else:
            target_val = np.datetime64(target_date, "D")

        # 2. Find the exact date index (using binary search since dates are sorted)
        idx = np.searchsorted(self.dates, target_val)

        if idx >= self.dates.size or self.dates[idx] != target_val:
            raise KeyError(
                f"Date {target_date} not found in historical matrix."
            )

        # 3. Extract the 1D cross-section for this date
        levels = self.values[idx, :]

        # 4. Mask out NaNs (factors that didn't exist or lacked data on this date)
        valid_mask = ~np.isnan(levels)

        valid_ids = self.factor_ids[valid_mask]
        valid_levels = levels[valid_mask]

        # Arrays extracted via boolean masks are contiguous, but we pass them
        # straight to the snapshot which will enforce all constraints anyway.
        return RiskFactorLevelSnapshot(
            as_of_date=target_date,
            factor_ids=valid_ids,
            factor_levels=valid_levels,
        )
