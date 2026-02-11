# riskquant/market_data/risk_factors/matrix.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from riskquant.utils.dates import ordinal32_to_datetime64D, datetime64D_to_ordinal32


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
        # ---- Basic dimensionality checks (cheap) ----
        if self.dates.ndim != 1:
            raise ValueError("dates must be 1D!")
        if self.factor_ids.ndim != 1:
            raise ValueError("factor_ids must be 1D!")
        if self.values.ndim != 2:
            raise ValueError("values must be 2D!")

        # ---- dtype invariants ----
        if not np.issubdtype(self.values.dtype, np.floating):
            raise TypeError(f"values must be floating dtype; got {self.values.dtype}")

        # Dates encoding
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

        if self.factor_ids.dtype != np.int32:
            raise TypeError(f"factor_ids must be int32; got {self.factor_ids.dtype}")

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
        if not self.values.flags["C_CONTIGUOUS"]:
            raise ValueError(
                "values must be C-contiguous (use np.ascontiguousarray at construction)."
            )

        # ---- dates: require sorted unique (alloc-free checks) ----
        if T > 1:
            if np.any(self.dates[1:] < self.dates[:-1]):
                raise ValueError("dates must be sorted ascending.")
            if np.any(self.dates[1:] == self.dates[:-1]):
                raise ValueError("dates must be unique!")

        # ---- factor_ids: canonicalize to sorted unique with minimal extra work ----
        if N > 1:
            # If already sorted, uniqueness check is alloc-free.
            if not np.any(self.factor_ids[1:] < self.factor_ids[:-1]):
                if np.any(self.factor_ids[1:] == self.factor_ids[:-1]):
                    raise ValueError("factor_ids must be unique!")
                return

            # Not sorted: build a permutation (O(N) memory), then reorder ids and values.
            perm = np.argsort(
                self.factor_ids, kind="mergesort"
            )  # stable, deterministic
            sorted_ids = self.factor_ids[perm]

            if np.any(sorted_ids[1:] == sorted_ids[:-1]):
                raise ValueError("factor_ids must be unique!")

            # Reorder columns; enforce C-contiguity after fancy indexing copy.
            sorted_vals = np.ascontiguousarray(self.values[:, perm])

            object.__setattr__(self, "factor_ids", sorted_ids)
            object.__setattr__(self, "values", sorted_vals)

    # -------------------- Convenience --------------------

    def to_df(self) -> pd.DataFrame:
        """Convert to wide DataFrame with dates as index and factor_ids as columns."""
        if np.issubdtype(self.dates.dtype, np.integer):
            dts = ordinal32_to_datetime64D(self.dates)
            index = pd.DatetimeIndex(dts)
        else:
            index = pd.DatetimeIndex(self.dates)

        df = pd.DataFrame(data=self.values, index=index, columns=self.factor_ids)
        df.index.name = "date"
        df.columns.name = "factor_id"
        return df

    def head(self, n: int = 5) -> pd.DataFrame:
        return self.to_df().head(n=n)

    def tail(self, n: int = 5) -> pd.DataFrame:
        return self.to_df().tail(n=n)

    def plot(self, **kwargs):
        """Plot the risk factor matrix."""
        return self.to_df().plot(**kwargs)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    @property
    def num_dates(self) -> int:
        return self.values.shape[0]

    @property
    def num_factors(self) -> int:
        return self.values.shape[1]

    def slice_dates(
        self,
        start: int | None = None,
        end: int | None = None,
    ) -> "RiskFactorMatrix":
        """Return a sliced matrix along the time dimension (dates remain sorted)."""
        if start is None:
            start = 0
        if end is None:
            end = self.num_dates

        return RiskFactorMatrix(
            dates=self.dates[start:end],
            factor_ids=self.factor_ids,
            values=self.values[start:end, :],
        )

    def slice_factors(self, factor_indices: np.ndarray) -> "RiskFactorMatrix":
        """
        Return a sliced matrix along the factor dimension.

        Note: the resulting factor_ids may no longer be sorted depending on factor_indices.
        We rely on __post_init__ to re-canonicalize (sort + permute columns) if needed.
        """
        return RiskFactorMatrix(
            dates=self.dates,
            factor_ids=self.factor_ids[factor_indices],
            values=self.values[:, factor_indices],
        )

    @staticmethod
    def from_df(
        *,
        df: pd.DataFrame,
        name_to_id_map: Optional[Mapping[str, int]] = None,
        use_int32_dates: bool = True,
        float_dtype: type[np.floating] = np.float32,
    ) -> "RiskFactorMatrix":
        """
        Construct a RiskFactorMatrix from a wide DataFrame.

        Efficiency:
          - values allocated once (df -> contiguous numpy array)
          - factor axis canonicalized HERE (to avoid an extra full matrix copy in __post_init__)
          - dates extracted as a view; optional ordinal conversion (may allocate)

        Notes:
          - This expects df.index to be date-like and already sorted unique.
          - If not, the constructor will raise.
        """
        # Dates (view when possible)
        dts_64D = df.index.to_numpy(dtype="datetime64[D]", copy=False)
        dates = datetime64D_to_ordinal32(dts_64D) if use_int32_dates else dts_64D

        cols = df.columns
        n = cols.size

        if name_to_id_map is None:
            # Columns already are IDs (int-like)
            factor_ids = np.fromiter((int(c) for c in cols), dtype=np.int32, count=n)
        else:
            _validate_mapping_dict(name_to_id_map)
            factor_ids = np.fromiter(
                (_require_mapping_key(name_to_id_map, str(c)) for c in cols),
                dtype=np.int32,
                count=n,
            )

        # Canonicalize factor axis BEFORE allocating values (best total cost)
        if n > 1 and np.any(factor_ids[1:] < factor_ids[:-1]):
            perm = np.argsort(factor_ids, kind="mergesort")
            factor_ids = factor_ids[perm]

            # Detect duplicates early (no need to allocate values if invalid)
            if np.any(factor_ids[1:] == factor_ids[:-1]):
                raise ValueError("factor_ids must be unique!")

            # Reorder df columns to match canonical factor order, then allocate values once
            df = df.iloc[:, perm]
        else:
            # Already sorted: quick duplicate check
            if n > 1 and np.any(factor_ids[1:] == factor_ids[:-1]):
                raise ValueError("factor_ids must be unique!")

        # Values: one tight contiguous allocation
        values = np.ascontiguousarray(
            df.to_numpy(dtype=float_dtype, copy=True, na_value=np.nan)
        )

        # Return RFM
        return RiskFactorMatrix(
            dates=dates,
            factor_ids=factor_ids,
            values=values,
        )


# @dataclass(frozen=True, slots=True)
# class RiskFactorMatrix:
#     dates: np.ndarray  # datetime64[D] or int32 ordinal
#     factor_ids: np.ndarray  # int32, shape (N,)
#     values: np.ndarray  # float32/float64, shape (T, N), contiguous
#
#     def __post_init__(self):
#         # Dim
#         if self.dates.ndim != 1:
#             raise ValueError("dates must be 1D!")
#         if self.factor_ids.ndim != 1:
#             raise ValueError("factor_ids must be 1D!")
#         if self.values.ndim != 2:
#             raise ValueError("values must be 2D!")
#
#         # Enforce floating dtype invariant
#         if not np.issubdtype(self.values.dtype, np.floating):
#             raise TypeError(f"values must be floating dtype; got {self.values.dtype}")
#
#         # Shapes
#         T, N = self.values.shape
#         if self.dates.size != T:
#             raise ValueError(
#                 f"dates length ({self.dates.size}) must match values.shape[0] ({T})"
#             )
#         if self.factor_ids.size != N:
#             raise ValueError(
#                 f"factor_ids length ({self.factor_ids.size}) must match values.shape[1] ({N})"
#             )
#
#         # Enforce canonical date encodings
#         if np.issubdtype(self.dates.dtype, np.integer):
#             if self.dates.dtype != np.int32:
#                 raise TypeError(
#                     f"int date encoding must be int32 days since DATE_EPOCH; got {self.dates.dtype}"
#                 )
#         else:
#             if self.dates.dtype != np.dtype("datetime64[D]"):
#                 raise TypeError(
#                     f"datetime date encoding must be datetime64[D]; got {self.dates.dtype}"
#                 )
#
#         if self.factor_ids.dtype != np.int32:
#             raise TypeError(f"factor_ids must be int32; got {self.factor_ids.dtype}")
#
#         # Contiguity (you want dense contiguous arrays for VaR)
#         if not self.values.flags["C_CONTIGUOUS"]:
#             raise ValueError(
#                 "values must be C-contiguous (use np.ascontiguousarray at construction)."
#             )
#
#         # Uniqueness with minimal allocations:
#         # - For dates: if you expect sorted input (typical), adjacent compare is O(T) and alloc-free.
#         #   If you cannot guarantee sortedness, keep your sort+adjacent approach instead.
#         if self.dates.size > 1:
#             # enforce monotonic non-decreasing
#             if np.any(self.dates[1:] < self.dates[:-1]):
#                 raise ValueError("dates must be sorted ascending.")
#             # then uniqueness is equivalent to “no equal adjacent pairs”
#             if np.any(self.dates[1:] == self.dates[:-1]):
#                 raise ValueError("dates must be unique!")
#
#         # - For factor_ids: not necessarily sorted; use unique counts.
#         if self.factor_ids.size > 1:
#             ids = np.sort(self.factor_ids)
#             if np.any(ids[1:] == ids[:-1]):
#                 raise ValueError("factor_ids must be unique!")
#
#     def to_df(self) -> pd.DataFrame:
#         """Convert to pandas DataFrame with dates as index and factor_ids as columns."""
#         if np.issubdtype(self.dates.dtype, np.integer):
#             dts = ordinal32_to_datetime64D(self.dates)
#             index = pd.DatetimeIndex(dts)
#         else:
#             index = pd.DatetimeIndex(self.dates)
#
#         df = pd.DataFrame(data=self.values, index=index, columns=self.factor_ids)
#         df.index.name = "date"
#         df.columns.name = "factor_id"
#         return df
#
#     def plot(self, **kwargs):
#         """Plot the risk factor matrix."""
#         return self.to_df().plot(**kwargs)
#
#     @property
#     def shape(self) -> tuple:
#         """Return (T, N) shape of the matrix."""
#         return self.values.shape
#
#     @property
#     def num_dates(self) -> int:
#         """Number of time periods."""
#         return self.values.shape[0]
#
#     @property
#     def num_factors(self) -> int:
#         """Number of factors."""
#         return self.values.shape[1]
#
#     def slice_dates(self, start: int = None, end: int = None) -> "RiskFactorMatrix":
#         """Return a sliced matrix along the time dimension."""
#         if start is None:
#             start = 0
#         if end is None:
#             end = self.num_dates
#
#         return RiskFactorMatrix(
#             dates=self.dates[start:end],
#             factor_ids=self.factor_ids,
#             values=self.values[start:end, :],
#         )
#
#     def slice_factors(self, factor_indices: np.ndarray) -> "RiskFactorMatrix":
#         """Return a sliced matrix along the factor dimension."""
#         return RiskFactorMatrix(
#             dates=self.dates,
#             factor_ids=self.factor_ids[factor_indices],
#             values=self.values[:, factor_indices],
#         )
#
#     @staticmethod
#     def from_df(
#         *,
#         df: pd.DataFrame,
#         name_to_id_map: Optional[Mapping[str, int]] = None,
#         use_int32_dates: bool = True,
#         float_dtype: type[np.floating] = np.float32,
#     ) -> "RiskFactorMatrix":
#         """
#         Construct a RiskFactorMatrix from a wide DataFrame.
#
#         Modes:
#           1) name_to_id_map is provided -> df.columns are names; map to int32 factor_ids
#           2) name_to_id_map is None     -> df.columns are already factor IDs (int-like)
#
#         Memory/CPU:
#           - one unavoidable allocation for values (df -> contiguous numpy array)
#           - dates taken as numpy view; optional ordinal conversion
#         """
#         # Dates (alloc-free view from index)
#         dts_64D = df.index.to_numpy(dtype="datetime64[D]", copy=False)
#         dates = datetime64D_to_ordinal32(dts_64D) if use_int32_dates else dts_64D
#
#         # Values: one tight contiguous allocation
#         values = np.ascontiguousarray(
#             df.to_numpy(dtype=float_dtype, copy=True, na_value=np.nan)
#         )
#
#         # Factor IDs
#         cols = df.columns
#         n = cols.size
#
#         if name_to_id_map is None:
#             # Columns already are IDs (int-like). Build int32 array once.
#             factor_ids = np.fromiter((int(c) for c in cols), dtype=np.int32, count=n)
#
#             # Construct and return.
#             return RiskFactorMatrix(dates=dates, factor_ids=factor_ids, values=values)
#
#         # Columns are names; map to IDs
#         _validate_mapping_dict(name_to_id_map)
#
#         # Map factor names to factor IDs.
#         factor_ids = np.fromiter(
#             (_require_mapping_key(name_to_id_map, str(c)) for c in cols),
#             dtype=np.int32,
#             count=n,
#         )
#
#         # Construct and return.
#         return RiskFactorMatrix(dates=dates, factor_ids=factor_ids, values=values)


def _validate_mapping_dict(m: Mapping[str, Any]) -> None:
    """
    Validate mapping dict for:
      - non-empty string keys
      - integer-like values
      - no duplicate IDs
    Designed for small dictionaries; called once at load time, not per run-loop.
    """
    # Empty/blank key detection (single pass)
    for k in m.keys():
        if not isinstance(k, str) or not k.strip():
            raise ValueError(f"Mapping contains invalid/blank key: {k!r}")

    seen: dict[int, str] = {}
    for k, v in m.items():
        # accept numpy int, Python int; reject floats/strings
        if isinstance(v, (bool,)):  # bool is subclass of int; exclude
            raise ValueError(
                f"Mapping value for '{k}' must be an int id, got bool={v!r}"
            )
        if not isinstance(v, (int, np.integer)):
            raise ValueError(
                f"Mapping value for '{k}' must be an int id, got {type(v).__name__}={v!r}"
            )

        iv = int(v)
        prev = seen.get(iv)
        if prev is not None:
            raise ValueError(
                "Mapping contains duplicate ID assignment:\n"
                f"  '{prev}' and '{k}' both map to ID={iv}\n"
                "Each risk factor must have a unique risk_factor_name_id."
            )
        seen[iv] = k


def _require_mapping_key(m: Mapping[str, Any], key: str, *, preview_n: int = 25) -> Any:
    """
    Fetch a key with explicit diagnostics on failure.

    Memory efficient:
      - normal path: single hash lookup
      - error path: only then materializes a small preview list of keys
    """
    try:
        return m[key]
    except KeyError:
        # Provide a bounded preview to avoid huge error strings
        keys = list(m.keys())
        n = len(keys)
        preview = keys[:preview_n]
        preview_str = ", ".join(preview) + (" ..." if n > preview_n else "")

        raise ValueError(
            "Risk factor ID mapping lookup failed.\n"
            f"Missing key: '{key}'\n\n"
            "Fix:\n"
            "  Add this risk factor to the STATIC config table that defines risk_factor_name_id.\n"
            "  Example row to add:\n"
            f"    risk_factor_name = {key}\n"
            "    risk_factor_name_id = <new unique integer>\n\n"
            f"Known keys preview ({min(preview_n, n)} of {n}):\n"
            f"  {preview_str}\n"
        ) from None


if __name__ == "__main__":
    # Build a tiny example (3 dates x 4 factors)
    df_test = pd.DataFrame(
        data=[
            [1.0, 2.0, 3.0, 4.0],
            [1.1, 2.1, 3.1, 4.1],
            [1.2, 2.2, 3.2, 4.2],
        ],
        index=pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-04"]),
        columns=[10, 20, 30, 40],
        dtype=np.float32,
    )

    # Absolute name -> id mapping for test
    name_map = {"10": 1, "20": 2, "30": 3, "40": 4}

    # Compile into matrix
    m = RiskFactorMatrix.from_df(df=df_test, name_to_id_map=name_map)

    print("Matrix shape:", m.shape)
    print(m.to_df())
