# riskquant/market_data/risk_factors/matrix.py

# ----------------------------
# 1) Imports: remove unused + add dtype helper
# ----------------------------
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from riskquant.utils.dates import (
    coerce_as_of_date_to_pd_timestamp,
    ordinal32_to_datetime64D,
    datetime64D_to_ordinal32,
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
                "values must be C-contiguous (use np.ascontiguousarray at construction). If you sliced, "
                "call np.ascontiguousarray(...) or use the provided slice_* helpers."
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

    @property
    def index_dates(self) -> pd.DatetimeIndex:
        """Get a DatetimeIndex view of the dates."""
        if np.issubdtype(self.dates.dtype, np.integer):
            dts = ordinal32_to_datetime64D(self.dates)
            return pd.DatetimeIndex(dts)
        else:
            return pd.DatetimeIndex(self.dates)

    @property
    def index_factors(self) -> pd.Index:
        """Get a Index view of the factor_ids."""
        return pd.Index(self.factor_ids, name="factor_id")

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
    def df(self) -> pd.DataFrame:
        """Get the risk factor matrix as a DataFrame."""
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

    # ----------------------------
    # 2) slice_dates: enforce C-contiguous values to satisfy invariant
    # ----------------------------
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

        vals = self.values[start:end, :]
        # Slices can be non-contiguous views; enforce invariant cheaply if needed
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
        """
        Return a sliced matrix along the factor dimension.

        Note: the resulting factor_ids may no longer be sorted depending on factor_indices.
        We rely on __post_init__ to re-canonicalize (sort + permute columns) if needed.
        """
        vals = self.values[:, factor_indices]
        if not vals.flags["C_CONTIGUOUS"]:
            vals = np.ascontiguousarray(vals)

        return RiskFactorMatrix(
            dates=self.dates,
            factor_ids=self.factor_ids[factor_indices],
            values=vals,
        )

    # ----------------------------
    # 4) from_df: accept "datetime-like" indices + enforce sorted/unique cleanly
    # ----------------------------
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

        Notes:
        - Expects df.index to be date-like and already sorted unique.
        - If not, we raise (we do not auto-sort to avoid O(T*N) row copies).
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Expected a DataFrame!")

        # Accept a wider set of date-like index types; coerce if needed.
        if not is_datetime64_any_dtype(df.index):
            # Convert (keeps the same rows; does not sort)
            df = df.copy(deep=False)
            df.index = pd.to_datetime(df.index, errors="raise")

        # Enforce invariants early with clear errors (and avoid needless allocations)
        if df.index.has_duplicates:
            raise ValueError("DataFrame index (dates) must be unique!")
        if not df.index.is_monotonic_increasing:
            raise ValueError("DataFrame index (dates) must be sorted ascending!")

        # Dates (view when possible)
        dts_64D = df.index.to_numpy(dtype="datetime64[D]", copy=False)
        dates = datetime64D_to_ordinal32(dts_64D) if use_int32_dates else dts_64D

        # Columns -> factor_ids: either direct integer parsing or mapping lookup. We validate the mapping dict fully if provided to fail fast on config issues.
        cols = df.columns

        # Numnber of factors (columns); used for preallocating arrays and validating mapping dict if provided.
        n = cols.size

        # Factor IDs: either direct integer parsing or mapping lookup. We validate the mapping dict fully if provided to fail fast on config issues.
        if name_to_id_map is None:
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

            if np.any(factor_ids[1:] == factor_ids[:-1]):
                raise ValueError("factor_ids must be unique!")

            df = df.iloc[:, perm]
        else:
            if n > 1 and np.any(factor_ids[1:] == factor_ids[:-1]):
                raise ValueError("factor_ids must be unique!")

        values = np.ascontiguousarray(
            df.to_numpy(dtype=float_dtype, copy=True, na_value=np.nan)
        )

        return RiskFactorMatrix(
            dates=dates,
            factor_ids=factor_ids,
            values=values,
        )

    # ----------------------------
    # 5) from_xlsx: coerce index to datetime + enforce sorted/unique + use Path typing
    # ----------------------------
    @staticmethod
    def from_xlsx(
        *,
        path: Union[str, Path],
        sheet_name: Union[str, int] = 0,
        name_to_id_map: Optional[Mapping[str, int]] = None,
        use_int32_dates: bool = True,
        float_dtype: type[np.floating] = np.float32,
        as_of_date: Optional[pd.Timestamp] = None,
    ) -> "RiskFactorMatrix":
        """
        Convenience constructor to load a RiskFactorMatrix directly from an Excel file.
        """
        df = pd.read_excel(str(path), sheet_name=sheet_name, index_col=0)

        # Convert index to datetime if needed (common for Excel imports)
        if not is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index, errors="raise")

        # Optional filter (preserves monotonicity if it was monotonic)
        if as_of_date is not None:
            cutoff = coerce_as_of_date_to_pd_timestamp(as_of_date)
            df = df.loc[df.index <= cutoff]

        # Enforce sorted/unique (same contract as from_df)
        if df.index.has_duplicates:
            raise ValueError("Excel sheet index (dates) must be unique!")
        if not df.index.is_monotonic_increasing:
            raise ValueError("Excel sheet index (dates) must be sorted ascending!")

        # Create from the DataFrame using the existing from_df method (handles name_to_id mapping and canonicalization)
        return RiskFactorMatrix.from_df(
            df=df,
            name_to_id_map=name_to_id_map,
            use_int32_dates=use_int32_dates,
            float_dtype=float_dtype,
        )


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
