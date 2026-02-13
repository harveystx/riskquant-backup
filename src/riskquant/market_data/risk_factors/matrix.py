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
        """Construct a RiskFactorMatrix from a wide pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Wide DataFrame with dates on the index and one column per risk factor.
        name_to_id_map : Optional[Mapping[str, int]]
            Optional mapping from column name (string) to integer factor id. If
            omitted, columns are parsed as integers directly.
        use_int32_dates : bool
            If True convert dates to int32 ordinal representation; otherwise
            keep datetime64[D] dtype.
        float_dtype : numpy floating type
            dtype used for the `values` array (default: np.float32).

        Returns
        -------
        RiskFactorMatrix
            Canonicalized matrix with validated shapes, dtypes and contiguity.

        Raises
        ------
        ValueError
            If the index is not date-like, not monotonic increasing, or contains
            duplicates; or if factor id mapping contains errors.

        Notes
        -----
        The function intentionally does not auto-sort the DataFrame rows
        (dates). If the provided index is unsorted we raise an error because
        auto-sorting would require an expensive O(T*N) copy of the values.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Expected a DataFrame!")

        # Accept a wider set of date-like index types; coerce if needed.
        if not is_datetime64_any_dtype(df.index):
            # Convert in-place where possible; this preserves row order and
            # avoids unnecessary copying of the data matrix.
            df = df.copy(deep=False)
            df.index = pd.to_datetime(df.index, errors="raise")

        # Enforce invariants early with clear errors (and avoid needless allocations)
        if df.index.has_duplicates:
            raise ValueError("DataFrame index (dates) must be unique!")
        if not df.index.is_monotonic_increasing:
            raise ValueError(
                "DataFrame index (dates) must be sorted ascending!"
            )

        # Dates: obtain a datetime64[D] view and optionally convert to int32 ordinal
        dts_64D = df.index.to_numpy(dtype="datetime64[D]", copy=False)
        dates = (
            datetime64D_to_ordinal32(dts_64D) if use_int32_dates else dts_64D
        )

        # Columns -> factor_ids: either direct integer parsing or mapping lookup.
        cols = df.columns
        n = cols.size

        if name_to_id_map is None:
            # Parse column labels as integers; will raise on non-integer labels.
            factor_ids = np.fromiter(
                (int(c) for c in cols), dtype=np.int32, count=n
            )
        else:
            # Validate mapping once (cheap for small config dicts) and then map.
            _validate_mapping_dict(name_to_id_map)
            factor_ids = np.fromiter(
                (_require_mapping_key(name_to_id_map, str(c)) for c in cols),
                dtype=np.int32,
                count=n,
            )

        # Canonicalize factor axis BEFORE allocating `values` (avoids extra copies).
        if n > 1 and np.any(factor_ids[1:] < factor_ids[:-1]):
            perm = np.argsort(factor_ids, kind="mergesort")
            factor_ids = factor_ids[perm]

            if np.any(factor_ids[1:] == factor_ids[:-1]):
                raise ValueError("factor_ids must be unique!")

            # Reorder the DataFrame columns to match the canonical factor order.
            df = df.iloc[:, perm]
        else:
            if n > 1 and np.any(factor_ids[1:] == factor_ids[:-1]):
                raise ValueError("factor_ids must be unique!")

        # Materialize the numeric matrix as a C-contiguous array with requested dtype.
        values = np.ascontiguousarray(
            df.to_numpy(dtype=float_dtype, copy=True, na_value=np.nan)
        )

        return RiskFactorMatrix(
            dates=dates, factor_ids=factor_ids, values=values
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
        """Load a RiskFactorMatrix from an Excel workbook/sheet.

        Parameters
        ----------
        path : str | Path
            Path to the Excel file.
        sheet_name : str | int
            Sheet name or index to read (forwarded to pandas.read_excel).
        name_to_id_map : Optional[Mapping[str, int]]
            Optional mapping for column name -> factor id (same semantics as
            `from_df`).
        use_int32_dates : bool
            If True convert dates to int32 ordinal representation.
        float_dtype : numpy floating type
            dtype used for the `values` array.
        as_of_date : Optional[pd.Timestamp]
            If provided, rows with date > as_of_date are filtered out.

        Returns
        -------
        RiskFactorMatrix
            Canonicalized matrix constructed from the Excel sheet.
        """
        # Read sheet with the first column treated as the index (dates expected).
        df = pd.read_excel(str(path), sheet_name=sheet_name, index_col=0)

        # Convert index to datetime if needed (common for Excel imports).
        if not is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index, errors="raise")

        # Optional filter by as_of_date; preserves monotonicity if present.
        if as_of_date is not None:
            cutoff = coerce_as_of_date_to_pd_timestamp(as_of_date)
            df = df.loc[df.index <= cutoff]

        # Enforce same sorted/unique contract as `from_df`.
        if df.index.has_duplicates:
            raise ValueError("Excel sheet index (dates) must be unique!")
        if not df.index.is_monotonic_increasing:
            raise ValueError(
                "Excel sheet index (dates) must be sorted ascending!"
            )

        # Reuse the robust `from_df` constructor for the remaining work.
        return RiskFactorMatrix.from_df(
            df=df,
            name_to_id_map=name_to_id_map,
            use_int32_dates=use_int32_dates,
            float_dtype=float_dtype,
        )


def _validate_mapping_dict(m: Mapping[str, Any]) -> None:
    """Validate a small static mapping of risk factor name -> integer id.

    Checks performed:
      - keys must be non-empty strings
      - values must be integer-like (reject bool, float, str)
      - no two keys map to the same integer id

    This helper is optimized for configuration-time validation (small dicts),
    not for per-row lookups in tight loops.
    """
    # Empty/blank key detection (single pass)
    for k in m.keys():
        if not isinstance(k, str) or not k.strip():
            raise ValueError(f"Mapping contains invalid/blank key: {k!r}")

    # Ensure values are integer-like and detect duplicates by tracking seen ids.
    seen: dict[int, str] = {}
    for k, v in m.items():
        # `bool` is a subclass of int in Python; explicitly reject it here because
        # booleans are not valid factor ids.
        if isinstance(v, (bool,)):
            raise ValueError(
                f"Mapping value for '{k}' must be an int id, got bool={v!r}"
            )

        # Accept Python ints and NumPy integer scalars; reject other types.
        if not isinstance(v, (int, np.integer)):
            raise ValueError(
                f"Mapping value for '{k}' must be an int id, got {type(v).__name__}={v!r}"
            )

        iv = int(v)
        prev = seen.get(iv)
        if prev is not None:
            # Clear, actionable error message when multiple names map to same id.
            raise ValueError(
                "Mapping contains duplicate ID assignment:\n"
                f"  '{prev}' and '{k}' both map to ID={iv}\n"
                "Each risk factor must have a unique risk_factor_name_id."
            )
        seen[iv] = k


def _require_mapping_key(
    m: Mapping[str, Any], key: str, *, preview_n: int = 25
) -> Any:
    """Lookup `key` in mapping `m` and raise a detailed error if missing.

    This helper provides user-friendly diagnostics on configuration errors by
    showing a short preview of known keys and a suggested fix. Designed to be
    inexpensive on the success path (single hash lookup); the preview list is
    materialized only on error.

    Parameters
    ----------
    m : Mapping[str, Any]
        Mapping from risk factor name to id.
    key : str
        Key to lookup (column label converted to string prior to lookup).
    preview_n : int
        Maximum number of known keys to show in the error message.
    """
    try:
        return m[key]
    except KeyError:
        # Materialize a small preview of keys only when diagnosing a failure --
        # prevents huge error messages for very large config maps.
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
