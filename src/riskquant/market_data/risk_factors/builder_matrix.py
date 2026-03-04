# src\riskquant\market_data\risk_factors\builder_matrix.py
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from riskquant.market_data.risk_factors.builder_utils import (
    _require_mapping_key,
    _validate_mapping_dict,
)
from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.utils.dates import (
    coerce_as_of_date_to_pd_timestamp,
    datetime64D_to_ordinal32,
)

# --- BUILD A RISK FACTOR MATRIX FROM A DATAFRAME


def build_risk_factor_matrix_from_df(
    *,
    df: pd.DataFrame,
    name_to_id_map: Optional[Mapping[str, int]] = None,
    use_int32_dates: bool = True,
    float_dtype: type[np.floating] = np.float32,
) -> RiskFactorMatrix:
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
        raise ValueError("DataFrame index (dates) must be sorted ascending!")

    # Dates: obtain a datetime64[D] view and optionally convert to int32 ordinal
    dts_64D = df.index.to_numpy(dtype="datetime64[D]", copy=False)
    dates = datetime64D_to_ordinal32(dts_64D) if use_int32_dates else dts_64D

    # Columns -> factor_ids: either direct integer parsing or mapping lookup.
    cols = df.columns
    n = cols.size

    if name_to_id_map is None:
        # Parse column labels as integers; will raise on non-integer labels.
        factor_ids = np.fromiter(
            (int(c) for c in cols),
            dtype=np.int32,
            count=n,
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

    return RiskFactorMatrix(dates=dates, factor_ids=factor_ids, values=values)


# --- BUILD A RISK FACTOR MATRIX DIRECTLY FROM AN XLSX FILE


def build_risk_factor_matrix_from_xlsx(
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
    df = pd.read_excel(
        str(path),
        sheet_name=sheet_name,
        index_col=0,
    )

    # If empty.
    if df.empty:
        raise ValueError("dataframe is empty - check: {path}")

    # Convert index to datetime if needed (common for Excel imports).
    if not is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index, errors="raise")

    # Optional filter by as_of_date; preserves monotonicity if present.
    if as_of_date is not None:
        cutoff = coerce_as_of_date_to_pd_timestamp(as_of_date)
        df = df.loc[df.index <= cutoff]

    # Enforce same sorted/unique contract as `from_df`.

    # Duplicated dates.
    if df.index.has_duplicates:
        duplicated = df.index[df.index.duplicated(keep=False)]
        unique_dupes = duplicated.unique()
        sample = unique_dupes[:10]
        raise ValueError(
            f"Excel sheet index (dates) must be unique.\n"
            f"Found {len(unique_dupes)} duplicated date(s).\n"
            f"Sample duplicates: {list(sample)}"
        )

    # Non-monotonic dates
    if not df.index.is_monotonic_increasing:
        raise ValueError("Excel sheet index (dates) must be sorted ascending!")

    # Reuse the robust `from_df` constructor for the remaining work.
    return build_risk_factor_matrix_from_df(
        df=df,
        name_to_id_map=name_to_id_map,
        use_int32_dates=use_int32_dates,
        float_dtype=float_dtype,
    )
