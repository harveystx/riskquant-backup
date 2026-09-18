# src/riskquant/market_data/risk_factors/builder_matrix.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from riskquant.config.static_config.static_config import RiskStaticConfig
from riskquant.market_data.risk_factors.compile_snapshot import canonical_unit_scale
from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.utils.dates import (
    coerce_as_of_date_to_pd_timestamp,
    datetime64D_to_ord32,
)

# --- BUILD A RISK FACTOR MATRIX FROM A DATAFRAME


def build_risk_factor_matrix_from_df(
    *,
    df: pd.DataFrame,
    static_config: RiskStaticConfig,
    use_int32_dates: bool = True,
    float_dtype: type[np.floating] = np.float64,
) -> RiskFactorMatrix:
    """Construct a canonical-unit RiskFactorMatrix from a wide source-unit frame.

    Unit contract (D069 / AUD-06)
    -----------------------------
    This is the history route's normalisation edge. Column values are in each
    factor's DECLARED source `unit_scale` (e.g. PERCENT 18.3). They are
    multiplied once by `canonical_unit_scale` (the same engine as the level
    snapshot compiler), so the matrix holds canonical base units (0.183).
    Every consumer downstream -- scenario generators (DIFFERENCE shocks are
    therefore canonical), `get_snapshot`, `from_snapshot` -- sees canonical
    units and never rescales. A matrix built directly with the
    `RiskFactorMatrix` constructor is canonical by contract.

    Parameters
    ----------
    df : pd.DataFrame
        Wide DataFrame with dates on the index and one column per risk factor,
        labelled by relative risk-factor name, values in declared source units.
    static_config : RiskStaticConfig
        Resolves names to rf_ids (`rel_name_to_rf_id`) and supplies the
        per-rf unit scale. An unknown name or unit slot raises.
    use_int32_dates : bool
        If True convert dates to int32 ordinal representation; otherwise
        keep datetime64[D] dtype.
    float_dtype : numpy floating type
        dtype used for the `values` array (default: np.float64).

        float64 is the default because the downstream historical-shock
        kernel (`simlab.scenarios.historical._kernel_fused_h_day_shock`) is
        dtype-pinned to an F-contiguous float64 `values` array: an h-day
        shock differences two nearby levels, which is a cancelling
        operation that must not lose significance before the result is
        narrowed to the float32 shock buffer. A float32 matrix raises
        `TypeError: No matching definition` from that kernel, so a float32
        default silently produced matrices that could not drive the
        canonical historical generator at all.

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

    # Check for duplicated column labels (factor names); this would cause silent data corruption
    if df.columns.has_duplicates:
        duplicated = df.columns[df.columns.duplicated(keep=False)]
        unique_dupes = duplicated.unique()
        sample = unique_dupes[:10]
        raise ValueError(
            f"DataFrame columns (factor names) must be unique.\n"
            f"Found {len(unique_dupes)} duplicated column(s).\n"
            f"Sample duplicates: {list(sample)}"
        )

    # Dates: obtain a datetime64[D] view and optionally convert to int32 ordinal
    dts_64D = df.index.to_numpy(dtype="datetime64[D]", copy=False)
    dates = datetime64D_to_ord32(dts_64D) if use_int32_dates else dts_64D

    # Columns -> factor_ids: either direct integer parsing or mapping lookup.
    cols = df.columns
    n = cols.size

    name_to_rf_id = static_config.rel_name_to_rf_id
    unknown = [str(c) for c in cols if str(c) not in name_to_rf_id]
    if unknown:
        raise KeyError(
            f"History columns do not resolve through static_config.rel_name_to_rf_id: "
            f"{unknown[:10]}"
        )
    factor_ids = np.fromiter(
        (int(name_to_rf_id[str(c)]) for c in cols), dtype=np.int32, count=n
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

    # D069: one multiplication from declared source units to canonical units,
    # computed in float64 per column (once per factor, not per observation
    # loop), then materialised in the requested dtype.
    scale = canonical_unit_scale(rf_ids=factor_ids, static_config=static_config)
    values = np.ascontiguousarray(
        (df.to_numpy(dtype=np.float64, copy=True, na_value=np.nan) * scale).astype(
            float_dtype, copy=False
        )
    )

    return RiskFactorMatrix(dates=dates, factor_ids=factor_ids, values=values)


# --- BUILD A RISK FACTOR MATRIX DIRECTLY FROM AN XLSX FILE


def build_risk_factor_matrix_from_xlsx(
    *,
    path: str | Path,
    static_config: RiskStaticConfig,
    sheet_name: str | int = 0,
    use_int32_dates: bool = True,
    float_dtype: type[np.floating] = np.float64,
    as_of_date: pd.Timestamp | None = None,
) -> RiskFactorMatrix:
    """Load a RiskFactorMatrix from an Excel workbook/sheet.

    Parameters
    ----------
    path : str | Path
        Path to the Excel file.
    sheet_name : str | int
        Sheet name or index to read (forwarded to pandas.read_excel).
    static_config : RiskStaticConfig
        Name resolution and per-rf unit scale; values are declared source
        units and are normalised exactly once (same contract as `from_df`).
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
        static_config=static_config,
        use_int32_dates=use_int32_dates,
        float_dtype=float_dtype,
    )
