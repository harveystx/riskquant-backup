# src/riskquant/config/config_utils.py
"""
Utility functions for config loading/validation.
"""

import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, Optional

# ===================== #
# --- SMALL HELPERS --- #
# ===================== #


def _require_cols(df: pd.DataFrame, cols: tuple[str, ...], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} sheet missing required columns: {missing}")


def _build_str_to_int_dict(
    *,
    df: pd.DataFrame,
    key_col: str,
    id_col: str,
    sheet_name: str,
    key_transform: Optional[Callable[[str], str]] = None,
    require_unique_keys: bool = True,
    require_non_null_keys: bool = True,
) -> Dict[str, int]:
    """
    Build Dict[str,int] from df[key_col] -> df[id_col], fail-fast on NaN IDs.

    - key_transform: optional callable applied to key strings (e.g. str.upper)
    - require_unique_keys: if True, fail-fast if key_col has duplicates
    - require_non_null_keys: if True, fail-fast if key_col has NaN/None
    """
    _require_cols(df, (key_col, id_col), sheet_name)

    key_s = df[key_col]
    id_s = df[id_col]

    # Fail-fast on missing keys (recommended for config)
    if require_non_null_keys and key_s.isna().any():
        bad_rows = df.loc[key_s.isna(), :].head(10)
        raise ValueError(
            f"{sheet_name} sheet has missing {key_col} values. Example rows:\n{bad_rows}"
        )

    # Keys as strings (copy=False where possible)
    keys = key_s.astype(str, copy=False)

    # Apply transform if provided
    if key_transform is not None:
        # .map is fine here; config tables are small and it preserves index alignment
        keys = keys.map(key_transform)

    # Fail-fast on NaN IDs
    if id_s.isna().any():
        bad = keys[id_s.isna()].unique().tolist()
        raise ValueError(f"{sheet_name} sheet has missing {id_col} for: {bad}")

    # Duplicate key check
    if require_unique_keys and not keys.is_unique:
        dup = keys[keys.duplicated()].unique().tolist()
        raise ValueError(
            f"{sheet_name} sheet has duplicate {key_col} values: {dup}"
        )

    # Safe integer cast (pandas -> int64), then Python int in dict
    ids = id_s.astype("int64", copy=False)

    return {str(k): int(v) for k, v in zip(keys, ids)}


def _is_na(x: Any) -> bool:
    # Fast NA check for typical pandas scalar values
    return x is None or (isinstance(x, float) and np.isnan(x))
