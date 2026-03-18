# src/riskquant/market_data/risk_factors/builder_snapshot.py
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict, Mapping, Optional, Union

import numpy as np
import pandas as pd

from riskquant.market_data.risk_factors.snapshot import RiskFactorLevelSnapshot

from riskquant.market_data.risk_factors.builder_utils import (
    _require_mapping_key,
    _validate_mapping_dict,
)


def build_risk_factor_level_snapshot_from_dict(
    *,
    as_of_date: date,
    levels_dict: Mapping[str, float],
    name_to_id_map: Optional[Mapping[str, int]] = None,
    float_dtype: type[np.floating] = np.float32,
) -> RiskFactorLevelSnapshot:
    """
    Build a snapshot from a dictionary mapping risk factor names to their levels.
    """
    if not levels_dict:
        raise ValueError("levels_dict cannot be empty.")

    n = len(levels_dict)

    if name_to_id_map is None:
        # Assume the keys in the dictionary are already integer IDs formatted as strings.
        ids = np.fromiter(
            (int(k) for k in levels_dict.keys()), dtype=np.int32, count=n
        )
    else:
        # Resolve names to IDs
        _validate_mapping_dict(name_to_id_map)
        ids = np.fromiter(
            (
                _require_mapping_key(name_to_id_map, str(k))
                for k in levels_dict.keys()
            ),
            dtype=np.int32,
            count=n,
        )

    # Extract levels into a contiguous array
    levels = np.fromiter(levels_dict.values(), dtype=float_dtype, count=n)

    return RiskFactorLevelSnapshot(
        as_of_date=as_of_date,
        factor_ids=ids,
        factor_levels=levels,
    )


def build_risk_factor_level_snapshot_from_df(
    *,
    as_of_date: date,
    df: pd.DataFrame,
    factor_col: str = "risk_factor_name",
    level_col: str = "level",
    name_to_id_map: Optional[Mapping[str, int]] = None,
    float_dtype: type[np.floating] = np.float32,
) -> RiskFactorLevelSnapshot:
    """
    Build a snapshot from a vertical pandas DataFrame.
    """
    if df.empty:
        raise ValueError("DataFrame is empty.")

    if factor_col not in df.columns or level_col not in df.columns:
        raise ValueError(
            f"DataFrame must contain columns '{factor_col}' and '{level_col}'."
        )

    # Drop any rows where the level is NaN
    df_clean = df.dropna(subset=[level_col])

    # There must not be duplicate factor names after dropping NaNs, as this would lead to ambiguity in the snapshot.
    if df_clean[factor_col].duplicated().any():
        duplicated_factors = df_clean[factor_col][
            df_clean[factor_col].duplicated()
        ].unique()
        raise ValueError(
            f"Duplicate factor names found in column '{factor_col}' after dropping NaNs: {duplicated_factors}"
        )

    # Extract as a dictionary to reuse the highly-optimized dictionary builder
    # zip is faster than df.set_index().to_dict() or df.itertuples()
    levels_dict = dict(
        zip(df_clean[factor_col].astype(str), df_clean[level_col])
    )

    return build_risk_factor_level_snapshot_from_dict(
        as_of_date=as_of_date,
        levels_dict=levels_dict,
        name_to_id_map=name_to_id_map,
        float_dtype=float_dtype,
    )


def build_risk_factor_level_snapshot_from_xlsx(
    *,
    as_of_date: date,
    path: Union[str, Path],
    sheet_name: Union[str, int] = 0,
    factor_col: str = "risk_factor_name",
    level_col: str = "level",
    name_to_id_map: Optional[Mapping[str, int]] = None,
    float_dtype: type[np.floating] = np.float32,
) -> RiskFactorLevelSnapshot:
    """
    Build a snapshot from a two-column Excel sheet.
    """
    df = pd.read_excel(str(path), sheet_name=sheet_name)

    return build_risk_factor_level_snapshot_from_df(
        as_of_date=as_of_date,
        df=df,
        factor_col=factor_col,
        level_col=level_col,
        name_to_id_map=name_to_id_map,
        float_dtype=float_dtype,
    )
