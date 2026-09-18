# src/riskquant/market_data/risk_factors/builder_snapshot.py
"""
Edge adapters that turn tabular level inputs into a `RiskFactorLevelSnapshot`.

D069: there is exactly ONE unit-normalisation boundary for level snapshots,
`compile_risk_factor_level_snapshot_from_levels_dict`. Every adapter here only
reshapes its input into a `{relative_name: level}` mapping and delegates, so a
level quoted `PERCENT` becomes the same canonical decimal value whichever
public route ingests it. A separate unit-agnostic builder previously kept the
raw value (18.3 for an 18.3 % volatility) while the compiler scaled it (0.183).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from pathlib import Path

import pandas as pd

from riskquant.config.static_config.static_config import RiskStaticConfig
from riskquant.market_data.risk_factors.compile_snapshot import (
    compile_risk_factor_level_snapshot_from_levels_dict,
)
from riskquant.market_data.risk_factors.snapshot import RiskFactorLevelSnapshot


def build_risk_factor_level_snapshot_from_dict(
    *,
    as_of_date: date,
    levels_dict: Mapping[str, float],
    static_config: RiskStaticConfig,
) -> RiskFactorLevelSnapshot:
    """Build a canonical (unit-normalised, float64) snapshot from `{name: level}`."""
    return compile_risk_factor_level_snapshot_from_levels_dict(
        as_of_date=as_of_date,
        levels_dict=levels_dict,
        static_config=static_config,
    )


def build_risk_factor_level_snapshot_from_df(
    *,
    as_of_date: date,
    df: pd.DataFrame,
    static_config: RiskStaticConfig,
    factor_col: str = "risk_factor_name",
    level_col: str = "level",
) -> RiskFactorLevelSnapshot:
    """
    Build a canonical snapshot from a vertical frame.

    A row with a missing level is an error, not an omission: dropping it would
    let an absent market quote surface later as a generic "missing factor".
    """
    if df.empty:
        raise ValueError("DataFrame is empty.")
    if factor_col not in df.columns or level_col not in df.columns:
        raise ValueError(
            f"DataFrame must contain columns '{factor_col}' and '{level_col}'."
        )
    missing = df[level_col].isna().to_numpy()
    if missing.any():
        bad = df.loc[missing, factor_col].astype(str).tolist()[:10]
        raise ValueError(f"Missing '{level_col}' for risk factors: {bad}")
    if df[factor_col].duplicated().any():
        dup = df.loc[df[factor_col].duplicated(), factor_col].unique().tolist()[:10]
        raise ValueError(f"Duplicate factor names in column '{factor_col}': {dup}")

    levels_dict = dict(zip(df[factor_col].astype(str), df[level_col], strict=True))
    return compile_risk_factor_level_snapshot_from_levels_dict(
        as_of_date=as_of_date,
        levels_dict=levels_dict,
        static_config=static_config,
    )


def build_risk_factor_level_snapshot_from_xlsx(
    *,
    as_of_date: date,
    path: str | Path,
    static_config: RiskStaticConfig,
    sheet_name: str | int = 0,
    factor_col: str = "risk_factor_name",
    level_col: str = "level",
) -> RiskFactorLevelSnapshot:
    """Build a canonical snapshot from a two-column Excel sheet."""
    df = pd.read_excel(str(path), sheet_name=sheet_name)
    return build_risk_factor_level_snapshot_from_df(
        as_of_date=as_of_date,
        df=df,
        static_config=static_config,
        factor_col=factor_col,
        level_col=level_col,
    )
