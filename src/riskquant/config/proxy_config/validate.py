# src/riskquant/config/proxy_config/validate.py

import pandas as pd


def validate_proxy_config_df(df: pd.DataFrame) -> None:
    """
    Performs vectorized validation on the raw proxy configuration DataFrame.
    Checks for schema correctness, uniqueness constraints, and logical errors.

    Args:
        df: The active rows of the proxy configuration DataFrame.

    Raises:
        ValueError: If any validation constraints are violated.
    """
    if df.empty:
        return  # An empty proxy map is valid (no proxies applied)

    # 1. Check Required Columns
    required_columns = {"proxying_level", "required_name", "proxy_name"}
    missing_cols = required_columns - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Proxy config validation failed: Missing required columns {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    # 2. Check for Missing/Null Values in Key Columns
    has_nulls = (
        df[["proxying_level", "required_name", "proxy_name"]].isnull().any()
    )
    if has_nulls.any():
        null_cols = has_nulls[has_nulls].index.tolist()
        raise ValueError(
            f"Proxy config validation failed: Found null values in required columns: {null_cols}. "
            "Names and proxying levels cannot be blank."
        )

    # 3. Check Domain Constraints on proxying_level
    valid_levels = {"container", "risk_factor"}
    invalid_levels = df[~df["proxying_level"].isin(valid_levels)]
    if not invalid_levels.empty:
        bad_values = invalid_levels["proxying_level"].unique().tolist()
        raise ValueError(
            f"Proxy config validation failed: Invalid proxying_level(s) found: {bad_values}. "
            f"Must be one of {valid_levels}."
        )

    # 4. Check Uniqueness Constraints
    # A single required_name cannot be proxied to multiple different proxies within the same level
    duplicates = df[
        df.duplicated(subset=["proxying_level", "required_name"], keep=False)
    ]
    if not duplicates.empty:
        raise ValueError(
            "Proxy config validation failed: Found duplicate mapping rules for the same required_name. "
            f"Conflicting rows:\n{duplicates[['proxying_level', 'required_name', 'proxy_name']].to_string(index=False)}"
        )

    # 5. Check Logical Constraints (Self-Referencing)
    # A proxy cannot proxy to itself (this would create a pointless loop)
    self_referencing = df[df["required_name"] == df["proxy_name"]]
    if not self_referencing.empty:
        raise ValueError(
            "Proxy config validation failed: A required_name cannot proxy to itself. "
            f"Invalid rows:\n{self_referencing[['proxying_level', 'required_name']].to_string(index=False)}"
        )

    # 6. Required names should not be duplicated across levels (e.g. a container and a risk factor with the same name)
    name_counts = df["required_name"].value_counts()
    duplicated_names = name_counts[name_counts > 1].index.tolist()
    if duplicated_names:
        raise ValueError(
            "Proxy config validation failed: Found required_name(s) duplicated across proxying levels. "
            f"Duplicated names: {duplicated_names}. Required names must be unique across the entire config."
        )
