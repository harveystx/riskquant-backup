# src/riskquant/config/validate.py
"""
High-performance vectorized validation for risk static config data.

Design goals:
  - Fail-fast before memory allocation of runtime objects.
  - Zero native Python loops over dataframe rows.
  - Utilize NumPy/Pandas C-level vectorization for constraints (Unique, FK).
"""

from typing import Dict

import numpy as np
import pandas as pd

from riskquant.market_data.curves.extrap_methods import EXTRAP_METHOD_NAMES
from riskquant.market_data.curves.interp_methods import INTERP_METHOD_NAMES


def _validate_risk_factor_tokens(dict_of_dfs: dict[str, pd.DataFrame]) -> None:
    """
    Validates that every tenor_token in the risk_factors sheet strictly conforms
    to the token_regex defined for its specific tenor_type_name in the schema.
    """
    rf_df = dict_of_dfs.get("risk_factors")
    cm_df = dict_of_dfs.get("container_meta")
    ts_df = dict_of_dfs.get("tenor_schema")

    if any(df is None or df.empty for df in (rf_df, cm_df, ts_df)):
        return

    # 1. Filter out scalars/FX spot (which legitimately have no tenor_token)
    tokens_df = rf_df[rf_df["tenor_token"].notna()].copy()
    if tokens_df.empty:
        return

    # 2. Join the hierarchy to figure out which regex each token SHOULD use
    # container_name -> tenor_type_name
    cm_mapping = cm_df.set_index("container_name")["tenor_type_name"]
    tokens_df["tenor_type_name"] = tokens_df["container_name"].map(cm_mapping)

    # tenor_type_name -> token_regex
    ts_mapping = ts_df.set_index("tenor_type_name")["token_regex"]
    tokens_df["expected_regex"] = tokens_df["tenor_type_name"].map(ts_mapping)

    # print(tokens_df.to_string())

    # 3. Vectorized validation using groupby
    # We group by the regex pattern so we can use Pandas' fast vectorized C-level string matching
    valid_mask = pd.Series(False, index=tokens_df.index)

    for regex_pattern, group in tokens_df.groupby(
        "expected_regex", dropna=False
    ):
        if pd.isna(regex_pattern):
            # If a container maps to a tenor_type_name that has no regex, mark as invalid
            continue

        # Apply the regex to the entire group at once
        valid_mask.loc[group.index] = (
            group["tenor_token"].astype(str).str.match(str(regex_pattern))
        )

    # 4. Catch violators and fail fast
    if not valid_mask.all():
        violators_df = tokens_df[~valid_mask]

        # # Format a clean error message showing exactly what failed
        # sample = (
        #     violators_df[
        #         [
        #             "risk_factor_name",
        #             "tenor_token",
        #             "tenor_type_name",
        #             "expected_regex",
        #         ]
        #     ]
        #     .to_dict("records")
        # )

        raise ValueError(
            f"Config Validation Failed: Found risk factors with invalid tenor_tokens.\n"
            f"Tokens must strictly match the token_regex defined in the tenor_schema.\n"
            f"Sample violators:\n"
            f"{violators_df.to_string(index=False)}"
        )


def _validate_container_meta(dict_of_dfs):
    # Extract the curve schema dataframe
    container_meta_df = dict_of_dfs.get("container_meta")

    # If empty.
    if container_meta_df is None or container_meta_df.empty:
        return

    # ---> INTERP METHODS <--- #

    required_cols = ["interp_method_name_x", "interp_method_name_y"]
    missing_cols = [
        c for c in required_cols if c not in container_meta_df.columns
    ]
    if missing_cols:
        raise ValueError(
            f"container_meta missing required columns: {missing_cols}"
        )

    # Normalise
    interp_df = (
        container_meta_df[required_cols]
        .astype(str)
        .apply(lambda col: col.str.strip().str.upper())
    )

    # Flatten unique values across both axes
    unique_methods = set(interp_df.values.ravel())

    # Remove blanks / null-like artifacts
    unique_methods = {
        m
        for m in unique_methods
        if m
        not in {
            "",
            "NONE",
            "NAN",
            "NA",
            "N/A",
            "<NA>",
            "NULL",
        }
    }

    # Define allowed methods (ideally centralised elsewhere)
    VALID_INTERP_METHODS = set(INTERP_METHOD_NAMES)

    # Validate
    invalid_methods = unique_methods - VALID_INTERP_METHODS
    if invalid_methods:
        raise ValueError(
            f"Invalid interpolation methods found: {sorted(invalid_methods)}. "
            f"Allowed methods: {sorted(VALID_INTERP_METHODS)}"
        )

    # ---> EXTRAP METHODS <--- #

    required_cols = ["extrap_method_name_x", "extrap_method_name_y"]
    missing_cols = [
        c for c in required_cols if c not in container_meta_df.columns
    ]
    if missing_cols:
        raise ValueError(
            f"container_meta missing required columns: {missing_cols}"
        )

    # Normalise
    extrap_df = (
        container_meta_df[required_cols]
        .astype(str)
        .apply(lambda col: col.str.strip().str.upper())
    )

    # Flatten unique values across both axes
    unique_methods = set(extrap_df.values.ravel())

    # Remove blanks / null-like artifacts
    unique_methods = {
        m
        for m in unique_methods
        if m
        not in {
            "",
            "NONE",
            "NAN",
            "NA",
            "N/A",
            "<NA>",
            "NULL",
        }
    }

    # Define allowed methods (ideally centralised elsewhere)
    VALID_EXTRAP_METHODS = set(EXTRAP_METHOD_NAMES)

    # Validate
    invalid_methods = unique_methods - VALID_EXTRAP_METHODS
    if invalid_methods:
        raise ValueError(
            f"Invalid extrapolation methods found: {sorted(invalid_methods)}. "
            f"Allowed methods: {sorted(VALID_EXTRAP_METHODS)}"
        )

    return


def validate_static_config_dfs(dict_of_dfs: Dict[str, pd.DataFrame]) -> None:
    """
    Executes vectorized schema checks (uniqueness and foreign keys) across the
    raw configuration dataframes.

    Args:
        dict_of_dfs: The raw loaded dataframes mapped by sheet name.

    Raises:
        ValueError: On the first violation of schema constraints.
    """

    # --------------------------------------------------------- #
    # 1. UNIQUENESS CONSTRAINTS (Primary Keys / Alt Keys)
    # --------------------------------------------------------- #
    # Structure: sheet_name -> tuple of column names that must be unique
    unique_constraints = {
        "container_meta": ("container_name", "container_name_id"),
        "risk_factors": ("risk_factor_name", "risk_factor_name_id"),
        "asset_classes": ("asset_class", "asset_class_id"),
        "asset_subclasses": ("asset_subclass", "asset_subclass_id"),
        "unit_types": ("unit_type_name", "unit_type_name_id"),
        "unit_scales": ("unit_scale_name", "unit_scale_name_id"),
        "ccy": ("ccy", "ccy_id"),
        "shift_types": ("shift_type", "shift_type_id"),
        # "container_proxies": ("container_name",),
        # "risk_factor_proxies": ("risk_factor_name",),
    }

    for sheet_name, cols in unique_constraints.items():
        df = dict_of_dfs.get(sheet_name)
        if df is None or df.empty:
            continue

        for col in cols:
            if col not in df.columns:
                continue  # Missing columns handled separately if strictly required

            # .is_unique is an O(N) C-level check.
            if not df[col].is_unique:
                # Only pay the cost of finding the exact duplicates if we fail
                duplicates = df[col][df[col].duplicated()].unique()
                raise ValueError(
                    f"Validation failed: '{sheet_name}.{col}' must be unique. "
                    f"Duplicates found: {duplicates.tolist()}"
                )

    # --------------------------------------------------------- #
    # 2. FOREIGN KEY CONSTRAINTS (Referential Integrity)
    # --------------------------------------------------------- #
    # Structure: tuple(child_sheet, child_col) -> tuple(parent_sheet, parent_col)
    fk_constraints = {
        ("risk_factors", "container_name"): (
            "container_meta",
            "container_name",
        ),
        ("asset_subclasses", "asset_class"): ("asset_classes", "asset_class"),
    }

    for (c_sheet, c_col), (p_sheet, p_col) in fk_constraints.items():
        c_df = dict_of_dfs.get(c_sheet)
        p_df = dict_of_dfs.get(p_sheet)

        if (
            c_df is None
            or p_df is None
            or c_col not in c_df.columns
            or p_col not in p_df.columns
        ):
            continue

        # Extract unique arrays directly. Dropping NA ensures we don't flag optional/blank FKs.
        child_vals = c_df[c_col].dropna().unique()
        parent_vals = p_df[p_col].dropna().unique()

        # np.setdiff1d is a highly optimized C-level array operation.
        # It is significantly faster and uses less memory than Python set(A) - set(B).
        missing_fks = np.setdiff1d(child_vals, parent_vals, assume_unique=True)

        if missing_fks.size > 0:
            raise ValueError(
                f"Validation failed: '{c_sheet}.{c_col}' contains values not present in "
                f"'{p_sheet}.{p_col}'. Missing: {missing_fks.tolist()}"
            )

    # --------------------------------------------------------- #
    # 3. DOMAIN CONSTRAINTS (Syntax & Business Logic)
    # --------------------------------------------------------- #
    # Now that we know containers and schemas exist and map correctly,
    # we enforce the strict data-driven regex rules on the inputs.

    _validate_risk_factor_tokens(dict_of_dfs)

    # ----------------------------------------- #
    # 4. VALIDATE THE INTERPOLATION/EXTRAPOLATION METHODS IN THE CURVE SCHEMA
    # ----------------------------------------- #

    # Validate.
    _validate_container_meta(dict_of_dfs)

    # You can add future domain checks here
    # (e.g., validate_shift_types, validate_currency_codes)
