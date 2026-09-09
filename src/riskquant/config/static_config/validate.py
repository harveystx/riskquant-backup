# src/riskquant/config/static_config/validate.py
"""
Vectorized validation for the god-tier static config DataFrames.

The new 7-table schema makes most of the legacy "if dim==0 then this
column must be null" checks unnecessary: every table's columns are
required by definition of the table. This validator only needs to
enforce:

  Tier 1 — Required sheets present.
  Tier 2 — Uniqueness (single + composite primary keys).
  Tier 3 — Enum membership for every typed string column.
  Tier 4 — Foreign-key referential integrity across tables.
  Tier 5 — Bespoke domain invariants:
             * FX node naming convention
             * (XAxisKind, RollMethod) allowed-pair table
             * tenor-token grammar matches parent's x_axis_kind
             * discount-curve default & basis invariants
             * every active currency has at least one discount curve
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from riskquant.config.enums.asset_classes import (
    ASSET_SUBCLASS_TO_CLASS,
    VALID_ASSET_CLASSES,
    VALID_ASSET_SUBCLASSES,
    AssetClass,
    AssetSubclass,
)
from riskquant.config.enums.discount_basis import (
    DISCOUNT_BASES_EXCLUDING_SENTINEL,
    FIRM_DEFAULT_DISCOUNT_BASIS,
    VALID_DISCOUNT_BASES,
    DiscountBasis,
)
from riskquant.config.enums.quote_types import (
    VALID_QUOTE_TYPES,
    VALID_UNIT_SCALES,
    QuoteType,
)
from riskquant.config.enums.roll_methods import (
    ALLOWED_X_AXIS_KIND_ROLL_METHOD_PAIRS,
    VALID_ROLL_METHODS,
)
from riskquant.config.enums.x_axis_kinds import (
    VALID_X_AXIS_KINDS,
    X_AXIS_KINDS_REQUIRING_ROLL,
    X_AXIS_TOKEN_REGEX,
    XAxisKind,
)
from riskquant.config.enums.y_axis_kinds import VALID_Y_AXIS_KINDS
from riskquant.config.namespace import GlobalIdNamespace, enforce_id_namespace
from riskquant.exceptions import ConfigurationError
from riskquant.market_data.curves.extrap_methods import EXTRAP_METHOD_NAMES
from riskquant.market_data.curves.interp_methods import INTERP_METHOD_NAMES
from riskquant.market_data.risk_factors.shift_types import SHIFT_TYPE_NAMES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_SHEETS: frozenset[str] = frozenset(
    {
        "ccy",
        "holiday_calendars",
        "spot_roll_rule",
        "node_meta",
        "curve_meta",
        "curve_point_meta",
        "surface_meta",
        "surface_point_meta",
    }
)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _check_required_sheets(dict_of_dfs: dict[str, pd.DataFrame]) -> None:
    missing = REQUIRED_SHEETS - dict_of_dfs.keys()
    if missing:
        raise ConfigurationError(
            f"Missing required sheet(s): {sorted(missing)}.\n"
            f"Required: {sorted(REQUIRED_SHEETS)}"
        )


def _check_columns(df: pd.DataFrame, cols: frozenset[str], sheet_name: str) -> None:
    missing = cols - set(df.columns)
    if missing:
        raise ConfigurationError(
            f"Sheet {sheet_name!r} missing required columns: {sorted(missing)}.\n"
            f"Found: {sorted(df.columns)}"
        )


def _check_unique(df: pd.DataFrame, col: str, sheet_name: str) -> None:
    if not df[col].is_unique:
        dupes = df[col][df[col].duplicated()].unique().tolist()
        raise ConfigurationError(
            f"Uniqueness violation: {sheet_name}.{col} must be unique. "
            f"Duplicates: {dupes}"
        )


def _check_composite_unique(
    df: pd.DataFrame,
    cols: tuple[str, ...],
    sheet_name: str,
) -> None:
    dupes_mask = df.duplicated(subset=list(cols), keep=False)
    if dupes_mask.any():
        offending = df.loc[dupes_mask, list(cols)]
        raise ConfigurationError(
            f"Composite uniqueness violation: "
            f"{sheet_name}({', '.join(cols)}) must be unique.\n"
            f"Duplicates:\n{offending.to_string(index=False)}"
        )


def _check_enum_col(
    df: pd.DataFrame,
    col: str,
    valid_set: frozenset[str],
    sheet_name: str,
    enum_name: str,
    *,
    nullable: bool,
) -> None:
    if col not in df.columns:
        return
    series = df[col]
    populated = series.dropna()
    if not nullable and len(populated) != len(series):
        n_null = int(series.isna().sum())
        raise ConfigurationError(
            f"Enum column {sheet_name}.{col} ({enum_name}) "
            f"contains {n_null} null value(s) but is non-nullable."
        )
    if populated.empty:
        return
    invalid = set(populated.astype(str)) - valid_set
    if invalid:
        raise ConfigurationError(
            f"Enum violation in {sheet_name}.{col}: values not in {enum_name}: "
            f"{sorted(invalid)}.\nValid: {sorted(valid_set)}"
        )


def _check_fk(
    child_df: pd.DataFrame,
    child_col: str,
    parent_df: pd.DataFrame,
    parent_col: str,
    *,
    child_sheet: str,
    parent_sheet: str,
) -> None:
    child_vals = child_df[child_col].dropna().to_numpy()
    parent_vals = parent_df[parent_col].dropna().to_numpy()
    missing = np.setdiff1d(child_vals, parent_vals, assume_unique=False)
    if missing.size > 0:
        raise ConfigurationError(
            f"FK violation: {child_sheet}.{child_col} references values absent "
            f"from {parent_sheet}.{parent_col}.\nMissing: {missing.tolist()[:25]}"
        )


def _check_no_blank_strings(
    df: pd.DataFrame, col: str, sheet_name: str, *, nullable: bool
) -> None:
    if col not in df.columns:
        return
    series = df[col]
    if nullable:
        series = series.dropna()
    if series.empty:
        return
    is_blank = series.astype(str).str.strip() == ""
    if is_blank.any():
        idx = series.index[is_blank].tolist()[:10]
        raise ConfigurationError(
            f"Blank string violation: {sheet_name}.{col} has empty values at "
            f"rows: {idx}"
        )


# ---------------------------------------------------------------------------
# Tier 2 — per-sheet uniqueness
# ---------------------------------------------------------------------------


def _validate_uniqueness(dict_of_dfs: dict[str, pd.DataFrame]) -> None:
    single_pk: dict[str, tuple[str, ...]] = {
        "ccy": ("ccy_name", "ccy_id"),
        "spot_roll_rule": ("rule_name",),
        "node_meta": ("node_id", "node_name"),
        "curve_meta": ("curve_id", "curve_name"),
        "curve_point_meta": ("rf_id",),
        "surface_meta": ("surface_id", "surface_name"),
        "surface_point_meta": ("rf_id",),
    }
    for sheet_name, cols in single_pk.items():
        df = dict_of_dfs[sheet_name]
        for col in cols:
            _check_unique(df, col, sheet_name)

    _check_composite_unique(
        dict_of_dfs["curve_point_meta"],
        ("curve_id", "tenor_token"),
        "curve_point_meta",
    )
    _check_composite_unique(
        dict_of_dfs["surface_point_meta"],
        ("surface_id", "x_tenor_token", "y_label"),
        "surface_point_meta",
    )


# ---------------------------------------------------------------------------
# Tier 3 — enum membership
# ---------------------------------------------------------------------------


def _validate_enums(dict_of_dfs: dict[str, pd.DataFrame]) -> None:
    interp_names = frozenset(INTERP_METHOD_NAMES)
    extrap_names = frozenset(EXTRAP_METHOD_NAMES)
    shift_names = frozenset(SHIFT_TYPE_NAMES)

    # --- node_meta ---
    node_df = dict_of_dfs["node_meta"]
    _check_enum_col(
        node_df,
        "asset_class",
        VALID_ASSET_CLASSES,
        "node_meta",
        "AssetClass",
        nullable=False,
    )
    _check_enum_col(
        node_df,
        "asset_subclass",
        VALID_ASSET_SUBCLASSES,
        "node_meta",
        "AssetSubclass",
        nullable=False,
    )
    _check_enum_col(
        node_df,
        "quote_type",
        VALID_QUOTE_TYPES,
        "node_meta",
        "UnitType",
        nullable=False,
    )
    _check_enum_col(
        node_df,
        "unit_scale",
        VALID_UNIT_SCALES,
        "node_meta",
        "UnitScale",
        nullable=False,
    )
    _check_enum_col(
        node_df, "shift_type", shift_names, "node_meta", "ShiftType", nullable=False
    )

    # --- curve_meta ---
    curve_df = dict_of_dfs["curve_meta"]
    _check_enum_col(
        curve_df,
        "asset_class",
        VALID_ASSET_CLASSES,
        "curve_meta",
        "AssetClass",
        nullable=False,
    )
    _check_enum_col(
        curve_df,
        "asset_subclass",
        VALID_ASSET_SUBCLASSES,
        "curve_meta",
        "AssetSubclass",
        nullable=False,
    )
    _check_enum_col(
        curve_df,
        "quote_type",
        VALID_QUOTE_TYPES,
        "curve_meta",
        "UnitType",
        nullable=False,
    )
    _check_enum_col(
        curve_df,
        "unit_scale",
        VALID_UNIT_SCALES,
        "curve_meta",
        "UnitScale",
        nullable=False,
    )
    _check_enum_col(
        curve_df, "shift_type", shift_names, "curve_meta", "ShiftType", nullable=False
    )
    _check_enum_col(
        curve_df,
        "x_axis_kind",
        VALID_X_AXIS_KINDS,
        "curve_meta",
        "XAxisKind",
        nullable=False,
    )
    _check_enum_col(
        curve_df,
        "x_interp_method",
        interp_names,
        "curve_meta",
        "InterpMethod",
        nullable=False,
    )
    _check_enum_col(
        curve_df,
        "x_extrap_method",
        extrap_names,
        "curve_meta",
        "ExtrapMethod",
        nullable=False,
    )
    _check_enum_col(
        curve_df,
        "discount_basis",
        VALID_DISCOUNT_BASES,
        "curve_meta",
        "DiscountBasis",
        nullable=False,
    )

    # --- surface_meta ---
    surf_df = dict_of_dfs["surface_meta"]
    _check_enum_col(
        surf_df,
        "asset_class",
        VALID_ASSET_CLASSES,
        "surface_meta",
        "AssetClass",
        nullable=False,
    )
    _check_enum_col(
        surf_df,
        "asset_subclass",
        VALID_ASSET_SUBCLASSES,
        "surface_meta",
        "AssetSubclass",
        nullable=False,
    )
    _check_enum_col(
        surf_df,
        "quote_type",
        VALID_QUOTE_TYPES,
        "surface_meta",
        "UnitType",
        nullable=False,
    )
    _check_enum_col(
        surf_df,
        "unit_scale",
        VALID_UNIT_SCALES,
        "surface_meta",
        "UnitScale",
        nullable=False,
    )
    _check_enum_col(
        surf_df, "shift_type", shift_names, "surface_meta", "ShiftType", nullable=False
    )
    _check_enum_col(
        surf_df,
        "x_axis_kind",
        VALID_X_AXIS_KINDS,
        "surface_meta",
        "XAxisKind",
        nullable=False,
    )
    _check_enum_col(
        surf_df,
        "x_interp_method",
        interp_names,
        "surface_meta",
        "InterpMethod",
        nullable=False,
    )
    _check_enum_col(
        surf_df,
        "x_extrap_method",
        extrap_names,
        "surface_meta",
        "ExtrapMethod",
        nullable=False,
    )
    _check_enum_col(
        surf_df,
        "y_axis_kind",
        VALID_Y_AXIS_KINDS,
        "surface_meta",
        "YAxisKind",
        nullable=False,
    )
    _check_enum_col(
        surf_df,
        "y_interp_method",
        interp_names,
        "surface_meta",
        "InterpMethod",
        nullable=False,
    )
    _check_enum_col(
        surf_df,
        "y_extrap_method",
        extrap_names,
        "surface_meta",
        "ExtrapMethod",
        nullable=False,
    )

    # --- spot_roll_rule ---
    srr_df = dict_of_dfs["spot_roll_rule"]
    _check_enum_col(
        srr_df,
        "roll_method",
        VALID_ROLL_METHODS,
        "spot_roll_rule",
        "RollMethod",
        nullable=False,
    )


# ---------------------------------------------------------------------------
# Tier 4 — foreign keys
# ---------------------------------------------------------------------------


def _validate_fks(dict_of_dfs: dict[str, pd.DataFrame]) -> None:
    ccy_df = dict_of_dfs["ccy"]
    node_df = dict_of_dfs["node_meta"]
    curve_df = dict_of_dfs["curve_meta"]
    curve_pt_df = dict_of_dfs["curve_point_meta"]
    surf_df = dict_of_dfs["surface_meta"]
    surf_pt_df = dict_of_dfs["surface_point_meta"]
    srr_df = dict_of_dfs["spot_roll_rule"]

    # ccy FKs
    _check_fk(
        node_df,
        "quote_ccy",
        ccy_df,
        "ccy_name",
        child_sheet="node_meta",
        parent_sheet="ccy",
    )
    _check_fk(
        curve_df,
        "quote_ccy",
        ccy_df,
        "ccy_name",
        child_sheet="curve_meta",
        parent_sheet="ccy",
    )
    _check_fk(
        surf_df,
        "quote_ccy",
        ccy_df,
        "ccy_name",
        child_sheet="surface_meta",
        parent_sheet="ccy",
    )

    # node_meta.fx_base_ccy is nullable; FK only on populated rows
    populated_fx_base = node_df.loc[node_df["fx_base_ccy"].notna(), ["fx_base_ccy"]]
    if not populated_fx_base.empty:
        _check_fk(
            populated_fx_base,
            "fx_base_ccy",
            ccy_df,
            "ccy_name",
            child_sheet="node_meta",
            parent_sheet="ccy",
        )

    # Curve / surface spot_roll_rule_name FKs (only populated rows)
    cm_pop = curve_df.loc[
        curve_df["spot_roll_rule_name"].notna(), ["spot_roll_rule_name"]
    ]
    if not cm_pop.empty:
        _check_fk(
            cm_pop,
            "spot_roll_rule_name",
            srr_df,
            "rule_name",
            child_sheet="curve_meta",
            parent_sheet="spot_roll_rule",
        )

    sm_pop = surf_df.loc[
        surf_df["spot_roll_rule_name"].notna(), ["spot_roll_rule_name"]
    ]
    if not sm_pop.empty:
        _check_fk(
            sm_pop,
            "spot_roll_rule_name",
            srr_df,
            "rule_name",
            child_sheet="surface_meta",
            parent_sheet="spot_roll_rule",
        )

    # Curve / surface point parent FKs
    _check_fk(
        curve_pt_df,
        "curve_id",
        curve_df,
        "curve_id",
        child_sheet="curve_point_meta",
        parent_sheet="curve_meta",
    )
    _check_fk(
        surf_pt_df,
        "surface_id",
        surf_df,
        "surface_id",
        child_sheet="surface_point_meta",
        parent_sheet="surface_meta",
    )


# ---------------------------------------------------------------------------
# Tier 5 — bespoke domain invariants
# ---------------------------------------------------------------------------


def _validate_global_name_uniqueness(dict_of_dfs: dict[str, pd.DataFrame]) -> None:
    """No name appears in more than one of {node, curve, surface} namespaces."""
    node_names = set(dict_of_dfs["node_meta"]["node_name"].astype(str))
    curve_names = set(dict_of_dfs["curve_meta"]["curve_name"].astype(str))
    surface_names = set(dict_of_dfs["surface_meta"]["surface_name"].astype(str))

    nc = node_names & curve_names
    ns = node_names & surface_names
    cs = curve_names & surface_names

    overlap = nc | ns | cs
    if overlap:
        raise ConfigurationError(
            f"Name collision across container namespaces: {sorted(overlap)}.\n"
            f"Every node_name / curve_name / surface_name must be globally unique."
        )


def _validate_rf_id_uniqueness(dict_of_dfs: dict[str, pd.DataFrame]) -> None:
    """rf_ids (node_id ∪ curve_point.rf_id ∪ surface_point.rf_id) must be globally unique."""
    node_ids = dict_of_dfs["node_meta"]["node_id"].to_numpy()
    cp_ids = dict_of_dfs["curve_point_meta"]["rf_id"].to_numpy()
    sp_ids = dict_of_dfs["surface_point_meta"]["rf_id"].to_numpy()

    all_rf_ids = np.concatenate([node_ids, cp_ids, sp_ids])

    if np.unique(all_rf_ids).size != all_rf_ids.size:
        unique, counts = np.unique(all_rf_ids, return_counts=True)
        dupes = unique[counts > 1].tolist()
        raise ConfigurationError(
            f"CRITICAL: rf_id collision across atomic namespaces: {dupes[:25]}.\n"
            f"node_id, curve_point.rf_id, and surface_point.rf_id share one ID space "
            f"and cannot overlap."
        )


def _validate_container_id_uniqueness(dict_of_dfs: dict[str, pd.DataFrame]) -> None:
    """Ensure internal primary key uniqueness within container tables."""
    # Cross-container collisions are prevented by the namespace bounds.
    # We only need to check that the tables themselves don't have duplicate IDs.
    for table, col in [("curve_meta", "curve_id"), ("surface_meta", "surface_id")]:
        ids = dict_of_dfs[table][col].to_numpy()
        if np.unique(ids).size != ids.size:
            raise ConfigurationError(
                f"CRITICAL: Duplicate primary keys found in {table}.{col}."
            )


def _validate_global_id_namespace(dict_of_dfs: dict[str, pd.DataFrame]) -> None:
    """IDs that share a namespace must be globally bounded."""
    # 1. Gather ALL atomic Risk Factor IDs (Dimension = 0)
    node_ids = dict_of_dfs["node_meta"][
        "node_id"
    ].to_numpy()  # Node IDs are RF IDs by definition
    cp_ids = dict_of_dfs["curve_point_meta"]["rf_id"].to_numpy()
    sp_ids = dict_of_dfs["surface_point_meta"]["rf_id"].to_numpy()
    all_rf_ids = np.concatenate([node_ids, cp_ids, sp_ids])

    # 2. Gather Containers
    curve_ids = dict_of_dfs["curve_meta"]["curve_id"].to_numpy()
    surface_ids = dict_of_dfs["surface_meta"]["surface_id"].to_numpy()

    # 3. Enforce the mathematical bounds
    enforce_id_namespace(ids=all_rf_ids, bounds=GlobalIdNamespace.NODES)
    enforce_id_namespace(ids=curve_ids, bounds=GlobalIdNamespace.CURVES)
    enforce_id_namespace(ids=surface_ids, bounds=GlobalIdNamespace.SURFACES)


def _validate_asset_class_subclass_consistency(
    dict_of_dfs: dict[str, pd.DataFrame],
) -> None:
    """asset_subclass must map to the declared asset_class for all 3 meta tables."""

    def _expected_class(subclass_val: str) -> str | None:
        try:
            return ASSET_SUBCLASS_TO_CLASS[AssetSubclass(subclass_val)].value
        except (ValueError, KeyError):
            # AssetSubclass(...) raises ValueError on an unknown value;
            # the dict lookup raises KeyError if a member is unmapped.
            # (Unreachable once _validate_enums has run, but kept honest.)
            return None

    for sheet in ("node_meta", "curve_meta", "surface_meta"):
        df = dict_of_dfs[sheet]
        if "asset_class" not in df.columns or "asset_subclass" not in df.columns:
            continue
        valid = df[["asset_class", "asset_subclass"]].dropna()
        if valid.empty:
            continue
        expected = valid["asset_subclass"].map(_expected_class)
        bad_mask = expected.notna() & (valid["asset_class"] != expected)
        if bad_mask.any():
            offending = valid.loc[bad_mask].copy()
            offending["expected_class"] = expected[bad_mask]
            raise ConfigurationError(
                f"Domain violation: {sheet} has inconsistent "
                f"asset_class / asset_subclass.\n"
                f"{offending.to_string(index=False)}"
            )


def _validate_fx_node_invariants(dict_of_dfs: dict[str, pd.DataFrame]) -> None:
    """
    FX rate node contract:

    1. unit_type == FX_RATE  =>  fx_base_ccy must be populated.
    2. unit_type != FX_RATE  =>  fx_base_ccy must be null.
    3. FX_RATE node_name must equal fx_base_ccy + quote_ccy.
    4. FX_RATE pairs (base, quote) must be unique.
    """
    df = dict_of_dfs["node_meta"]

    is_fx_rate = df["quote_type"] == QuoteType.FX_RATE.value

    missing_base = is_fx_rate & df["fx_base_ccy"].isna()
    if missing_base.any():
        offending = df.loc[
            missing_base,
            [
                "node_name",
                "quote_type",
                "fx_base_ccy",
                "quote_ccy",
            ],
        ]
        raise ConfigurationError(
            "Domain violation: FX_RATE nodes must specify fx_base_ccy.\n"
            f"{offending.to_string(index=False)}"
        )

    unexpected_base = ~is_fx_rate & df["fx_base_ccy"].notna()
    if unexpected_base.any():
        offending = df.loc[
            unexpected_base,
            [
                "node_name",
                "asset_class",
                "quote_type",
                "fx_base_ccy",
            ],
        ]
        raise ConfigurationError(
            "Domain violation: only FX_RATE nodes may specify fx_base_ccy.\n"
            f"{offending.to_string(index=False)}"
        )

    fx_rate_rows = df.loc[is_fx_rate]

    if fx_rate_rows.empty:
        return

    expected_names = fx_rate_rows["fx_base_ccy"].astype(str) + fx_rate_rows[
        "quote_ccy"
    ].astype(str)

    name_mismatch = fx_rate_rows["node_name"] != expected_names

    if name_mismatch.any():
        offending = fx_rate_rows.loc[
            name_mismatch,
            ["node_name", "fx_base_ccy", "quote_ccy"],
        ]

        raise ConfigurationError(
            "Domain violation: FX_RATE node_name must equal "
            "fx_base_ccy + quote_ccy.\n"
            f"{offending.to_string(index=False)}"
        )

    from collections import Counter

    pairs = list(
        zip(
            fx_rate_rows["fx_base_ccy"],
            fx_rate_rows["quote_ccy"],
            strict=True,
        )
    )

    duplicate_pairs = [pair for pair, count in Counter(pairs).items() if count > 1]

    if duplicate_pairs:
        raise ConfigurationError(
            "FX pair duplication: "
            f"(base, quote) pairs appear more than once: "
            f"{duplicate_pairs}"
        )


def _validate_axis_roll_pairs(dict_of_dfs: dict[str, pd.DataFrame]) -> None:
    """
    (XAxisKind, RollMethod) compatibility:

      - LABEL curves must have spot_roll_rule_name == NULL.
      - OFFSET_* curves must have spot_roll_rule_name populated.
      - The resolved (XAxisKind, RollMethod) pair must be in the
        ALLOWED_X_AXIS_KIND_ROLL_METHOD_PAIRS table.

    Vectorized: a single pass over each meta table, no row loops.
    """
    srr_df = dict_of_dfs["spot_roll_rule"]
    rule_to_method: dict[str, str] = dict(
        zip(
            srr_df["rule_name"].astype(str),
            srr_df["roll_method"].astype(str),
            strict=True,
        )
    )

    # Pre-compute the allowed-pair set as plain strings (cheap O(K) build).
    allowed_pair_names: frozenset[tuple[str, str]] = frozenset(
        (xk.name, rm.name) for xk, rm in ALLOWED_X_AXIS_KIND_ROLL_METHOD_PAIRS
    )

    for sheet, name_col in (
        ("curve_meta", "curve_name"),
        ("surface_meta", "surface_name"),
    ):
        df = dict_of_dfs[sheet]
        if df.empty:
            continue

        x_kind = df["x_axis_kind"].astype(str)
        rule = df["spot_roll_rule_name"]
        rule_populated = rule.notna()
        needs_roll = x_kind.isin(X_AXIS_KINDS_REQUIRING_ROLL)

        # Missing rule on OFFSET container.
        missing = needs_roll & ~rule_populated
        if missing.any():
            offending = df.loc[missing, [name_col, "x_axis_kind"]]
            raise ConfigurationError(
                f"{sheet}: OFFSET containers missing spot_roll_rule_name:\n"
                f"{offending.to_string(index=False)}"
            )

        # Populated rule on LABEL container.
        extraneous = ~needs_roll & rule_populated
        if extraneous.any():
            offending = df.loc[
                extraneous, [name_col, "x_axis_kind", "spot_roll_rule_name"]
            ]
            raise ConfigurationError(
                f"{sheet}: LABEL containers with populated spot_roll_rule_name:\n"
                f"{offending.to_string(index=False)}"
            )

        # For OFFSET rows, check the (x_kind, method) pair is allowed.
        offset_rows = df.loc[needs_roll].copy()
        if offset_rows.empty:
            continue
        method = offset_rows["spot_roll_rule_name"].astype(str).map(rule_to_method)
        # Vectorized pair check: encode each (x_kind, roll_method) tuple as a
        # single "X|M" string and isin() against the precomputed allowed set.
        # Drops the Python-level list comprehension in the validation hot path.
        joined = offset_rows["x_axis_kind"].astype(str) + "|" + method.astype(str)
        allowed_join = frozenset(f"{xk}|{rm}" for xk, rm in allowed_pair_names)
        bad_mask = ~joined.isin(allowed_join).to_numpy(copy=False)
        if bad_mask.any():
            offending = offset_rows.loc[
                bad_mask, [name_col, "x_axis_kind", "spot_roll_rule_name"]
            ].copy()
            offending["roll_method"] = method[bad_mask].values
            raise ConfigurationError(
                f"{sheet}: disallowed (x_axis_kind, roll_method) combinations:\n"
                f"{offending.to_string(index=False)}"
            )


def _validate_fh_bh_mmdd(dict_of_dfs: dict[str, pd.DataFrame]) -> None:
    fh_bh_mmdd = dict_of_dfs["spot_roll_rule"]["fh_bh_mmdd"]
    if fh_bh_mmdd.empty:
        return
    fh_bh_mmdd = fh_bh_mmdd.dropna().astype(str)

    # Assert correct length of 9 (e.g., 0401|1001)
    bad_length = fh_bh_mmdd.str.len() != 9
    if bad_length.any():
        offending = fh_bh_mmdd[bad_length]
        raise ConfigurationError(
            f"Invalid fh_bh_mmdd length:\n{offending.to_string(index=False)}"
        )

    # Assert correct format of 4 digits, pipe, 4 digits.
    bad_format = ~fh_bh_mmdd.str.match(r"^\d{4}\|\d{4}$")
    if bad_format.any():
        offending = fh_bh_mmdd[bad_format]
        raise ConfigurationError(
            f"Invalid fh_bh_mmdd format (expected 'MMDD|MMDD'):\n"
            f"{offending.to_string(index=False)}"
        )


def _validate_curve_point_token_grammar(
    dict_of_dfs: dict[str, pd.DataFrame],
) -> None:
    """
    Every curve_point.tenor_token matches its parent's x_axis_kind regex.

    Vectorized: group rows by parent x_axis_kind, apply pd.Series.str.match
    with the kind-specific regex to each group in one vectorized call.
    """
    cm_df = dict_of_dfs["curve_meta"]
    cp_df = dict_of_dfs["curve_point_meta"]
    if cp_df.empty:
        return

    # Join parent x_axis_kind onto each point row via map (vectorized).
    parent_kind_by_id = cm_df.set_index("curve_id")["x_axis_kind"].astype(str)
    work = cp_df[["curve_id", "rf_id", "tenor_token"]].copy()
    work["x_axis_kind"] = work["curve_id"].map(parent_kind_by_id)

    bad_pieces: list[pd.DataFrame] = []
    for kind_name, group in work.groupby("x_axis_kind"):
        # Explicitly cast kind_name to a string
        kind_name_str = str(kind_name)
        regex = X_AXIS_TOKEN_REGEX[XAxisKind[kind_name_str]]
        good = group["tenor_token"].astype(str).str.match(regex)
        bad = ~good
        if bad.any():
            bad_pieces.append(group.loc[bad])

    if bad_pieces:
        offending = pd.concat(bad_pieces, ignore_index=True)
        raise ConfigurationError(
            "Curve point token grammar violations:\n"
            f"{offending.head(25).to_string(index=False)}"
        )


def _validate_surface_point_token_grammar(
    dict_of_dfs: dict[str, pd.DataFrame],
) -> None:
    """
    Every surface_point.x_tenor_token matches its parent's x_axis_kind regex.

    Vectorized: same per-kind-group str.match pattern as the curve-point
    validator.
    """
    sm_df = dict_of_dfs["surface_meta"]
    sp_df = dict_of_dfs["surface_point_meta"]
    if sp_df.empty:
        return

    parent_kind_by_id = sm_df.set_index("surface_id")["x_axis_kind"].astype(str)
    work = sp_df[["surface_id", "rf_id", "x_tenor_token"]].copy()
    work["x_axis_kind"] = work["surface_id"].map(parent_kind_by_id)

    bad_pieces: list[pd.DataFrame] = []
    for kind_name, group in work.groupby("x_axis_kind"):
        # Explicitly cast kind_name to a string
        kind_name_str = str(kind_name)
        regex = X_AXIS_TOKEN_REGEX[XAxisKind[kind_name_str]]
        good = group["x_tenor_token"].astype(str).str.match(regex)
        bad = ~good
        if bad.any():
            bad_pieces.append(group.loc[bad])

    if bad_pieces:
        offending = pd.concat(bad_pieces, ignore_index=True)
        raise ConfigurationError(
            "Surface point token grammar violations:\n"
            f"{offending.head(25).to_string(index=False)}"
        )


def _validate_discount_invariants(dict_of_dfs: dict[str, pd.DataFrame]) -> None:
    """
    Discount-curve rules:
      - is_default_discount_for_ccy=True curves must have a real basis
        (not NOT_A_DISCOUNT_CURVE).
      - Non-discount curves (basis=NOT_A_DISCOUNT_CURVE) cannot be default.
      - Discount curves must have asset_class=='IR'.
      - The default's basis must equal FIRM_DEFAULT_DISCOUNT_BASIS.
      - Exactly one default per currency that has any discount curve.
      - (ccy, basis) is unique among discount curves.
      - Every currency in the ccy table that appears as a quote_ccy on
        any other container must have at least one discount curve.
    """
    curve_df = dict_of_dfs["curve_meta"]
    node_df = dict_of_dfs["node_meta"]
    surf_df = dict_of_dfs["surface_meta"]
    # ccy_df = dict_of_dfs["ccy"]

    is_disc = curve_df["discount_basis"] != DiscountBasis.NOT_A_DISCOUNT_CURVE.name
    is_default = curve_df["is_default_discount_for_ccy"].astype(bool)

    # Default => is a real discount curve.
    bad = is_default & ~is_disc
    if bad.any():
        offending = curve_df.loc[bad, ["curve_name", "discount_basis"]]
        raise ConfigurationError(
            "Default discount flag set on non-discount curve(s):\n"
            f"{offending.to_string(index=False)}"
        )

    # Discount curves must be IR.
    bad_class = is_disc & (curve_df["asset_class"] != AssetClass.IR.value)
    if bad_class.any():
        offending = curve_df.loc[bad_class, ["curve_name", "asset_class"]]
        raise ConfigurationError(
            "Discount curves must have asset_class=='IR'.\n"
            f"{offending.to_string(index=False)}"
        )

    # Default basis must equal the firm standard.
    bad_basis = is_default & (
        curve_df["discount_basis"] != FIRM_DEFAULT_DISCOUNT_BASIS.name
    )
    if bad_basis.any():
        offending = curve_df.loc[bad_basis, ["curve_name", "discount_basis"]]
        raise ConfigurationError(
            f"Default discount curves must use the firm-standard basis "
            f"{FIRM_DEFAULT_DISCOUNT_BASIS.name!r}.\n"
            f"{offending.to_string(index=False)}"
        )

    # Exactly one default per currency that has any discount curve.
    disc_curves = curve_df.loc[
        is_disc,
        ["curve_name", "quote_ccy", "discount_basis", "is_default_discount_for_ccy"],
    ]
    default_counts = (
        disc_curves.groupby("quote_ccy")["is_default_discount_for_ccy"]
        .sum()
        .astype(int)
    )
    bad_ccys = default_counts[default_counts != 1]
    if not bad_ccys.empty:
        raise ConfigurationError(
            f"Each currency with a discount curve must flag exactly one as "
            f"default. Currencies with !=1 default:\n"
            f"{bad_ccys.to_string()}"
        )

    # (ccy, basis) is unique among discount curves.
    composite = disc_curves[["quote_ccy", "discount_basis"]]
    dupe_mask = composite.duplicated(keep=False)
    if dupe_mask.any():
        offending = disc_curves.loc[dupe_mask]
        raise ConfigurationError(
            "Duplicate (quote_ccy, discount_basis) among discount curves:\n"
            f"{offending.to_string(index=False)}"
        )

    # Coverage: every currency that's a quote_ccy somewhere must have a
    # discount curve. (Skips currencies that exist in `ccy` but aren't used.)
    quote_ccys_used = pd.Index(
        pd.concat([curve_df["quote_ccy"], surf_df["quote_ccy"], node_df["quote_ccy"]])
        .dropna()
        .unique()
    )
    ccys_with_discount = set(disc_curves["quote_ccy"].astype(str))
    missing = sorted(set(quote_ccys_used.astype(str)) - ccys_with_discount)
    if missing:
        raise ConfigurationError(
            f"Currencies with no discount curve: {missing}. "
            f"Every active quote_ccy must have at least one discount curve."
        )

    # Sanity: discount_basis used by discount curves must be one of the real bases.
    real_bases = set(disc_curves["discount_basis"].astype(str))
    bogus = real_bases - DISCOUNT_BASES_EXCLUDING_SENTINEL
    if bogus:
        raise ConfigurationError(
            f"Unexpected discount_basis on discount curves: {sorted(bogus)}. "
            f"Allowed: {sorted(DISCOUNT_BASES_EXCLUDING_SENTINEL)}"
        )


def _validate_no_blanks(dict_of_dfs: dict[str, pd.DataFrame]) -> None:
    """Hard fail on empty / whitespace-only strings in non-nullable string cols."""
    string_cols_by_sheet: dict[str, dict[str, bool]] = {
        "ccy": {"ccy_name": False},
        "node_meta": {
            "node_name": False,
            "quote_ccy": False,
            "asset_class": False,
            "asset_subclass": False,
            "quote_type": False,
            "unit_scale": False,
            "shift_type": False,
            "fx_base_ccy": True,
        },
        "curve_meta": {
            "curve_name": False,
            "quote_ccy": False,
            "asset_class": False,
            "asset_subclass": False,
            "quote_type": False,
            "unit_scale": False,
            "shift_type": False,
            "x_axis_kind": False,
            "x_interp_method": False,
            "x_extrap_method": False,
            "spot_roll_rule_name": True,
            "discount_basis": False,
        },
        "curve_point_meta": {"tenor_token": False},
        "surface_meta": {
            "surface_name": False,
            "quote_ccy": False,
            "asset_class": False,
            "asset_subclass": False,
            "quote_type": False,
            "unit_scale": False,
            "shift_type": False,
            "x_axis_kind": False,
            "x_interp_method": False,
            "x_extrap_method": False,
            "spot_roll_rule_name": True,
            "y_axis_kind": False,
            "y_interp_method": False,
            "y_extrap_method": False,
        },
        "surface_point_meta": {"x_tenor_token": False, "y_label": False},
        "spot_roll_rule": {
            "rule_name": False,
            "roll_method": False,
            "fh_bh_mmdd": True,
        },
    }
    for sheet_name, cols in string_cols_by_sheet.items():
        df = dict_of_dfs[sheet_name]
        for col, nullable in cols.items():
            _check_no_blank_strings(df, col, sheet_name, nullable=nullable)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _validate_ccy_id_bounds(dict_of_dfs: dict[str, pd.DataFrame]) -> None:
    """
    Validate that every ``ccy_id`` fits within the ``np.int8`` range [0, 127].

    ``_build_currency_state`` stores currency ids as ``np.int8`` (the universe of
    currencies is small). A ``ccy_id`` outside [0, 127] would either raise an
    opaque ``OverflowError`` from numpy during state construction or silently
    wrap into a negative id and misindex the FX matrix, and a large positive id
    would size the dense ``(n, n)`` ``fx_spot_matrix`` far beyond what is
    intended. ``GlobalIdNamespace`` bounds rf/curve/surface ids but not currency
    ids, so this is the gate that keeps the narrowing cast safe. (RQ-AUD-014.)
    """
    ccy_df = dict_of_dfs["ccy"]
    ids = ccy_df["ccy_id"].to_numpy()
    max_int8 = int(np.iinfo(np.int8).max)  # 127
    out_of_bounds = ids[(ids < 0) | (ids > max_int8)]
    if out_of_bounds.size > 0:
        raise ConfigurationError(
            f"ccy_id values out of the valid range [0, {max_int8}]: "
            f"{sorted({int(v) for v in out_of_bounds})}. "
            f"ccy_id is stored as np.int8; values must satisfy 0 <= ccy_id <= "
            f"{max_int8}. Exceeding this also over-sizes the dense fx_spot_matrix."
        )


def validate_static_config_dfs(dict_of_dfs: dict[str, pd.DataFrame]) -> None:
    """
    Execute all validation tiers against the loaded config DataFrames.

    Raises ConfigurationError on the first violation with a descriptive message.
    """
    _check_required_sheets(dict_of_dfs)

    expected_columns: dict[str, frozenset[str]] = {
        "ccy": frozenset({"ccy_name", "ccy_id"}),
        "holiday_calendars": frozenset({"holiday_calendar_name", "holiday_date"}),
        "spot_roll_rule": frozenset(
            {
                "rule_name",
                "roll_method",
                "offset_in_tenor_units",
                "roll_month",
                "roll_day",
                "fh_bh_mmdd",
                "post_roll_shift",
            }
        ),
        "node_meta": frozenset(
            {
                "node_id",
                "node_name",
                "quote_ccy",
                "asset_class",
                "asset_subclass",
                "quote_type",
                "unit_scale",
                "shift_type",
                "fx_base_ccy",
            }
        ),
        "curve_meta": frozenset(
            {
                "curve_id",
                "curve_name",
                "quote_ccy",
                "asset_class",
                "asset_subclass",
                "quote_type",
                "unit_scale",
                "shift_type",
                "x_axis_kind",
                "x_interp_method",
                "x_extrap_method",
                "spot_roll_rule_name",
                "discount_basis",
                "is_default_discount_for_ccy",
            }
        ),
        "curve_point_meta": frozenset({"rf_id", "curve_id", "tenor_token"}),
        "surface_meta": frozenset(
            {
                "surface_id",
                "surface_name",
                "quote_ccy",
                "asset_class",
                "asset_subclass",
                "quote_type",
                "unit_scale",
                "shift_type",
                "x_axis_kind",
                "x_interp_method",
                "x_extrap_method",
                "spot_roll_rule_name",
                "y_axis_kind",
                "y_interp_method",
                "y_extrap_method",
            }
        ),
        "surface_point_meta": frozenset(
            {
                "rf_id",
                "surface_id",
                "x_tenor_token",
                "y_label",
            }
        ),
    }
    for sheet_name, cols in expected_columns.items():
        _check_columns(dict_of_dfs[sheet_name], cols, sheet_name)

    _validate_no_blanks(dict_of_dfs)
    _validate_uniqueness(dict_of_dfs)
    _validate_ccy_id_bounds(dict_of_dfs)
    _validate_enums(dict_of_dfs)
    _validate_fks(dict_of_dfs)

    # Tier 5 — domain
    _validate_rf_id_uniqueness(dict_of_dfs)
    _validate_container_id_uniqueness(dict_of_dfs)
    _validate_global_id_namespace(dict_of_dfs)
    _validate_asset_class_subclass_consistency(dict_of_dfs)
    _validate_fx_node_invariants(dict_of_dfs)
    _validate_axis_roll_pairs(dict_of_dfs)
    _validate_curve_point_token_grammar(dict_of_dfs)
    _validate_surface_point_token_grammar(dict_of_dfs)
    _validate_discount_invariants(dict_of_dfs)
    _validate_fh_bh_mmdd(dict_of_dfs)
