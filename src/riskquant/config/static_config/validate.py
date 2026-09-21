# src/riskquant/config/static_config/validate.py
"""
Vectorized validation for the god-tier static config DataFrames.

The 7-table schema makes most of the legacy "if dim==0 then this
column must be null" checks unnecessary: every table's columns are
required by definition of the table. This validator only needs to
enforce:

  Tier 1 — Required sheets present.
  Tier 2 — Uniqueness (single + composite primary keys).
  Tier 3 — Enum membership for every typed string column.
  Tier 4 — Foreign-key referential integrity across tables.
  Tier 5 — Bespoke domain invariants:
             * FX node naming convention
             * (XAxisKind, RollMethod) allowed-pair table, declared
               directly on curve_meta / surface_meta
             * method-specific roll-parameter presence/absence + ranges
             * tenor-token grammar matches parent's x_axis_kind
             * discount-curve default & basis invariants
             * every active currency has at least one discount curve
"""

from __future__ import annotations

from calendar import monthrange
from collections import Counter

import numpy as np
import polars as pl

from riskquant.config.enums.asset_classes import (
    ASSET_SUBCLASS_TO_CLASS,
    VALID_ASSET_CLASSES,
    VALID_ASSET_SUBCLASSES,
    AssetClass,
    AssetSubclass,
)
from riskquant.config.enums.curve_input_types import VALID_CURVE_INPUT_TYPES
from riskquant.config.enums.discount_basis import (
    DISCOUNT_BASES_EXCLUDING_SENTINEL,
    FIRM_DEFAULT_DISCOUNT_BASIS,
    VALID_DISCOUNT_BASES,
    DiscountBasis,
)
from riskquant.config.enums.financial_conventions import (
    VALID_BUSINESS_DAY_CONVENTIONS,
    VALID_COMPOUNDING_CONVENTIONS,
)
from riskquant.config.enums.quote_types import (
    VALID_QUOTE_TYPES,
    VALID_UNIT_SCALES,
    QuoteType,
)
from riskquant.config.enums.roll_methods import (
    ALLOWED_X_AXIS_KIND_ROLL_METHOD_PAIRS,
    ROLL_METHOD_REQUIRED_PARAMS,
    VALID_ROLL_METHODS,
    RollMethod,
    parse_fh_bh_mmdd,
    roll_method_forbidden_params,
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

REQUIRED_SHEETS: frozenset[str] = frozenset({
    "ccy",
    "holiday_calendars",
    "node_meta",
    "curve_meta",
    "curve_point_meta",
    "surface_meta",
    "surface_point_meta",
})


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _check_required_sheets(dict_of_dfs: dict[str, pl.DataFrame]) -> None:
    missing = REQUIRED_SHEETS - dict_of_dfs.keys()
    if missing:
        raise ConfigurationError(
            f"Missing required sheet(s): {sorted(missing)}.\n"
            f"Required: {sorted(REQUIRED_SHEETS)}"
        )


def _check_columns(df: pl.DataFrame, cols: frozenset[str], sheet_name: str) -> None:
    missing = cols - set(df.columns)
    if missing:
        raise ConfigurationError(
            f"Sheet {sheet_name!r} missing required columns: {sorted(missing)}.\n"
            f"Found: {sorted(df.columns)}"
        )


def _check_unique(df: pl.DataFrame, col: str, sheet_name: str) -> None:
    series = df[col]
    if series.n_unique() != series.len():
        dupes = series.filter(series.is_duplicated()).unique().to_list()
        raise ConfigurationError(
            f"Uniqueness violation: {sheet_name}.{col} must be unique. "
            f"Duplicates: {dupes}"
        )


def _check_composite_unique(
    df: pl.DataFrame,
    cols: tuple[str, ...],
    sheet_name: str,
) -> None:
    dupes_mask = df.select(list(cols)).is_duplicated()
    if dupes_mask.any():
        offending = df.filter(dupes_mask).select(list(cols))
        raise ConfigurationError(
            f"Composite uniqueness violation: "
            f"{sheet_name}({', '.join(cols)}) must be unique.\n"
            f"Duplicates:\n{offending}"
        )


def _check_enum_col(
    df: pl.DataFrame,
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
    populated = series.drop_nulls()
    if not nullable and populated.len() != series.len():
        n_null = series.null_count()
        raise ConfigurationError(
            f"Enum column {sheet_name}.{col} ({enum_name}) "
            f"contains {n_null} null value(s) but is non-nullable."
        )
    if populated.len() == 0:
        return
    invalid = set(populated.cast(pl.Utf8).to_list()) - valid_set
    if invalid:
        raise ConfigurationError(
            f"Enum violation in {sheet_name}.{col}: values not in {enum_name}: "
            f"{sorted(invalid)}.\nValid: {sorted(valid_set)}"
        )


def _check_fk(
    child_df: pl.DataFrame,
    child_col: str,
    parent_df: pl.DataFrame,
    parent_col: str,
    *,
    child_sheet: str,
    parent_sheet: str,
) -> None:
    child_vals = child_df[child_col].drop_nulls().to_numpy()
    parent_vals = parent_df[parent_col].drop_nulls().to_numpy()
    missing = np.setdiff1d(child_vals, parent_vals, assume_unique=False)
    if missing.size > 0:
        raise ConfigurationError(
            f"FK violation: {child_sheet}.{child_col} references values absent "
            f"from {parent_sheet}.{parent_col}.\nMissing: {missing.tolist()[:25]}"
        )


def _check_no_blank_strings(
    df: pl.DataFrame, col: str, sheet_name: str, *, nullable: bool
) -> None:
    if col not in df.columns:
        return
    series = df[col]
    if nullable:
        series = series.drop_nulls()
    if series.len() == 0:
        return
    is_blank = series.cast(pl.Utf8).str.strip_chars() == ""
    if is_blank.any():
        idx = [i for i, v in enumerate(is_blank.to_list()) if v][:10]
        raise ConfigurationError(
            f"Blank string violation: {sheet_name}.{col} has empty values at "
            f"rows: {idx}"
        )


# ---------------------------------------------------------------------------
# Tier 2 — per-sheet uniqueness
# ---------------------------------------------------------------------------


def _validate_uniqueness(dict_of_dfs: dict[str, pl.DataFrame]) -> None:
    single_pk: dict[str, tuple[str, ...]] = {
        "ccy": ("ccy_name", "ccy_id"),
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


def _validate_enums(dict_of_dfs: dict[str, pl.DataFrame]) -> None:
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
    # D079 clause 3. Optional and nullable: a curve that has not declared its
    # economic input representation is still a valid market-data input; only
    # future curve construction needs the declaration, and that path is
    # fail-closed under D061. A populated value must be a known member.
    _check_enum_col(
        curve_df,
        "curve_input_type",
        VALID_CURVE_INPUT_TYPES,
        "curve_meta",
        "CurveInputType",
        nullable=True,
    )
    # D080 construction metadata. Optional and nullable for the same reason as
    # curve_input_type: only a curve declared constructible carries them, and
    # the construction layer rejects a partially declared curve.
    _check_enum_col(
        curve_df,
        "curve_compounding",
        VALID_COMPOUNDING_CONVENTIONS,
        "curve_meta",
        "CompoundingConvention",
        nullable=True,
    )
    _check_enum_col(
        curve_df,
        "business_day_convention",
        VALID_BUSINESS_DAY_CONVENTIONS,
        "curve_meta",
        "BusinessDayConvention",
        nullable=True,
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

    # --- roll_method (declared directly on curve_meta / surface_meta) ---
    # Nullable: LABEL axes carry no roll_method. Presence/absence versus the
    # x_axis_kind is enforced in _validate_axis_roll_pairs.
    _check_enum_col(
        curve_df,
        "roll_method",
        VALID_ROLL_METHODS,
        "curve_meta",
        "RollMethod",
        nullable=True,
    )
    _check_enum_col(
        surf_df,
        "roll_method",
        VALID_ROLL_METHODS,
        "surface_meta",
        "RollMethod",
        nullable=True,
    )


# ---------------------------------------------------------------------------
# Tier 4 — foreign keys
# ---------------------------------------------------------------------------


def _validate_fks(dict_of_dfs: dict[str, pl.DataFrame]) -> None:
    ccy_df = dict_of_dfs["ccy"]
    node_df = dict_of_dfs["node_meta"]
    curve_df = dict_of_dfs["curve_meta"]
    curve_pt_df = dict_of_dfs["curve_point_meta"]
    surf_df = dict_of_dfs["surface_meta"]
    surf_pt_df = dict_of_dfs["surface_point_meta"]

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
    populated_fx_base = node_df.filter(pl.col("fx_base_ccy").is_not_null()).select([
        "fx_base_ccy"
    ])
    if not populated_fx_base.is_empty():
        _check_fk(
            populated_fx_base,
            "fx_base_ccy",
            ccy_df,
            "ccy_name",
            child_sheet="node_meta",
            parent_sheet="ccy",
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


def _validate_global_name_uniqueness(dict_of_dfs: dict[str, pl.DataFrame]) -> None:
    """No name appears in more than one of {node, curve, surface} namespaces."""
    node_names = set(dict_of_dfs["node_meta"]["node_name"].cast(pl.Utf8).to_list())
    curve_names = set(dict_of_dfs["curve_meta"]["curve_name"].cast(pl.Utf8).to_list())
    surface_names = set(
        dict_of_dfs["surface_meta"]["surface_name"].cast(pl.Utf8).to_list()
    )

    nc = node_names & curve_names
    ns = node_names & surface_names
    cs = curve_names & surface_names

    overlap = nc | ns | cs
    if overlap:
        raise ConfigurationError(
            f"Name collision across container namespaces: {sorted(overlap)}.\n"
            f"Every node_name / curve_name / surface_name must be globally unique."
        )


def _validate_rf_id_uniqueness(dict_of_dfs: dict[str, pl.DataFrame]) -> None:
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


def _validate_container_id_uniqueness(dict_of_dfs: dict[str, pl.DataFrame]) -> None:
    """Ensure internal primary key uniqueness within container tables."""
    # Cross-container collisions are prevented by the namespace bounds.
    # We only need to check that the tables themselves don't have duplicate IDs.
    for table, col in [("curve_meta", "curve_id"), ("surface_meta", "surface_id")]:
        ids = dict_of_dfs[table][col].to_numpy()
        if np.unique(ids).size != ids.size:
            raise ConfigurationError(
                f"CRITICAL: Duplicate primary keys found in {table}.{col}."
            )


def _validate_global_id_namespace(dict_of_dfs: dict[str, pl.DataFrame]) -> None:
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
    dict_of_dfs: dict[str, pl.DataFrame],
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
        valid = df.select(["asset_class", "asset_subclass"]).drop_nulls()
        if valid.is_empty():
            continue
        subclass_to_class = {
            sc: _expected_class(sc) for sc in valid["asset_subclass"].unique().to_list()
        }
        expected = valid["asset_subclass"].replace_strict(
            subclass_to_class, default=None
        )
        bad_mask = expected.is_not_null() & (valid["asset_class"] != expected)
        if bad_mask.any():
            offending = valid.filter(bad_mask).with_columns(
                expected.filter(bad_mask).alias("expected_class")
            )
            raise ConfigurationError(
                f"Domain violation: {sheet} has inconsistent "
                f"asset_class / asset_subclass.\n"
                f"{offending}"
            )


def _validate_fx_node_invariants(dict_of_dfs: dict[str, pl.DataFrame]) -> None:
    """
    FX rate node contract:

    1. unit_type == FX_RATE  =>  fx_base_ccy must be populated.
    2. unit_type != FX_RATE  =>  fx_base_ccy must be null.
    3. FX_RATE node_name must equal fx_base_ccy + quote_ccy.
    4. FX_RATE pairs (base, quote) must be unique.
    """
    df = dict_of_dfs["node_meta"]

    is_fx_rate = df["quote_type"] == QuoteType.FX_RATE.value

    missing_base = is_fx_rate & df["fx_base_ccy"].is_null()
    if missing_base.any():
        offending = df.filter(missing_base).select([
            "node_name",
            "quote_type",
            "fx_base_ccy",
            "quote_ccy",
        ])
        raise ConfigurationError(
            f"Domain violation: FX_RATE nodes must specify fx_base_ccy.\n{offending}"
        )

    unexpected_base = ~is_fx_rate & df["fx_base_ccy"].is_not_null()
    if unexpected_base.any():
        offending = df.filter(unexpected_base).select([
            "node_name",
            "asset_class",
            "quote_type",
            "fx_base_ccy",
        ])
        raise ConfigurationError(
            "Domain violation: only FX_RATE nodes may specify fx_base_ccy.\n"
            f"{offending}"
        )

    fx_rate_rows = df.filter(is_fx_rate)

    if fx_rate_rows.is_empty():
        return

    expected_names = fx_rate_rows["fx_base_ccy"].cast(pl.Utf8) + fx_rate_rows[
        "quote_ccy"
    ].cast(pl.Utf8)

    name_mismatch = fx_rate_rows["node_name"] != expected_names

    if name_mismatch.any():
        offending = fx_rate_rows.filter(name_mismatch).select([
            "node_name",
            "fx_base_ccy",
            "quote_ccy",
        ])

        raise ConfigurationError(
            "Domain violation: FX_RATE node_name must equal "
            "fx_base_ccy + quote_ccy.\n"
            f"{offending}"
        )

    pairs = list(
        zip(
            fx_rate_rows["fx_base_ccy"].to_list(),
            fx_rate_rows["quote_ccy"].to_list(),
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


def _validate_axis_roll_pairs(dict_of_dfs: dict[str, pl.DataFrame]) -> None:
    """
    (XAxisKind, RollMethod) compatibility, declared directly on each meta:

      - LABEL containers must have roll_method == NULL.
      - OFFSET_* containers must have roll_method populated.
      - The (XAxisKind, RollMethod) pair must be in
        ALLOWED_X_AXIS_KIND_ROLL_METHOD_PAIRS.

    Vectorized: one pass over each meta table, no rule-table join, no row
    loops. Reads roll_method straight off the container.
    """
    allowed_join: frozenset[str] = frozenset(
        f"{xk.name}|{rm.name}" for xk, rm in ALLOWED_X_AXIS_KIND_ROLL_METHOD_PAIRS
    )

    for sheet, name_col in (
        ("curve_meta", "curve_name"),
        ("surface_meta", "surface_name"),
    ):
        df = dict_of_dfs[sheet]
        if df.is_empty():
            continue

        x_kind = df["x_axis_kind"].cast(pl.Utf8)
        roll_method = df["roll_method"]
        roll_populated = roll_method.is_not_null()
        needs_roll = x_kind.is_in(list(X_AXIS_KINDS_REQUIRING_ROLL))

        # Missing roll_method on OFFSET container.
        missing = needs_roll & ~roll_populated
        if missing.any():
            offending = df.filter(missing).select([name_col, "x_axis_kind"])
            raise ConfigurationError(
                f"{sheet}: OFFSET containers missing roll_method:\n{offending}"
            )

        # Populated roll_method on LABEL container.
        extraneous = ~needs_roll & roll_populated
        if extraneous.any():
            offending = df.filter(extraneous).select([
                name_col,
                "x_axis_kind",
                "roll_method",
            ])
            raise ConfigurationError(
                f"{sheet}: LABEL containers with populated roll_method:\n{offending}"
            )

        # For OFFSET rows, the (x_kind, roll_method) pair must be allowed.
        # Encode each tuple as a single "X|M" string and is_in() against the
        # precomputed allowed set — no Python-level per-row comprehension.
        offset_rows = df.filter(needs_roll)
        if offset_rows.is_empty():
            continue
        joined = (
            offset_rows["x_axis_kind"].cast(pl.Utf8)
            + "|"
            + offset_rows["roll_method"].cast(pl.Utf8)
        )
        bad_mask = ~joined.is_in(list(allowed_join))
        if bad_mask.any():
            offending = offset_rows.filter(bad_mask).select([
                name_col,
                "x_axis_kind",
                "roll_method",
            ])
            raise ConfigurationError(
                f"{sheet}: disallowed (x_axis_kind, roll_method) combinations:\n"
                f"{offending}"
            )


def _validate_roll_params(dict_of_dfs: dict[str, pl.DataFrame]) -> None:
    """
    Method-specific roll-parameter presence, absence, and ranges (fail-fast).

    For every curve_meta / surface_meta row:

      - roll_method NULL (LABEL axis): every roll parameter must be NULL.
      - roll_method populated (OFFSET axis):
          * the method's required parameters must be non-NULL,
          * every other roll parameter must be NULL (no meaningless params),
          * ROLL_ON_MMDD: roll_month in [1, 12] and roll_day valid for month,
          * HALF_YEAR_BY_*: fh_bh_mmdd parses as a valid 'MMDD|MMDD'.

    The required/forbidden contract is ROLL_METHOD_REQUIRED_PARAMS — the same
    table the resolver reads — so validation and resolution cannot drift.

    Iterates rows in Python: these meta tables are tiny (order 10s of rows)
    and this edge check must name the offending table, container, field, and
    rule precisely. Enum validity of roll_method is already enforced upstream
    in _validate_enums, so RollMethod[...] is safe here.
    """
    for sheet, name_col in (
        ("curve_meta", "curve_name"),
        ("surface_meta", "surface_name"),
    ):
        df = dict_of_dfs[sheet]
        if df.is_empty():
            continue

        for row in df.iter_rows(named=True):
            name = row[name_col]
            roll_method_name = row["roll_method"]
            vals: dict[str, object | None] = {
                "anchor_offset": row["anchor_offset"],
                "roll_month": row["roll_month"],
                "roll_day": row["roll_day"],
                "fh_bh_mmdd": row["fh_bh_mmdd"],
            }

            # A Float64 roll column (e.g. from an Excel round-trip) makes a
            # non-integral value like 1.5 reachable here. Reject it; never let
            # a later int() truncate it to 1. Applies whether the row carries
            # the method or not -- meaningless non-integers still fail loud.
            for num_col in ("anchor_offset", "roll_month", "roll_day"):
                num_val = vals[num_col]
                if isinstance(num_val, float) and not num_val.is_integer():
                    raise ConfigurationError(
                        f"{sheet}.{name}: {num_col} is non-integral ({num_val!r}). "
                        f"{num_col} is a discrete integer count; non-integral "
                        f"values are rejected, not truncated."
                    )

            # LABEL axis: no roll metadata at all.
            if roll_method_name is None:
                populated = sorted(k for k, v in vals.items() if v is not None)
                if populated:
                    raise ConfigurationError(
                        f"{sheet}.{name}: roll_method is NULL (LABEL axis) but roll "
                        f"parameters are populated: {populated}. LABEL axes carry no "
                        f"roll metadata."
                    )
                continue

            method = RollMethod[roll_method_name]
            required = ROLL_METHOD_REQUIRED_PARAMS[method]
            forbidden = roll_method_forbidden_params(method)

            missing = sorted(k for k in required if vals[k] is None)
            if missing:
                raise ConfigurationError(
                    f"{sheet}.{name}: roll_method {method.name} requires "
                    f"{sorted(required)} but these are NULL: {missing}."
                )

            extraneous = sorted(k for k in forbidden if vals[k] is not None)
            if extraneous:
                raise ConfigurationError(
                    f"{sheet}.{name}: roll_method {method.name} forbids "
                    f"{sorted(forbidden)} but these are populated: {extraneous}. "
                    f"{method.name} uses only {sorted(required)}."
                )

            if method is RollMethod.CALENDAR_PERIOD:
                # anchor_offset shifts the as-of anchor forward to the spot
                # period; a negative shift would place the spot in the past.
                anchor = int(row["anchor_offset"])
                if anchor < 0:
                    raise ConfigurationError(
                        f"{sheet}.{name}: anchor_offset {anchor} is negative. "
                        f"anchor_offset is a non-negative count of periods "
                        f"added to the as-of anchor."
                    )
            elif method is RollMethod.ROLL_ON_MMDD:
                # Both are guaranteed non-null here by the required-param check
                # above and integral by the gate above; read straight off the
                # (Any-typed) polars row.
                mm = int(row["roll_month"])
                dd = int(row["roll_day"])
                if not 1 <= mm <= 12:
                    raise ConfigurationError(
                        f"{sheet}.{name}: roll_month {mm} out of range [1, 12]."
                    )
                max_day = monthrange(2001, mm)[1]
                if not 1 <= dd <= max_day:
                    raise ConfigurationError(
                        f"{sheet}.{name}: roll_day {dd} out of range [1, {max_day}] "
                        f"for month {mm}."
                    )
            elif method in (
                RollMethod.HALF_YEAR_BY_DATE,
                RollMethod.HALF_YEAR_BY_LABEL,
            ):
                try:
                    parse_fh_bh_mmdd(str(vals["fh_bh_mmdd"]))
                except ValueError as exc:
                    raise ConfigurationError(
                        f"{sheet}.{name}: invalid fh_bh_mmdd: {exc}"
                    ) from exc


def _validate_curve_point_token_grammar(
    dict_of_dfs: dict[str, pl.DataFrame],
) -> None:
    """
    Every curve_point.tenor_token matches its parent's x_axis_kind regex.

    Vectorized: join parent x_axis_kind onto each point row, then group by
    kind and apply the kind-specific regex to each group in one vectorized
    call per group.
    """
    cm_df = dict_of_dfs["curve_meta"]
    cp_df = dict_of_dfs["curve_point_meta"]
    if cp_df.is_empty():
        return

    parent_kind = cm_df.select([
        pl.col("curve_id"),
        pl.col("x_axis_kind").cast(pl.Utf8),
    ])
    work = cp_df.select(["curve_id", "rf_id", "tenor_token"]).join(
        parent_kind, on="curve_id", how="left"
    )

    bad_pieces: list[pl.DataFrame] = []
    for (kind_name,), group in work.group_by(["x_axis_kind"]):
        kind_name_str = str(kind_name)
        regex = X_AXIS_TOKEN_REGEX[XAxisKind[kind_name_str]]
        good = group["tenor_token"].cast(pl.Utf8).str.contains(regex.pattern)
        bad = ~good
        if bad.any():
            bad_pieces.append(group.filter(bad))

    if bad_pieces:
        offending = pl.concat(bad_pieces)
        raise ConfigurationError(
            f"Curve point token grammar violations:\n{offending.head(25)}"
        )


def _validate_surface_point_token_grammar(
    dict_of_dfs: dict[str, pl.DataFrame],
) -> None:
    """
    Every surface_point.x_tenor_token matches its parent's x_axis_kind regex.

    Vectorized: same per-kind-group str-match pattern as the curve-point
    validator.
    """
    sm_df = dict_of_dfs["surface_meta"]
    sp_df = dict_of_dfs["surface_point_meta"]
    if sp_df.is_empty():
        return

    parent_kind = sm_df.select([
        pl.col("surface_id"),
        pl.col("x_axis_kind").cast(pl.Utf8),
    ])
    work = sp_df.select(["surface_id", "rf_id", "x_tenor_token"]).join(
        parent_kind, on="surface_id", how="left"
    )

    bad_pieces: list[pl.DataFrame] = []
    for (kind_name,), group in work.group_by(["x_axis_kind"]):
        kind_name_str = str(kind_name)
        regex = X_AXIS_TOKEN_REGEX[XAxisKind[kind_name_str]]
        good = group["x_tenor_token"].cast(pl.Utf8).str.contains(regex.pattern)
        bad = ~good
        if bad.any():
            bad_pieces.append(group.filter(bad))

    if bad_pieces:
        offending = pl.concat(bad_pieces)
        raise ConfigurationError(
            f"Surface point token grammar violations:\n{offending.head(25)}"
        )


def _validate_discount_invariants(dict_of_dfs: dict[str, pl.DataFrame]) -> None:
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

    is_disc = curve_df["discount_basis"] != DiscountBasis.NOT_A_DISCOUNT_CURVE.name
    is_default = curve_df["is_default_discount_for_ccy"].cast(pl.Boolean)

    # Default => is a real discount curve.
    bad = is_default & ~is_disc
    if bad.any():
        offending = curve_df.filter(bad).select(["curve_name", "discount_basis"])
        raise ConfigurationError(
            f"Default discount flag set on non-discount curve(s):\n{offending}"
        )

    # Discount curves must be IR.
    bad_class = is_disc & (curve_df["asset_class"] != AssetClass.IR.value)
    if bad_class.any():
        offending = curve_df.filter(bad_class).select(["curve_name", "asset_class"])
        raise ConfigurationError(
            f"Discount curves must have asset_class=='IR'.\n{offending}"
        )

    # Default basis must equal the firm standard.
    bad_basis = is_default & (
        curve_df["discount_basis"] != FIRM_DEFAULT_DISCOUNT_BASIS.name
    )
    if bad_basis.any():
        offending = curve_df.filter(bad_basis).select(["curve_name", "discount_basis"])
        raise ConfigurationError(
            f"Default discount curves must use the firm-standard basis "
            f"{FIRM_DEFAULT_DISCOUNT_BASIS.name!r}.\n"
            f"{offending}"
        )

    # Exactly one default per currency that has any discount curve.
    disc_curves = curve_df.filter(is_disc).select([
        "curve_name",
        "quote_ccy",
        "discount_basis",
        "is_default_discount_for_ccy",
    ])
    default_counts = disc_curves.group_by("quote_ccy").agg(
        pl.col("is_default_discount_for_ccy").cast(pl.Int64).sum().alias("n_default")
    )
    bad_ccys = default_counts.filter(pl.col("n_default") != 1)
    if not bad_ccys.is_empty():
        raise ConfigurationError(
            f"Each currency with a discount curve must flag exactly one as "
            f"default. Currencies with !=1 default:\n"
            f"{bad_ccys}"
        )

    # (ccy, basis) is unique among discount curves.
    composite = disc_curves.select(["quote_ccy", "discount_basis"])
    dupe_mask = composite.is_duplicated()
    if dupe_mask.any():
        offending = disc_curves.filter(dupe_mask)
        raise ConfigurationError(
            f"Duplicate (quote_ccy, discount_basis) among discount curves:\n{offending}"
        )

    # Coverage: every currency that's a quote_ccy somewhere must have a
    # discount curve. (Skips currencies that exist in `ccy` but aren't used.)
    quote_ccys_used = (
        pl
        .concat([
            curve_df["quote_ccy"].cast(pl.Utf8),
            surf_df["quote_ccy"].cast(pl.Utf8),
            node_df["quote_ccy"].cast(pl.Utf8),
        ])
        .drop_nulls()
        .unique()
    )
    ccys_with_discount = set(disc_curves["quote_ccy"].cast(pl.Utf8).to_list())
    missing = sorted(set(quote_ccys_used.to_list()) - ccys_with_discount)
    if missing:
        raise ConfigurationError(
            f"Currencies with no discount curve: {missing}. "
            f"Every active quote_ccy must have at least one discount curve."
        )

    # Sanity: discount_basis used by discount curves must be one of the real bases.
    real_bases = set(disc_curves["discount_basis"].cast(pl.Utf8).to_list())
    bogus = real_bases - DISCOUNT_BASES_EXCLUDING_SENTINEL
    if bogus:
        raise ConfigurationError(
            f"Unexpected discount_basis on discount curves: {sorted(bogus)}. "
            f"Allowed: {sorted(DISCOUNT_BASES_EXCLUDING_SENTINEL)}"
        )


def _validate_no_blanks(dict_of_dfs: dict[str, pl.DataFrame]) -> None:
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
            "roll_method": True,
            "fh_bh_mmdd": True,
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
            "roll_method": True,
            "fh_bh_mmdd": True,
            "y_axis_kind": False,
            "y_interp_method": False,
            "y_extrap_method": False,
        },
        "surface_point_meta": {"x_tenor_token": False, "y_label": False},
    }
    for sheet_name, cols in string_cols_by_sheet.items():
        df = dict_of_dfs[sheet_name]
        for col, nullable in cols.items():
            _check_no_blank_strings(df, col, sheet_name, nullable=nullable)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _validate_ccy_id_bounds(dict_of_dfs: dict[str, pl.DataFrame]) -> None:
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


def validate_static_config_dfs(dict_of_dfs: dict[str, pl.DataFrame]) -> None:
    """
    Execute all validation tiers against the loaded config DataFrames.

    Raises ConfigurationError on the first violation with a descriptive message.
    """
    _check_required_sheets(dict_of_dfs)

    expected_columns: dict[str, frozenset[str]] = {
        "ccy": frozenset({"ccy_name", "ccy_id"}),
        "holiday_calendars": frozenset({"holiday_calendar_name", "holiday_date"}),
        "node_meta": frozenset({
            "node_id",
            "node_name",
            "quote_ccy",
            "asset_class",
            "asset_subclass",
            "quote_type",
            "unit_scale",
            "shift_type",
            "fx_base_ccy",
        }),
        "curve_meta": frozenset({
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
            "roll_method",
            "anchor_offset",
            "roll_month",
            "roll_day",
            "fh_bh_mmdd",
            "discount_basis",
            "is_default_discount_for_ccy",
        }),
        "curve_point_meta": frozenset({"rf_id", "curve_id", "tenor_token"}),
        "surface_meta": frozenset({
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
            "roll_method",
            "anchor_offset",
            "roll_month",
            "roll_day",
            "fh_bh_mmdd",
            "y_axis_kind",
            "y_interp_method",
            "y_extrap_method",
        }),
        "surface_point_meta": frozenset({
            "rf_id",
            "surface_id",
            "x_tenor_token",
            "y_label",
        }),
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
    _validate_roll_params(dict_of_dfs)
