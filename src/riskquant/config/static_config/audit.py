# src/riskquant/config/static_config/audit.py
"""
Tabular audit of a compiled RiskStaticConfig.

Auditing is an edge-only diagnostic, not a runtime path. Never imported by
valuation, simlab, or any kernel. The tables are Polars frames (D006) and print
through ``riskquant.utils.printer.print_df``.

Layout: one DataFrame per logical table the 7-table schema exposes,
plus reverse-mapped derived structures (FX spot lookup, discount-curve
lookup, dense per-rf attribute arrays). Each section prints `head(n)`
with a configurable row cap.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict
from typing import Any

import numpy as np
import polars as pl

from riskquant.config.enums.container_topology import ContainerTopologyType
from riskquant.config.enums.discount_basis import DiscountBasis
from riskquant.config.enums.x_axis_kinds import XAxisKind
from riskquant.config.static_config.compile import (
    compile_risk_static_config_from_xlsx,
)
from riskquant.config.static_config.static_config import RiskStaticConfig
from riskquant.utils.printer import print_df


def _print_title(title: str) -> None:
    len_title = len(title) + 4
    ruler = "#" * len_title
    print(f"\n{ruler}")
    print(f"# {title} #")
    print(f"{ruler}")


def _print_subheader(title: str) -> None:
    ruler = "-" * (len(title) + 0)
    print("\n")
    print(f"{title}")
    print(f"{ruler}")


def _df_from_rows(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """
    Build a frame from row dicts; no rows give an empty frame.

    ``infer_schema_length=None`` reads every row for the schema. The Polars
    default reads 100 rows and fails on a later value of another type.
    ``pl.from_dicts`` raises ``NoDataError`` for an empty list.
    """
    if not rows:
        return pl.DataFrame()
    return pl.from_dicts(rows, infer_schema_length=None)


def _df_from_dataclass_values(values: Iterable[Any]) -> pl.DataFrame:
    """Build a frame from an iterable of frozen dataclasses."""
    return _df_from_rows([asdict(v) for v in values])


def _print_table(df: pl.DataFrame) -> None:
    """Print one audit table: every cell as its native text, no index."""
    print_df(df, float_format=None, add_space_before=False, add_space_after=False)


# ============================================================
# Per-table renderers
# ============================================================


def _audit_ccy(rsc: RiskStaticConfig, n: int) -> None:
    _print_subheader("1. Currencies (ccy)")
    df = _df_from_dataclass_values(rsc.ccy_dict.values())
    if df.is_empty():
        print("(empty)")
        return
    _print_table(df.head(n))
    print(f"  Total: {rsc.num_ccys}")


def _audit_holiday_calendars(rsc: RiskStaticConfig, n: int) -> None:
    _print_subheader("2. Holiday calendars (counts per calendar)")
    rows = [
        {
            "holiday_calendar_name": name,
            "n_holidays": int(cal.holiday_dates_ord.size),
            "first_ord": int(cal.holiday_dates_ord[0])
            if cal.holiday_dates_ord.size
            else None,
            "last_ord": int(cal.holiday_dates_ord[-1])
            if cal.holiday_dates_ord.size
            else None,
        }
        for name, cal in rsc.holiday_calendars_dict.items()
    ]
    df = _df_from_rows(rows)
    if df.is_empty():
        print("(empty)")
        return
    _print_table(df.head(n))
    print(f"  Total calendars: {len(rsc.holiday_calendars_dict)}")


def _audit_node_meta(rsc: RiskStaticConfig, n: int) -> None:
    _print_subheader(
        "3. Node meta (scalar risk factors; FX + equity + credit single names)"
    )
    df = _df_from_dataclass_values(rsc.node_meta_dict.values())
    if df.is_empty():
        print("(empty)")
        return
    _print_table(df.head(n))
    print(
        f"  Total nodes: {rsc.num_nodes}  (FX: {sum(1 for n_ in rsc.node_meta_dict.values() if n_.fx_base_ccy is not None)})"
    )


def _audit_curve_meta(rsc: RiskStaticConfig, n: int) -> None:
    _print_subheader("4. Curve meta (1D functions; discount + non-discount)")
    df = _df_from_dataclass_values(rsc.curve_meta_dict.values())
    if df.is_empty():
        print("(empty)")
        return
    _print_table(df.head(n))
    is_disc = df.get_column("discount_basis") != DiscountBasis.NOT_A_DISCOUNT_CURVE.name
    print(
        f"  Total curves: {rsc.num_curves}  "
        f"(discount: {int(is_disc.sum())}, "
        f"default-flagged: {int(df.get_column('is_default_discount_for_ccy').sum())})"
    )


def _audit_curve_point_meta(rsc: RiskStaticConfig, n: int) -> None:
    _print_subheader("5. Curve point counts (per parent curve)")
    counts: dict[int, int] = {}
    for pt in rsc.curve_point_meta_dict.values():
        counts[pt.curve_id] = counts.get(pt.curve_id, 0) + 1
    rows = []
    for cid, cm in rsc.curve_meta_dict.items():
        rows.append({
            "curve_id": cid,
            "curve_name": cm.curve_name,
            "x_axis_kind": cm.x_axis_kind,
            "n_points": counts.get(cid, 0),
        })
    df = _df_from_rows(rows)
    if df.is_empty():
        print("(empty)")
        return
    _print_table(df.head(n))
    print(f"  Total curve points: {rsc.num_curve_points}")


def _audit_surface_meta(rsc: RiskStaticConfig, n: int) -> None:
    _print_subheader("6. Surface meta (2D functions; vol surfaces)")
    df = _df_from_dataclass_values(rsc.surface_meta_dict.values())
    if df.is_empty():
        print("(empty)")
        return
    _print_table(df.head(n))
    print(f"  Total surfaces: {rsc.num_surfaces}")


def _audit_surface_point_meta(rsc: RiskStaticConfig, n: int) -> None:
    _print_subheader("7. Surface point counts (per parent surface)")
    counts: dict[int, int] = {}
    for pt in rsc.surface_point_meta_dict.values():
        counts[pt.surface_id] = counts.get(pt.surface_id, 0) + 1
    rows = []
    for sid, sm in rsc.surface_meta_dict.items():
        rows.append({
            "surface_id": sid,
            "surface_name": sm.surface_name,
            "x_axis_kind": sm.x_axis_kind,
            "y_axis_kind": sm.y_axis_kind,
            "n_points": counts.get(sid, 0),
        })
    df = _df_from_rows(rows)
    if df.is_empty():
        print("(empty)")
        return
    _print_table(df.head(n))
    print(f"  Total surface points: {rsc.num_surface_points}")


# ============================================================
# Derived structures
# ============================================================


def _audit_fx_spot_lookup(rsc: RiskStaticConfig, n: int) -> None:
    _print_subheader("8. FX spot lookup ((base_ccy_id, quote_ccy_id) -> rf_id)")
    if not rsc.fx_spot_lookup:
        print("(no FX nodes)")
        return
    rows = []
    for (base_id, quote_id), rf_id in rsc.fx_spot_lookup.items():
        base_name = (
            rsc.ccy_id_to_name[int(base_id)]
            if int(base_id) < len(rsc.ccy_id_to_name)
            else "?"
        )
        quote_name = (
            rsc.ccy_id_to_name[int(quote_id)]
            if int(quote_id) < len(rsc.ccy_id_to_name)
            else "?"
        )
        node = rsc.node_meta_dict.get(int(rf_id))
        rows.append({
            "base_ccy_id": int(base_id),
            "base_ccy": base_name,
            "quote_ccy_id": int(quote_id),
            "quote_ccy": quote_name,
            "rf_id": int(rf_id),
            "node_name": node.node_name if node else "?",
        })
    df = _df_from_rows(rows).sort(["base_ccy", "quote_ccy"], maintain_order=True)
    _print_table(df.head(n))
    print(f"  Total FX pairs: {rsc.num_fx_pairs}")


def _audit_discount_curves(rsc: RiskStaticConfig, n: int) -> None:
    _print_subheader("9. Discount-curve lookups")

    # 10a — per-(ccy, basis) full table
    print("\n  (a) (ccy_id, basis_id) -> curve_id")
    rows = []
    for (ccy_id, basis_id), curve_id in rsc.discount_curve_lookup.items():
        cm = rsc.curve_meta_dict[int(curve_id)]
        ccy_name = rsc.ccy_id_to_name[int(ccy_id)]
        basis_name = DiscountBasis(int(basis_id)).name
        rows.append({
            "ccy_id": int(ccy_id),
            "ccy": ccy_name,
            "basis_id": int(basis_id),
            "basis": basis_name,
            "curve_id": int(curve_id),
            "curve_name": cm.curve_name,
            "is_default": cm.is_default_discount_for_ccy,
        })
    if rows:
        df = _df_from_rows(rows).sort(["ccy", "basis"], maintain_order=True)
        _print_table(df.head(n))
    else:
        print("  (no discount curves)")

    # 10b — per-ccy default
    print("\n  (b) ccy_id -> default discount curve_id")
    rows = []
    for ccy_id_int, curve_id in enumerate(rsc.ccy_id_to_default_discount_curve_id):
        cid = int(curve_id)
        if cid < 0:
            continue
        ccy_name = rsc.ccy_id_to_name[ccy_id_int]
        cm = rsc.curve_meta_dict[cid]
        rows.append({
            "ccy_id": ccy_id_int,
            "ccy": ccy_name,
            "curve_id": cid,
            "curve_name": cm.curve_name,
            "basis": cm.discount_basis,
        })
    if rows:
        df = _df_from_rows(rows).sort("ccy", maintain_order=True)
        _print_table(df.head(n))
    else:
        print("  (no default-flagged discount curves)")


def _audit_dense_rf_arrays(rsc: RiskStaticConfig, n: int) -> None:
    _print_subheader("10. Dense per-rf attribute arrays (head)")
    n_total = rsc.max_rf_id + 1
    # Mask out empty slots (rf_id not occupied).
    occupied = rsc.rf_id_to_parent_topology_type >= ContainerTopologyType.NODE.value
    if not occupied.any():
        print("(no risk factors)")
        return

    df = pl.DataFrame({
        "rf_id": np.arange(n_total, dtype=np.int32),
        "parent_topology_type": rsc.rf_id_to_parent_topology_type,
        "parent_id": rsc.rf_id_to_parent_id,
        "ccy_id": rsc.rf_id_to_ccy_id,
        "shift_type_id": rsc.rf_id_to_shift_type_id,
        "unit_scale_x": rsc.rf_id_to_unit_scale_factor,
    })
    df = df.filter(pl.Series(occupied))
    _print_table(df.head(n))
    print(f"  Occupied rf_id slots: {int(occupied.sum())} / {n_total} array length")


def _audit_dense_container_arrays(rsc: RiskStaticConfig, n: int) -> None:
    _print_subheader("11. Per-container attribute arrays (slot-indexed; head)")
    n_total = rsc.num_containers
    occupied = rsc.container_topology_type > ContainerTopologyType.NODE.value
    if not occupied.any():
        print("(no containers)")
        return

    x_kind_names = [
        XAxisKind(int(k)).name if int(k) >= 0 else "-"
        for k in rsc.container_x_axis_kind
    ]
    # roll_method is owned by the curve / surface meta directly; source it
    # from the meta dicts by container_id (no per-container SoA roll array
    # exists — nothing in the numerical core consumes one).
    roll_by_container: dict[int, str] = {}
    for cm in rsc.curve_meta_dict.values():
        roll_by_container[int(cm.curve_id)] = cm.roll_method or "-"
    for sm in rsc.surface_meta_dict.values():
        roll_by_container[int(sm.surface_id)] = sm.roll_method or "-"
    roll_method_col = [
        roll_by_container.get(int(cid), "-") for cid in rsc.container_ids
    ]
    df = pl.DataFrame({
        "container_id": rsc.container_ids,
        "topology_type": rsc.container_topology_type,  # 1=curve, 2=surface
        "ccy_id": rsc.container_ccy_id,
        "interp_x_id": rsc.container_interp_x_id,
        "extrap_x_id": rsc.container_extrap_x_id,
        "interp_y_id": rsc.container_interp_y_id,
        "extrap_y_id": rsc.container_extrap_y_id,
        "x_axis_kind": x_kind_names,
        "roll_method": roll_method_col,
    })
    df = df.filter(pl.Series(occupied))
    _print_table(df.head(n))
    print(f"  Containers: {int(occupied.sum())} / {n_total} slots")


def _audit_tenor_index(rsc: RiskStaticConfig, n: int) -> None:
    _print_subheader("12. Per-curve tenor index summary")
    rows = []
    for curve_id, idx in rsc.curve_id_to_tenor_index.items():
        cm = rsc.curve_meta_dict[curve_id]
        rows.append({
            "curve_id": curve_id,
            "curve_name": cm.curve_name,
            "x_axis_kind": cm.x_axis_kind,
            "n_offsets": int(idx.sorted_offsets.size),
            "n_labels": len(idx.label_to_rf_id),
            "min_offset": int(idx.sorted_offsets[0])
            if idx.sorted_offsets.size
            else None,
            "max_offset": int(idx.sorted_offsets[-1])
            if idx.sorted_offsets.size
            else None,
        })
    df = _df_from_rows(rows)
    if df.is_empty():
        print("(empty)")
        return
    _print_table(df.head(n))


# ============================================================
# Public API
# ============================================================


def audit_risk_static_config(rsc: RiskStaticConfig, n: int = 10) -> None:
    """
    Print a tabular audit of every table and derived structure on `rsc`.

    Parameters
    ----------
    rsc
        Compiled static config to audit.
    n
        Row cap per section. Use a large number for full dumps.
    """
    _print_title("RiskStaticConfig Audit")

    print("Summary:")
    print(f"  Currencies          : {rsc.num_ccys}")
    print(f"  Holiday calendars   : {len(rsc.holiday_calendars_dict)}")
    print(f"  Nodes               : {rsc.num_nodes}")
    print(f"  Curves              : {rsc.num_curves}")
    print(f"  Curve points        : {rsc.num_curve_points}")
    print(f"  Surfaces            : {rsc.num_surfaces}")
    print(f"  Surface points      : {rsc.num_surface_points}")
    print(f"  Risk factors total  : {rsc.num_risk_factors}")
    print(f"  Containers total    : {rsc.num_containers}")
    print(f"  FX pairs            : {rsc.num_fx_pairs}")
    print(f"  Max rf_id           : {rsc.max_rf_id}")
    print(f"  Max container_id    : {rsc.max_container_id}")

    _audit_ccy(rsc, n)
    _audit_holiday_calendars(rsc, n)
    _audit_node_meta(rsc, n)
    _audit_curve_meta(rsc, n)
    _audit_curve_point_meta(rsc, n)
    _audit_surface_meta(rsc, n)
    _audit_surface_point_meta(rsc, n)
    _audit_fx_spot_lookup(rsc, n)
    _audit_discount_curves(rsc, n)
    _audit_dense_rf_arrays(rsc, n)
    _audit_dense_container_arrays(rsc, n)
    _audit_tenor_index(rsc, n)

    _print_title("End of RiskStaticConfig Audit")


if __name__ == "__main__":
    from riskquant.utils.printer import print_calling_file, print_success_message

    print_calling_file()

    DEV_DATA_DIR: str = (
        "C:/Users/mharvey/Documents/code/riskquant/data/canonical_examples"
    )
    DEV_DATA_FILE_NAME_STATIC_CONFIG: str = "canonical_risk_static_config.xlsx"

    rsc = compile_risk_static_config_from_xlsx(
        path=f"{DEV_DATA_DIR}/{DEV_DATA_FILE_NAME_STATIC_CONFIG}"
    )

    rsc.summary()
    audit_risk_static_config(rsc=rsc, n=200)

    print_success_message()
