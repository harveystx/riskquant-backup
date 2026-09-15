# tests/ingest_fixtures/config/risk_static_config/test_parts_unit.py
"""
Per-table unit tests for the individual `_parts/*.py` fixture modules.

These are finer-grained than `test_canonical_fixture.py`'s whole-fixture
tests: each test here targets exactly one table's internal structural
invariants, independent of the compile/validate pipeline (which already
has its own extensive coverage in
`tests/config/static_config/test_validate_tier5_domain_invariants.py`).
The goal is to catch a broken _parts file at the source, with a message
that points straight at the offending table.
"""

from __future__ import annotations

from riskquant.ingest_fixtures.config.risk_static_config._parts.ccy import (
    CANONICAL_CCY_DATA,
)
from riskquant.ingest_fixtures.config.risk_static_config._parts.curve_meta import (
    CANONICAL_CURVE_META_DATA,
)
from riskquant.ingest_fixtures.config.risk_static_config._parts.curve_point_meta import (
    CANONICAL_CURVE_POINT_META_DATA,
)
from riskquant.ingest_fixtures.config.risk_static_config._parts.holiday_calendars import (
    CANONICAL_HOLIDAY_CALENDARS_DATA,
)
from riskquant.ingest_fixtures.config.risk_static_config._parts.node_meta import (
    CANONICAL_NODE_META_DATA,
)
from riskquant.ingest_fixtures.config.risk_static_config._parts.surface_meta import (
    CANONICAL_SURFACE_META_DATA,
)
from riskquant.ingest_fixtures.config.risk_static_config._parts.surface_point_meta import (
    CANONICAL_SURFACE_POINT_META_DATA,
)


def _col_lengths(data: dict) -> set[int]:
    return {len(v) for v in data.values()}


# ---------------------------------------------------------------------------
# ccy
# ---------------------------------------------------------------------------


def test_ccy_names_are_unique():
    names = CANONICAL_CCY_DATA["ccy_name"]
    assert len(names) == len(set(names))


def test_ccy_ids_are_unique_and_within_int8():
    ids = CANONICAL_CCY_DATA["ccy_id"]
    assert len(ids) == len(set(ids))
    assert all(0 <= i <= 127 for i in ids)


# ---------------------------------------------------------------------------
# node_meta
# ---------------------------------------------------------------------------


def test_node_meta_column_lengths_consistent():
    assert len(_col_lengths(CANONICAL_NODE_META_DATA)) == 1


def test_node_ids_are_dense_from_zero():
    ids = CANONICAL_NODE_META_DATA["node_id"]
    assert list(ids) == list(range(len(ids))), (
        "node_id is documented as 0..n_nodes-1 (see node_meta.py's rf_id "
        "allocation comment); a gap or reorder here would silently shift "
        "every FK-referencing test/demo that assumes this."
    )


def test_fx_nodes_have_base_ccy_non_fx_nodes_do_not():
    for name, quote_type, fx_base in zip(
        CANONICAL_NODE_META_DATA["node_name"],
        CANONICAL_NODE_META_DATA["quote_type"],
        CANONICAL_NODE_META_DATA["fx_base_ccy"],
        strict=True,
    ):
        if quote_type == "FX_RATE":
            assert fx_base is not None, f"{name}: FX_RATE node missing fx_base_ccy"
        else:
            assert fx_base is None, f"{name}: non-FX_RATE node has fx_base_ccy set"


def test_fx_node_names_equal_base_plus_quote_ccy():
    for name, quote_type, fx_base, quote_ccy in zip(
        CANONICAL_NODE_META_DATA["node_name"],
        CANONICAL_NODE_META_DATA["quote_type"],
        CANONICAL_NODE_META_DATA["fx_base_ccy"],
        CANONICAL_NODE_META_DATA["quote_ccy"],
        strict=True,
    ):
        if quote_type == "FX_RATE":
            assert name == f"{fx_base}{quote_ccy}"


# ---------------------------------------------------------------------------
# curve_meta
# ---------------------------------------------------------------------------


def test_curve_meta_column_lengths_consistent():
    assert len(_col_lengths(CANONICAL_CURVE_META_DATA)) == 1


def test_curve_ids_are_unique_and_in_curves_namespace():
    from riskquant.config.namespace import GlobalIdNamespace

    ids = CANONICAL_CURVE_META_DATA["curve_id"]
    assert len(ids) == len(set(ids))
    for cid in ids:
        assert GlobalIdNamespace.CURVES.min_id <= cid <= GlobalIdNamespace.CURVES.max_id


def test_label_curves_have_no_roll_method_offset_curves_do():
    for name, x_kind, roll_method in zip(
        CANONICAL_CURVE_META_DATA["curve_name"],
        CANONICAL_CURVE_META_DATA["x_axis_kind"],
        CANONICAL_CURVE_META_DATA["roll_method"],
        strict=True,
    ):
        if x_kind == "LABEL":
            assert roll_method is None, f"{name}: LABEL curve has roll_method set"
        else:
            assert roll_method is not None, f"{name}: OFFSET curve missing roll_method"


def test_non_discount_curves_carry_the_sentinel_basis():
    for name, asset_class, basis, is_default in zip(
        CANONICAL_CURVE_META_DATA["curve_name"],
        CANONICAL_CURVE_META_DATA["asset_class"],
        CANONICAL_CURVE_META_DATA["discount_basis"],
        CANONICAL_CURVE_META_DATA["is_default_discount_for_ccy"],
        strict=True,
    ):
        if asset_class != "IR":
            assert basis == "NOT_A_DISCOUNT_CURVE", (
                f"{name}: non-IR curve has a real basis"
            )
            assert is_default is False, (
                f"{name}: non-IR curve flagged as default discount"
            )


def test_exactly_one_default_discount_curve_per_ir_currency():
    from collections import Counter

    defaults_by_ccy = Counter()
    for ccy, asset_class, is_default in zip(
        CANONICAL_CURVE_META_DATA["quote_ccy"],
        CANONICAL_CURVE_META_DATA["asset_class"],
        CANONICAL_CURVE_META_DATA["is_default_discount_for_ccy"],
        strict=True,
    ):
        if asset_class == "IR" and is_default:
            defaults_by_ccy[ccy] += 1

    ir_ccys = {
        ccy
        for ccy, asset_class in zip(
            CANONICAL_CURVE_META_DATA["quote_ccy"],
            CANONICAL_CURVE_META_DATA["asset_class"],
            strict=True,
        )
        if asset_class == "IR"
    }
    for ccy in ir_ccys:
        assert defaults_by_ccy[ccy] == 1, (
            f"{ccy}: expected exactly 1 default discount curve"
        )


# ---------------------------------------------------------------------------
# curve_point_meta
# ---------------------------------------------------------------------------


def test_curve_point_meta_column_lengths_consistent():
    assert len(_col_lengths(CANONICAL_CURVE_POINT_META_DATA)) == 1


def test_curve_point_rf_ids_are_unique():
    ids = CANONICAL_CURVE_POINT_META_DATA["rf_id"]
    assert len(ids) == len(set(ids))


def test_every_curve_point_references_a_real_curve():
    valid_curve_ids = set(CANONICAL_CURVE_META_DATA["curve_id"])
    for cid in CANONICAL_CURVE_POINT_META_DATA["curve_id"]:
        assert cid in valid_curve_ids


def test_every_curve_has_at_least_one_point():
    curve_ids_with_points = set(CANONICAL_CURVE_POINT_META_DATA["curve_id"])
    all_curve_ids = set(CANONICAL_CURVE_META_DATA["curve_id"])
    assert all_curve_ids <= curve_ids_with_points, (
        f"Curves with zero points: {all_curve_ids - curve_ids_with_points}"
    )


# ---------------------------------------------------------------------------
# surface_meta / surface_point_meta
# ---------------------------------------------------------------------------


def test_surface_meta_column_lengths_consistent():
    assert len(_col_lengths(CANONICAL_SURFACE_META_DATA)) == 1


def test_surface_point_meta_column_lengths_consistent():
    assert len(_col_lengths(CANONICAL_SURFACE_POINT_META_DATA)) == 1


def test_every_surface_point_references_a_real_surface():
    valid_surface_ids = set(CANONICAL_SURFACE_META_DATA["surface_id"])
    for sid in CANONICAL_SURFACE_POINT_META_DATA["surface_id"]:
        assert sid in valid_surface_ids


def test_surface_point_rf_ids_are_unique():
    ids = CANONICAL_SURFACE_POINT_META_DATA["rf_id"]
    assert len(ids) == len(set(ids))


def test_each_surface_is_a_complete_grid():
    """Every surface's points form a full x_tenor_token x y_label grid (no
    missing cells) -- important because the surface interpolator assumes
    dense pillars."""
    from collections import defaultdict

    points_by_surface: dict[int, set[tuple[str, str]]] = defaultdict(set)
    for sid, x_tok, y_lbl in zip(
        CANONICAL_SURFACE_POINT_META_DATA["surface_id"],
        CANONICAL_SURFACE_POINT_META_DATA["x_tenor_token"],
        CANONICAL_SURFACE_POINT_META_DATA["y_label"],
        strict=True,
    ):
        points_by_surface[sid].add((x_tok, y_lbl))

    for sid, points in points_by_surface.items():
        x_toks = {p[0] for p in points}
        y_lbls = {p[1] for p in points}
        expected_grid_size = len(x_toks) * len(y_lbls)
        assert len(points) == expected_grid_size, (
            f"surface_id={sid}: {len(points)} points but "
            f"{len(x_toks)} x-tokens x {len(y_lbls)} y-labels = "
            f"{expected_grid_size} expected -- grid has a hole."
        )


# ---------------------------------------------------------------------------
# rf_id global uniqueness across ALL three atomic tables
# ---------------------------------------------------------------------------


def test_rf_ids_are_globally_unique_across_nodes_curve_points_surface_points():
    node_ids = set(CANONICAL_NODE_META_DATA["node_id"])
    curve_pt_ids = set(CANONICAL_CURVE_POINT_META_DATA["rf_id"])
    surface_pt_ids = set(CANONICAL_SURFACE_POINT_META_DATA["rf_id"])

    assert node_ids.isdisjoint(curve_pt_ids), "node_id / curve_point.rf_id overlap"
    assert node_ids.isdisjoint(surface_pt_ids), "node_id / surface_point.rf_id overlap"
    assert curve_pt_ids.isdisjoint(surface_pt_ids), (
        "curve_point.rf_id / surface_point.rf_id overlap"
    )


# ---------------------------------------------------------------------------
# holiday_calendars
# ---------------------------------------------------------------------------


def test_holiday_calendars_column_lengths_consistent():
    assert len(_col_lengths(CANONICAL_HOLIDAY_CALENDARS_DATA)) == 1


def test_holiday_calendars_are_currently_unreferenced_by_anything_else():
    """
    Documents a real audit finding: holiday_calendars is a genuinely
    standalone table today -- no other table's column references
    holiday_calendar_name, and (per repo-wide grep during the audit) no
    production code reads `RiskStaticConfig.holiday_calendars_dict` at all.
    It compiles correctly and is presumably forward-looking scaffolding for
    a not-yet-implemented business-day/roll-adjustment engine.

    This test is intentionally a documentation pin, not a correctness
    check: if a future change wires holiday_calendars into another table
    (e.g. a `holiday_calendar_name` FK on `ccy`), this test should be
    deleted rather than "fixed", since its premise (nothing references
    this table) will no longer hold.
    """
    calendar_names = set(CANONICAL_HOLIDAY_CALENDARS_DATA["holiday_calendar_name"])
    # Naming convention is currently a mix of country codes (US, GB, CH, AU)
    # and one currency code (EUR) -- harmless while nothing consumes the
    # table by name, but worth a heads-up for whoever wires it up next.
    assert calendar_names == {"US", "GB", "CH", "AU", "EUR"}
