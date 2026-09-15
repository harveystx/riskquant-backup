# scripts/demos/config/static_config/demo_walkthrough.py
"""
A verbose, narrated, end-to-end walkthrough of `riskquant.config.static_config`.

This script is written for a reader who has never touched this package
before. Every stage prints a plain-English explanation of what is about to
happen and why, before showing the actual code and its output. Read it
top to bottom; each section builds on the previous one.

Run it directly:

    python scripts/demos/config/static_config/demo_walkthrough.py

Sections
--------
  1. What is static_config, and what does it compile FROM?
  2. Loading the canonical ingest fixture (Polars DataFrames).
  3. Compiling: validate -> typed dataclasses -> RiskStaticConfig.
  4. Inspecting the compiled RiskStaticConfig (nodes, curves, surfaces, FX).
  5. Resolving risk-factor names for a specific as-of date.
  6. The three roll-method combinations the canonical fixture does NOT
     exercise -- standalone worked examples for each.
  7. What validation failure actually looks like (intentionally broken
     input, on purpose, to show the error message a real typo would
     produce).
  8. Loading from an actual .xlsx workbook (the production entry point).
"""

from __future__ import annotations

from datetime import date
from typing import Any, cast

import polars as pl

from riskquant.config.static_config.compile import (
    compile_risk_static_config_from_dict_of_dfs,
)
from riskquant.config.static_config.rf_resolution import (
    build_risk_factor_name_mapping_for_as_of_date,
)
from riskquant.exceptions import ConfigurationError
from riskquant.ingest_fixtures.config.risk_static_config.canonical_risk_static_config import (
    CANONICAL_RISK_STATIC_CONFIG_DATA,
)


def _section(title: str) -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def _sub(title: str) -> None:
    print()
    print(f"--- {title} ---")


# =============================================================================
# 1. WHAT IS static_config?
# =============================================================================


def section_1_what_is_static_config() -> None:
    _section("1. What is riskquant.config.static_config?")
    print(
        """
`riskquant.config.static_config` compiles a firm's risk-factor universe
(currencies, curves, surfaces, and every individual point on them) from a
human-editable source -- an Excel workbook, or an equivalent dict of
tables -- into ONE immutable, typed, dataframe-free runtime object:
`RiskStaticConfig`.

That compiled object is what every downstream valuation, scenario, and
risk calculation in RiskQuant reads. Nothing downstream ever re-parses a
spreadsheet at runtime; everything downstream reads dense NumPy arrays and
plain dicts off `RiskStaticConfig`.

The pipeline, in one line:

    7 source tables (Polars DataFrames)
        --validate-->  ConfigurationError if anything is wrong
        --compile-->   typed per-row dataclasses (NodeMeta, CurveMeta, ...)
        --assemble-->  RiskStaticConfig (dense arrays, O(1) lookups)

The 7 source tables are:
  ccy                 -- the currency universe
  holiday_calendars   -- named holiday-date lists (see section 4 for a note
                          on this table's current status)
  node_meta           -- scalar risk factors (an FX rate, a single equity
                          price, a single ATM vol)
  curve_meta          -- named curves (e.g. a discount curve, a commodity
                          forward curve) and their axis conventions
  curve_point_meta    -- the individual points ON each curve
  surface_meta        -- named 2D vol surfaces and their axis conventions
  surface_point_meta  -- the individual (x, y) points ON each surface

Roll / resolution metadata is embedded directly in curve_meta and
surface_meta (roll_method plus method-specific parameters). There is no
separate roll-rule table.
"""
    )


# =============================================================================
# 2. LOADING THE CANONICAL FIXTURE
# =============================================================================


def section_2_load_fixture() -> dict[str, pl.DataFrame]:
    _section("2. Loading the canonical ingest fixture")
    print(
        """
For this walkthrough we use the CANONICAL fixture -- a hand-built,
internally-consistent example universe shipped at:

    src/riskquant/ingest_fixtures/config/risk_static_config/

It exists for exactly this purpose: a small, realistic, always-valid
reference dataset that tests and demos can build on without needing a real
firm's data. It covers:
  - 5 currencies (AUD, CHF, EUR, GBP, USD)
  - 10 scalar risk factors (6 FX rates, 1 equity price, 3 ATM vols)
  - 13 curves (8 IR discount curves, 5 commodity forward curves)
  - 98 individual curve points
  - 3 vol surfaces (commodity, FX, equity -- each with a DIFFERENT y-axis
    convention, to show the surface schema is not FX/equity-specific)
  - 60 individual surface points

Each table is Python-native data (dicts of lists) defined in its own
`_parts/*.py` file, purely for readability -- there is no requirement that
your own data start this way. What static_config actually consumes is a
`dict[str, polars.DataFrame]`, one DataFrame per table above. We build
that here explicitly so the conversion step is visible.
"""
    )

    dict_of_dfs: dict[str, pl.DataFrame] = {
        table_name: pl.DataFrame(cast(dict[str, Any], table_data))
        for table_name, table_data in CANONICAL_RISK_STATIC_CONFIG_DATA.items()
    }

    _sub("The 7 tables, as loaded")
    for name, df in dict_of_dfs.items():
        print(f"  {name:<20s} shape={df.shape}")

    _sub("A closer look at one table: node_meta (the scalar risk factors)")
    print(dict_of_dfs["node_meta"])

    return dict_of_dfs


# =============================================================================
# 3. COMPILING
# =============================================================================


def section_3_compile(dict_of_dfs: dict[str, pl.DataFrame]):
    _section("3. Compiling: validate -> typed dataclasses -> RiskStaticConfig")
    print(
        """
`compile_risk_static_config_from_dict_of_dfs` does three things, in order:

  1. VALIDATE (riskquant.config.static_config.validate). Roughly 15 checks
     run against the raw tables: every required column is present, every
     enum-typed column (asset_class, quote_type, x_axis_kind, ...) only
     contains values the system actually understands, every foreign key
     resolves (a curve's quote_ccy must exist in the ccy table; a curve
     point's curve_id must exist in curve_meta), and a set of bespoke
     domain rules fire (an FX-rate node's name must equal
     fx_base_ccy+quote_ccy; every currency with a discount curve must have
     EXACTLY one flagged as the default; and more).

     If ANY check fails, this raises `ConfigurationError` with a message
     that names the exact table, column, and offending value(s) -- see
     section 7 for a live example.

  2. COMPILE. Once validation passes, every raw table row becomes a typed,
     immutable dataclass instance (NodeMeta, CurveMeta, CurvePointMeta,
     ...). This is the ONLY place a raw cell value (which arrives as a
     loosely-typed Python object) is coerced into its final Python type
     (str, int, float, bool).

  3. ASSEMBLE. `RiskStaticConfig.__init__` takes the typed dataclass dicts
     and builds every derived, dense, O(1)-lookup structure it exposes:
     name<->id maps, a currency-pair FX spot matrix, slot indices for
     curves/surfaces, and more. This happens exactly once, at startup --
     never again during a risk run.
"""
    )

    rsc = compile_risk_static_config_from_dict_of_dfs(dict_of_dfs)

    print("Compiled successfully. No ConfigurationError was raised.")
    return rsc


# =============================================================================
# 4. INSPECTING THE COMPILED CONFIG
# =============================================================================


def section_4_inspect(rsc) -> None:
    _section("4. Inspecting the compiled RiskStaticConfig")
    print(
        """
`RiskStaticConfig` exposes everything downstream code needs as plain
dicts and dense NumPy arrays -- no strings are re-parsed, no DataFrame
exists anywhere in this object.
"""
    )

    _sub("Headline counts")
    print(f"  num_ccys            = {rsc.num_ccys}")
    print(f"  num_nodes           = {rsc.num_nodes}")
    print(f"  num_curves          = {rsc.num_curves}")
    print(f"  num_curve_points    = {rsc.num_curve_points}")
    print(f"  num_surfaces        = {rsc.num_surfaces}")
    print(f"  num_surface_points  = {rsc.num_surface_points}")
    print(
        f"  num_risk_factors    = {rsc.num_risk_factors}  "
        f"(= nodes + curve_points + surface_points)"
    )
    print(f"  num_fx_pairs        = {rsc.num_fx_pairs}")

    _sub("Currency lookups (ccy_name -> ccy_id, both directions)")
    for name, cid in sorted(rsc.ccy_name_to_id.items()):
        print(f"  {name} <-> {int(cid)}")

    _sub("Every named risk factor (node), and its rf_id")
    for node in rsc.node_meta_dict.values():
        print(
            f"  rf_id={node.node_id:<3d}  {node.node_name:<22s}  "
            f"asset_class={node.asset_class:<10s}  quote_type={node.quote_type}"
        )

    _sub("Every named curve, its axis convention, and point count")
    points_per_curve: dict[int, int] = {}
    for pt in rsc.curve_point_meta_dict.values():
        points_per_curve[pt.curve_id] = points_per_curve.get(pt.curve_id, 0) + 1
    for curve in rsc.curve_meta_dict.values():
        n_pts = points_per_curve.get(curve.curve_id, 0)
        print(
            f"  {curve.curve_name:<22s}  x_axis_kind={curve.x_axis_kind:<10s}  "
            f"discount_basis={curve.discount_basis:<20s}  points={n_pts}"
        )

    _sub("The FX spot lookup: (base_ccy_id, quote_ccy_id) -> rf_id")
    print(
        "  Only pairs with an explicit FX_RATE node in node_meta resolve;\n"
        "  everything else is intentionally absent (there is no 'inverse\n"
        "  quote' synthesis at this layer)."
    )
    for (base_id, quote_id), rf_id in sorted(rsc.fx_spot_lookup.items()):
        base_name = rsc.ccy_id_to_name[base_id]
        quote_name = rsc.ccy_id_to_name[quote_id]
        print(f"  ({base_name}, {quote_name}) -> rf_id={rf_id}")

    _sub("The default discount curve for each currency")
    print(
        "  ccy_id_to_default_discount_curve_id is a dense array indexed by\n"
        "  ccy_id, with a negative sentinel (-999999, INVALID_UNIT_SCALE-style)\n"
        "  for any currency that has no discount curve at all."
    )
    for ccy_name, ccy_id in sorted(rsc.ccy_name_to_id.items(), key=lambda kv: kv[1]):
        curve_id = int(rsc.ccy_id_to_default_discount_curve_id[int(ccy_id)])
        if curve_id < 0:
            print(f"  {ccy_name} -> (no discount curve)")
        else:
            curve_name = rsc.curve_meta_dict[curve_id].curve_name
            print(f"  {ccy_name} -> {curve_name}")


# =============================================================================
# 5. RESOLVING RISK-FACTOR NAMES FOR AN AS-OF DATE
# =============================================================================


def section_5_resolve_names(rsc) -> None:
    _section("5. Resolving risk-factor names for a specific as-of date")
    print(
        """
Every curve/surface point is declared with a RELATIVE tenor token, e.g.
"[M+0]" ("the front month") or "[HY+1]" ("the half-year period after the
current one"). Those tokens are NOT dated -- the same static config is
valid on any as-of date; only the ABSOLUTE name a relative token resolves
to changes from day to day.

`build_risk_factor_name_mapping_for_as_of_date` performs that resolution
for every relative name in the compiled config, for one as-of date, and
returns a bidirectional lookup: absolute name <-> relative name <-> rf_id.

Nodes need no resolution at all (an FX rate's name IS its absolute name,
every day). Curve/surface points with a LABEL axis (e.g. "[3M]", a fixed
market-convention label) also don't change. Only OFFSET-axis points
(monthly/annual/half-yearly rolling tenors) actually shift.
"""
    )

    as_of = date(2026, 5, 23)
    mapping = build_risk_factor_name_mapping_for_as_of_date(
        risk_static_config=rsc, as_of_date=as_of
    )

    print(
        f"Resolved for as_of_date = {as_of} ({mapping.num_risk_factors} risk factors)"
    )

    _sub("A LABEL curve point (does not move with as_of_date)")
    print(
        f"  relative 'USD_SOFR_OIS[3M]' -> absolute "
        f"'{mapping.get_abs_name_for_rel_name('USD_SOFR_OIS[3M]')}'"
    )

    _sub("An OFFSET_M curve point (front-month commodity forward)")
    for tok in ("[M+0]", "[M+1]", "[M+3]"):
        rel = f"BRENT_CRUDE_FWD{tok}"
        print(
            f"  relative '{rel}' -> absolute '{mapping.get_abs_name_for_rel_name(rel)}'"
        )

    _sub("An OFFSET_Y curve point (annual carbon roll, April-begin)")
    for tok in ("[Y+0]", "[Y+1]"):
        rel = f"CO2_EUA_ANNUAL_FWD{tok}"
        print(
            f"  relative '{rel}' -> absolute '{mapping.get_abs_name_for_rel_name(rel)}'"
        )

    _sub("An OFFSET_HY curve point (half-year power, Apr/Oct roll)")
    for tok in ("[HY+0]", "[HY+1]"):
        rel = f"POWER_CH_BASELOAD_HY{tok}"
        print(
            f"  relative '{rel}' -> absolute '{mapping.get_abs_name_for_rel_name(rel)}'"
        )

    _sub("Same query, one year later: the OFFSET names shift, LABEL does not")
    mapping_next_year = build_risk_factor_name_mapping_for_as_of_date(
        risk_static_config=rsc, as_of_date=date(2027, 5, 23)
    )
    print(
        f"  USD_SOFR_OIS[3M]        : "
        f"{mapping.get_abs_name_for_rel_name('USD_SOFR_OIS[3M]')}  "
        f"(2026) vs "
        f"{mapping_next_year.get_abs_name_for_rel_name('USD_SOFR_OIS[3M]')}  (2027)"
    )
    print(
        f"  BRENT_CRUDE_FWD[M+0]    : "
        f"{mapping.get_abs_name_for_rel_name('BRENT_CRUDE_FWD[M+0]')}  "
        f"(2026) vs "
        f"{mapping_next_year.get_abs_name_for_rel_name('BRENT_CRUDE_FWD[M+0]')}  (2027)"
    )


# =============================================================================
# 6. THE THREE ROLL-METHOD COMBINATIONS THE CANONICAL FIXTURE DOES NOT COVER
# =============================================================================


def section_6_supplementary_roll_methods() -> None:
    _section("6. Roll-method combinations NOT exercised by the canonical fixture")
    print(
        """
The resolution engine supports 6 (XAxisKind, RollMethod) combinations.
The canonical fixture, by deliberate design (each curve owns its roll
convention directly; the goal is a fixture that matches runtime needs,
not a feature catalogue), only exercises 3 of them:

    (OFFSET_M,  CALENDAR_PERIOD)     <- BRENT_CRUDE_FWD etc.  (section 5)
    (OFFSET_Y,  ROLL_ON_MMDD)        <- CO2_EUA_ANNUAL_FWD    (section 5)
    (OFFSET_HY, HALF_YEAR_BY_LABEL)  <- POWER_CH_BASELOAD_HY  (section 5)

The other 3 are demonstrated below with STANDALONE example data (built
directly from the resolver's own internal building blocks, not by
extending the canonical fixture -- several other tests assert exact
counts against that fixture, so growing it would silently break them).
This is tracked explicitly in
`tests/ingest_fixtures/config/risk_static_config/test_canonical_static_config_fixture.py::test_documented_roll_method_coverage_gap`.
"""
    )

    from riskquant.config.static_config.rf_resolution import (
        _resolve_offset_hy_half_year_by_date,
        _resolve_offset_m_half_year_by_date,
        _resolve_offset_y_calendar_period,
        _ResolverContext,
    )

    as_of = date(2026, 5, 23)

    _sub("(OFFSET_M, HALF_YEAR_BY_DATE)  -- monthly axis, half-year-date roll")
    print(
        "  A curve declared with x_axis_kind=OFFSET_M but rolled on the\n"
        "  half-year DATE convention (rather than CALENDAR_PERIOD) -- the\n"
        "  absolute name becomes a full YYYYMMDD settlement date, not a\n"
        "  YYYYMM period."
    )
    ctx = _ResolverContext(
        container_name="EXAMPLE_CURVE",
        relative_name="EXAMPLE_CURVE[M+0]",
        token_offset=0,
        as_of_date=as_of,
        anchor_offset=None,
        roll_month=None,
        roll_day=None,
        fh_bh_mmdd="0401|1001",
    )
    suffix = _resolve_offset_m_half_year_by_date(ctx)
    print(
        f"  as_of={as_of}, fh_bh_mmdd='0401|1001', token=[M+0]"
        f"  ->  EXAMPLE_CURVE_{suffix}"
    )

    _sub("(OFFSET_Y, CALENDAR_PERIOD)  -- annual axis, plain calendar-year roll")
    print(
        "  An annual curve rolled by a simple +N-years offset from the\n"
        "  as-of year (no fixed month/day roll boundary), as opposed to\n"
        "  ROLL_ON_MMDD's fixed-date year boundary (section 5's CO2 example)."
    )
    ctx2 = _ResolverContext(
        container_name="EXAMPLE_CURVE",
        relative_name="EXAMPLE_CURVE[Y+0]",
        token_offset=0,
        as_of_date=as_of,
        anchor_offset=1,
        roll_month=None,
        roll_day=None,
        fh_bh_mmdd=None,
    )
    suffix2 = _resolve_offset_y_calendar_period(ctx2)
    print(f"  as_of={as_of}, anchor_offset=1, token=[Y+0]  ->  EXAMPLE_CURVE_{suffix2}")

    _sub("(OFFSET_HY, HALF_YEAR_BY_DATE)  -- half-year axis, date-form roll")
    print(
        "  Half-year axis but resolved to a full settlement DATE rather\n"
        "  than a '<year>_FH'/'<year>_BH' LABEL (section 5's power example\n"
        "  used HALF_YEAR_BY_LABEL; this is the DATE-form sibling)."
    )
    ctx3 = _ResolverContext(
        container_name="EXAMPLE_CURVE",
        relative_name="EXAMPLE_CURVE[HY+0]",
        token_offset=0,
        as_of_date=as_of,
        anchor_offset=None,
        roll_month=None,
        roll_day=None,
        fh_bh_mmdd="0401|1001",
    )
    suffix3 = _resolve_offset_hy_half_year_by_date(ctx3)
    print(
        f"  as_of={as_of}, fh_bh_mmdd='0401|1001', token=[HY+0]"
        f"  ->  EXAMPLE_CURVE_{suffix3}"
    )


# =============================================================================
# 7. VALIDATION FAILURE, LIVE
# =============================================================================


def section_7_validation_failure(dict_of_dfs: dict[str, pl.DataFrame]) -> None:
    _section("7. What a validation failure actually looks like")
    print(
        """
Every check in validate.py is designed to name the EXACT problem, not
just say "invalid config". Below we intentionally break the canonical
fixture in a way a real typo would (misspelling a currency on one curve)
and show the resulting error, unmodified.
"""
    )

    broken = dict(dict_of_dfs)  # shallow copy of the table-name -> DataFrame dict
    broken["curve_meta"] = dict_of_dfs["curve_meta"].with_columns(
        pl.when(pl.col("curve_name") == "USD_SOFR_OIS")
        .then(pl.lit("USDD"))  # typo: not a real currency
        .otherwise(pl.col("quote_ccy"))
        .alias("quote_ccy")
    )

    _sub("Attempting to compile the broken config")
    try:
        compile_risk_static_config_from_dict_of_dfs(broken)
        print("  (unexpectedly succeeded -- this should not happen)")
    except ConfigurationError as exc:
        print("  Raised ConfigurationError, exactly as intended:")
        print()
        for line in str(exc).splitlines():
            print(f"    {line}")


# =============================================================================
# 8. LOADING FROM A REAL .xlsx WORKBOOK
# =============================================================================


def section_8_xlsx_roundtrip() -> None:
    _section("8. Loading from a real .xlsx workbook (the production path)")
    print(
        """
Everything above used in-memory Polars DataFrames directly. In production,
the source is almost always an Excel workbook with one sheet per table.
`compile_risk_static_config_from_xlsx` wraps the loader
(`load_risk_static_config_xlsx_into_dict`, which reads every required
sheet via `polars.read_excel`) and the same validate/compile pipeline used
above -- there is no behavioural difference, only the source of the raw
tables changes.

We write the canonical fixture out to a real .xlsx workbook, then read it
straight back in, to prove the round trip is lossless.
"""
    )
    import tempfile
    from pathlib import Path

    from riskquant.config.static_config.compile import (
        compile_risk_static_config_from_xlsx,
    )
    from riskquant.ingest_fixtures.config.risk_static_config.canonical_risk_static_config import (
        CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        xlsx_path = Path(tmp_dir) / "demo_static_config.xlsx"

        import xlsxwriter

        workbook = xlsxwriter.Workbook(str(xlsx_path))
        for sheet_name, df in CANONICAL_RISK_STATIC_CONFIG_DICT_OF_DFS.items():
            df.write_excel(workbook=workbook, worksheet=sheet_name)
        workbook.close()

        print(f"  Wrote: {xlsx_path}")

        rsc_from_xlsx = compile_risk_static_config_from_xlsx(xlsx_path)
        print(
            f"  Compiled from xlsx: num_risk_factors="
            f"{rsc_from_xlsx.num_risk_factors} "
            f"(matches the in-memory compile in sections 3-5)"
        )


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    section_1_what_is_static_config()
    dict_of_dfs = section_2_load_fixture()
    rsc = section_3_compile(dict_of_dfs)
    section_4_inspect(rsc)
    section_5_resolve_names(rsc)
    section_6_supplementary_roll_methods()
    section_7_validation_failure(dict_of_dfs)
    section_8_xlsx_roundtrip()

    _section("Done")
    print(
        """
You have now seen the full static_config lifecycle: load -> validate ->
compile -> inspect -> resolve names for a date -> see validation failure
-> round-trip through xlsx. For the terser reference version of this same
pipeline (no narration), see demo_full.py in this same directory.
"""
    )


if __name__ == "__main__":
    from riskquant_lab.polars_display import configure_polars_display

    configure_polars_display()
    main()
