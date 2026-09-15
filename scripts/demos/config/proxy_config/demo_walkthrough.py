# scripts/demos/config/proxy_config/demo_walkthrough.py
"""
A verbose, narrated, end-to-end walkthrough of `riskquant.config.proxy_config`.

This script is written for a reader who has never touched this package
before. Every stage prints a plain-English explanation of what is about to
happen and why, before showing the actual code and its output. Read it
top to bottom; each section builds on the previous one.

Run it directly:

    python scripts/demos/config/proxy_config/demo_walkthrough.py

This walkthrough assumes basic familiarity with `riskquant.config.static_config`
(see scripts/demos/config/static_config/demo_walkthrough.py if not) -- a
compiled `RiskStaticConfig` is proxy_config's mandatory input, since every
proxy mapping is validated by resolving names against it.

Sections
--------
  1. What is proxy_config, and why does it exist?
  2. Building the prerequisite: a compiled RiskStaticConfig.
  3. Loading the canonical proxy fixture.
  4. Compiling: validate -> ProxyTopology.
  5. Inspecting the compiled ProxyTopology.
  6. Resolving a required id through a proxy -- container level and rf level.
  7. Overriding a proxy mapping at runtime (with_overrides).
  8. The one proxy pattern the canonical fixture does NOT exercise.
  9. What validation failure looks like (three different intentional breaks).
  10. Loading from a real .xlsx workbook.
"""

from __future__ import annotations

from typing import Any, cast

import polars as pl

from riskquant.config.proxy_config.compile import compile_proxy_topology_from_df
from riskquant.config.proxy_config.schema_objects import ProxyMap
from riskquant.config.static_config.compile import (
    compile_risk_static_config_from_dict_of_dfs,
)
from riskquant.ingest_fixtures.config.risk_proxy_config.canonical_risk_proxy_config import (
    CANONICAL_RISK_PROXY_CONFIG_DF,
)
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
# 1. WHAT IS proxy_config?
# =============================================================================


def section_1_what_is_proxy_config() -> None:
    _section("1. What is riskquant.config.proxy_config?")
    print(
        """
A firm's risk-factor universe (see static_config) inevitably contains
names with insufficient or unreliable historical data: a newly-listed
instrument, an illiquid tenor, a thinly-traded cross-currency pair. Rather
than run a scenario/VaR calculation on garbage-in-garbage-out data for
those names, `proxy_config` lets you declare: "when you need historical
data for name X, use name Y's data instead" -- optionally scaled by a
`vol_scaling_factor` to account for a known relative-vol difference
between the two series.

Two levels of substitution are supported:

  CONTAINER level  -- substitute an entire curve or surface's worth of
                       historical shocks with another curve/surface's.
                       E.g.: "WTI_CRUDE_FWD has no reliable history yet;
                       use BRENT_CRUDE_FWD's shocks instead."

  RISK_FACTOR level -- substitute a single scalar risk factor's history
                        (one FX rate, one curve point, one surface point)
                        with another single risk factor's.
                        E.g.: "USDGBP is illiquid; use USDEUR's shocks."

Rules, strictly enforced by the validator:
  - Single-hop only. A proxy target must itself be a TERMINAL name -- it
    cannot ALSO be proxied (no A->B->C chains, which also rules out
    cycles).
  - No self-reference (A cannot proxy itself).
  - No cross-level conflict: a container cannot be container-proxied while
    one of its own child risk factors is ALSO separately rf-proxied --
    those are two competing policies for the same underlying data.

The compiled output, `ProxyTopology`, is -- like `RiskStaticConfig` --
completely dataframe-free: two dense NumPy arrays (one for the container
axis, one for the rf axis) with a `-1` sentinel meaning "no proxy, pass
through unchanged". A single `resolve` gather per id, no iteration, on
the hot path.
"""
    )


# =============================================================================
# 2. PREREQUISITE: A COMPILED RiskStaticConfig
# =============================================================================


def section_2_build_static_config():
    _section("2. Building the prerequisite: a compiled RiskStaticConfig")
    print(
        """
Every proxy mapping references names by string ("WTI_CRUDE_FWD",
"USDGBP[3M]", ...). Validating those references -- confirming they
actually exist, resolving them to ids, checking the cross-level rule --
requires the full compiled risk-factor universe. So proxy_config always
takes a `RiskStaticConfig` as an explicit dependency; it never re-derives
one internally.

We build one here from the SAME canonical static_config fixture used in
static_config's own walkthrough.
"""
    )
    dict_of_dfs = {
        name: pl.DataFrame(cast(Any, data))
        for name, data in CANONICAL_RISK_STATIC_CONFIG_DATA.items()
    }
    rsc = compile_risk_static_config_from_dict_of_dfs(dict_of_dfs)
    print(
        f"Compiled: {rsc.num_nodes} nodes, {rsc.num_curves} curves, "
        f"{rsc.num_surfaces} surfaces, {rsc.num_risk_factors} total risk factors."
    )
    return rsc


# =============================================================================
# 3. LOADING THE CANONICAL PROXY FIXTURE
# =============================================================================


def section_3_load_fixture() -> pl.DataFrame:
    _section("3. Loading the canonical proxy fixture")
    print(
        """
Like static_config, a small, always-valid canonical fixture ships at:

    src/riskquant/ingest_fixtures/config/risk_proxy_config/

It declares 6 proxy mappings -- 3 container-level, 3 risk-factor-level --
each with a documented rationale (see the module comment in
canonical_risk_proxy_config.py). What `compile_proxy_topology_from_df`
actually consumes is a `polars.DataFrame` with exactly 4 columns:
proxying_level, required_name, proxy_name, vol_scaling_factor.
"""
    )
    print(CANONICAL_RISK_PROXY_CONFIG_DF)
    return CANONICAL_RISK_PROXY_CONFIG_DF


# =============================================================================
# 4. COMPILING
# =============================================================================


def section_4_compile(proxy_df: pl.DataFrame, rsc):
    _section("4. Compiling: validate -> ProxyTopology")
    print(
        """
`compile_proxy_topology_from_df` runs two validation stages before
building anything:

  Stage 1 (validate_proxy_config_df) -- DataFrame-tier checks: required
    columns present, no nulls/blanks, proxying_level is CONTAINER or
    RISK_FACTOR, vol_scaling_factor in (0, 10], (proxying_level,
    required_name) is unique, no self-reference.

  Stage 2 (validate_proxy_maps) -- cross-table checks, run AFTER the
    6 rows are partitioned by level: every required_name/proxy_name
    resolves against the static config (FK), no proxy target is itself
    a required_name in the same level (single-hop/no-chain/no-cycle), and
    no container-level proxy conflicts with an rf-level proxy on one of
    its own children (cross-level).

Only once both stages pass does `ProxyTopology.build` assemble the dense
lookup arrays.
"""
    )
    topology = compile_proxy_topology_from_df(df=proxy_df, static_config=rsc)
    print("Compiled successfully. No ValueError was raised.")
    return topology


# =============================================================================
# 5. INSPECTING THE COMPILED TOPOLOGY
# =============================================================================


def section_5_inspect(topology) -> None:
    _section("5. Inspecting the compiled ProxyTopology")

    _sub("Headline counts")
    print(f"  num_container_proxies = {topology.num_container_proxies}")
    print(f"  num_rf_proxies        = {topology.num_rf_proxies}")
    print(f"  num_total_proxies     = {topology.num_total_proxies}")

    _sub("Every container-level proxy mapping")
    for pm in topology.container_proxies.values():
        print(
            f"  {pm.required_name:<22s} -> {pm.proxy_name:<20s} "
            f"(scale={pm.vol_scaling_factor})"
        )

    _sub("Every risk-factor-level proxy mapping")
    for pm in topology.rf_proxies.values():
        print(
            f"  {pm.required_name:<22s} -> {pm.proxy_name:<20s} "
            f"(scale={pm.vol_scaling_factor})"
        )

    _sub("The dense hot-path arrays (what a kernel actually reads)")
    print(
        "  container_proxy_id is SLOT-indexed (position in container_ids,\n"
        "  not the raw container_id itself -- container ids live at 250M+,\n"
        "  so a dense-by-raw-id array would be multi-GB). rf_id_to_proxy_rf_id\n"
        "  is dense by raw rf_id (rf ids are small, so this is fine)."
    )
    print(f"  container_proxy_id.shape       = {topology.container_proxy_id.shape}")
    print(f"  rf_id_to_proxy_rf_id.shape     = {topology.rf_id_to_proxy_rf_id.shape}")
    n_set_c = int((topology.container_proxy_id >= 0).sum())
    n_set_rf = int((topology.rf_id_to_proxy_rf_id >= 0).sum())
    print(
        f"  container slots actually set   = {n_set_c}  (matches num_container_proxies)"
    )
    print(f"  rf slots actually set          = {n_set_rf}  (matches num_rf_proxies)")

    topology.summary()


# =============================================================================
# 6. RESOLVING A REQUIRED ID THROUGH A PROXY
# =============================================================================


def section_6_resolve(topology, rsc) -> None:
    _section("6. Resolving a required id through a proxy")
    print(
        """
`resolve_container` / `resolve_rf` are single-id convenience helpers for
tests and interactive sessions -- NOT the batch-gather pattern a real
kernel uses, but the clearest way to see the substitution semantics.

Contract: if no proxy is registered for an id, the id and a vol_scale of
1.0 (identity) are returned unchanged -- this includes ids that don't
even exist in the static config. The kernel never needs a separate
"is this id proxied" branch; it always calls resolve and gets back
something safe to use.
"""
    )

    _sub("A container-level proxy in action")
    wti_id = rsc.get_container_id("WTI_CRUDE_FWD")
    proxy_id, scale = topology.resolve_container(wti_id)
    proxy_name = rsc.curve_meta_dict[proxy_id].curve_name
    print(f"  WTI_CRUDE_FWD (container_id={wti_id})")
    print(f"    -> proxy container_id={proxy_id} ({proxy_name}), vol_scale={scale}")

    _sub("An UNPROXIED container -- passes through unchanged")
    ois_id = rsc.get_container_id("USD_SOFR_OIS")
    proxy_id, scale = topology.resolve_container(ois_id)
    print(f"  USD_SOFR_OIS (container_id={ois_id})")
    print(f"    -> {proxy_id} (itself), vol_scale={scale}  (no proxy registered)")

    _sub("An rf-level proxy in action (FX cross substitution)")
    usdgbp_id = rsc.get_rf_id("USDGBP")
    proxy_id, scale = topology.resolve_rf(usdgbp_id)
    proxy_name = rsc.rel_name_to_rf_id  # (dict; just to show the reverse lookup exists)
    print(f"  USDGBP (rf_id={usdgbp_id})")
    print(f"    -> proxy rf_id={proxy_id} (USDEUR), vol_scale={scale}")

    _sub("An rf-level proxy on a CURVE POINT, not just an FX node")
    tenor_id = rsc.get_rf_id("USD_LIBOR_3M[10Y]")
    proxy_id, scale = topology.resolve_rf(tenor_id)
    print(f"  USD_LIBOR_3M[10Y] (rf_id={tenor_id})")
    print(f"    -> proxy rf_id={proxy_id} (USD_SOFR_OIS[10Y]), vol_scale={scale}")
    print(
        "  This shows proxying is not FX-specific: any two risk factors of\n"
        "  the same kind (both curve points here) can be substituted."
    )


# =============================================================================
# 7. OVERRIDING A PROXY MAPPING AT RUNTIME
# =============================================================================


def section_7_overrides(topology, rsc) -> None:
    _section("7. Overriding a proxy mapping at runtime")
    print(
        """
`with_overrides` returns a BRAND NEW ProxyTopology with the supplied
overrides layered on top of the base one -- the base topology is never
mutated (it's a frozen dataclass). This is the mechanism for a desk to
say "for THIS run only, also proxy this one extra name" without touching
the firm-wide canonical config.

The full validation pipeline (FK, chain, cross-level) re-runs against the
MERGED mapping set, so an override that would introduce a chain or a
conflict is rejected immediately, exactly like a fresh compile would be.
"""
    )
    override = ProxyMap(
        required_name="AAPL_ATM_VOL", proxy_name="AAPL_EQ", vol_scaling_factor=2.0
    )
    updated = topology.with_overrides(
        static_config=rsc, rf_overrides={"AAPL_ATM_VOL": override}
    )
    print(f"  Base topology rf proxies:    {topology.num_rf_proxies}")
    print(f"  Overridden topology rf proxies: {updated.num_rf_proxies}  (+1)")

    vol_id = rsc.get_rf_id("AAPL_ATM_VOL")
    proxy_id, scale = updated.resolve_rf(vol_id)
    eq_id = rsc.get_rf_id("AAPL_EQ")
    print(
        f"  AAPL_ATM_VOL (rf_id={vol_id}) now resolves to "
        f"rf_id={proxy_id} (AAPL_EQ, id={eq_id}), scale={scale}"
    )

    print()
    print("  An override that would create a chain is rejected immediately:")
    bad_override = ProxyMap(
        required_name="EURCHF", proxy_name="USDGBP", vol_scaling_factor=1.0
    )
    try:
        topology.with_overrides(
            static_config=rsc, rf_overrides={"EURCHF": bad_override}
        )
        print("    (unexpectedly succeeded)")
    except ValueError as exc:
        for line in str(exc).splitlines()[:2]:
            print(f"    {line}")


# =============================================================================
# 8. THE PROXY PATTERN THE CANONICAL FIXTURE DOES NOT EXERCISE
# =============================================================================


def section_8_supplementary_example(rsc) -> None:
    _section("8. The proxy pattern the canonical fixture does NOT exercise")
    print(
        """
All 3 container-level rows in the canonical fixture proxy CURVE
containers. No SURFACE container is proxied, and no surface-point rf_id
is used as an rf-level required_name -- both are genuinely supported
(the validator's cross-level check explicitly handles both CURVE and
SURFACE container kinds; see `_check_cross_level_conflict`), just not
represented in the small canonical example.

This is tracked explicitly in
tests/ingest_fixtures/config/risk_proxy_config/test_canonical_proxy_config_fixture.py
::test_documented_proxy_coverage_gap, and demonstrated below with a
standalone example rather than by growing the canonical fixture (several
tests assert its exact row count).
"""
    )
    surface_example = pl.DataFrame(
        {
            "proxying_level": ["CONTAINER"],
            "required_name": ["USDEUR_VOL_SURF"],
            "proxy_name": ["AAPL_VOL_SURF"],
            "vol_scaling_factor": [1.0],
        }
    )
    example_topology = compile_proxy_topology_from_df(
        df=surface_example, static_config=rsc
    )
    surf_id = rsc.get_container_id("USDEUR_VOL_SURF")
    proxy_id, scale = example_topology.resolve_container(surf_id)
    proxy_name = rsc.surface_meta_dict[proxy_id].surface_name
    print(f"  USDEUR_VOL_SURF (surface container_id={surf_id})")
    print(f"    -> proxy container_id={proxy_id} ({proxy_name}), vol_scale={scale}")
    print(
        "  Same mechanism as the curve case in section 6 -- proxying is\n"
        "  container-kind-agnostic at the compile/resolve layer."
    )


# =============================================================================
# 9. VALIDATION FAILURE, LIVE
# =============================================================================


def section_9_validation_failures(rsc) -> None:
    _section("9. What validation failure actually looks like")
    print(
        """
Three intentional breaks below, each targeting a different validation
tier, to show the error messages a real config mistake would produce.
"""
    )

    _sub("1. Self-reference (a name proxying itself)")
    bad_self = pl.DataFrame(
        {
            "proxying_level": ["RISK_FACTOR"],
            "required_name": ["USDGBP"],
            "proxy_name": ["USDGBP"],
            "vol_scaling_factor": [1.0],
        }
    )
    try:
        compile_proxy_topology_from_df(df=bad_self, static_config=rsc)
    except ValueError as exc:
        for line in str(exc).splitlines():
            print(f"    {line}")

    _sub("2. Chain (A proxies B, but B is itself proxied elsewhere)")
    bad_chain = pl.DataFrame(
        {
            "proxying_level": ["RISK_FACTOR", "RISK_FACTOR"],
            "required_name": ["USDGBP", "EURCHF"],
            "proxy_name": ["EURCHF", "USDEUR"],
            "vol_scaling_factor": [1.0, 1.0],
        }
    )
    try:
        compile_proxy_topology_from_df(df=bad_chain, static_config=rsc)
    except ValueError as exc:
        for line in str(exc).splitlines()[:4]:
            print(f"    {line}")

    _sub("3. Cross-level conflict (container AND one of its own points proxied)")
    bad_cross = pl.DataFrame(
        {
            "proxying_level": ["CONTAINER", "RISK_FACTOR"],
            "required_name": ["BRENT_CRUDE_FWD", "BRENT_CRUDE_FWD[M+0]"],
            "proxy_name": ["WTI_CRUDE_FWD", "WTI_CRUDE_FWD[M+0]"],
            "vol_scaling_factor": [1.0, 1.0],
        }
    )
    try:
        compile_proxy_topology_from_df(df=bad_cross, static_config=rsc)
    except ValueError as exc:
        for line in str(exc).splitlines()[:4]:
            print(f"    {line}")


# =============================================================================
# 10. LOADING FROM A REAL .xlsx WORKBOOK
# =============================================================================


def section_10_xlsx_roundtrip(rsc) -> None:
    _section("10. Loading from a real .xlsx workbook (the production path)")
    print(
        """
`compile_proxy_topology_from_xlsx` wraps `load_proxy_config_xlsx_into_df`
(which requires an `active` column -- rows are filtered to active=True,
and the column itself is dropped before validation) and the same
validate/compile pipeline used above.

We write the canonical fixture to a real .xlsx workbook, then read it
straight back in, to prove the round trip is lossless.
"""
    )
    import tempfile
    from pathlib import Path

    from riskquant.config.proxy_config.compile import (
        compile_proxy_topology_from_xlsx,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        xlsx_path = Path(tmp_dir) / "demo_proxy_config.xlsx"

        import xlsxwriter

        workbook = xlsxwriter.Workbook(str(xlsx_path))
        CANONICAL_RISK_PROXY_CONFIG_DF.write_excel(
            workbook=workbook, worksheet="proxy_config"
        )
        workbook.close()

        print(f"  Wrote: {xlsx_path}")

        topology_from_xlsx = compile_proxy_topology_from_xlsx(
            path=xlsx_path, static_config=rsc, sheet_name="proxy_config"
        )
        print(
            f"  Compiled from xlsx: {topology_from_xlsx.num_total_proxies} "
            f"total proxies (matches sections 4-6)"
        )


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    section_1_what_is_proxy_config()
    rsc = section_2_build_static_config()
    proxy_df = section_3_load_fixture()
    topology = section_4_compile(proxy_df, rsc)
    section_5_inspect(topology)
    section_6_resolve(topology, rsc)
    section_7_overrides(topology, rsc)
    section_8_supplementary_example(rsc)
    section_9_validation_failures(rsc)
    section_10_xlsx_roundtrip(rsc)

    _section("Done")
    print(
        """
You have now seen the full proxy_config lifecycle: build the prerequisite
static config -> load the canonical fixture -> compile -> inspect ->
resolve a required id through a proxy (both container and rf level) ->
override at runtime -> see the one deliberately-uncovered pattern ->
see three kinds of validation failure -> round-trip through xlsx.

For the terser reference version of this same pipeline (no narration),
see demo_full.py in this same directory.
"""
    )


if __name__ == "__main__":
    from riskquant_lab.polars_display import configure_polars_display

    configure_polars_display()
    main()
