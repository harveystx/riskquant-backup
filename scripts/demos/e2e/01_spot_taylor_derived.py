# scripts/demos/e2e/01_spot_taylor_derived.py
"""
RiskQuant SPOT -- chapter 01: TAYLOR/DERIVED risk.

    full valuation representation (the SpotBlock and its pricer)
        -> run_sensitivities            : closed-form partials at the base
        -> build_sensitivity_book_spec  : the book, on the run axis's valuation prefix
        -> derive_taylor_declaration    : order + F(p) + base binding
        -> SensitivityPnlCalculator     : compiled, certified book
        -> TaylorBlock -> run_scenario_var_es (TAYLOR/DERIVED partition)

What RiskQuant derives, and why
-------------------------------
For SPOT, V = q * P (* X for a quanto). The sensitizer differentiates the
pricer's own formula with respect to the LEVELS it reads:

    dV/dP = q * X      dV/dX = q * P      d2V/(dP dX) = q      d2V/dP2 = 0

F(p), the dependency set, is the pricer's STRUCTURAL read set: an FX row
exists iff settlement != quote currency, never because a value is non-zero.
The declaration then states a Taylor ORDER, and the base BINDING pins the
book to the exact base it was derived at (as-of, base levels, discounting,
static config, valuation contract, V_0). The run recomputes that binding
from its own ValuationContext and refuses any book bound elsewhere (P1-A/B).

Why a Taylor representation exists at all: it prices a scenario with a few
multiply-adds per sensitivity instead of a revaluation. For SPOT the full
second order is EXACT (the payoff is bilinear), so the orders below show
precisely which term each approximation drops. Part 6 repeats the comparison
against FULL_REVAL at realistic scale, where the dtype contracts decide the
remaining difference.

The admission boundary (part 7)
-------------------------------
    TAYLOR/DERIVED, certified, bound to the run's base   admitted
    TAYLOR/DERIVED, bound to another base               refused (P1-A/B)
    TAYLOR/DERIVED, uncertified (no declaration)        refused (P1-A)
    TAYLOR/SUPPLIED (caller-supplied Greeks)            refused (owner policy P-1)
    TAYLOR/DERIVED on an inverted-FX SPOT leg           refused (D084, chapter 03)

Run:  python scripts/demos/e2e/01_spot_taylor_derived.py
"""

from __future__ import annotations

import dataclasses
import time

import _spot_world as sw
import numpy as np

from riskquant.sensitivities.engine import run_sensitivities
from riskquant.solutions.risk_run.api import run_scenario_var_es
from riskquant.solutions.risk_run.schemas import (
    CompositionContractError,
    MethodologyDeclaration,
    TaylorBlock,
)
from riskquant.solutions.scenario_models.schemas import HistoricalGeneratorConfig
from riskquant.solutions.sensitivity_pnl.api import (
    SensitivityPnlCalculator,
    build_sensitivity_book_spec,
    derive_taylor_declaration,
)
from riskquant.solutions.sensitivity_pnl.taylor_contract import (
    TaylorContractError,
    TaylorOrder,
)

CONFIDENCE: float = 0.75
BATCH_SIZE: int = 3


def main() -> None:
    ok = sw.Checks()
    sw.title("RiskQuant SPOT executable guide - 01 Taylor-derived risk")
    fixture = sw.tiny_fixture()
    world, history = fixture.world, fixture.history

    # ---------------------------------------------------------------- 1
    sw.step("1", "Derive sensitivities from the valuation representation")
    sens = run_sensitivities(
        portfolio=world.portfolio, uix_mapper=world.uix_mapper, market_data=world.market_data
    )
    sw.table(
        ["position", "factor", "dV/dx", "d2V/dx2", "hand"],
        [
            (int(u), world.name_of(int(f)), sw.num(d1), sw.num(d2), hand)
            for u, f, d1, d2, hand in zip(
                sens.position_uids.tolist(), sens.factor_rf_ids.tolist(),
                sens.first_order.tolist(), sens.second_order.tolist(),
                ["q = 10", "q = -4", "q X = 8 x 1.25", "q P = 8 x 120"], strict=True,
            )
        ],
        align="rlrrl",
    )
    sw.table(
        ["position", "factor i", "factor j", "cross d2V", "hand"],
        [(int(u), world.name_of(int(i)), world.name_of(int(j)), sw.num(c), "q = 8")
         for u, i, j, c in zip(sens.cross_position_uids.tolist(), sens.cross_factor_rf_ids_i.tolist(),
                               sens.cross_factor_rf_ids_j.tolist(), sens.cross_second_order.tolist(), strict=True)],
        align="rllrl",
    )
    ok.check("first orders equal the closed-form partials exactly",
             sens.first_order.tolist() == [10.0, -4.0, 10.0, 960.0])
    ok.check("pure second orders are zero (SPOT is linear in each level)",
             not sens.second_order.any())
    ok.check("the quanto's cross term is q = 8; only 103 depends on EURUSD",
             sens.cross_second_order.tolist() == [8.0] and sens.cross_position_uids.tolist() == [103])
    sw.para(
        """
        Note what is absent. 102 is EUR-settled and reported in USD, but its
        sensitivities contain SAP_EQ only: reporting FX is applied after P&L
        (D081), it is not a Taylor factor.
        """
    )
    ok.check("102 has no EURUSD sensitivity (reporting FX is not a Taylor factor)",
             world.rf("EURUSD") not in sens.factor_rf_ids[sens.position_uids == 102].tolist())

    # ---------------------------------------------------------------- 2
    sw.step("2", "The base binding")
    binding = sw.run_base_binding(world, world.uids.tolist())
    sw.kv([(f.name, str(getattr(binding, f.name))[:40]) for f in dataclasses.fields(binding)])
    sw.para(
        """
        Every field is a content identity, not a label. The run derives the
        same six fields from its own ValuationContext and base levels, and
        requires equality field by field.
        """
    )

    # ---------------------------------------------------------------- 3
    sw.step("3", "Build, declare, compile - three Taylor orders")
    orders = (TaylorOrder.FIRST_ORDER, TaylorOrder.SECOND_ORDER_TRUNCATED, TaylorOrder.SECOND_ORDER_FULL)
    blocks: dict[TaylorOrder, TaylorBlock] = {}
    for order in orders:
        # cross terms belong ONLY to the full second order; the other orders
        # omit them BY DECLARATION (completeness is proven at compile)
        sub = sw.restrict_sensitivities(sens, world.uids.tolist(), keep_cross=order is TaylorOrder.SECOND_ORDER_FULL)
        spec = build_sensitivity_book_spec(sensitivities=sub, required_factor_ids=world.valuation_factor_ids)
        declaration = derive_taylor_declaration(sensitivities=sub, order=order, base_binding=binding)
        blocks[order] = TaylorBlock(
            calculator=SensitivityPnlCalculator.compile(dataclasses.replace(spec, taylor_declaration=declaration))
        )
    sw.table(
        ["order", "terms kept", "cross entries"],
        [
            ("FIRST_ORDER", "d1 dx", blocks[orders[0]].calculator.num_cross_entries),
            ("SECOND_ORDER_TRUNCATED", "d1 dx + 1/2 d2 dx^2", blocks[orders[1]].calculator.num_cross_entries),
            ("SECOND_ORDER_FULL", "... + c_ij dx_i dx_j", blocks[orders[2]].calculator.num_cross_entries),
        ],
        align="llr",
    )
    ok.check("every book is compiled on the valuation prefix of the run axis, row for row",
             all(b.calculator.axis.keys.to_list() == world.valuation_factor_ids.tolist() for b in blocks.values()),
             "reporting FX is never a Taylor factor, so the Taylor axis is the valuation prefix (D082, D085)")

    # ---------------------------------------------------------------- 4
    sw.step("4", "Run each certified book through the production entry point")
    scenario_config = HistoricalGeneratorConfig(model="historical", horizon=1)
    method = MethodologyDeclaration(
        production_position_uids=world.uids,
        full_reval_uids=np.empty(0, dtype=np.int64),
        taylor_supplied_uids=np.empty(0, dtype=np.int64),
        taylor_derived_uids=world.uids,
    )
    reporting = sw.reporting_spec(world, reporting_ccy="USD")
    runs = {}
    anat = {}
    for order in orders:
        runs[order] = run_scenario_var_es(
            scenario_config=scenario_config,
            data=history,
            required_factor_ids=world.required_factor_ids,
            required_shift_type_id=world.required_shift_type_id,
            proxy_terms=world.proxy_terms,
            base_levels=world.base_levels,
            methodology=method,
            valuation_context=sw.valuation_context(world),
            taylor_derived=blocks[order],
            reporting=reporting,
            confidence_level=CONFIDENCE,
            batch_size=BATCH_SIZE,
        )
        anat[order] = sw.scenario_pnl_anatomy(
            world, scenario_config=scenario_config, data=history, method=method,
            taylor_derived=blocks[order], reporting=reporting,
            confidence_level=CONFIDENCE, batch_size=BATCH_SIZE,
        )
        ok.check(f"{order.value}: anatomy equals the RiskRun bitwise",
                 sw.anatomy_matches_run(anat[order], runs[order]))
    ok.check("each run carries a Taylor provenance stamp (levels contract)",
             all(r.pnl_provenance is not None and r.pnl_provenance.market_contract == "levels" for r in runs.values()))

    # ---------------------------------------------------------------- 5
    sw.step("5", "Taylor P&L against the exact increment")
    shocked = anat[orders[0]].shocked_levels
    dp = shocked[world.row("SAP_EQ")] - 120.0
    dx = shocked[world.row("EURUSD")] - 1.25
    exact_103 = 8.0 * (1.25 * dp + 120.0 * dx + dp * dx)
    row = anat[orders[0]].row(103)
    sw.para(
        """
        The quanto 103 is the only position with a second-order term. Its
        exact increment is q (X0 dP + P0 dX + dP dX). FIRST and TRUNCATED
        both drop the cross term q dP dX; FULL keeps it.
        """
    )
    sw.table(
        ["scenario", "dP", "dX", "exact", "FIRST", "TRUNC", "FULL", "q dP dX"],
        [
            (s, sw.num(dp[s], 1), sw.num(dx[s]), sw.num(exact_103[s], 2),
             *(sw.num(anat[o].settlement_pnl[row, s], 2) for o in orders), sw.num(8.0 * dp[s] * dx[s], 2))
            for s in range(dp.shape[0])
        ],
        align="rrrrrrrr",
    )
    ok.check("SECOND_ORDER_FULL equals the exact increment for 103, exactly",
             np.array_equal(anat[orders[2]].settlement_pnl[row], exact_103))
    ok.check("FIRST_ORDER misses exactly the cross term q dP dX",
             np.array_equal(exact_103 - anat[orders[0]].settlement_pnl[row], 8.0 * dp * dx))
    ok.check("TRUNCATED == FIRST for SPOT (every pure second order is zero)",
             np.array_equal(anat[orders[1]].settlement_pnl, anat[orders[0]].settlement_pnl))
    sw.table(
        ["order", "portfolio VaR", "portfolio ES", "103 VaR"],
        [(o.value, sw.num(runs[o].portfolio_var, 2), sw.num(runs[o].portfolio_es, 2),
          sw.num(runs[o].var[runs[o].position_uids.tolist().index(103)], 2)) for o in orders],
        align="lrrr",
    )
    sw.para(
        """
        Identity follows the COMPILED numbers. `hash_compiled_book` digests
        the CSR arrays the kernel consumes, not the declaration label. For
        SPOT, FIRST and TRUNCATED compile to identical arrays (all pure second
        orders are zero), so they share one run identity; FULL adds a cross
        entry and gets a new identity.
        """
    )
    ok.check("FIRST and TRUNCATED: identical compiled books, identical identity",
             runs[orders[0]].identity == runs[orders[1]].identity)
    ok.check("FULL: different compiled book, different identity",
             runs[orders[2]].identity != runs[orders[0]].identity)

    # ---------------------------------------------------------------- 6
    sw.step("6", "Realistic scale: DERIVED against FULL_REVAL")
    t0 = time.perf_counter()
    real = sw.realistic_fixture()
    w = real.world
    uids = w.uids
    rep = sw.reporting_spec(w, reporting_ccy="USD")
    quanto = set(
        w.positions.loc[
            w.positions["underlying"].map(lambda n: n.startswith(("EU_", "UK_")))
            & (w.positions["settlement_ccy"] == "USD"),
            "uid",
        ].astype(int)
    )
    sw.kv([
        ("positions", uids.shape[0]),
        ("factors on the run axis", w.required_factor_ids.shape[0]),
        ("shift type", "LOG_RETURN for every factor"),
        ("quanto (EUR/GBP asset, USD settled)", len(quanto)),
        ("scenarios", real.num_observations - 1),
    ])
    # Local names only: `method` and `reporting` of parts 4-5 are reused in part 7.
    out = {}
    for label, method_r, block_r in (
        ("FULL_REVAL", sw.methodology(full=uids.tolist()), None),
        ("DERIVED FULL", sw.methodology(derived=uids.tolist()),
         sw.certified_taylor_block(w, uids.tolist(), order=TaylorOrder.SECOND_ORDER_FULL)),
        ("DERIVED FIRST", sw.methodology(derived=uids.tolist()),
         sw.certified_taylor_block(w, uids.tolist(), order=TaylorOrder.FIRST_ORDER)),
    ):
        run_r = run_scenario_var_es(
            scenario_config=scenario_config,
            data=real.history,
            required_factor_ids=w.required_factor_ids,
            required_shift_type_id=w.required_shift_type_id,
            proxy_terms=w.proxy_terms,
            base_levels=w.base_levels,
            methodology=method_r,
            valuation_context=sw.valuation_context(w),
            taylor_derived=block_r,
            reporting=rep,
            confidence_level=0.99,
            batch_size=256,
        )
        an_r = sw.scenario_pnl_anatomy(w, scenario_config=scenario_config, data=real.history, method=method_r,
                                       taylor_derived=block_r, reporting=rep, confidence_level=0.99, batch_size=256)
        ok.check(f"{label}: anatomy equals the RiskRun bitwise", sw.anatomy_matches_run(an_r, run_r))
        out[label] = (run_r, an_r)

    full = out["FULL_REVAL"][1].settlement_pnl
    derived = out["DERIVED FULL"][1].settlement_pnl
    first = out["DERIVED FIRST"][1].settlement_pnl
    # ---- a priori error bound from the dtype contract ----------------------
    # Taylor kernel (sensitivity_pnl.kernels): dx is formed in float64 and
    # rounded ONCE to float32 (relative error <= u = 2^-24); coefficients and
    # the accumulator are float64. Hence, per position p and scenario s,
    #   |Taylor - exact| <= u * sum_k |d1_k dx_k| + 2u * sum_ij |c_ij dx_i dx_j|
    # plus one float32 rounding of each side into the P&L grid (<= 1 ulp of
    # the larger magnitude in total). The float64 terms are orders smaller
    # and are absorbed by the factor-2 margin on the cross term.
    u = 2.0 ** -24
    sens_real = run_sensitivities(portfolio=w.portfolio, uix_mapper=w.uix_mapper, market_data=w.market_data)
    dxs = out["FULL_REVAL"][1].shocked_levels - w.base_levels[:, None]
    row_of = {int(r): i for i, r in enumerate(w.required_factor_ids)}
    pos_of = {int(x): i for i, x in enumerate(uids.tolist())}
    bound = np.zeros_like(full)
    for p_uid, f_id, d1 in zip(sens_real.position_uids.tolist(), sens_real.factor_rf_ids.tolist(),
                               sens_real.first_order.tolist(), strict=True):
        bound[pos_of[p_uid]] += u * abs(d1) * np.abs(dxs[row_of[f_id]])
    for p_uid, fi, fj, c in zip(sens_real.cross_position_uids.tolist(), sens_real.cross_factor_rf_ids_i.tolist(),
                                sens_real.cross_factor_rf_ids_j.tolist(), sens_real.cross_second_order.tolist(),
                                strict=True):
        bound[pos_of[p_uid]] += 2.0 * u * abs(c) * np.abs(dxs[row_of[fi]] * dxs[row_of[fj]])
    bound += np.spacing(np.maximum(np.abs(full), np.abs(derived)).astype(np.float32)).astype(np.float64)
    within = np.abs(full - derived) <= bound
    sw.para(
        """
        Rule (derived from the Taylor kernel's dtype contract, not tuned):
        the kernel rounds each displacement dx to float32 once (relative
        error <= 2^-24) and accumulates in float64. So per position and
        scenario |DERIVED FULL - FULL_REVAL| <= 2^-24 sum|d1 dx| +
        2^-23 sum|c dx_i dx_j|, plus one float32 ulp for rounding both
        values into the grid. In exact arithmetic the two are the same
        number; this bound only admits the documented float32 steps.
        """
    )
    q_rows = np.array([u_ in quanto for u_ in uids.tolist()])
    sw.table(
        ["comparison", "max |diff|", "max bound", "elements ok"],
        [
            ("FULL_REVAL vs DERIVED FULL", f"{np.abs(full - derived).max():.3e}", f"{bound.max():.3e}",
             f"{int(within.sum()):,} / {within.size:,}"),
            ("FIRST vs DERIVED FULL (non-quanto)", f"{np.abs(first - derived)[~q_rows].max():.3e}", "exact",
             "-"),
            ("FIRST vs FULL_REVAL (quanto)", f"{np.abs(full - first)[q_rows].max():.3e}", "-", "-"),
        ],
        align="lrrr",
    )
    ok.check("FULL_REVAL vs DERIVED FULL: every element within the stated bound", bool(within.all()))
    ok.check("non-quanto rows: FIRST and second-order FULL are bitwise equal "
             "(same coefficients, same float32 dx, same kernel)",
             np.array_equal(first[~q_rows], derived[~q_rows]))
    ok.check("quanto rows: FIRST_ORDER omits q dP dX, a genuine approximation error",
             float(np.abs(full - first)[q_rows].max()) > 1.0)
    sw.table(
        ["methodology", "portfolio VaR 99%", "portfolio ES 99%"],
        [(k, sw.num(r.portfolio_var, 2), sw.num(r.portfolio_es, 2)) for k, (r, _) in out.items()],
        align="lrr",
    )
    sw.para(
        f"""
        The difference between FULL_REVAL and FIRST_ORDER is not noise: it is
        the dropped cross term, largest in scenarios where the asset and its
        FX leg move together. Methodology is part of the run identity even
        where the numbers agree (chapter 02). Part 6 wall time:
        {time.perf_counter() - t0:.2f} s.
        """
    )

    # ---------------------------------------------------------------- 7
    sw.step("7", "The admission boundary: what the run refuses")
    run_kwargs = dict(
        scenario_config=scenario_config,
        data=history,
        required_factor_ids=world.required_factor_ids,
        required_shift_type_id=world.required_shift_type_id,
        proxy_terms=world.proxy_terms,
        base_levels=world.base_levels,
        valuation_context=sw.valuation_context(world),
        reporting=reporting,
        confidence_level=CONFIDENCE,
        batch_size=BATCH_SIZE,
    )
    sw.para(
        """
        Only a certified representation, bound to the run's own base,
        produces production P&L. Each refusal below uses the tiny book of
        parts 1-5 and names the gate that holds the line.
        """
    )
    yesterday = sw.tiny_fixture(levels={"EURUSD": 1.25, "AAPL_EQ": 196.0, "SAP_EQ": 118.0}).world
    stale = sw.certified_taylor_block(yesterday, yesterday.uids.tolist())
    ok.expect_rejection(
        "a Taylor book certified at yesterday's levels (AAPL 196, SAP 118) is refused today",
        TaylorContractError,
        lambda: run_scenario_var_es(methodology=method, taylor_derived=stale, **run_kwargs),
        reason="Greeks and V_0 are valid only at the exact base they were "
        "produced at (D081 section 10); the run values today's V_0 itself, "
        "rebuilds the binding and names every field that differs",
        stage="P1-A gate, after the base valuation (S = 1), before scenarios",
    )
    uncertified = SensitivityPnlCalculator.compile(
        build_sensitivity_book_spec(sensitivities=sens, required_factor_ids=world.valuation_factor_ids)
    )
    ok.expect_rejection(
        "the same book compiled WITHOUT its TaylorDeclaration is refused",
        TaylorContractError,
        lambda: run_scenario_var_es(
            methodology=method, taylor_derived=TaylorBlock(calculator=uncertified), **run_kwargs
        ),
        reason="P1-A: order, dependency set F(p) and base binding are unproven, "
        "so the book cannot produce production P&L",
        stage="P1-A gate, after the base valuation (S = 1), before scenarios",
    )
    supplied = MethodologyDeclaration(
        production_position_uids=world.uids,
        full_reval_uids=np.empty(0, dtype=np.int64),
        taylor_supplied_uids=world.uids,
        taylor_derived_uids=np.empty(0, dtype=np.int64),
    )
    ok.expect_rejection(
        "TAYLOR/SUPPLIED positions (caller-supplied Greeks) are refused",
        CompositionContractError,
        lambda: run_scenario_var_es(methodology=supplied, **run_kwargs),
        reason="owner policy P-1 (supplied representations) is unresolved; a "
        "supplied book is never silently priced as DERIVED or FULL_REVAL",
        stage="composition contract, before identity and numerics",
    )
    sw.para(
        """
        Supplied sensitivities can still be priced OUTSIDE a RiskRun, in
        settlement currency, with `riskquant.solutions.sensitivity_pnl`
        (`run_sensitivity_pnl`). That result is not a production VaR. The
        inverted-FX refusal belongs to the FX boundary (chapter 03).
        """
    )
    ok.done()


if __name__ == "__main__":
    main()
