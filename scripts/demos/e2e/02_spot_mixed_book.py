# scripts/demos/e2e/02_spot_mixed_book.py
"""
RiskQuant SPOT -- chapter 02: a mixed-methodology book (D081 section 11).

One run, one explicit production universe U, partitioned:

    U = U_F (FULL_REVAL) + U_S (TAYLOR/SUPPLIED) + U_D (TAYLOR/DERIVED), disjoint

Supported combinations in the current repository:

    FULL_REVAL        yes
    TAYLOR/DERIVED    yes  (certified book, bound to the run's base)
    TAYLOR/SUPPLIED   refused while owner policy P-1 is open (chapter 01)

So the mixed book here is FULL_REVAL + TAYLOR/DERIVED. The architecture it
teaches:

    ONE scenario stream -> ONE shocked-level buffer
      -> FULL block    : dV = V(x_s) - V(x0)          (rows U_F)
      -> DERIVED block : dV = certified Taylor book     (rows U_D)
    ONE float32 PnlGrid [U_F | U_S | U_D] -> scattered to sorted(U)
    -> ONE D081 conversion -> ONE VaR/ES call

Composition is proven, not asserted: every position's row in the mixed run
is bitwise the row the same position gets in a pure run of its own
methodology, and the portfolio is their sum. To make the two methodologies
numerically distinguishable, the DERIVED book here is FIRST_ORDER.

Run:  python scripts/demos/e2e/02_spot_mixed_book.py
"""

from __future__ import annotations

import _spot_world as sw
import numpy as np

from riskquant.solutions.risk_run.api import run_scenario_var_es
from riskquant.solutions.risk_run.schemas import (
    CompositionContractError,
    MethodologyDeclaration,
    RiskRun,
    TaylorBlock,
)
from riskquant.solutions.scenario_models.schemas import HistoricalGeneratorConfig
from riskquant.solutions.sensitivity_pnl.taylor_contract import TaylorOrder

SCENARIOS = HistoricalGeneratorConfig(model="historical", horizon=1)
CONFIDENCE: float = 0.75


def _run(world: sw.SpotWorld, history, method: MethodologyDeclaration, block: TaylorBlock | None,
         reporting, *, confidence: float = CONFIDENCE, batch_size: int = 3) -> RiskRun:
    return run_scenario_var_es(
        scenario_config=SCENARIOS,
        data=history,
        required_factor_ids=world.required_factor_ids,
        required_shift_type_id=world.required_shift_type_id,
        source_factor_ids=world.source_factor_ids,
        shock_scale=world.shock_scale,
        base_levels=world.base_levels,
        methodology=method,
        valuation_context=sw.valuation_context(world),
        taylor_derived=block,
        reporting=reporting,
        confidence_level=confidence,
        batch_size=batch_size,
    )


def main() -> None:
    ok = sw.Checks()
    sw.title("RiskQuant SPOT executable guide - 02 mixed-methodology book")

    # ---------------------------------------------------------------- 1
    sw.step("1", "Declare the partition")
    tiny = sw.tiny_fixture()
    world = tiny.world
    full_uids, derived_uids = [102], [101, 103]
    method = MethodologyDeclaration(
        production_position_uids=np.array([101, 102, 103], dtype=np.int64),
        full_reval_uids=np.array(full_uids, dtype=np.int64),
        taylor_supplied_uids=np.empty(0, dtype=np.int64),
        taylor_derived_uids=np.array(derived_uids, dtype=np.int64),
    )
    block = sw.certified_taylor_block(world, derived_uids, order=TaylorOrder.FIRST_ORDER)
    sw.para(
        """
        U is stated EXPLICITLY; it is never inferred as the union of the
        partitions, so a forgotten position cannot silently vanish. The
        declaration proves U_F, U_S and U_D are disjoint and cover U exactly.
        The DERIVED block must price exactly U_D, on the run axis.
        """
    )
    sw.table(
        ["uid", "position", "valuation treatment", "P&L methodology"],
        [
            (101, "+10 AAPL, USD", "Greeks derived at x0", "TAYLOR/DERIVED (1st)"),
            (102, "-4 SAP, EUR", "revalued per scenario", "FULL_REVAL"),
            (103, "+8 SAP, USD quanto", "Greeks derived at x0", "TAYLOR/DERIVED (1st)"),
        ],
        align="rlll",
    )
    sw.kv([
        ("U", method.production_position_uids.tolist()),
        ("blocks (U_F, U_S, U_D)", [b.tolist() for b in method.blocks()]),
        ("partition digest", method.digest()[:24] + "..."),
        ("book prices", block.calculator.book.position_uids.tolist()),
    ])
    ok.check("the DERIVED book prices exactly U_D",
             block.calculator.book.position_uids.tolist() == derived_uids)

    # ---------------------------------------------------------------- 2
    sw.step("2", "One production call, then the pure references")
    reporting = sw.reporting_spec(world, reporting_ccy="USD")
    mixed = _run(world, tiny.history, method, block, reporting)
    all_uids = world.uids
    pure_full_method = sw.methodology(full=all_uids.tolist())
    pure_derived_method = sw.methodology(derived=all_uids.tolist())
    pure_block = sw.certified_taylor_block(world, all_uids.tolist(), order=TaylorOrder.FIRST_ORDER)
    pure_full = _run(world, tiny.history, pure_full_method, None, reporting)
    pure_derived = _run(world, tiny.history, pure_derived_method, pure_block, reporting)
    a_mixed = sw.scenario_pnl_anatomy(world, scenario_config=SCENARIOS, data=tiny.history, method=method,
                                      taylor_derived=block, reporting=reporting, confidence_level=CONFIDENCE, batch_size=3)
    a_full = sw.scenario_pnl_anatomy(world, scenario_config=SCENARIOS, data=tiny.history, method=pure_full_method,
                                     taylor_derived=None, reporting=reporting, confidence_level=CONFIDENCE, batch_size=3)
    a_der = sw.scenario_pnl_anatomy(world, scenario_config=SCENARIOS, data=tiny.history, method=pure_derived_method,
                                    taylor_derived=pure_block, reporting=reporting, confidence_level=CONFIDENCE, batch_size=3)
    for label, a, r in (("mixed", a_mixed, mixed), ("pure FULL", a_full, pure_full), ("pure DERIVED", a_der, pure_derived)):
        ok.check(f"{label}: anatomy equals the RiskRun bitwise", sw.anatomy_matches_run(a, r))

    # ---------------------------------------------------------------- 3
    sw.step("3", "Component contributions")
    rows = []
    for u in all_uids.tolist():
        meth = "FULL_REVAL" if u in full_uids else "DERIVED"
        i = a_mixed.row(u)
        rows.append((u, meth, " ".join(sw.num(x, 1) for x in a_mixed.reporting.converted_pnl[i]),
                     sw.num(mixed.var[i], 2)))
    rows.append(("PORT", "sum", " ".join(sw.num(x, 1) for x in a_mixed.reporting.portfolio_pnl),
                 sw.num(mixed.portfolio_var, 2)))
    sw.table(["uid", "method", "dW (USD) s=0..3", "VaR"], rows, align="rllr")
    for u in all_uids.tolist():
        ref, ref_run = (a_full, pure_full) if u in full_uids else (a_der, pure_derived)
        ok.check(f"uid {u}: mixed row == pure {'FULL' if u in full_uids else 'DERIVED'} row (P&L and VaR, bitwise)",
                 np.array_equal(a_mixed.reporting.converted_pnl[a_mixed.row(u)], ref.reporting.converted_pnl[ref.row(u)])
                 and mixed.var[a_mixed.row(u)] == ref_run.var[ref.row(u)])
    ok.check("portfolio P&L = sum of the composed rows, every scenario",
             np.array_equal(a_mixed.reporting.portfolio_pnl, a_mixed.reporting.converted_pnl.sum(axis=0)))
    sw.table(
        ["run", "portfolio VaR", "portfolio ES"],
        [("pure FULL_REVAL", sw.num(pure_full.portfolio_var, 2), sw.num(pure_full.portfolio_es, 2)),
         ("pure DERIVED FIRST", sw.num(pure_derived.portfolio_var, 2), sw.num(pure_derived.portfolio_es, 2)),
         ("mixed", sw.num(mixed.portfolio_var, 2), sw.num(mixed.portfolio_es, 2))],
        align="lrr",
    )
    sw.para(
        """
        Here 102 has no second-order term, so its FULL row equals its FIRST
        row, and the mixed portfolio equals the pure DERIVED portfolio. The
        identities still differ: the partition is part of the identity.
        """
    )
    ok.check("mixed, pure FULL and pure DERIVED have three distinct identities",
             len({mixed.identity, pure_full.identity, pure_derived.identity}) == 3)

    # ---------------------------------------------------------------- 4
    sw.step("4", "The same composition at realistic scale")
    real = sw.realistic_fixture()
    w = real.world
    u = w.uids
    f_uids, d_uids = u[0::2].tolist(), u[1::2].tolist()
    rep = sw.reporting_spec(w, reporting_ccy="USD")
    m_mixed = sw.methodology(full=f_uids, derived=d_uids)
    b_mixed = sw.certified_taylor_block(w, d_uids)
    b_all = sw.certified_taylor_block(w, u.tolist())
    r_mixed = _run(w, real.history, m_mixed, b_mixed, rep, confidence=0.99, batch_size=128)
    r_full = _run(w, real.history, sw.methodology(full=u.tolist()), None, rep, confidence=0.99, batch_size=128)
    r_der = _run(w, real.history, sw.methodology(derived=u.tolist()), b_all, rep, confidence=0.99, batch_size=128)
    pos = {int(x): i for i, x in enumerate(r_mixed.position_uids.tolist())}
    sw.kv([
        ("U", f"{u.shape[0]} positions: {len(f_uids)} FULL, {len(d_uids)} DERIVED"),
        ("scenarios", real.num_observations - 1),
        ("portfolio VaR 99% (mixed)", sw.num(r_mixed.portfolio_var, 2)),
    ])
    ok.check("every FULL position's VaR equals its pure FULL_REVAL VaR bitwise",
             all(r_mixed.var[pos[x]] == r_full.var[pos[x]] for x in f_uids))
    ok.check("every DERIVED position's VaR equals its pure DERIVED VaR bitwise",
             all(r_mixed.var[pos[x]] == r_der.var[pos[x]] for x in d_uids))

    # ---------------------------------------------------------------- 5
    sw.step("5", "Wiring defects fail closed")
    ok.expect_rejection(
        "positions declared TAYLOR/DERIVED but no DERIVED block supplied",
        CompositionContractError,
        lambda: _run(world, tiny.history, method, None, reporting),
        reason="a declared methodology without its representation cannot be priced",
        stage="composition contract, before identity and numerics",
    )
    ok.expect_rejection(
        "a DERIVED book that prices a FULL_REVAL position as well",
        CompositionContractError,
        lambda: _run(world, tiny.history, method, pure_block, reporting),
        reason="the book must price exactly U_D; a position priced twice would "
        "be double counted",
        stage="composition contract, before identity and numerics",
    )
    ok.expect_rejection(
        "U omits a position the valuation context compiles (103)",
        CompositionContractError,
        lambda: _run(world, tiny.history, sw.methodology(full=[101, 102]), None, reporting),
        reason="valuation rows are never dropped and absent positions are never "
        "zero-filled: U must equal the compiled economic positions exactly",
        stage="composition contract, before identity and numerics",
    )
    ok.expect_rejection(
        "a position declared both FULL_REVAL and TAYLOR/DERIVED",
        CompositionContractError,
        lambda: MethodologyDeclaration(
            production_position_uids=np.array([101, 102, 103], dtype=np.int64),
            full_reval_uids=np.array([102, 103], dtype=np.int64),
            taylor_supplied_uids=np.empty(0, dtype=np.int64),
            taylor_derived_uids=np.array([101, 103], dtype=np.int64),
        ),
        reason="a position has exactly one methodology",
        stage="MethodologyDeclaration construction",
    )
    ok.done()


if __name__ == "__main__":
    main()
