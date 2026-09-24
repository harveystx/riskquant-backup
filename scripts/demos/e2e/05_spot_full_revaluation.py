# scripts/demos/e2e/05_spot_full_revaluation.py
"""
RiskQuant SPOT -- chapter 05: FULL_REVAL, and how it compares with Taylor.

    base valuation V0 = V(x0)            (the run values it itself, S = 1)
        -> shocked market levels x_s     (ONE canonical (K, S) buffer)
        -> revaluation V_s = V(x_s)      (the SAME route: flat levels ->
                                          market data -> valuation state ->
                                          run_valuation)
        -> dV = V_s - V0, stored in the float32 P&L grid
        -> D081 conversion -> VaR / ES

Part A (tiny, exact): FULL_REVAL, TAYLOR/DERIVED second order FULL and
TAYLOR/DERIVED FIRST_ORDER on the chapter-00 book. With dyadic inputs every
number is exact, so FULL_REVAL and the full second order agree bitwise and
FIRST_ORDER differs by exactly the quanto cross term.

Part B (realistic, LOG_RETURN): the same three methodologies on 120
positions and 750 historical scenarios. Here the two exact representations
agree up to the rounding their dtype contracts allow. The comparison rule is
stated and derived from those contracts, not tuned: the Taylor kernel rounds
each displacement to float32 once, and both methods round their P&L once into
the float32 grid. The resulting per-element bound is computed and printed.

Run:  python scripts/demos/e2e/05_spot_full_revaluation.py
"""

from __future__ import annotations

import time

import _spot_world as sw
import numpy as np

from riskquant.sensitivities.engine import run_sensitivities
from riskquant.solutions.risk_run.api import run_scenario_var_es
from riskquant.solutions.risk_run.schemas import MethodologyDeclaration, RiskRun, TaylorBlock
from riskquant.solutions.scenario_models.schemas import HistoricalGeneratorConfig
from riskquant.solutions.sensitivity_pnl.taylor_contract import TaylorOrder

SCENARIOS = HistoricalGeneratorConfig(model="historical", horizon=1)


def _run(world: sw.SpotWorld, data, method: MethodologyDeclaration, block: TaylorBlock | None,
         reporting, confidence: float, batch_size: int) -> RiskRun:
    """Tiny local wrapper: the production call with this chapter's fixed inputs."""
    return run_scenario_var_es(
        scenario_config=SCENARIOS,
        data=data,
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


def _declare(uids: np.ndarray, *, derived: bool) -> MethodologyDeclaration:
    empty = np.empty(0, dtype=np.int64)
    return MethodologyDeclaration(
        production_position_uids=uids,
        full_reval_uids=empty if derived else uids,
        taylor_supplied_uids=empty,
        taylor_derived_uids=uids if derived else empty,
    )


def main() -> None:
    ok = sw.Checks()
    sw.title("RiskQuant SPOT executable guide - 05 full revaluation")

    # ================================================================ A
    sw.step("A1", "Tiny book: the revaluation route, step by step")
    tiny = sw.tiny_fixture()
    world = tiny.world
    shocked = sw.shocked_levels_for(world, scenario_config=SCENARIOS, data=tiny.history, batch_size=4)
    vs = sw.revalue(world, shocked)
    v0 = sw.revalue(world, world.base_levels[:, None].copy())
    sw.para(
        """
        `revalue` is the run's own valuation route applied to any (K, S)
        level slice. At the base column (S = 1) it returns V0; at the shocked
        buffer it returns V_s. Same pricer, same fold, same kernel.
        """
    )
    sw.table(
        ["uid", "V0", *[f"V_s s={s}" for s in range(shocked.shape[1])]],
        [(u, sw.num(v0[u][0], 2), *(sw.num(x, 2) for x in vs[u])) for u in world.uids.tolist()],
    )
    ok.check("revaluation at x0 reproduces the ingest-time V0 exactly",
             all(v0[u][0] == world.v0_by_uid[u] for u in world.uids.tolist()))

    sw.step("A2", "Tiny book: FULL_REVAL vs TAYLOR/DERIVED")
    reporting = sw.reporting_spec(world, reporting_ccy="USD")
    uids = world.uids
    full_block = sw.certified_taylor_block(world, uids.tolist(), order=TaylorOrder.SECOND_ORDER_FULL)
    first_block = sw.certified_taylor_block(world, uids.tolist(), order=TaylorOrder.FIRST_ORDER)
    variants = {
        "FULL_REVAL": (_declare(uids, derived=False), None),
        "DERIVED second-order FULL": (_declare(uids, derived=True), full_block),
        "DERIVED FIRST_ORDER": (_declare(uids, derived=True), first_block),
    }
    results = {}
    for label, (method, block) in variants.items():
        run = _run(world, tiny.history, method, block, reporting, 0.75, 4)
        anatomy = sw.scenario_pnl_anatomy(world, scenario_config=SCENARIOS, data=tiny.history, method=method,
                                          taylor_derived=block, reporting=reporting, confidence_level=0.75, batch_size=4)
        ok.check(f"{label}: anatomy equals the RiskRun bitwise", sw.anatomy_matches_run(anatomy, run))
        results[label] = (run, anatomy)
    sw.table(
        ["methodology", "port. VaR", "port. ES", "103 dV, s=0..3"],
        [(k, sw.num(r.portfolio_var, 2), sw.num(r.portfolio_es, 2),
          " ".join(sw.num(x, 0) for x in a.settlement_pnl[a.row(103)])) for k, (r, a) in results.items()],
        align="lrrl",
    )
    f_run, f_an = results["FULL_REVAL"]
    t_run, t_an = results["DERIVED second-order FULL"]
    l_run, l_an = results["DERIVED FIRST_ORDER"]
    ok.check("FULL_REVAL and second-order FULL agree bitwise (bilinear payoff, exact inputs)",
             np.array_equal(f_an.settlement_pnl, t_an.settlement_pnl)
             and np.array_equal(f_run.var, t_run.var) and np.array_equal(f_run.es, t_run.es))
    ok.check("FIRST_ORDER differs only on the quanto 103",
             all(np.array_equal(f_an.settlement_pnl[f_an.row(u)], l_an.settlement_pnl[l_an.row(u)])
                 for u in (101, 102))
             and not np.array_equal(f_an.settlement_pnl[f_an.row(103)], l_an.settlement_pnl[l_an.row(103)]))
    ok.check("same economics, different methodology: different run identity",
             f_run.identity != t_run.identity)
    sw.para(
        """
        Methodology is part of the run identity (partition digest and book
        digest), even when the numbers agree: a reproduction must use the
        same representation, not merely one that happens to match.
        """
    )

    # ================================================================ B
    sw.step("B1", "Realistic book: 120 positions, 750 historical scenarios")
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
    blocks = {
        "FULL": sw.certified_taylor_block(w, uids.tolist(), order=TaylorOrder.SECOND_ORDER_FULL),
        "FIRST": sw.certified_taylor_block(w, uids.tolist(), order=TaylorOrder.FIRST_ORDER),
    }
    out = {}
    for label, method, block in (
        ("FULL_REVAL", _declare(uids, derived=False), None),
        ("DERIVED FULL", _declare(uids, derived=True), blocks["FULL"]),
        ("DERIVED FIRST", _declare(uids, derived=True), blocks["FIRST"]),
    ):
        run = _run(w, real.history, method, block, rep, 0.99, 256)
        an = sw.scenario_pnl_anatomy(w, scenario_config=SCENARIOS, data=real.history, method=method,
                                     taylor_derived=block, reporting=rep, confidence_level=0.99, batch_size=256)
        ok.check(f"{label}: anatomy equals the RiskRun bitwise", sw.anatomy_matches_run(an, run))
        out[label] = (run, an)

    sw.step("B2", "Compare - with an explicit rule")
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
    sens = run_sensitivities(portfolio=w.portfolio, uix_mapper=w.uix_mapper, market_data=w.market_data)
    shocked_b = out["FULL_REVAL"][1].shocked_levels
    dxs = shocked_b - w.base_levels[:, None]
    row_of = {int(r): i for i, r in enumerate(w.required_factor_ids)}
    pos_of = {int(x): i for i, x in enumerate(uids.tolist())}
    bound = np.zeros_like(full)
    for p_uid, f_id, d1 in zip(sens.position_uids.tolist(), sens.factor_rf_ids.tolist(),
                               sens.first_order.tolist(), strict=True):
        bound[pos_of[p_uid]] += u * abs(d1) * np.abs(dxs[row_of[f_id]])
    for p_uid, fi, fj, c in zip(sens.cross_position_uids.tolist(), sens.cross_factor_rf_ids_i.tolist(),
                                sens.cross_factor_rf_ids_j.tolist(), sens.cross_second_order.tolist(), strict=True):
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
        FX leg move together. It is explained, not hidden. Part B wall time:
        {time.perf_counter() - t0:.2f} s.
        """
    )
    ok.done()


if __name__ == "__main__":
    main()
