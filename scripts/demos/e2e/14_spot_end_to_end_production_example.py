# scripts/demos/e2e/14_spot_end_to_end_production_example.py
"""
RiskQuant SPOT -- chapter 14: a production-style end-to-end example.

The template to copy for an integration. It follows the order a real daily
run takes:

    1  receive the desk's inputs     (positions, levels, history, static
                                      config, proxy config, run request)
    2  build the valuation world     (ingest -> mapper -> axis -> V0)
    3  choose each position's methodology by an explicit policy
    4  build the scenario model from the request (untyped -> typed config)
    5  derive the run axis           (reporting route pre-check, cheap)
    6  run RiskQuant                 (ONE call: run_scenario_var_es)
    7  inspect the result
    8  verify reproducibility        (reproduce= the persisted manifest)
    9  persist and reload the RiskRun (bitwise JSON)

Replace `sw.realistic_fixture()` in step 1 with the firm's data feeds; every
later step is production API usage exactly as an integration would write it.
Methodology policy used here (an example, not a recommendation): positions
whose pricer reads an FX leg (quanto) are FULL_REVAL; single-factor
positions are TAYLOR/DERIVED with a certified second-order book.

Run:  python scripts/demos/e2e/14_spot_end_to_end_production_example.py
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import _spot_world as sw
import numpy as np

from riskquant.solutions.risk_run.api import derive_run_factor_axis, run_scenario_var_es
from riskquant.solutions.risk_run.schemas import MethodologyDeclaration, RiskRun
from riskquant.solutions.scenario_models.api import parse_scenario_model_config
from riskquant.solutions.sensitivity_pnl.taylor_contract import TaylorOrder

#: The run request as it would arrive from a scheduler, a JSON file or an
#: Excel named range: plain strings and numbers only.
RUN_REQUEST: dict[str, object] = {
    "scenario_model": {
        "model": "filtered_bootstrap",
        "horizon": 1,
        "total_scenarios": 5000,
        "seed": 20270104,
    },
    "confidence_level": 0.99,
    "reporting_ccy": "USD",
    "batch_size": 1024,
}


def main() -> None:
    ok = sw.Checks()
    sw.title("RiskQuant SPOT executable guide - 14 production-style example")
    t_start = time.perf_counter()

    # ---------------------------------------------------------------- 1
    sw.step("1", "Receive the inputs")
    feed = sw.realistic_fixture()  # <- replace with the firm's data feeds
    positions = feed.world.positions
    sw.kv([
        ("positions", f"{len(positions)} rows, columns {list(positions.columns)}"),
        ("settlement ccys", positions["settlement_ccy"].value_counts().to_dict()),
        ("level snapshot", f"{len(feed.world.levels)} factors, as-of {feed.world.as_of}"),
        ("history", f"{feed.num_observations} observations x {feed.history.factor_ids.shape[0]} factors"),
        ("proxy config", "WTI_CRUDE <- BRENT_CRUDE x 0.95"),
        ("run request", RUN_REQUEST["scenario_model"]),
    ])

    # ---------------------------------------------------------------- 2
    sw.step("2", "Build the valuation world")
    world = sw.build_world(
        rsc=feed.world.rsc,
        proxy_topology=feed.world.proxy_topology,
        positions=positions,
        levels=feed.world.levels,
        as_of=feed.world.as_of,
        history_factor_names=[feed.world.name_of(int(r)) for r in feed.history.factor_ids],
        reporting_ccy=str(RUN_REQUEST["reporting_ccy"]),
    )
    context = sw.valuation_context(world)
    sw.kv([
        ("run axis", f"{world.required_factor_ids.shape[0]} factors"),
        ("proxied factors on this axis",
         [world.axis_names()[i] for i in range(world.required_factor_ids.shape[0])
          if world.source_factor_ids[i] != world.required_factor_ids[i]]
         or "none (no position reads WTI_CRUDE; see chapter 07)"),
        ("V0 computed for", f"{len(world.v0_by_uid)} positions (settlement ccy)"),
    ])
    ok.check("every ingested position has exactly one V0",
             sorted(world.v0_by_uid) == sorted(int(u) for u in positions["uid"]))

    # ---------------------------------------------------------------- 3
    sw.step("3", "Methodology by explicit policy")
    # The ingest batch records, per position, the FX leg its pricer reads
    # (fx_u2s_ids < 0: none). The policy reads that production fact directly.
    reads_fx = world.ingest_batch.fx_u2s_ids >= 0
    batch_uids = world.ingest_batch.uids
    full_uids = sorted(int(u) for u in batch_uids[reads_fx])
    derived_uids = sorted(int(u) for u in batch_uids[~reads_fx])
    method = MethodologyDeclaration(
        production_position_uids=world.uids,
        full_reval_uids=np.array(full_uids, dtype=np.int64),
        taylor_supplied_uids=np.empty(0, dtype=np.int64),
        taylor_derived_uids=np.array(derived_uids, dtype=np.int64),
    )
    taylor = sw.certified_taylor_block(world, derived_uids, order=TaylorOrder.SECOND_ORDER_FULL)
    sw.kv([("FULL_REVAL (quanto)", len(full_uids)), ("TAYLOR/DERIVED (single factor)", len(derived_uids)),
           ("TAYLOR/SUPPLIED", "0 (refused while P-1 is open)")])

    # ---------------------------------------------------------------- 4
    sw.step("4", "Scenario model from the request")
    scenario_config = parse_scenario_model_config(RUN_REQUEST["scenario_model"])  # type: ignore[arg-type]
    sw.kv([(k, v) for k, v in scenario_config.model_dump().items()])

    # ---------------------------------------------------------------- 5
    sw.step("5", "Derive the run axis (the reporting pre-check)")
    # The caller chooses ONLY the reporting currency (D085). The run derives
    # every settlement currency and reporting FX route from the compiled book;
    # `derive_run_factor_axis` runs that derivation up front, so an
    # unroutable currency fails here, before any scenario data is prepared.
    reporting = sw.reporting_spec(world, reporting_ccy=str(RUN_REQUEST["reporting_ccy"]))
    run_axis = derive_run_factor_axis(valuation_context=context, reporting_ccy=reporting.reporting_ccy)
    sw.kv([
        ("valuation factors", run_axis.num_valuation_factors),
        ("reporting-only FX factors", [world.name_of(int(r)) for r in run_axis.reporting_factor_ids] or "none"),
    ])
    ok.check("the world's scenario data is bound to exactly the derived run axis",
             np.array_equal(run_axis.factor_ids, world.required_factor_ids)
             and np.array_equal(run_axis.shift_type_ids, world.required_shift_type_id))

    # ---------------------------------------------------------------- 6
    sw.step("6", "Run RiskQuant")
    t_run = time.perf_counter()
    run = run_scenario_var_es(
        scenario_config=scenario_config,
        data=feed.history,
        required_factor_ids=world.required_factor_ids,
        required_shift_type_id=world.required_shift_type_id,
        source_factor_ids=world.source_factor_ids,
        shock_scale=world.shock_scale,
        base_levels=world.base_levels,
        methodology=method,
        valuation_context=context,
        taylor_derived=taylor,
        reporting=reporting,
        confidence_level=float(RUN_REQUEST["confidence_level"]),  # type: ignore[arg-type]
        batch_size=int(RUN_REQUEST["batch_size"]),  # type: ignore[arg-type]
    )
    run_s = time.perf_counter() - t_run
    sw.kv([("wall time", f"{run_s:.2f} s"), ("identity", run.identity)])

    # ---------------------------------------------------------------- 7
    sw.step("7", "Inspect the result")
    c = float(RUN_REQUEST["confidence_level"])  # type: ignore[arg-type]
    sw.kv([
        ("portfolio VaR", f"{sw.num(run.portfolio_var, 2)} {run.reporting_ccy} at {c:.0%}"),
        ("portfolio ES", f"{sw.num(run.portfolio_es, 2)} {run.reporting_ccy}"),
        ("sum of standalone VaRs", sw.num(float(run.var[:-1].sum()), 2)),
        ("reproducible", run.reproducible),
        ("Taylor provenance", f"{run.pnl_provenance.num_positions} positions, "
                              f"{run.pnl_provenance.num_active_factors} active factors"
         if run.pnl_provenance else "none"),
    ])
    worst = np.argsort(run.var[:-1])[:5]
    frame = positions.set_index("uid")
    sw.table(
        ["uid", "underlying", "qty", "settle", "method", "standalone VaR"],
        [(int(run.position_uids[i]), frame.loc[int(run.position_uids[i]), "underlying"],
          sw.num(frame.loc[int(run.position_uids[i]), "quantity"], 0),
          frame.loc[int(run.position_uids[i]), "settlement_ccy"],
          "FULL" if int(run.position_uids[i]) in full_uids else "DERIVED", sw.num(run.var[i], 2)) for i in worst],
        align="rlrllr",
    )
    anatomy = sw.scenario_pnl_anatomy(world, scenario_config=scenario_config, data=feed.history, method=method,
                                      taylor_derived=taylor, reporting=reporting, confidence_level=c,
                                      batch_size=int(RUN_REQUEST["batch_size"]))  # type: ignore[arg-type]
    v, e, _ = sw.oracle_var_es(anatomy.reporting.portfolio_pnl, c)
    ok.check("anatomy equals the RiskRun bitwise", sw.anatomy_matches_run(anatomy, run))
    ok.check("portfolio VaR equals the independent order statistic; ES within the summation bound",
             run.portfolio_var == v
             and abs(run.portfolio_es - e) <= sw.es_summation_bound(anatomy.reporting.portfolio_pnl, c))
    ok.check("one VaR/ES per position plus the portfolio; ES at least as severe as VaR",
             run.var.shape == (len(positions) + 1,) and bool((run.es <= run.var).all()))

    # ---------------------------------------------------------------- 8
    sw.step("8", "Persist, reload, reproduce")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"riskrun_{run.identity[:16]}.json"
        path.write_text(run.to_json(), encoding="utf-8")
        stored = RiskRun.from_json(path.read_text(encoding="utf-8"))
        size_kb = path.stat().st_size / 1024
    sw.kv([("artifact", f"riskrun_<identity>.json, {size_kb:.1f} KiB (temporary dir, removed)")])
    ok.check("the reloaded RiskRun is bitwise identical", np.array_equal(stored.var, run.var)
             and np.array_equal(stored.es, run.es) and stored.identity == run.identity)
    replay = run_scenario_var_es(
        scenario_config=scenario_config,
        data=feed.history,
        required_factor_ids=world.required_factor_ids,
        required_shift_type_id=world.required_shift_type_id,
        source_factor_ids=world.source_factor_ids,
        shock_scale=world.shock_scale,
        base_levels=world.base_levels,
        methodology=method,
        valuation_context=context,
        taylor_derived=taylor,
        reporting=reporting,
        confidence_level=c,
        batch_size=257,  # a different partition on purpose
        reproduce=stored.manifest,
    )
    ok.check("reproduce= against the persisted manifest: same identity, bitwise same numbers",
             replay.identity == stored.identity and np.array_equal(replay.var, stored.var)
             and np.array_equal(replay.es, stored.es))

    # ---------------------------------------------------------------- 9
    sw.step("9", "The RiskRun, presented")
    m = run.manifest
    sw.kv([
        ("identity", run.identity[:32] + "..."),
        ("scenario model", m.scenario_config),
        ("data", f"{m.data.kind}, digest {m.data.digest[:12]}..."),
        ("base levels", f"digest {m.pnl.base_levels_digest[:12]}..."),
        ("Taylor book", f"digest {m.pnl.sensitivities_digest[:12]}..."),
        ("partition", f"digest {m.pnl.composition.methodology.partition_digest[:12]}..."),
        ("as-of / discounting", f"{m.pnl.composition.context.as_of_ord32} / "
                                f"{m.pnl.composition.context.discounting_mode}"),
        ("reporting", f"{m.reporting.reporting_currency}, plan {m.reporting.fx_digest[:12]}..."),
        ("risk measure", f"confidence {m.risk_measure.confidence_level}"),
        ("engine", f"python {m.engine.python_version}, numpy {m.engine.numpy_version}, "
                   f"numba {m.engine.numba_version}"),
    ])
    sw.write_evidence("14_spot_end_to_end_production_example", {"riskrun": run.to_json()})
    sw.para(f"Total wall time: {time.perf_counter() - t_start:.2f} s.")
    ok.done()


if __name__ == "__main__":
    main()
