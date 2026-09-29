# scripts/demos/e2e/07_spot_production_example.py
"""
RiskQuant SPOT -- chapter 07: a production-style end-to-end example.

The template to copy for an integration. It follows the order a real daily
run takes, and it writes every production call inline:

    1  receive the desk's inputs     (static config, proxy topology,
                                      positions frame, levels, history,
                                      run request)
    2  build the valuation context   (ingest -> mapper -> ValuationContext)
    3  bind scenario data to the run axis
    4  choose each position's methodology by an explicit policy
    5  certify the TAYLOR/DERIVED book on U_D
    6  run RiskQuant                 (ONE call: run_scenario_var_es)
    7  inspect the result
    8  persist, reload and reproduce (bitwise JSON, reproduce=)
    9  present the RiskRun and its manifest

Only step 1 uses the demo helper (`sw.realistic_fixture()`), and only as a
data feed: the edge tables it returns are what a firm's feeds deliver.
Nothing else from the helper world is used.

Import surface, stated plainly
------------------------------
The run, its contracts, the run-axis derivation, the scenario-model parser
and the Taylor contract come from `riskquant.solutions.*`. Assembling the
run's inputs still needs core-package calls: ingest, the UixMapper, the level
snapshot, the authored Proxy V1 terms and, for Taylor certification only, the U_D base
valuation and the sensitizer. No solution facade exists for them today. The
run itself values every V_0; the caller values V_0 only to certify a Taylor
book at the base the run will derive (P1-B).

Methodology policy used here (an example, not a recommendation): positions
whose pricer reads an FX leg (quanto) are FULL_REVAL; single-factor
positions are TAYLOR/DERIVED with a certified second-order book.

Run:  python scripts/demos/e2e/07_spot_production_example.py
"""

from __future__ import annotations

import dataclasses
import tempfile
import time
from pathlib import Path

import _spot_world as sw
import numpy as np

from riskquant.config.enums.discounting_modes import DiscountingMode
from riskquant.market_data.assemble import (
    build_market_data_matrix_from_snapshot,
    build_universe_mapping,
    extract_levels_from_snapshot,
)
from riskquant.market_data.compile import assemble_valuation_state
from riskquant.market_data.risk_factors.builder_snapshot import (
    build_risk_factor_level_snapshot_from_dict,
)
from riskquant.portfolio.engine import (
    build_uix_mapper,
    compile_portfolio_from_ingest_batch,
)
from riskquant.portfolio.ingest.pipeline import compile_raw_df_into_ingest_batch
from riskquant.config.proxy_model.compile import compile_proxy_model
from riskquant.sensitivities.engine import run_sensitivities
from riskquant.simlab.shocks.axis_operator import lower_axis_operator, world_request
from riskquant.solutions.risk_run.api import derive_run_factor_axis, run_scenario_var_es
from riskquant.solutions.risk_run.schemas import (
    MethodologyDeclaration,
    ReportingSpec,
    RiskRun,
    TaylorBlock,
    ValuationContext,
)
from riskquant.solutions.scenario_models.api import parse_scenario_model_config
from riskquant.solutions.sensitivity_pnl.api import (
    SensitivityPnlCalculator,
    build_sensitivity_book_spec,
    derive_taylor_declaration,
)
from riskquant.solutions.sensitivity_pnl.taylor_contract import (
    TaylorBaseBinding,
    TaylorOrder,
    base_levels_digest,
    base_settlement_values_digest,
    static_config_content_digest,
    valuation_contract_fingerprint,
)
from riskquant.utils.dates import date_to_ord32
from riskquant.valuation.engine import run_valuation

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

#: SPOT reads no discount factor.
DISCOUNTING: DiscountingMode = DiscountingMode.NONE


def main() -> None:
    ok = sw.Checks()
    sw.title("RiskQuant SPOT executable guide - 07 production-style example")
    t_start = time.perf_counter()

    # ---------------------------------------------------------------- 1
    sw.step("1", "Receive the inputs")
    feed = sw.realistic_fixture()  # <- replace with the firm's data feeds
    rsc = feed.world.rsc  # compiled static configuration
    proxy_terms = feed.world.proxy_terms  # authored Proxy V1 terms (D088)
    positions = feed.world.positions  # human-readable positions frame
    levels = dict(feed.world.levels)  # name -> level at the as-of
    as_of = feed.world.as_of
    history = feed.history  # RiskFactorMatrix: the scenario model's data
    sw.kv([
        ("positions", f"{len(positions)} rows, columns {list(positions.columns)}"),
        ("settlement ccys", positions["settlement_ccy"].value_counts().to_dict()),
        ("level snapshot", f"{len(levels)} factors, as-of {as_of}"),
        ("history", f"{feed.num_observations} observations x {history.factor_ids.shape[0]} factors"),
        ("proxy config", "WTI_CRUDE <- BRENT_CRUDE x 0.95"),
        ("run request", RUN_REQUEST["scenario_model"]),
    ])

    # ---------------------------------------------------------------- 2
    sw.step("2", "Build the valuation context")
    # Names die here: ingest resolves every name to an rf_id. Ingest mutates
    # its input frame in place unless told otherwise, so it receives a copy.
    ingest_batch = compile_raw_df_into_ingest_batch(raw_df=positions.copy(), static_config=rsc)
    as_of_ord = int(date_to_ord32(as_of))
    uix_mapper = build_uix_mapper([ingest_batch], as_of_ord32=as_of_ord)
    context = ValuationContext(
        static_config=rsc,
        ingest_batch=ingest_batch,
        uix_mapper=uix_mapper,
        discounting_mode=DISCOUNTING,
    )
    snapshot = build_risk_factor_level_snapshot_from_dict(
        as_of_date=as_of, levels_dict=levels, static_config=rsc
    )
    sw.para(
        """
        The caller hands over the ingest output, never a compiled portfolio:
        the ValuationContext compiles the book itself (N1), and the run
        values every V_0 from it. The caller computes no V_0 for the run.
        """
    )
    ok.check("the context compiled one valuation row per ingested position",
             np.array_equal(np.sort(np.asarray(context.portfolio.trade_uids, dtype=np.int64)),
                            np.sort(np.asarray(positions["uid"], dtype=np.int64))))

    # ---------------------------------------------------------------- 3
    sw.step("3", "Bind the scenario data to the run axis")
    # The caller chooses ONLY the reporting currency (D085). The run derives
    # every settlement currency and reporting FX route from the compiled book;
    # `derive_run_factor_axis` performs that derivation up front, so an
    # unroutable currency fails here, before any scenario data is prepared.
    reporting = ReportingSpec(reporting_ccy=str(RUN_REQUEST["reporting_ccy"]))
    run_axis = derive_run_factor_axis(valuation_context=context, reporting_ccy=reporting.reporting_ccy)
    # The axis is a read-only descriptor; the scenario building blocks take
    # the caller's own writable arrays bound to it, row for row.
    required_factor_ids = np.array(run_axis.factor_ids, copy=True)
    required_shift_type_id = np.array(run_axis.shift_type_ids, copy=True)
    base_levels = extract_levels_from_snapshot(required_ids=required_factor_ids, snapshot=snapshot)
    # Preview of the run's own Proxy V1 composition (the run repeats it): the
    # authored terms compiled over the full axis, lowered onto it, and the
    # (driver, driver chart) world columns the rows read.
    preview = lower_axis_operator(
        model=compile_proxy_model(
            authored_terms=proxy_terms,
            static_config=rsc,
            as_of_date=as_of,
            run_axis_rf_ids=required_factor_ids,
        ),
        axis_rf_ids=required_factor_ids,
        axis_shift_type_ids=required_shift_type_id,
    )
    request_drivers, _ = world_request(preview)
    k_val = run_axis.num_valuation_factors
    on_axis = set(required_factor_ids.tolist())
    proxied = [t.target_name for t in proxy_terms if rsc.rel_name_to_rf_id[t.target_name] in on_axis]
    sw.kv([
        ("run axis", f"{required_factor_ids.shape[0]} factors: {k_val} valuation, "
                     f"{required_factor_ids.shape[0] - k_val} reporting-only"),
        ("proxied factors on this axis", proxied or "none (no position reads WTI_CRUDE)"),
        ("base levels", f"{base_levels.dtype}, one per axis row"),
    ])
    ok.check("every world column the compiled model reads is a column of the history",
             bool(np.isin(request_drivers, history.factor_ids).all()))

    # ---------------------------------------------------------------- 4
    sw.step("4", "Methodology by explicit policy")
    # The ingest batch records, per position, the FX leg its pricer reads
    # (fx_u2s_ids < 0: none). The policy reads that production fact directly.
    reads_fx = ingest_batch.fx_u2s_ids >= 0
    full_uids = sorted(int(u) for u in ingest_batch.uids[reads_fx])
    derived_uids = sorted(int(u) for u in ingest_batch.uids[~reads_fx])
    method = MethodologyDeclaration(
        production_position_uids=np.sort(np.asarray(positions["uid"], dtype=np.int64)),
        full_reval_uids=np.array(full_uids, dtype=np.int64),
        taylor_supplied_uids=np.empty(0, dtype=np.int64),
        taylor_derived_uids=np.array(derived_uids, dtype=np.int64),
    )
    sw.kv([("FULL_REVAL (quanto)", len(full_uids)), ("TAYLOR/DERIVED (single factor)", len(derived_uids)),
           ("TAYLOR/SUPPLIED", "0 (refused while P-1 is open, chapter 01)")])

    # ---------------------------------------------------------------- 5
    sw.step("5", "Certify the TAYLOR/DERIVED book on U_D")
    sw.para(
        """
        Sensitise U_D ONLY, on its own compiled sub-portfolio: a position
        outside U_D (for example an inverted-FX leg, FULL_REVAL only, D084)
        never reaches the sensitizer. Value the same sub-portfolio at the
        base to bind the book to V_0, then compile the book on the valuation
        prefix of the run axis. The run re-derives this binding from its own
        context and refuses any other (P1-A/B).
        """
    )
    d_frame = positions.loc[positions["uid"].isin(derived_uids)].reset_index(drop=True)
    d_batch = compile_raw_df_into_ingest_batch(raw_df=d_frame, static_config=rsc)
    d_mapper = build_uix_mapper([d_batch], as_of_ord32=as_of_ord)
    d_portfolio = compile_portfolio_from_ingest_batch(ingest_batch=d_batch, uix_mapper=d_mapper)
    d_market = build_market_data_matrix_from_snapshot(
        discounting_mode=DISCOUNTING,
        as_of_date=as_of,
        uix_mapper=d_mapper,
        universe_map=build_universe_mapping(
            discounting_mode=DISCOUNTING, static_config=rsc, uix_mapper=d_mapper
        ),
        snapshot=snapshot,
        num_scenarios=1,
    )
    sensitivities = run_sensitivities(portfolio=d_portfolio, uix_mapper=d_mapper, market_data=d_market)
    v0_rows = np.empty((d_portfolio.total_trades, 1), dtype=np.float64)
    run_valuation(
        portfolio=d_portfolio,
        valuation_state=assemble_valuation_state(matrix=d_market, uix_mapper=d_mapper),
        out=v0_rows,
    )
    row_uids = np.asarray(d_portfolio.trade_uids, dtype=np.int64)
    order = np.argsort(row_uids, kind="stable")
    book_uids, book_v0 = row_uids[order], np.ascontiguousarray(v0_rows[order, 0])
    if np.unique(book_uids).shape[0] != book_uids.shape[0]:
        raise ValueError("a SPOT uid has more than one valuation row")
    valuation_factor_ids = np.array(required_factor_ids[:k_val], copy=True)
    binding = TaylorBaseBinding(
        as_of_ord32=int(uix_mapper.as_of_ord32),
        base_levels_digest=base_levels_digest(
            factor_ids=valuation_factor_ids, base_levels=base_levels[:k_val]
        ),
        discounting_mode=DISCOUNTING.value,
        static_config_digest=static_config_content_digest(rsc),
        valuation_contract_fingerprint=valuation_contract_fingerprint(),
        base_settlement_values_digest=base_settlement_values_digest(
            position_uids=book_uids, values=book_v0
        ),
    )
    spec = build_sensitivity_book_spec(sensitivities=sensitivities, required_factor_ids=valuation_factor_ids)
    declaration = derive_taylor_declaration(
        sensitivities=sensitivities, order=TaylorOrder.SECOND_ORDER_FULL, base_binding=binding
    )
    taylor = TaylorBlock(
        calculator=SensitivityPnlCalculator.compile(dataclasses.replace(spec, taylor_declaration=declaration))
    )
    ok.check("the certified book prices exactly U_D, on the valuation prefix of the run axis",
             np.array_equal(taylor.calculator.book.position_uids, np.array(derived_uids, dtype=np.int64))
             and taylor.calculator.axis.keys.to_list() == valuation_factor_ids.tolist())

    # ---------------------------------------------------------------- 6
    sw.step("6", "Run RiskQuant")
    scenario_config = parse_scenario_model_config(RUN_REQUEST["scenario_model"])  # type: ignore[arg-type]
    confidence = float(RUN_REQUEST["confidence_level"])  # type: ignore[arg-type]
    t_run = time.perf_counter()
    run = run_scenario_var_es(
        scenario_config=scenario_config,
        data=history,
        required_factor_ids=required_factor_ids,
        required_shift_type_id=required_shift_type_id,
        proxy_terms=proxy_terms,
        base_levels=base_levels,
        methodology=method,
        valuation_context=context,
        taylor_derived=taylor,
        reporting=reporting,
        confidence_level=confidence,
        batch_size=int(RUN_REQUEST["batch_size"]),  # type: ignore[arg-type]
    )
    run_s = time.perf_counter() - t_run
    sw.kv([("scenario model", scenario_config.model_dump()), ("wall time", f"{run_s:.2f} s"),
           ("identity", run.identity)])

    # ---------------------------------------------------------------- 7
    sw.step("7", "Inspect the result")
    sw.kv([
        ("portfolio VaR", f"{sw.num(run.portfolio_var, 2)} {run.reporting_ccy} at {confidence:.0%}"),
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
    ok.check("one VaR/ES per position of U plus the portfolio; ES at least as severe as VaR",
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
        data=history,
        required_factor_ids=required_factor_ids,
        required_shift_type_id=required_shift_type_id,
        proxy_terms=proxy_terms,
        base_levels=base_levels,
        methodology=method,
        valuation_context=context,
        taylor_derived=taylor,
        reporting=reporting,
        confidence_level=confidence,
        batch_size=257,  # a different partition on purpose (chapter 06)
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
    sw.write_evidence("07_spot_production_example", {"riskrun": run.to_json()})
    sw.para(f"Total wall time: {time.perf_counter() - t_start:.2f} s.")
    ok.done()


if __name__ == "__main__":
    main()
