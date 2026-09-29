# tests/solutions/risk_run/_certfix.py
"""Certification fixture: production-representative, drives the OFFICIAL API.

P1-B: a real valuation world, not a synthetic Taylor book. A static
configuration with 249 equity nodes and one FX node (USDEUR, id 250) is compiled
from the demo sheets; 200 SPOT positions (alternately settled in EUR and USD)
are ingested and compiled; the run axis is the universe map the run builds; the
TAYLOR/DERIVED book is certified at the run base on the producer path, from the
same context the run values. `FIDS` / `K` are the run axis and its length."""

import dataclasses
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import _mixedfix as mf

from riskquant.market_data.assemble import (
    build_market_data_matrix_from_flat_levels,
    build_universe_mapping,
)
from riskquant.market_data.compile import assemble_valuation_state
from riskquant.market_data.risk_factors.matrix import RiskFactorMatrix
from riskquant.market_data.risk_factors.shift_types import SHIFT_LOG_RETURN_ID
from riskquant.simlab.scenarios.parametric_data import (
    FORECAST_PERIOD_UNIT_BUSINESS_DAY_ID,
    ParametricRiskData,
)
from riskquant.solutions.risk_run.schemas import (
    MethodologyDeclaration,
    ReportingSpec,
    TaylorBlock,
    ValuationContext,
)
from riskquant.solutions.run_identity.hashing import hash_parametric_risk_data
from riskquant.solutions.scenario_models.api import parse_scenario_model_config
from riskquant.utils.business_days import ord_to_date
from riskquant.valuation.engine import run_valuation

FX = mf.FX
NONE = FX.DiscountingMode.NONE

# Dimensions: the release gate (scripts/release/simlab) may resize the world.
K_TOTAL = int(os.environ.get("RQ_CERT_FACTORS", 250))
T = int(os.environ.get("RQ_CERT_OBSERVATIONS", 600))
S = int(os.environ.get("RQ_CERT_SCENARIOS", 2000))
M = int(os.environ.get("RQ_CERT_POSITIONS", 200))

ALL_IDS = np.arange(1, K_TOTAL + 1, dtype=np.int32)  # 250 = USDEUR
NAMES = [f"EQ{i:04d}" for i in range(1, K_TOTAL)] + ["USDEUR"]
AS_OF = int(FX.date_to_ord32(FX._SNAPSHOT_AS_OF_DATE))


def _static_config(first_equity_shift: str = "LOG_RETURN"):
    """The certification static configuration.

    `first_equity_shift` changes ONE load-bearing field (the shift type of
    the first equity node) for mutation tests; everything else is identical.
    """
    sheets = dict(FX._RISK_STATIC_CONFIG_DICT_OF_DFS)
    key = next(k for k, v in sheets.items() if v is FX._NODE_META_DF)
    eq, n = K_TOTAL - 1, K_TOTAL

    sheets[key] = pl.DataFrame({
        "node_id": ALL_IDS.tolist(),
        "node_name": NAMES,
        "quote_ccy": ["USD"] * eq + ["EUR"],
        "asset_class": ["EQUITY"] * eq + ["FX"],
        "asset_subclass": ["EQ_SINGLE_NAME"] * eq + ["FX_G10"],
        "quote_type": ["PRICE"] * eq + ["FX_RATE"],
        "unit_scale": ["DECIMAL"] * n,
        "shift_type": [first_equity_shift] + ["LOG_RETURN"] * (n - 1),
        "fx_base_ccy": [None] * eq + ["USD"],
    }).cast(FX._NODE_META_DF.schema)

    return FX.compile_risk_static_config_from_dict_of_dfs(sheets)


def _vals():
    r = np.random.default_rng(0)

    v = 100.0 * np.exp(
        np.cumsum(
            r.normal(0, 0.01, (T, K_TOTAL)),
            axis=0,
        )
    )

    v[:, K_TOTAL - 1] = 0.9 * np.exp(
        np.cumsum(
            r.normal(0, 0.005, T),
        )
    )

    return np.asfortranarray(v)


VALS = _vals()
RSC = _static_config()

_r = np.random.default_rng(2)

# Hostile economic UIDs: non-contiguous, and the RAW FRAME rows are in an
# unsorted UID order, so the compiled valuation rows are unsorted too.
_UID_POOL = np.sort(
    _r.choice(
        np.arange(1, 50_000),
        size=M,
        replace=False,
    )
).astype(np.int64)

_ORDER = _r.permutation(M)

POSITIONS = pd.DataFrame({
    "uid": _UID_POOL[_ORDER].tolist(),
    "instrument": ["SPOT"] * M,
    "quantity": _r.normal(0, 100, M).round(2)[_ORDER].tolist(),
    "underlying": [NAMES[i] for i in _r.integers(0, K_TOTAL - 1, M)[_ORDER]],
    "settlement_ccy": [("EUR" if i % 2 else "USD") for i in _ORDER],
})

UIDS = np.sort(_UID_POOL)  # U is ascending; the frame is not

_BATCH = FX.compile_raw_df_into_ingest_batch(
    raw_df=POSITIONS.copy(),
    static_config=RSC,
)

MAPPER = FX.build_uix_mapper(
    [_BATCH],
    as_of_ord32=AS_OF,
)

PORTFOLIO = FX.compile_portfolio_from_ingest_batch(
    ingest_batch=_BATCH,
    uix_mapper=MAPPER,
)

UNIVERSE_MAP = build_universe_mapping(
    static_config=RSC,
    uix_mapper=MAPPER,
    discounting_mode=NONE,
)

FIDS = np.ascontiguousarray(
    UNIVERSE_MAP.required_factor_ids,
    dtype=np.int32,
)  # the run axis

K = int(FIDS.shape[0])

BASE = np.ascontiguousarray(
    VALS[-1][FIDS - 1].astype(np.float64),
)  # ids are 1-based


def matrix():
    return RiskFactorMatrix(
        dates=np.arange(T, dtype=np.int32),
        factor_ids=ALL_IDS,
        values=VALS,
    )


# BLAS-attack isolation. When set, `parametric_data()` LOADS the exact bytes a
# pre-attack process froze with `freeze_parametric_data` instead of rebuilding
# them. Rebuilding is BLAS-sensitive (see `build_parametric_data`), so a child
# whose BLAS thread configuration is under attack would otherwise certify a
# different economic input, and its manifest would correctly say so.
PARAMETRIC_DATA_PATH_ENV = "RQ_CERT_PARAMETRIC_DATA"
_PARAMETRIC_FIELDS = (
    "factor_ids",
    "cholesky_matrix",
    "volatilities",
    "effective_period_count",
    "effective_period_unit",
    "chart_ids",
)


def build_parametric_data():
    """Construct the certification parametric data.

    `a @ a.T` (BLAS syrk/gemm) and `np.linalg.cholesky` (LAPACK potrf) take
    their reduction order from the BLAS thread configuration, so the float64
    bytes of `cholesky_matrix` are fixed only WITHIN one configuration. Same
    seed and same code do not imply same bytes across configurations.
    """
    r = np.random.default_rng(1)
    a = r.normal(0, 1, (K_TOTAL, K_TOTAL))
    # D091: L factors a correlation matrix; the covariance is normalised first
    # (still through BLAS and LAPACK, as the docstring above requires).
    cov = a @ a.T / K_TOTAL + np.eye(K_TOTAL) * 0.05
    inv_sd = 1.0 / np.sqrt(np.diag(cov))
    corr = cov * np.outer(inv_sd, inv_sd)

    return ParametricRiskData(
        factor_ids=ALL_IDS,
        cholesky_matrix=np.ascontiguousarray(np.linalg.cholesky(corr)),
        volatilities=np.ascontiguousarray(np.full(K_TOTAL, 0.02)),
        effective_period_count=1,
        effective_period_unit=FORECAST_PERIOD_UNIT_BUSINESS_DAY_ID,
        chart_ids=np.full(K_TOTAL, SHIFT_LOG_RETURN_ID, dtype=np.int8),
    )


def freeze_parametric_data(path: str) -> str:
    """Build the data ONCE and persist its exact arrays.

    NumPy `.npz`: dtype, shape and bytes preserved. Returns the content digest
    the manifest uses.
    """
    data = build_parametric_data()

    np.savez(
        path,
        **{name: getattr(data, name) for name in _PARAMETRIC_FIELDS},
    )

    return hash_parametric_risk_data(data)


def load_parametric_data(path: str) -> ParametricRiskData:
    """Load frozen data without recasting or changing layout.

    `ParametricRiskData` rejects a wrong dtype, shape or layout, and unexpected
    content fails.
    """
    with np.load(path, allow_pickle=False) as frozen:
        if sorted(frozen.files) != sorted(_PARAMETRIC_FIELDS):
            raise ValueError(
                f"frozen parametric data {path!r} holds "
                f"{sorted(frozen.files)}, expected exactly "
                f"{sorted(_PARAMETRIC_FIELDS)}."
            )

        # A 0-d array holds a scalar field (the D090 period); `[()]` restores the
        # NumPy scalar with its stored dtype, no recasting.
        return ParametricRiskData(
            **{
                name: frozen[name][()] if frozen[name].ndim == 0 else frozen[name]
                for name in _PARAMETRIC_FIELDS
            }
        )


def parametric_data():
    """The frozen data when `PARAMETRIC_DATA_PATH_ENV` is set, else a build."""
    path = os.environ.get(PARAMETRIC_DATA_PATH_ENV)

    return build_parametric_data() if path is None else load_parametric_data(path)


def v0(base=None):
    """Producer-side base values of U at a base (default: the run base)."""
    return _price(BASE if base is None else np.asarray(base, dtype=np.float64))[1]


def _price(base):
    market = build_market_data_matrix_from_flat_levels(
        as_of_date=ord_to_date(AS_OF),
        uix_mapper=MAPPER,
        universe_map=UNIVERSE_MAP,
        flat_levels=np.ascontiguousarray(base[:, None]),
        num_scenarios=1,
        discounting_mode=NONE,
    )

    rows = np.empty(
        (PORTFOLIO.total_trades, 1),
        dtype=np.float64,
    )

    run_valuation(
        portfolio=PORTFOLIO,
        out=rows,
        valuation_state=assemble_valuation_state(
            matrix=market,
            uix_mapper=MAPPER,
        ),
    )

    pv = dict(
        zip(
            PORTFOLIO.trade_uids.tolist(),
            rows[:, 0].tolist(),
            strict=True,
        )
    )

    return market, np.array(
        [pv[int(u)] for u in UIDS],
        dtype=np.float64,
    )


def derived(calc):
    return TaylorBlock(calculator=calc)


def _subset(sens, uids):
    """The sensitivities of a UID subset, by identity."""
    keep = np.isin(
        sens.position_uids,
        np.asarray(uids, np.int64),
    )

    kx = np.isin(
        sens.cross_position_uids,
        np.asarray(uids, np.int64),
    )

    return dataclasses.replace(
        sens,
        position_uids=sens.position_uids[keep],
        factor_rf_ids=sens.factor_rf_ids[keep],
        first_order=sens.first_order[keep],
        second_order=sens.second_order[keep],
        cross_position_uids=sens.cross_position_uids[kx],
        cross_factor_rf_ids_i=sens.cross_factor_rf_ids_i[kx],
        cross_factor_rf_ids_j=sens.cross_factor_rf_ids_j[kx],
        cross_second_order=sens.cross_second_order[kx],
    )


def partition(mixed: bool):
    """U_F / U_D over U (ascending). `mixed` splits it in two halves."""
    if not mixed:
        return np.zeros(0, np.int64), UIDS

    return (
        UIDS[::2].copy(),
        UIDS[1::2].copy(),
    )


def calculator(base=None, variant=0, uids=None):
    """Producer side: certify the DERIVED book at a base.

    `variant=1` certifies the same positions as SECOND_ORDER_TRUNCATED
    (cross terms omitted by declaration): a different compiled book, same
    base.
    """
    base = BASE if base is None else np.asarray(base, dtype=np.float64)

    market, values = _price(base)

    sens = FX.run_sensitivities(
        portfolio=PORTFOLIO,
        market_data=market,
        uix_mapper=MAPPER,
    )

    book_uids = UIDS if uids is None else np.asarray(uids, np.int64)

    if uids is not None:
        sens = _subset(
            sens,
            book_uids,
        )
        values = values[
            np.isin(
                UIDS,
                book_uids,
            )
        ]

    if variant == 1:
        sens = dataclasses.replace(
            sens,
            **{
                f.name: getattr(sens, f.name)[:0]
                for f in dataclasses.fields(sens)
                if f.name.startswith("cross_")
                and isinstance(
                    getattr(sens, f.name),
                    np.ndarray,
                )
            },
        )

    binding = FX.TaylorBaseBinding(
        as_of_ord32=AS_OF,
        base_levels_digest=FX.base_levels_digest(
            factor_ids=FIDS,
            base_levels=base,
        ),
        discounting_mode=NONE.value,
        static_config_digest=FX.static_config_content_digest(RSC),
        valuation_contract_fingerprint=(FX.valuation_contract_fingerprint()),
        base_settlement_values_digest=(
            FX.base_settlement_values_digest(
                position_uids=book_uids,
                values=values,
            )
        ),
    )

    spec = FX.build_sensitivity_book_spec(
        sensitivities=sens,
        required_factor_ids=FIDS,
    )

    declaration = FX.derive_taylor_declaration(
        sensitivities=sens,
        base_binding=binding,
        order=(
            FX.TaylorOrder.SECOND_ORDER_TRUNCATED
            if variant == 1
            else FX.TaylorOrder.SECOND_ORDER_FULL
        ),
    )

    return FX.SensitivityPnlCalculator.compile(
        dataclasses.replace(
            spec,
            taylor_declaration=declaration,
        )
    )


CFGS = {
    "historical": {
        "model": "historical",
        "horizon": 1,
    },
    "sbb": {
        "model": "sbb",
        "horizon": 1,
        "total_scenarios": S,
        "expected_block_size": 20.0,
        "seed": 7,
    },
    "filtered_bootstrap": {
        "model": "filtered_bootstrap",
        "horizon": 1,
        "total_scenarios": S,
        "seed": 7,
    },
    "filtered_historical": {
        "model": "filtered_historical",
        "horizon": 1,
    },
    "parametric": {
        "model": "parametric",
        "forecast_horizon": {"count": 1, "unit": "BUSINESS_DAY"},
        "total_scenarios": S,
        "distribution": {
            "kind": "student_t",
            "df": 5.0,
        },
        "seed": 7,
    },
}


def inputs(
    model="sbb",
    batch_size=256,
    seed=7,
    calc=None,
    mixed=False,
    total_scenarios=None,
):
    """Build the canonical production-path input bundle.

    `S` remains the canonical fixture scenario count. `total_scenarios`
    provides an explicit per-invocation override for release stress runs,
    without changing the canonical fixture dimensions.

    Only scenario models that already declare `total_scenarios` may override
    it. Historical models retain their data-defined scenario population.
    """
    cfg = dict(CFGS[model])

    u_full, u_derived = partition(mixed)

    if "seed" in cfg:
        cfg["seed"] = seed

    if total_scenarios is not None:
        if isinstance(total_scenarios, bool) or not isinstance(
            total_scenarios,
            (int, np.integer),
        ):
            raise TypeError("total_scenarios must be an integer")

        total_scenarios = int(total_scenarios)

        if total_scenarios <= 0:
            raise ValueError("total_scenarios must be > 0")

        if "total_scenarios" not in cfg:
            raise ValueError(f"{model!r} does not support an explicit scenario count")

        cfg["total_scenarios"] = total_scenarios

    return dict(
        scenario_config=parse_scenario_model_config(cfg),
        data=(parametric_data() if model == "parametric" else matrix()),
        required_factor_ids=FIDS,
        required_shift_type_id=np.ascontiguousarray(
            UNIVERSE_MAP.required_shift_type_ids,
            np.int8,
        ),
        source_factor_ids=FIDS.copy(),
        shock_scale=np.ones(
            K,
            np.float32,
        ),
        base_levels=BASE,
        methodology=MethodologyDeclaration(
            production_position_uids=UIDS,
            full_reval_uids=u_full,
            taylor_supplied_uids=[],
            taylor_derived_uids=u_derived,
        ),
        valuation_context=ValuationContext(
            static_config=RSC,
            ingest_batch=_BATCH,
            uix_mapper=MAPPER,
            discounting_mode=NONE,
        ),
        taylor_derived=TaylorBlock(
            calculator=(calculator(uids=u_derived) if calc is None else calc)
        ),
        reporting=ReportingSpec(reporting_ccy="USD"),
        confidence_level=0.99,
        batch_size=batch_size,
    )
