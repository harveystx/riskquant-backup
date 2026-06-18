# scripts/demo_rq_ingest_pipeline.py

"""
RiskQuant ingest pipeline — end-to-end demo.

Runs every bundled ingest fixture through the full edge pipeline and reports,
stage by stage, exactly how far each one gets:

    canonicalize -> materialize -> validate/dispatch -> compile -> pack
                 -> portfolio-compile (CompiledPortfolio)

Each stage is isolated in its own guard so a failure is reported with its
diagnostic rather than aborting the run. This is a status probe, not a test
suite: it shows the current state of the subsystem honestly, failures included.

Run
---
    python -m scripts.demo_ingest_pipeline       # (riskquant must be importable)

Notes
-----
- The risk-factor universe is mocked here. In production the two lookup tables
  come from static config; the demo hard-codes a universe covering exactly the
  names used by the fixtures.
- The `vanilla_forward_node` fixture ships settlement dates as full ISO-8601
  datetimes ("2026-12-01T00:00:00.000"). The date materializer expects strict
  YYYY-MM-DD, so the raw fixture is *expected* to fail at stage 2. A
  date-normalized variant is also run to show the pipeline clearing once the
  fixture data is clean.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# Fixtures (bundled ingest cases).
from riskquant.ingest_fixtures.credit.jtd.sf_copula_jtd.minimal_sf_copula_gaussian import (
    CANONICAL_GAUSSIAN_COPULA_JTD_INGEST_DATA,
)
from riskquant.ingest_fixtures.credit.jtd.sf_copula_jtd.minimal_sf_copula_students_t_global_chi import (
    CANONICAL_T_COPULA_JTD_INGEST_DATA,
)
from riskquant.ingest_fixtures.portfolio.exhaustive import (
    CANONICAL_EXHAUSTIVE_MARKET_RISK_PORTFOLIO_DATA,
)
from riskquant.ingest_fixtures.portfolio.spot import (
    CANONICAL_SPOT_MARKET_RISK_PORTFOLIO_DATA,
)
from riskquant.ingest_fixtures.portfolio.vanilla_forward_node import (
    CANONICAL_VANILLA_FORWARD_NODE_MARKET_RISK_PORTFOLIO_DATA,
)
from riskquant.portfolio.ingest.batch import IngestBatch
from riskquant.portfolio.ingest.canonicalize import canonicalize_raw_ingest_df
from riskquant.portfolio.ingest.compile import compile_ingest_soa_payload_from_df
from riskquant.portfolio.ingest.dispatcher import dispatch_ingest_validation
from riskquant.portfolio.ingest.materialize import (
    materialize_canonicalized_ingest_df,
)
from riskquant.portfolio.ingest.schema import DATE_COLS
from riskquant.utils.printer import print_calling_file, print_success_message

np.set_printoptions(linewidth=118, threshold=60)


# ==============================================================================
# MOCK RISK-FACTOR UNIVERSE
# ==============================================================================


def build_mock_universe() -> tuple[dict[str, int], dict[int, str]]:
    """Hard-coded universe covering every name used by the bundled fixtures."""
    rf_name_to_rf_id: dict[str, int] = {
        "TEST_BRENT_CRUDE_202701": 0,  # price factor (underlying)
        "TEST_USD_OIS": 1,  # discount curve
        "TEST_EUR_ESTR": 2,  # discount curve
        "USDEUR": 3,  # FX spot pair
        "EURUSD": 4,  # FX spot pair
        "GLOBAL_CREDIT": 5,  # systemic credit factor
        "IDIO_1": 6,
        "IDIO_2": 7,
        "IDIO_3": 8,
        "CHI_GLOBAL": 9,  # chi-squared mixing factor (t-copula)
    }
    rf_id_to_ccy: dict[int, str] = {
        0: "USD",  # Brent is quoted in USD
        1: "USD",
        2: "EUR",
        3: "EUR",
        4: "USD",
        5: "USD",
        6: "USD",
        7: "USD",
        8: "USD",
        9: "USD",
    }
    return rf_name_to_rf_id, rf_id_to_ccy


def normalize_demo_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Demo data hygiene: truncate ISO-8601 datetime strings to YYYY-MM-DD.

    Vectorized (`.str.slice`) — no `.apply`, no row loops.
    """
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = df[col].astype("string").str.slice(0, 10)
    return df


# ==============================================================================
# PIPELINE RUNNER
# ==============================================================================

_PIPELINE_STAGES: tuple[str, ...] = (
    "canonicalize",
    "materialize",
    "validate/dispatch",
    "compile",
    "pack",
    "portfolio-compile",
)


@dataclass(slots=True)
class CaseResult:
    name: str
    furthest_stage: str
    error: str | None = None
    batch: IngestBatch | None = None
    portfolio: Any | None = None


def _stage(label: str, fn: Callable[[], Any]) -> tuple[Any, str | None]:
    """Run one pipeline stage; return (result, error_string_or_None)."""
    try:
        result = fn()
    except Exception as exc:  # noqa: BLE001 - demo probe wants every failure
        print(f"   [FAIL] {label}  --  {type(exc).__name__}: {exc}")
        return None, f"{type(exc).__name__}: {exc}"
    print(f"   [ OK ] {label}")
    return result, None


def _compile_portfolio(batch: IngestBatch) -> Any:
    """Imported lazily so stages 1-5 stay runnable even without numba/scipy."""
    from riskquant.portfolio.engine import compile_portfolio_from_ingest_batch

    return compile_portfolio_from_ingest_batch(batch)


def run_pipeline(
    name: str,
    raw_df: pd.DataFrame,
    universe: tuple[dict[str, int], dict[int, str]],
) -> CaseResult:
    rf_name_to_rf_id, rf_id_to_ccy = universe
    df = raw_df.copy()

    print(f"\n{'-' * 78}")
    print(f"CASE: {name}   ({len(df)} rows, {len(df.columns)} raw columns)")
    print(f"{'-' * 78}")

    _, err = _stage(
        "stage 1  canonicalize raw frame", lambda: canonicalize_raw_ingest_df(df)
    )
    if err:
        return CaseResult(name, "canonicalize", err)

    _, err = _stage(
        "stage 2  materialize dtypes", lambda: materialize_canonicalized_ingest_df(df)
    )
    if err:
        return CaseResult(name, "materialize", err)

    instrument_ids, err = _stage(
        "stage 3  dispatch validation", lambda: dispatch_ingest_validation(df)
    )
    if err:
        return CaseResult(name, "validate/dispatch", err)

    compiled, err = _stage(
        "stage 4  compile DataFrame -> SoA",
        lambda: compile_ingest_soa_payload_from_df(
            df=df,
            instrument_ids=instrument_ids,
            rf_name_to_rf_id=rf_name_to_rf_id,
            rf_id_to_ccy=rf_id_to_ccy,
        ),
    )
    if err:
        return CaseResult(name, "compile", err)

    batch, err = _stage(
        "stage 5  pack -> IngestBatch",
        lambda: IngestBatch.from_compiled_ingest_soa(compiled),
    )
    if err:
        return CaseResult(name, "pack", err)

    _print_batch(batch)

    portfolio, err = _stage(
        "stage 6  compile -> CompiledPortfolio", lambda: _compile_portfolio(batch)
    )
    if err:
        return CaseResult(name, "portfolio-compile", err, batch=batch)

    _print_portfolio(portfolio)
    return CaseResult(name, "DONE", None, batch=batch, portfolio=portfolio)


# ==============================================================================
# REPORTING
# ==============================================================================


def _print_batch(batch: IngestBatch) -> None:
    """Show a slice of the SoA payload — proof that the strings are gone."""
    probe = (
        ("instrument_ids", batch.instrument_ids),
        ("uids", batch.uids),
        ("qty", batch.qty),
        ("strike_price", batch.strike_price),
        ("cp_mult", batch.cp_mult),
        ("price_rf_ids", batch.price_rf_ids),
        ("fx_under_to_settle_ids", batch.fx_under_to_settle_ids),
        ("fx_under_to_settle_invert", batch.fx_under_to_settle_invert),
        ("settle_date_uix", batch.settle_date_uix),
        ("under_discount_curve_ids", batch.under_discount_curve_ids),
        ("systemic_uix", batch.systemic_uix),
        ("beta", batch.beta),
    )
    print(
        f"\n   IngestBatch  (size={batch.size}, unique_dates={len(batch.unique_dates_ord)})"
    )
    print("   SoA arrays  --  every dtype below is numeric; no strings survive:")
    for field_name, arr in probe:
        values = np.asarray(arr)
        print(f"     {field_name:<26} {str(values.dtype):<9} {values}")


def _print_portfolio(portfolio: Any) -> None:
    print()
    portfolio.summary()


def print_run_summary(results: list[CaseResult]) -> None:
    print("\n" + "=" * 78)
    print("PIPELINE STATUS SUMMARY".center(78))
    print("=" * 78)
    print(f"{'case':<44}{'furthest stage':<22}{'status'}")
    print("-" * 78)
    for r in results:
        status = "PASS" if r.error is None else "FAIL"
        print(f"{r.name:<44}{r.furthest_stage:<22}{status}")
        if r.error is not None:
            print(f"{'':<44}reason: {r.error}")
    print("=" * 78)


# ==============================================================================
# MAIN
# ==============================================================================


def main() -> None:
    print_calling_file()

    universe = build_mock_universe()
    rf_name_to_rf_id, _ = universe

    print("=" * 78)
    print("RISKQUANT INGEST PIPELINE DEMO".center(78))
    print("=" * 78)
    print("\nMock risk-factor universe (name -> uix):")
    for name, uix in rf_name_to_rf_id.items():
        print(f"   {name:<28} -> {uix}")

    # The forward fixture intentionally ships ISO datetime strings; the raw
    # variant is expected to fail at materialize, the normalized variant clears.
    forward_raw = pd.DataFrame(
        CANONICAL_VANILLA_FORWARD_NODE_MARKET_RISK_PORTFOLIO_DATA
    )
    exhaustive_clean = normalize_demo_dates(
        pd.DataFrame(CANONICAL_EXHAUSTIVE_MARKET_RISK_PORTFOLIO_DATA)
    )
    forward_clean = normalize_demo_dates(
        pd.DataFrame(CANONICAL_VANILLA_FORWARD_NODE_MARKET_RISK_PORTFOLIO_DATA)
    )

    cases: list[tuple[str, pd.DataFrame]] = [
        ("spot", pd.DataFrame(CANONICAL_SPOT_MARKET_RISK_PORTFOLIO_DATA)),
        ("vanilla_forward_node  [raw fixture]", forward_raw),
        ("vanilla_forward_node  [date-normalized]", forward_clean),
        ("market_risk_exhaustive  [date-normalized]", exhaustive_clean),
        (
            "gaussian_copula_jtd",
            pd.DataFrame(CANONICAL_GAUSSIAN_COPULA_JTD_INGEST_DATA),
        ),
        ("t_copula_jtd", pd.DataFrame(CANONICAL_T_COPULA_JTD_INGEST_DATA)),
    ]

    results = [run_pipeline(name, df, universe) for name, df in cases]
    print_run_summary(results)

    print_success_message()


if __name__ == "__main__":
    main()
