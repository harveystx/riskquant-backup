# src/riskquant/ingest_fixtures/factor_regression/catalog.py
"""
Registry of the canonical factor-regression cases.

==============================  ===================================================
Case                            Library features it exercises
==============================  ===================================================
``fx_converted_cointegration``  LOG-scale declared combination (FX conversion
                                inside the combination), ADF CONSTANT, spurious
                                level regression diagnostics, exact-identity
                                ``DegenerateFitError``, joint Wald on betas.
``proxy_calibration``           Proxy V1 loadings without intercept, short and
                                gappy target history, non-synchronous closes,
                                ``step`` and ``stride``, overlap diagnostic, HAC.
``credit_factor_loadings``      Multi-factor loadings and R**2 for the credit
                                copula, ``OUTSIDE_WINDOW`` exclusions, weekly
                                non-overlapping rows, HC0 under a vol regime.
``collinear_drivers``           Condition-number diagnostic, rank gate on one
                                price in two units, Wald on a sum of loadings.
``rates_spread``                DIFFERENCE transform, LEVEL scale, levels that
                                cross zero (domain errors), CONSTANT versus
                                CONSTANT_TREND.
==============================  ===================================================

Series names are unique across cases, so ``CANONICAL_FACTOR_REGRESSION_SERIES_DF``
can be filtered by name alone.

Import direction (owner ruling, fix round of the demo)
------------------------------------------------------
The fixture builds inputs for the public facade
``riskquant.solutions.factor_regression.api`` and records the expected
outcomes of canonical analyses. ``observed.py`` and ``analyses.py`` import the
facade; the fixture imports no core module (``riskquant.factor_regression``);
no solution or core module imports the fixture.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import polars as pl

from riskquant.ingest_fixtures.factor_regression.case import FactorRegressionCase
from riskquant.ingest_fixtures.factor_regression.collinear_drivers import (
    COLLINEAR_DRIVERS_CASE,
)
from riskquant.ingest_fixtures.factor_regression.credit_factor_loadings import (
    CREDIT_FACTOR_LOADINGS_CASE,
)
from riskquant.ingest_fixtures.factor_regression.fx_converted_cointegration import (
    FX_CONVERTED_COINTEGRATION_CASE,
)
from riskquant.ingest_fixtures.factor_regression.proxy_calibration import (
    PROXY_CALIBRATION_CASE,
)
from riskquant.ingest_fixtures.factor_regression.rates_spread import (
    RATES_SPREAD_CASE,
)

__all__ = [
    "CANONICAL_FACTOR_REGRESSION_CASES",
    "CANONICAL_FACTOR_REGRESSION_SERIES_DF",
]


def _registry() -> Mapping[str, FactorRegressionCase]:
    cases = (
        FX_CONVERTED_COINTEGRATION_CASE,
        PROXY_CALIBRATION_CASE,
        CREDIT_FACTOR_LOADINGS_CASE,
        COLLINEAR_DRIVERS_CASE,
        RATES_SPREAD_CASE,
    )
    by_name: dict[str, FactorRegressionCase] = {}
    seen_series: dict[str, str] = {}
    for case in cases:
        if case.name in by_name:
            raise ValueError(f"repeated case name {case.name!r}")
        for series in case.series_names:
            if series in seen_series:
                raise ValueError(
                    f"series {series!r} appears in {seen_series[series]!r} "
                    f"and {case.name!r}"
                )
            seen_series[series] = case.name
        by_name[case.name] = case
    return MappingProxyType(by_name)


CANONICAL_FACTOR_REGRESSION_CASES: Final[Mapping[str, FactorRegressionCase]] = (
    _registry()
)

CANONICAL_FACTOR_REGRESSION_SERIES_DF: Final[pl.DataFrame] = pl.concat(
    [case.frame for case in CANONICAL_FACTOR_REGRESSION_CASES.values()],
    how="vertical",
)


def _debug_print() -> None:
    from riskquant.utils.printer import print_calling_file, print_success_message

    print_calling_file()
    for case in CANONICAL_FACTOR_REGRESSION_CASES.values():
        print(f"=== {case.name} ===")
        print(case.question)
        print(
            case.frame.group_by("series_name", "unit", maintain_order=True).agg(
                pl.len().alias("n_obs"),
                pl.col("date").min().alias("first"),
                pl.col("date").max().alias("last"),
                pl.col("value").min().alias("min"),
                pl.col("value").max().alias("max"),
            )
        )
        for key, value in case.truth.items():
            print(f"  truth {key} = {value:.6g}")
    print_success_message(add_space_before=False)


if __name__ == "__main__":
    _debug_print()
