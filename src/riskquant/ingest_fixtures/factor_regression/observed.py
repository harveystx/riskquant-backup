# src/riskquant/ingest_fixtures/factor_regression/observed.py
"""
Edge adapter: fixture long frame to facade ``ObservedSeries``.

This is the only place where the fixture crosses into
``riskquant.solutions.factor_regression``. Strings stay in the frame and in
the facade contracts; the arrays handed over are ``int32`` date ordinals and
``float64`` levels, exactly the dtypes ``ObservedSeries`` demands. No cast is
needed: ``pl.Date`` is an ``int32`` day count from 1970-01-01, the riskquant
date-ordinal epoch.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from riskquant.ingest_fixtures.factor_regression.case import (
    FACTOR_REGRESSION_SERIES_SCHEMA,
    FactorRegressionCase,
)
from riskquant.solutions.factor_regression.api import (
    AlignedLevels,
    ObservedSeries,
    align_levels,
)

__all__ = [
    "aligned_levels_for_case",
    "observed_series_for_case",
    "observed_series_from_frame",
]


def observed_series_from_frame(
    frame: pl.DataFrame, *, series_names: Sequence[str]
) -> tuple[ObservedSeries, ...]:
    """
    Build one ``ObservedSeries`` per name, in the order of ``series_names``.

    Raises
    ------
    ValueError
        ``frame`` does not have ``FACTOR_REGRESSION_SERIES_SCHEMA``, a name is
        absent, or one series carries more than one unit.
    """
    if frame.schema != FACTOR_REGRESSION_SERIES_SCHEMA:
        raise ValueError(f"frame schema {frame.schema} is not canonical")
    wanted = list(series_names)
    if len(set(wanted)) != len(wanted):
        raise ValueError(f"repeated name in series_names {wanted}")
    selected = (
        frame
        .filter(pl.col("series_name").is_in(wanted))
        .select(
            "series_name",
            "unit",
            pl.col("date").cast(pl.Int32).alias("date_ord"),
            "value",
        )
        .sort("series_name", "date_ord")
    )
    parts = selected.partition_by("series_name", as_dict=True, maintain_order=True)
    by_name = {str(key[0]): part for key, part in parts.items()}
    out: list[ObservedSeries] = []
    for name in wanted:
        part = by_name.get(name)
        if part is None:
            raise ValueError(f"series {name!r} is not in the frame")
        units = part.get_column("unit").unique()
        if units.len() != 1:
            raise ValueError(f"series {name!r} carries units {units.to_list()}")
        out.append(
            ObservedSeries(
                name=name,
                unit=str(units.item()),
                date_ord=np.ascontiguousarray(
                    part.get_column("date_ord").to_numpy(), dtype=np.int32
                ),
                values=np.ascontiguousarray(
                    part.get_column("value").to_numpy(), dtype=np.float64
                ),
            )
        )
    return tuple(out)


def observed_series_for_case(
    case: FactorRegressionCase,
) -> tuple[ObservedSeries, ...]:
    """Every series of ``case``, in ``case.series_names`` order."""
    return observed_series_from_frame(case.frame, series_names=case.series_names)


def aligned_levels_for_case(case: FactorRegressionCase) -> AlignedLevels:
    """``align_levels`` over every series of ``case`` on the case window."""
    return align_levels(
        observed_series_for_case(case),
        start_ord=case.start_ord,
        end_ord=case.end_ord,
    )
