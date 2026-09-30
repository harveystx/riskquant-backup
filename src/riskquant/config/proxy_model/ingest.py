# src/riskquant/config/proxy_model/ingest.py
"""
Tabular authoring edge for Proxy V1 terms (D088 §12.3; D093, PROPOSED).

One frame row is one authored term. The columns are the compiler's authored
input contract and nothing else:

    term_kind     Utf8     "NODE" | "CONTAINER"
    target_name   Utf8     target factor-node name (NODE) or container name
    driver_name   Utf8     driver factor-node name (NODE) or container name
    driver_chart  Utf8     "LOG_RETURN" | "SIMPLE_RETURN" | "DIFFERENCE"  (tau)
    loading       Float64  signed finite binary64 coefficient (beta)

No target chart (kappa is owned by the static configuration), no horizon (the
world owns h) and no identity marker (a V1 target without terms gets its
identity row at compile time) is authored.

This edge checks only the frame's structure: the exact column set, the
dtypes, no nulls, no blank names, the two vocabularies and finite loadings.
It maps each row to `NodeProxyTerm` / `ContainerProxyTerm` in row order and
performs no resolution, merging, dropping or sorting: name resolution at the
as-of, admissibility, canonicalisation and identity stay with
`compile_proxy_model`. Every rejection names the row and the column.
"""

from __future__ import annotations

import math
from typing import Final

import polars as pl

from riskquant.config.proxy_model.schema_objects import (
    AuthoredProxyTerm,
    ContainerProxyTerm,
    NodeProxyTerm,
)
from riskquant.exceptions import ConfigurationError
from riskquant.market_data.risk_factors.shift_types import ShiftType

__all__ = ["PROXY_MODEL_TERM_SCHEMA", "proxy_terms_from_df"]

#: The exact frame schema of the authoring edge.
PROXY_MODEL_TERM_SCHEMA: Final[dict[str, type[pl.DataType]]] = {
    "term_kind": pl.Utf8,
    "target_name": pl.Utf8,
    "driver_name": pl.Utf8,
    "driver_chart": pl.Utf8,
    "loading": pl.Float64,
}
_TERM_KINDS: Final[frozenset[str]] = frozenset({"NODE", "CONTAINER"})
_DRIVER_CHARTS: Final[dict[str, ShiftType]] = {
    "LOG_RETURN": ShiftType.LOG_RETURN,
    "SIMPLE_RETURN": ShiftType.SIMPLE_RETURN,
    "DIFFERENCE": ShiftType.DIFFERENCE,
}


def proxy_terms_from_df(df: pl.DataFrame) -> tuple[AuthoredProxyTerm, ...]:
    """
    Map an authored-term frame to Proxy V1 term records, in row order.

    Raises
    ------
    ConfigurationError
        Wrong column set or dtype, a null, a blank name, an unknown term kind
        or driver chart, or a non-finite loading. The message names the row
        and the column.
    """
    if not isinstance(df, pl.DataFrame):
        raise ConfigurationError(f"expected a polars DataFrame, got {type(df).__name__}.")
    if set(df.columns) != set(PROXY_MODEL_TERM_SCHEMA):
        raise ConfigurationError(
            f"proxy-model term frame columns must be exactly "
            f"{sorted(PROXY_MODEL_TERM_SCHEMA)}; got {sorted(df.columns)}."
        )
    for col, dtype in PROXY_MODEL_TERM_SCHEMA.items():
        if df.schema[col] != dtype:
            raise ConfigurationError(
                f"column {col!r} must have dtype {dtype}; got {df.schema[col]}."
            )
        nulls = df.get_column(col).is_null()
        if nulls.any():
            raise ConfigurationError(
                f"row {int(nulls.arg_true()[0])}, column {col!r}: null value."
            )

    terms: list[AuthoredProxyTerm] = []
    for i, row in enumerate(df.select(list(PROXY_MODEL_TERM_SCHEMA)).iter_rows()):
        kind, target, driver, chart, loading = row
        for col, value in (("target_name", target), ("driver_name", driver)):
            if not value.strip():
                raise ConfigurationError(f"row {i}, column {col!r}: blank name.")
        if kind not in _TERM_KINDS:
            raise ConfigurationError(
                f"row {i}, column 'term_kind': {kind!r} is not one of {sorted(_TERM_KINDS)}."
            )
        if chart not in _DRIVER_CHARTS:
            raise ConfigurationError(
                f"row {i}, column 'driver_chart': {chart!r} is not one of "
                f"{sorted(_DRIVER_CHARTS)}."
            )
        if not math.isfinite(loading):
            raise ConfigurationError(f"row {i}, column 'loading': {loading!r} is not finite.")
        if kind == "NODE":
            terms.append(NodeProxyTerm(
                target_name=target, driver_name=driver,
                driver_chart=_DRIVER_CHARTS[chart], loading=float(loading),
            ))
        else:
            terms.append(ContainerProxyTerm(
                target_container_name=target, driver_container_name=driver,
                driver_chart=_DRIVER_CHARTS[chart], loading=float(loading),
            ))
    return tuple(terms)
