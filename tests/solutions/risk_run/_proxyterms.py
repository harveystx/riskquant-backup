# tests/solutions/risk_run/_proxyterms.py
"""
Test-side migration of D013 routing tables into authored Proxy V1 terms.

Some fixtures take their world from the external demo package
(`scripts/demos/e2e/_fixtures.py`), whose `_run_shock_universe` still returns a
D013 routing table: per run-axis row a source factor and a float32 scale. The
production run no longer accepts that table (D088 Stage D). This helper states
the same scenario law as authored Proxy V1 terms, so these fixtures exercise the
production compiler:

    row k with source s != own factor, or scale != 1:
        NodeProxyTerm(target = row k, driver = s,
                      driver_chart = kappa_k, loading = float(scale_k))
    every other row: no authored term (its identity row is compiled).

The D013 path applied the scale inside the target chart on the source's
kappa_k column, so this term reproduces it exactly: float64(Y) * float64(s)
is the same product Slice 2 forms from 0.0 + beta * float64(Y). The driver
chart is written explicitly here; the production path never infers it.

This is not a production compatibility layer: the run receives only the
authored terms. The demo package should author its terms directly
(integration dependency, reported with D088 Stage D).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from riskquant.config.proxy_model.schema_objects import NodeProxyTerm
from riskquant.config.static_config.static_config import RiskStaticConfig
from riskquant.market_data.risk_factors.shift_types import ShiftType


def rf_name(static_config: RiskStaticConfig, rf_id: int) -> str:
    """The relative name of one rf_id in the static configuration."""
    names = {int(v): k for k, v in static_config.rel_name_to_rf_id.items()}
    return names[int(rf_id)]


def authored_terms_from_d013_routing(
    *,
    static_config: RiskStaticConfig,
    required_factor_ids: npt.NDArray[np.int32],
    required_shift_type_id: npt.NDArray[np.int8],
    source_factor_ids: npt.NDArray[np.int32],
    shock_scale: npt.NDArray[np.float32],
) -> tuple[NodeProxyTerm, ...]:
    terms = []
    for k in range(required_factor_ids.shape[0]):
        own, src = int(required_factor_ids[k]), int(source_factor_ids[k])
        scale = float(shock_scale[k])
        if src == own and scale == 1.0:
            continue  # identity row, compiled by default
        terms.append(
            NodeProxyTerm(
                target_name=rf_name(static_config, own),
                driver_name=rf_name(static_config, src),
                driver_chart=ShiftType(int(required_shift_type_id[k])),
                loading=scale,
            )
        )
    return tuple(terms)
