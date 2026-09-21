# tests/solutions/risk_run/_binding.py
"""
P1-A certification for the synthetic risk-run books.

These books have no RiskStaticConfig (they are synthetic Taylor books over a
three-factor axis), so their static-config digest is the digest of the EMPTY
configuration -- an explicit statement, not a placeholder. Everything else is
the real contract: F(p) from the book's rows, the declared order, the base
levels and V_0 the run will actually use.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from riskquant.solutions.sensitivity_pnl.taylor_contract import (
    TaylorBaseBinding,
    TaylorDeclaration,
    TaylorOrder,
    base_levels_digest,
    base_settlement_values_digest,
    static_config_content_digest,
    valuation_contract_fingerprint,
)

SYNTHETIC_AS_OF_ORD32 = 20_000


def synthetic_binding(
    *, factor_ids, base_levels, position_uids, v0
) -> TaylorBaseBinding:
    return TaylorBaseBinding(
        as_of_ord32=SYNTHETIC_AS_OF_ORD32,
        base_levels_digest=base_levels_digest(
            factor_ids=np.asarray(factor_ids, dtype=np.int32),
            base_levels=np.asarray(base_levels, dtype=np.float64),
        ),
        discounting_mode="NONE",
        static_config_digest=static_config_content_digest({}),
        valuation_contract_fingerprint=valuation_contract_fingerprint(),
        base_settlement_values_digest=base_settlement_values_digest(
            position_uids=position_uids, values=np.asarray(v0, dtype=np.float64)
        ),
    )


def synthetic_declaration(
    *, sensitivities: pl.DataFrame, order: TaylorOrder, binding
) -> TaylorDeclaration:
    return TaylorDeclaration(
        order=order,
        dependencies=sensitivities.select(
            pl.col("position_id").cast(pl.Int64), pl.col("factor_id").cast(pl.Int32)
        ).unique(maintain_order=True),
        zero_exposure_position_uids=(),
        base_binding=binding,
    )
