# src/riskquant/sensitivities/book_adapter.py
"""
Edge adapter: `BlockSensitivities` -> `SensitivityBookSpec`.

This is the boundary between the numerical sensitivity core and the validated
edge contract consumed by `solutions.sensitivity_pnl`. It is the ONLY module
in `riskquant.sensitivities` that imports Polars or touches the
Pydantic-validated spec: kernels, sensitizers, and the engine stay pure
NumPy SoA.

`SensitivityBookSpec` is a validated EDGE representation, not the natural
output of numerical work -- so the flow is deliberately

    kernel/block layer  ->  primitive contiguous arrays (BlockSensitivities)
                        ->  orchestration
                        ->  validated book specification (here)

rather than having kernels emit DataFrames.

Factor-axis contract
--------------------
The axis is built from the sorted-unique global `rf_id`s actually referenced
by the computed sensitivities, as an explicit Int32 Polars series. Dtype must
match the `factor_id` columns exactly: the downstream validator rejects an
Int32/Int64 mismatch between the axis and the sensitivity frames (it would
otherwise risk silently merging or failing to resolve factors). Because the
ids are sorted and unique, the axis takes the zero-copy DIRECT encoding, so

    factor_axis position -> rf_id

is the identity on the axis array, with no slot indirection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from riskquant.solutions.sensitivity_pnl.api import (
    DisplacementConvention,
    FactorAxis,
    SensitivityBookSpec,
)

if TYPE_CHECKING:
    from riskquant.sensitivities.result import BlockSensitivities


def build_sensitivity_book_spec(
    *,
    sensitivities: BlockSensitivities,
    displacement_convention: DisplacementConvention = DisplacementConvention.ABSOLUTE,
    quote_base: float = 1.0,
) -> SensitivityBookSpec:
    """
    Convert computed block sensitivities into the canonical, validated book.

    Parameters
    ----------
    sensitivities
        Portfolio-level SoA sensitivities from `sensitivities.engine`.
    displacement_convention
        Shock convention declared for every active factor. ABSOLUTE means a
        shock is an additive displacement in the factor's own quoted units,
        which is what the closed-form partials above are expressed against.
    quote_base
        Quotation scale folded into the coefficients by the book compiler
        (order-n scales by base**n). 1.0 leaves coefficients as computed.

    Returns
    -------
    SensitivityBookSpec
        Ready for `SensitivityPnlCalculator.compile`.
    """
    diag = pl.DataFrame(
        {
            "position_id": pl.Series(
                sensitivities.position_uids.tolist(), dtype=pl.Int64
            ),
            "factor_id": pl.Series(
                sensitivities.factor_rf_ids.tolist(), dtype=pl.Int32
            ),
            "first_order": pl.Series(
                sensitivities.first_order.tolist(), dtype=pl.Float64
            ),
            "second_order": pl.Series(
                sensitivities.second_order.tolist(), dtype=pl.Float64
            ),
        }
    )

    cross: pl.DataFrame | None = None
    if sensitivities.num_cross_entries:
        cross = pl.DataFrame(
            {
                "position_id": pl.Series(
                    sensitivities.cross_position_uids.tolist(), dtype=pl.Int64
                ),
                "factor_id_i": pl.Series(
                    sensitivities.cross_factor_rf_ids_i.tolist(), dtype=pl.Int32
                ),
                "factor_id_j": pl.Series(
                    sensitivities.cross_factor_rf_ids_j.tolist(), dtype=pl.Int32
                ),
                "cross_second_order": pl.Series(
                    sensitivities.cross_second_order.tolist(), dtype=pl.Float64
                ),
            }
        )

    active_rf_ids = sensitivities.active_factor_rf_ids
    factor_meta = pl.DataFrame(
        {
            "factor_id": pl.Series(active_rf_ids.tolist(), dtype=pl.Int32),
            "displacement_convention_id": pl.Series(
                [int(displacement_convention)] * active_rf_ids.shape[0],
                dtype=pl.Int8,
            ),
            "quote_base": pl.Series(
                [float(quote_base)] * active_rf_ids.shape[0], dtype=pl.Float64
            ),
        }
    )

    axis = FactorAxis.from_keys(
        pl.Series("factor_id", active_rf_ids.tolist(), dtype=pl.Int32)
    )

    return SensitivityBookSpec(
        sensitivities=diag,
        factor_meta=factor_meta,
        factor_axis=axis,
        position_uids=np.unique(sensitivities.position_uids).tolist(),
        cross_sensitivities=cross,
    )
