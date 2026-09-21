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

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import polars as pl

from riskquant.solutions.sensitivity_pnl.api import (
    DisplacementConvention,
    FactorAxis,
    SensitivityBookSpec,
)
from riskquant.solutions.sensitivity_pnl.taylor_contract import (
    TaylorBaseBinding,
    TaylorDeclaration,
    TaylorOrder,
)

if TYPE_CHECKING:
    from riskquant.sensitivities.result import BlockSensitivities


def build_sensitivity_book_spec(
    *,
    sensitivities: BlockSensitivities,
    displacement_convention: DisplacementConvention = DisplacementConvention.ABSOLUTE,
    quote_base: float = 1.0,
    base_levels_by_factor: Mapping[int, float] | None = None,
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
    # ---- Coordinate constitution (D081 §8, P1-A) ---------------------------
    # Every registered sensitizer differentiates with respect to the LEVEL
    # (ABSOLUTE is the native coordinate). Declaring another convention is a
    # coordinate CHANGE, not a label: it is performed exactly, at the bound base
    # level x0 of each factor, or refused.
    #   RELATIVE  y = x/x0 - 1 :  V_y = V_x x0,   V_yy = V_xx x0^2
    #   LOG       y = ln(x/x0) :  V_y = V_x x0,   V_yy = V_xx x0^2 + V_x x0
    #   cross (separable maps) :  V_{y_i y_j} = V_{x_i x_j} x0_i x0_j
    d1 = np.asarray(sensitivities.first_order, dtype=np.float64)
    d2 = np.asarray(sensitivities.second_order, dtype=np.float64)
    cx = np.asarray(sensitivities.cross_second_order, dtype=np.float64)
    convention = DisplacementConvention(displacement_convention)
    if convention is not DisplacementConvention.ABSOLUTE:
        if base_levels_by_factor is None:
            raise ValueError(
                f"displacement_convention={convention.name} requires the bound "
                "base levels (base_levels_by_factor): sensitizers emit ABSOLUTE "
                "derivatives and relabelling them is mathematically invalid."
            )
        rf = np.asarray(sensitivities.factor_rf_ids, dtype=np.int64)
        x0 = _base_levels_for(rf, base_levels_by_factor, convention)
        d1_native = d1
        d1 = d1_native * x0
        d2 = d2 * x0 * x0
        if convention is DisplacementConvention.LOG:
            d2 = d2 + d1_native * x0  # the mandatory first-derivative term
        if sensitivities.num_cross_entries:
            xi = _base_levels_for(
                np.asarray(sensitivities.cross_factor_rf_ids_i, dtype=np.int64),
                base_levels_by_factor,
                convention,
            )
            xj = _base_levels_for(
                np.asarray(sensitivities.cross_factor_rf_ids_j, dtype=np.int64),
                base_levels_by_factor,
                convention,
            )
            cx = cx * xi * xj
    diag = pl.DataFrame({
        "position_id": pl.Series(sensitivities.position_uids.tolist(), dtype=pl.Int64),
        "factor_id": pl.Series(sensitivities.factor_rf_ids.tolist(), dtype=pl.Int32),
        "first_order": pl.Series(d1.tolist(), dtype=pl.Float64),
        "second_order": pl.Series(d2.tolist(), dtype=pl.Float64),
    })

    cross: pl.DataFrame | None = None
    if sensitivities.num_cross_entries:
        cross = pl.DataFrame({
            "position_id": pl.Series(
                sensitivities.cross_position_uids.tolist(), dtype=pl.Int64
            ),
            "factor_id_i": pl.Series(
                sensitivities.cross_factor_rf_ids_i.tolist(), dtype=pl.Int32
            ),
            "factor_id_j": pl.Series(
                sensitivities.cross_factor_rf_ids_j.tolist(), dtype=pl.Int32
            ),
            "cross_second_order": pl.Series(cx.tolist(), dtype=pl.Float64),
        })

    active_rf_ids = sensitivities.active_factor_rf_ids
    factor_meta = pl.DataFrame({
        "factor_id": pl.Series(active_rf_ids.tolist(), dtype=pl.Int32),
        "displacement_convention_id": pl.Series(
            [int(displacement_convention)] * active_rf_ids.shape[0],
            dtype=pl.Int8,
        ),
        "quote_base": pl.Series(
            [float(quote_base)] * active_rf_ids.shape[0], dtype=pl.Float64
        ),
    })

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


def _base_levels_for(
    rf_ids: npt.NDArray[np.int64],
    base_levels_by_factor: Mapping[int, float],
    convention: DisplacementConvention,
) -> npt.NDArray[np.float64]:
    """Bound base level per row, with the coordinate's domain enforced."""
    missing = sorted({int(r) for r in rf_ids.tolist()} - set(base_levels_by_factor))
    if missing:
        raise ValueError(
            f"no bound base level for factor(s) {missing[:10]}; the coordinate "
            "change is evaluated at the base and cannot be performed."
        )
    x0 = np.array([float(base_levels_by_factor[int(r)]) for r in rf_ids.tolist()])
    if not np.isfinite(x0).all():
        raise ValueError("bound base levels must be finite.")
    if convention is DisplacementConvention.RELATIVE and np.any(x0 == 0.0):
        raise ValueError(
            "RELATIVE coordinates require a non-zero base level (x0 != 0)."
        )
    if convention is DisplacementConvention.LOG and np.any(x0 <= 0.0):
        raise ValueError(
            "LOG coordinates require a strictly positive base level (x0 > 0)."
        )
    return x0


def derive_taylor_declaration(
    *,
    sensitivities: BlockSensitivities,
    order: TaylorOrder,
    base_binding: TaylorBaseBinding,
) -> TaylorDeclaration:
    """
    TAYLOR/DERIVED declaration for a sensitizer's output.

    F(p) is the sensitizer's STRUCTURAL read set: registered sensitizers emit
    one row per factor their block's pricer reads, masked by topology (e.g. an
    FX row exists iff settlement != pricing currency), never by value -- so a
    zero Greek keeps its row and its factor stays in F(p). The declared `order`
    is then proven by `validate_taylor_completeness` at compile.
    """
    deps = pl.DataFrame({
        "position_id": pl.Series(sensitivities.position_uids.tolist(), dtype=pl.Int64),
        "factor_id": pl.Series(sensitivities.factor_rf_ids.tolist(), dtype=pl.Int32),
    }).unique(maintain_order=True)
    return TaylorDeclaration(
        order=order,
        dependencies=deps,
        zero_exposure_position_uids=(),
        base_binding=base_binding,
    )
