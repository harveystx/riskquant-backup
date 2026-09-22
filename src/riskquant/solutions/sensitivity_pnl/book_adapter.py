# src/riskquant/solutions/sensitivity_pnl/book_adapter.py
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

Factor-axis contract (F-1, D082)
--------------------------------
The factor axis is DECLARED by its owner, the production run
(`UniverseMapping.required_factor_ids`, in layout order), and passed in as
`required_factor_ids`. This adapter never invents, sorts, trims or pads it:
the spec carries exactly that axis, in exactly that order, as an explicit Int32
Polars series (the dtype must match the `factor_id` columns; the validator
rejects an Int32/Int64 mismatch).

Alignment of the sensitivities onto the axis is by FACTOR IDENTITY, never by
position. `FactorAxis.from_keys` keeps the given order (sorted-unique ids take
the zero-copy DIRECT encoding, any other order the SLOT encoding), and
`compile_book` relabels every factor key through `FactorAxis.relabel`
(`replace_strict`: an unmapped id raises; it is never dropped or zeroed).

Every factor the sensitivities reference must be on the axis, or this adapter
raises. An axis factor that no position references is an INACTIVE row: no
sensitivity entry and no coefficient is created for it. The Taylor
declaration's F(p), not this adapter, states that no position depends on it,
and P1-A completeness still rejects a declared dependency without its
derivative. The calculator then requires market axis == book axis, row for row.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import polars as pl

from riskquant.sensitivity_pnl.conventions import DisplacementConvention
from riskquant.solutions.sensitivity_pnl.market import FactorAxis
from riskquant.solutions.sensitivity_pnl.schemas import SensitivityBookSpec
from riskquant.solutions.sensitivity_pnl.taylor_contract import (
    TaylorBaseBinding,
    TaylorContractError,
    TaylorDeclaration,
    TaylorOrder,
    base_levels_digest,
)

if TYPE_CHECKING:
    from riskquant.sensitivities.result import BlockSensitivities


def build_sensitivity_book_spec(
    *,
    sensitivities: BlockSensitivities,
    required_factor_ids: npt.NDArray[np.int32],
    displacement_convention: DisplacementConvention = DisplacementConvention.ABSOLUTE,
    quote_base: float = 1.0,
    base_levels_by_factor: Mapping[int, float] | None = None,
    base_binding: TaylorBaseBinding | None = None,
) -> SensitivityBookSpec:
    """
    Convert computed block sensitivities into the canonical, validated book.

    Parameters
    ----------
    sensitivities
        Portfolio-level SoA sensitivities from `sensitivities.engine`.
    required_factor_ids
        The AUTHORITATIVE ordered factor axis the book is priced against: for a
        production run, the run's `UniverseMapping.required_factor_ids`. 1-D
        int32, unique, used exactly in the given order (never sorted). It must
        carry every factor the sensitivities reference; see the module notes
        for inactive factors.
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
    # ---- Factor-axis constitution (F-1, D082): identity checks come first ---
    axis_ids = _require_declared_axis(required_factor_ids, sensitivities)

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
    transform_base_digest: str | None = None
    if convention is not DisplacementConvention.ABSOLUTE:
        if base_levels_by_factor is None:
            raise ValueError(
                f"displacement_convention={convention.name} requires the bound "
                "base levels (base_levels_by_factor): sensitizers emit ABSOLUTE "
                "derivatives and relabelling them is mathematically invalid."
            )
        # P1-A closure: the transformation base IS the bound Taylor base. The
        # mapping must be the full bound axis, and its identity-keyed digest
        # (the one authoritative levels digest) must equal the binding's --
        # checked BEFORE any coefficient is transformed.
        transform_base_digest = _require_bound_transform_base(
            base_levels_by_factor, base_binding, convention
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

    # One metadata row per DECLARED axis factor, keyed by factor id.
    factor_meta = pl.DataFrame({
        "factor_id": pl.Series(axis_ids.tolist(), dtype=pl.Int32),
        "displacement_convention_id": pl.Series(
            [int(displacement_convention)] * axis_ids.shape[0],
            dtype=pl.Int8,
        ),
        "quote_base": pl.Series(
            [float(quote_base)] * axis_ids.shape[0], dtype=pl.Float64
        ),
    })

    axis = FactorAxis.from_keys(
        pl.Series("factor_id", axis_ids.tolist(), dtype=pl.Int32)
    )

    return SensitivityBookSpec(
        sensitivities=diag,
        factor_meta=factor_meta,
        factor_axis=axis,
        position_uids=np.unique(sensitivities.position_uids).tolist(),
        cross_sensitivities=cross,
        coordinate_transform_base_digest=transform_base_digest,
    )


def _require_declared_axis(
    required_factor_ids: npt.NDArray[np.int32], sensitivities: BlockSensitivities
) -> npt.NDArray[np.int32]:
    """
    Validate the declared axis, and the representation against it, by factor
    identity, before any coefficient is transformed. Returns an owned copy.
    """
    ids = required_factor_ids
    if not isinstance(ids, np.ndarray) or ids.ndim != 1 or ids.dtype != np.int32:
        detail = (
            type(ids).__name__
            if not isinstance(ids, np.ndarray)
            else f"{ids.dtype} with ndim={ids.ndim}"
        )
        raise TypeError(
            "required_factor_ids must be a 1-D int32 array (the run's ordered "
            f"factor-id axis); got {detail}."
        )
    if ids.shape[0] == 0:
        raise ValueError(
            "required_factor_ids is empty: a book is priced against at least one factor."
        )
    unique, counts = np.unique(ids, return_counts=True)
    repeated = unique[counts > 1]
    if repeated.size:
        raise ValueError(
            f"required_factor_ids declares {repeated.size} factor id(s) more than "
            f"once; a row would be ambiguous. First 10: {repeated[:10].tolist()}."
        )
    foreign = np.setdiff1d(sensitivities.active_factor_rf_ids, ids)
    if foreign.size:
        raise ValueError(
            f"{foreign.size} factor(s) the sensitivities reference are not on "
            f"required_factor_ids (first 10: {foreign[:10].tolist()}). A dependency "
            "is never dropped: price the book on an axis that carries it."
        )
    return np.array(ids, dtype=np.int32, copy=True)


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


def _require_bound_transform_base(
    base_levels_by_factor: Mapping[int, float],
    base_binding: TaylorBaseBinding | None,
    convention: DisplacementConvention,
) -> str:
    """Digest of the transformation base, proven equal to the bound base."""
    if base_binding is None:
        raise TaylorContractError(
            f"a {convention.name} coordinate transformation is evaluated at a base "
            "and must be bound to it: pass the TaylorBaseBinding the book will "
            "carry (base_binding)."
        )
    ids = np.fromiter((int(k) for k in base_levels_by_factor), dtype=np.int32)
    levels = np.fromiter(
        (float(base_levels_by_factor[int(k)]) for k in ids.tolist()), dtype=np.float64
    )
    digest = base_levels_digest(factor_ids=ids, base_levels=levels)
    if digest != base_binding.base_levels_digest:
        raise TaylorContractError(
            "coordinate transformation base differs from the bound Taylor base "
            "(base_levels_digest mismatch): transforming at one base and binding "
            "the result to another is mathematically invalid."
        )
    return digest
