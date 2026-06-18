# src/riskquant/credit/exposure/engine.py
"""
Exposure engine: model-id -> evaluator dispatch and the simulated exposure resolution loop.

`EXPOSURE_EVALUATORS` is the exposure-layer analogue of `valuation.engine.BLOCK_EVALUATORS`:
a frozen tuple indexed by `ExposureModel` id. `run_exposure` reads each view's
`EXPOSURE_MODEL_ID` class constant, indexes the table, and dispatches — no
`type()`, no isinstance ladder. Adding SA-CCR is: one enum member, one view,
one evaluator, one `_register` line. The import-time guard rejects any model
that ships unwired.
"""

from __future__ import annotations

from typing import Any, Final, Protocol, TypeVar, cast, runtime_checkable

import numpy as np
import numpy.typing as npt

from riskquant.credit.exposure.compiled import CompiledExposure
from riskquant.credit.exposure.evaluators.replacement_cost import (
    evaluate_replacement_cost,
)
from riskquant.credit.exposure.models import (
    NUM_EXPOSURE_MODELS,
    REPLACEMENT_COST_ID,
    ExposureModel,
)
from riskquant.credit.exposure.results import NettingSetExposurePaths
from riskquant.credit.exposure.views.base import _NettingSetView

V_contra = TypeVar("V_contra", bound=_NettingSetView, contravariant=True)


@runtime_checkable
class ExposureEvaluator(Protocol[V_contra]):
    """Structural contract for all exposure-model evaluators."""

    def __call__(
        self,
        *,
        view: V_contra,
        mtm: npt.NDArray[np.float64],
        out: npt.NDArray[np.float64],
    ) -> None: ...


_EXPOSURE_EVALUATORS_MUT: list[ExposureEvaluator[Any] | None] = [
    None
] * NUM_EXPOSURE_MODELS


def _register(model_id: int, evaluator: ExposureEvaluator[Any]) -> None:
    if not isinstance(evaluator, ExposureEvaluator):
        raise TypeError(
            f"Evaluator for model_id={model_id} does not satisfy the "
            f"ExposureEvaluator protocol: {evaluator!r}"
        )
    _EXPOSURE_EVALUATORS_MUT[model_id] = evaluator


_register(REPLACEMENT_COST_ID, evaluate_replacement_cost)

EXPOSURE_EVALUATORS: Final[tuple[ExposureEvaluator[Any], ...]] = cast(
    tuple[ExposureEvaluator[Any], ...], tuple(_EXPOSURE_EVALUATORS_MUT)
)

_unregistered = [
    ExposureModel(i).name for i, ev in enumerate(EXPOSURE_EVALUATORS) if ev is None
]
if _unregistered:
    raise RuntimeError(
        f"No evaluator registered for exposure models: {_unregistered}. "
        f"Register them in exposure/engine.py."
    )


def run_exposure(
    *,
    exposure: CompiledExposure,
    mtm: npt.NDArray[np.float64],
) -> NettingSetExposurePaths:
    """
    Resolve simulated exposure across all model views from a portfolio MtM matrix.

    Walks the compiled views in band order, dispatches each on its
    `EXPOSURE_MODEL_ID`, and writes into contiguous bands of one pre-allocated
    (n_netting_sets x n_scenarios) output. Mirror of `run_valuation`.
    """
    n_sets = exposure.total_netting_sets
    n_scenarios = mtm.shape[1]

    # Pre-allocated contiguous block for the evaluators to write into
    out_exposure: npt.NDArray[np.float64] = np.empty(
        (n_sets, n_scenarios), dtype=np.float64, order="C"
    )

    views = exposure.views
    offsets = exposure.view_offsets
    evaluators_table = EXPOSURE_EVALUATORS

    for i in range(exposure.total_views):
        view = views[i]
        start = int(offsets[i])
        end = start + view.num_netting_sets

        evaluator = evaluators_table[view.EXPOSURE_MODEL_ID]
        evaluator(view=view, mtm=mtm, out=out_exposure[start:end, :])

    return NettingSetExposurePaths(uid=exposure.netting_set_uids, exposure=out_exposure)
