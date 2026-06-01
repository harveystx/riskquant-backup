# src/riskquant/credit/exposure/engine.py
"""
Exposure engine: model-id -> evaluator dispatch and the EAD resolution loop.

`EAD_EVALUATORS` is the EAD-layer analogue of `valuation.engine.BLOCK_EVALUATORS`:
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
from riskquant.credit.exposure.results import EadResult
from riskquant.credit.exposure.views.base import _NettingSetView

V_contra = TypeVar("V_contra", bound=_NettingSetView, contravariant=True)


@runtime_checkable
class EadEvaluator(Protocol[V_contra]):
    """Structural contract for all EAD-model evaluators."""

    def __call__(
        self,
        *,
        view: V_contra,
        mtm: npt.NDArray[np.float64],
        out: npt.NDArray[np.float64],
    ) -> None: ...


_EAD_EVALUATORS_MUT: list[EadEvaluator[Any] | None] = [None] * NUM_EXPOSURE_MODELS


def _register(model_id: int, evaluator: EadEvaluator[Any]) -> None:
    if not isinstance(evaluator, EadEvaluator):
        raise TypeError(
            f"Evaluator for model_id={model_id} does not satisfy the "
            f"EadEvaluator protocol: {evaluator!r}"
        )
    _EAD_EVALUATORS_MUT[model_id] = evaluator


_register(REPLACEMENT_COST_ID, evaluate_replacement_cost)

EAD_EVALUATORS: Final[tuple[EadEvaluator[Any], ...]] = cast(
    tuple[EadEvaluator[Any], ...], tuple(_EAD_EVALUATORS_MUT)
)

_unregistered = [
    ExposureModel(i).name for i, ev in enumerate(EAD_EVALUATORS) if ev is None
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
) -> EadResult:
    """
    Resolve EAD across all model views from a portfolio MtM matrix.

    Walks the compiled views in band order, dispatches each on its
    `EXPOSURE_MODEL_ID`, and writes into contiguous bands of one pre-allocated
    (n_netting_sets x n_scenarios) output. Mirror of `run_valuation`.
    """
    n_sets = exposure.total_netting_sets
    n_scenarios = mtm.shape[1]

    ead: npt.NDArray[np.float64] = np.empty(
        (n_sets, n_scenarios), dtype=np.float64, order="C"
    )

    views = exposure.views
    offsets = exposure.view_offsets
    evaluators_table = EAD_EVALUATORS

    for i in range(exposure.total_views):
        view = views[i]
        start = int(offsets[i])
        end = start + view.num_netting_sets

        evaluator = evaluators_table[view.EXPOSURE_MODEL_ID]
        evaluator(view=view, mtm=mtm, out=ead[start:end, :])

    return EadResult(uid=exposure.netting_set_uids, ead=ead)
