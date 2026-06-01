# src/riskquant/credit/jtd/evaluators/base.py
"""
JTD evaluator dispatch contract.

`JtdEvaluator` is deliberately NOT the valuation `Evaluator` protocol. A pricer
maps (block, market data) -> PV; a JTD evaluator maps (block, EAD, latent
factors) -> loss. Every evaluator — copula or deterministic — honors the same
`(n_obligors, S)` `out` contract so the engine table stays homogeneous.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from riskquant.credit.factors.state import CreditFactorMatrix
    from riskquant.credit.jtd.blocks.base import _JtdBlock

B_contra = TypeVar("B_contra", bound="_JtdBlock", contravariant=True)


@runtime_checkable
class JtdEvaluator(Protocol[B_contra]):
    """Structural contract for all JTD-model evaluators."""

    def __call__(
        self,
        *,
        block: B_contra,
        ead: npt.NDArray[np.float64],
        credit_factors: CreditFactorMatrix,
        out: npt.NDArray[np.float64],
    ) -> None: ...
