# src/riskquant/credit/jtd/blocks/stress_regime_bernoulli.py

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from riskquant.credit.jtd.blocks.base import _JtdBlock
from riskquant.credit.jtd.models import (
    STRESS_REGIME_BERNOULLI_ID,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class StressRegimeBernoulliJtdBlock(_JtdBlock):
    """
    Stress regime bernoulli credit model.

    Stress Regime Bernoulli
    -----------------------
    Stress_s ∈ {0,1}

    PD_i,s =
        PD_i                    if Stress_s = 0
        PD_i * shock_mult_i     if Stress_s = 1

    default_i,s = 1(U_i,s < PD_i,s)

    loss_i,s = ead_i,s * lgd_i * default_i,s

    """

    JTD_MODEL_ID: ClassVar[int] = STRESS_REGIME_BERNOULLI_ID

    pd: npt.NDArray[np.float64]  # (N,)

    uniform_uix: npt.NDArray[np.int32]  # (N,)

    # stress_regime_indicator_uix: (N,) per-obligor index into the latent factor
    # matrix naming each obligor's stress-regime indicator factor. The indicator
    # is binary per scenario (1 if that obligor's regime is stressed, else 0).
    # Obligors sharing an index share a regime (correlated); obligors on distinct
    # indicators have independent regimes (block-diagonal correlation). All equal
    # -> the single global-stress case (all obligors share the same regime - default
    # correlation is global).
    stress_regime_indicator_uix: npt.NDArray[np.int32]  # (N,)

    # stress_pd_multiplier: (N,) multiplier to apply to the PD in stressed scenarios,
    # e.g. 2.0 means PD doubles in stressed scenarios. Each obligor can have a different
    # stress PD multiplier, but the same multiplier is applied across all scenarios for
    # a given obligor.
    stress_pd_multiplier: npt.NDArray[np.float64]  # (N,)
