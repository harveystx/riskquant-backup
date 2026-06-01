# src/riskquant/credit/factors/state.py
"""
Latent-factor container for copula credit models.

`CreditFactorMatrix` is the credit-layer analogue of `MarketDataMatrix`: a
single C-contiguous `(num_credit_factors, num_scenarios)` array holding every
simulated latent driver — systemic Z, idiosyncratic eps, chi-squared mixing W —
in one integer-indexed universe. Copula blocks index it by `*_uix`.

It is deliberately separate from `MarketDataMatrix`: latent copula draws and
market prices are different sample spaces. Separation also lets the credit
Monte Carlo run an independent scenario count `S_credit != S_market`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True, kw_only=True)
class CreditFactorMatrix:
    """
    Vectorized latent-factor container for copula JTD models.

    Dimension 0 is the credit-factor UIX space; dimension -1 is the scenario
    index. Z, eps and W share one UIX space, so a block's systemic/idio/chi
    indices all address the same array.
    """

    # (num_credit_factors, num_scenarios) — all latent drivers, scenario-major.
    latent_factors: npt.NDArray[np.float64]

    @property
    def num_credit_factors(self) -> int:
        return self.latent_factors.shape[0]

    @property
    def num_scenarios(self) -> int:
        return self.latent_factors.shape[1]
