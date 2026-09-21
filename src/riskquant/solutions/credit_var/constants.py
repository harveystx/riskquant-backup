# src/riskquant/solutions/credit_var/constants.py
"""Credit VaR solution constants."""

from __future__ import annotations

# Default correlation for single-factor Gaussian copula and single-factor Student-t
# copula.
SF_RHO = 0.15

# Degrees of freedom for Student-t copula
SF_NU = 7.0

# Beta-Binomial PD parameters (for the global PD factor)
SF_BETA_BINOMIAL_ALPHA = 1.5
SF_BETA_BINOMIAL_BETA = 6.0  # mean PD = alpha/(alpha+beta) = 0.20

# Marshall-Olkin shared common-shock probability. Owned here, like every other
# model default in this module: the catalog's ParamSpec (what the request
# validator resolves an omitted param to) and the recipe (what actually reaches
# the builder) both read THIS name. Two independent literals would let a request
# validate against one shock probability and then be modelled with another.
SF_MO_SYSTEMIC_PD = 0.02

# Stress-Regime Bernoulli parameters (for the global stress factor)
SF_STRESS_REGIME_PROB = 0.10
SF_STRESS_REGIME_MULT = 3.0
