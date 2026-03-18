# src/riskquant/valuation/models/black_scholes.py
import numpy as np
from scipy.stats import norm


def fv_black_scholes_vanilla_3d(
    *,
    fwd: np.ndarray,  # Shape: (N, S)
    strike: np.ndarray,  # Shape: (N, 1)
    vol: np.ndarray,  # Shape: (N, S)
    tte: np.ndarray,  # Shape: (N, 1)
    is_call: np.ndarray,  # Shape: (N, 1)
) -> np.ndarray:
    """
    Future Value (Undiscounted) of N options across S scenarios.
    Uses the Black (1976) forward formula. Represents the expected payoff at settlement.
    """
    safe_tte = np.maximum(tte, 1e-9)
    std_dev = vol * np.sqrt(safe_tte)

    d1 = (np.log(fwd / strike) + 0.5 * (vol**2) * safe_tte) / std_dev
    d2 = d1 - std_dev

    cp = np.where(is_call == 1, 1.0, -1.0)

    phi_d1 = norm.cdf(cp * d1)
    phi_d2 = norm.cdf(cp * d2)

    return cp * (fwd * phi_d1 - strike * phi_d2)


def pv_black_scholes_vanilla_3d(
    *,
    fwd: np.ndarray,  # Shape: (N, S)
    strike: np.ndarray,  # Shape: (N, 1)
    vol: np.ndarray,  # Shape: (N, S)
    tte: np.ndarray,  # Shape: (N, 1)
    is_call: np.ndarray,  # Shape: (N, 1)
    df_settle: np.ndarray,  # Shape: (N, S)
) -> np.ndarray:
    """
    Present Value of N options across S scenarios.
    """
    fv = fv_black_scholes_vanilla_3d(
        fwd=fwd,
        strike=strike,
        vol=vol,
        tte=tte,
        is_call=is_call,
    )

    return fv * df_settle
