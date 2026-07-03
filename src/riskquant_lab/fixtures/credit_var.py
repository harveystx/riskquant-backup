# src/riskquant_lab/fixtures/credit_var.py
"""
Generate synthetic Credit VaR fixtures.
"""

from __future__ import annotations

import numpy as np


def generate_credit_portfolio_data(
    *,
    n_obligors: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """
    S-tier synthetic credit portfolio generator for JTD / Credit VaR.

    Produces economically structured obligors with:
    - rating-driven PDs
    - sector adjustments
    - PD–LGD dependence
    - lognormal EAD distribution
    """

    # ------------------------------------------------------------------
    # 1. Ratings (core credit structure)
    # ------------------------------------------------------------------

    ratings = np.array(["AAA", "AA", "A", "BBB", "BB", "B", "CCC"])

    ratings_to_id_map = {r: i for i, r in enumerate(ratings)}

    rating_probs = np.array([0.02, 0.08, 0.20, 0.30, 0.20, 0.15, 0.05])

    rating = rng.choice(ratings, size=n_obligors, p=rating_probs)

    # Base PD term structure (1Y-ish stylised mapping)
    pd_map = {
        "AAA": 1e-4,
        "AA": 5e-4,
        "A": 2e-3,
        "BBB": 1e-2,
        "BB": 3e-2,
        "B": 8e-2,
        "CCC": 2e-1,
    }

    base_pd = np.array([pd_map[r] for r in rating])

    # ------------------------------------------------------------------
    # 2. Sector structure (systematic credit regimes)
    # ------------------------------------------------------------------

    sectors = np.array(["FINANCIALS", "ENERGY", "TECH", "CONSUMER", "INDUSTRIAL"])

    sectors_to_id_map = {s: i for i, s in enumerate(sectors)}

    sector_probs = np.array([0.15, 0.15, 0.25, 0.25, 0.20])

    sector = rng.choice(sectors, size=n_obligors, p=sector_probs)

    sector_pd_mult = {
        "FINANCIALS": 1.25,
        "ENERGY": 1.15,
        "TECH": 0.90,
        "CONSUMER": 1.00,
        "INDUSTRIAL": 1.05,
    }

    pd = base_pd * np.array([sector_pd_mult[s] for s in sector])

    # ------------------------------------------------------------------
    # 3. Idiosyncratic PD noise (bounded multiplicative)
    # ------------------------------------------------------------------

    pd = pd * np.exp(rng.normal(0.0, 0.25, size=n_obligors))

    pd = np.clip(pd, 1e-6, 0.5)

    # ------------------------------------------------------------------
    # 4. LGD model (PD-dependent + noise)
    # ------------------------------------------------------------------
    # Economic intuition:
    # worse credit quality → higher LGD due to lower recovery

    lgd = 0.25 + 0.55 * (pd**0.35)

    lgd += rng.normal(0.0, 0.05, size=n_obligors)

    lgd = np.clip(lgd, 0.1, 0.9)

    # ------------------------------------------------------------------
    # 5. EAD model (lognormal + segmentation)
    # ------------------------------------------------------------------

    ead = rng.lognormal(mean=12.0, sigma=1.0, size=n_obligors)

    size_bucket = rng.choice(
        ["SMALL", "MID", "LARGE"],
        size=n_obligors,
        p=[0.65, 0.25, 0.10],
    )

    ead[size_bucket == "SMALL"] *= 0.2
    ead[size_bucket == "MID"] *= 1.0
    ead[size_bucket == "LARGE"] *= 6.0

    # ------------------------------------------------------------------
    # 6. Optional weak systematic factor (stress coherence)
    # ------------------------------------------------------------------
    # This does NOT replace copulas, but improves realism of calibration

    system_factor = rng.normal(0.0, 1.0)

    pd = pd * np.exp(0.15 * system_factor)

    pd = np.clip(pd, 1e-6, 0.5)

    # ------------------------------------------------------------------
    # 7. Assemble portfolio
    # ------------------------------------------------------------------

    return {
        "uid": np.arange(n_obligors, dtype=np.int64),
        "rating": rating,
        "rating_id": np.array([ratings_to_id_map[r] for r in rating], dtype=np.int64),
        "sector": sector,
        "sector_id": np.array([sectors_to_id_map[s] for s in sector], dtype=np.int64),
        "ead": ead.astype(np.float64),
        "pd": pd.astype(np.float64),
        "lgd": lgd.astype(np.float64),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(99)
    portfolio = generate_credit_portfolio_data(n_obligors=100, rng=rng)
    for k, v in portfolio.items():
        print(f"{k}: {v[:5]} ... {v[-5:]}")
