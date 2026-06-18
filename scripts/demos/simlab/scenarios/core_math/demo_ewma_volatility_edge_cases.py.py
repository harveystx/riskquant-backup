"""
scripts/demos/simlab/scenarios/core_math/demo_ewma_volatility_edge_cases.py
"""

import numpy as np

from riskquant.market_data.risk_factors.shift_types import SHIFT_LOG_RETURN_ID
from riskquant.simlab.scenarios.core_math.ewma_volatility import kernel_ewma_vol_ratios


def print_case_summary(title: str, underlying: np.ndarray, ratios: np.ndarray):
    print(f"\n>>> Case Study: {title}")
    print(f"  Underlying Series Head (t=0..2) : {underlying[:3].round(4)}")
    print(f"  Underlying Series Tail (t=T-3..): {underlying[-3:].round(4)}")
    print("  Computed Z-Scores (sigma_today / sigma_t):")
    print(f"    t=0,1 (Algorithm Seed Fallback): {ratios[0]:.4f}, {ratios[1]:.4f}")
    print(
        f"    t={len(ratios) // 2} (Midpoint History)      : {ratios[len(ratios) // 2]:.4f}"
    )
    print(f"    t={len(ratios) - 1} (Most Recent T-1)     : {ratios[-1]:.4f}")


def run_case_studies():
    T = 200
    decay = 0.94
    max_ratio = 4.5

    # 1D configurations map directly onto native standard unit-strides ([::1])
    cols = np.array([0], dtype=np.int32)
    shift_types = np.array([SHIFT_LOG_RETURN_ID], dtype=np.int8)

    # --------------------------------------------------------------------------
    # CASE A: Standard Volatility Decay (High Historical Volatility -> Calm Present)
    # --------------------------------------------------------------------------
    returns_a = np.zeros(T)
    returns_a[:100] = np.random.normal(0, 0.09, 100)  # Distant high variance regime
    returns_a[100:] = np.random.normal(
        0, 0.004, 100
    )  # Current low variance environment
    price_a = 100.0 * np.exp(np.cumsum(returns_a))

    # Pre-allocating shape=(T, 2) uniquely enforces Fortran layout rules for Numba
    values_a = np.empty((T, 2), dtype=np.float64, order="F")
    values_a[:, 0] = price_a
    values_a[:, 1] = 1.0
    out_a = np.empty((T, 2), dtype=np.float64, order="F")

    kernel_ewma_vol_ratios(values_a, cols, shift_types, decay, out_a, max_ratio)
    print_case_summary(
        "Volatility Dissipation (Z-Scores expand over time)",
        values_a[:, 0],
        out_a[:, 0],
    )

    # --------------------------------------------------------------------------
    # CASE B: Volatility Explosion & Hard Clipping Limits
    # --------------------------------------------------------------------------
    returns_b = np.zeros(T)
    returns_b[:185] = np.random.normal(0, 0.001, 185)  # Historical flatline calm
    returns_b[185:] = np.random.normal(0, 0.16, 15)  # Late structural systemic shock
    price_b = 100.0 * np.exp(np.cumsum(returns_b))

    values_b = np.empty((T, 2), dtype=np.float64, order="F")
    values_b[:, 0] = price_b
    values_b[:, 1] = 1.0
    out_b = np.empty((T, 2), dtype=np.float64, order="F")

    kernel_ewma_vol_ratios(values_b, cols, shift_types, decay, out_b, max_ratio)
    print_case_summary(
        f"Market Volatility Shock (Hard Clamp at max_ratio={max_ratio})",
        values_b[:, 0],
        out_b[:, 0],
    )
    clipped_count = np.sum(out_b[:, 0] == max_ratio)
    print(f"  -> Total points capped at boundary: {clipped_count} / {T}")

    # --------------------------------------------------------------------------
    # CASE C: Asset Stagnation (Halted Tradings, Dead Pegs)
    # --------------------------------------------------------------------------
    values_c = np.empty((T, 2), dtype=np.float64, order="F")
    values_c[:, 0] = 74.50  # Constant pricing
    values_c[:, 1] = 1.0
    out_c = np.empty((T, 2), dtype=np.float64, order="F")

    kernel_ewma_vol_ratios(values_c, cols, shift_types, decay, out_c, max_ratio)
    print_case_summary(
        "Halted Asset Flight Check (MIN_EWMA_VARIANCE Interception)",
        values_c[:, 0],
        out_c[:, 0],
    )
    print(
        f"  -> Did internal check cleanly fall back to stable 1.0? {np.all(out_c[:, 0] == 1.0)}"
    )


if __name__ == "__main__":
    run_case_studies()
