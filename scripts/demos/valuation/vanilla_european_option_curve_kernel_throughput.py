# scripts/demos/valuation/vanilla_european_option_curve_throughput.py

import time

import numpy as np

from riskquant.constants import RNG_SEED
from riskquant.utils.printer import print_calling_file, print_success_message
from riskquant.valuation.kernels.vanilla_european_option_curve import (
    _kernel_price_vanilla_european_option_curve,
)


def _rng():
    return np.random.default_rng(seed=RNG_SEED)


def run_benchmark():
    print_calling_file()

    m_trades = 100_000
    s_scenarios = 10_000

    c_price_curves = 100
    v_surfaces = 50
    y_pillars = 11
    f_fx_spots = 50
    c_df_curves = 20
    t_tenors = 60

    print("--- S-TIER VANILLA EUROPEAN CURVE BENCHMARK ---")
    print(f"Trades:    {m_trades:,}")
    print(f"Scenarios: {s_scenarios:,}")
    print(f"Total PVs: {m_trades * s_scenarios:,} (1 Billion valuations)")
    print("-" * 51)

    print("Allocating engine memory...")
    out = np.zeros((m_trades, s_scenarios), dtype=np.float64)
    rng = _rng()

    price_curve_grid = rng.uniform(
        10.0, 100.0, (c_price_curves, t_tenors, s_scenarios)
    ).astype(np.float64)
    fx_rates = rng.uniform(0.5, 1.5, (f_fx_spots, s_scenarios)).astype(np.float64)
    df_grid = rng.uniform(0.8, 1.0, (c_df_curves, t_tenors, s_scenarios)).astype(
        np.float64
    )
    tte_grid = np.linspace(0.1, 10.0, t_tenors).astype(np.float64)

    # Volatility Surface Data
    vol_surface_strikes = np.tile(
        np.linspace(50.0, 150.0, y_pillars), (v_surfaces, 1)
    ).astype(np.float64)

    # DOD FIX: The grid is now generated as [surface, expiry, scenario, pillar]
    vol_surfaces = rng.uniform(
        0.1, 0.5, (v_surfaces, t_tenors, s_scenarios, y_pillars)
    ).astype(np.float64)

    qty = rng.uniform(-1000, 1000, m_trades).astype(np.float64)
    strike = rng.uniform(60.0, 140.0, m_trades).astype(np.float64)
    cp_mult = rng.choice([-1, 1], size=m_trades).astype(np.int8)

    y_axis_type_id = np.zeros(m_trades, dtype=np.int8)

    price_curve_uix = rng.integers(0, c_price_curves, m_trades).astype(np.int32)
    vol_surface_uix = rng.integers(0, v_surfaces, m_trades).astype(np.int32)
    delivery_date_uix = rng.integers(0, t_tenors, m_trades).astype(np.int32)
    expiry_date_uix = rng.integers(0, t_tenors, m_trades).astype(np.int32)
    settle_date_uix = rng.integers(0, t_tenors, m_trades).astype(np.int32)

    fx_u_eff_uix = rng.integers(-1, f_fx_spots, m_trades).astype(np.int32)
    df_u_uix = rng.integers(-1, c_df_curves, m_trades).astype(np.int32)
    fx_k_eff_uix = rng.integers(-1, f_fx_spots, m_trades).astype(np.int32)
    df_k_uix = rng.integers(-1, c_df_curves, m_trades).astype(np.int32)

    assert out.flags.c_contiguous
    assert vol_surfaces.flags.c_contiguous

    for _ in range(3):
        print("Warming up JIT compiler...")
        start_compile = time.perf_counter()

        # Warmup Call (Mock vol completely removed from signature)
        _kernel_price_vanilla_european_option_curve(
            out[:10, :],
            price_curve_grid,
            vol_surface_strikes,
            vol_surfaces,
            fx_rates,
            df_grid,
            tte_grid,
            qty[:10],
            strike[:10],
            cp_mult[:10],
            price_curve_uix[:10],
            vol_surface_uix[:10],
            y_axis_type_id[:10],
            fx_u_eff_uix[:10],
            df_u_uix[:10],
            fx_k_eff_uix[:10],
            df_k_uix[:10],
            delivery_date_uix[:10],
            expiry_date_uix[:10],
            settle_date_uix[:10],
        )

        print(f"Compilation finished in {time.perf_counter() - start_compile:.4f}s")
        print("-" * 51)

        print("Executing 1 Billion valuations...")
        runs = 10
        total_time = 0.0

        for r in range(runs):
            start_time = time.perf_counter()

            _kernel_price_vanilla_european_option_curve(
                out,
                price_curve_grid,
                vol_surface_strikes,
                vol_surfaces,
                fx_rates,
                df_grid,
                tte_grid,
                qty,
                strike,
                cp_mult,
                price_curve_uix,
                vol_surface_uix,
                y_axis_type_id,
                fx_u_eff_uix,
                df_u_uix,
                fx_k_eff_uix,
                df_k_uix,
                delivery_date_uix,
                expiry_date_uix,
                settle_date_uix,
            )

            run_time = time.perf_counter() - start_time
            total_time += run_time
            print(f"  Run {r + 1}: {run_time:.4f} seconds")

        avg_time = total_time / runs
        total_valuations = m_trades * s_scenarios
        throughput = total_valuations / avg_time
        bandwidth_gb_s = ((total_valuations * 8) / avg_time) / (1024**3)

        print("-" * 51)
        print(f"Average Time: {avg_time:.4f} seconds")
        print(f"Throughput:   {throughput / 1_000_000_000:.2f} Billion PVs/sec")
        print(f"Write B/W:    {bandwidth_gb_s:.2f} GB/s")
        print("-" * 51)

    print_success_message()


if __name__ == "__main__":
    run_benchmark()
