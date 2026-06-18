# scripts/demo_rq_valuation_engine_throughput_benchmarking.py

import time

import numpy as np

from riskquant.market_data.state import MarketDataMatrix
from riskquant.portfolio.compiled import CompiledPortfolio
from riskquant.utils.printer import print_calling_file, print_success_message
from riskquant.valuation.blocks.vanilla_forward_block import (
    VanillaForwardNodeBlock,
)
from riskquant.valuation.engine import run_valuation


def benchmark_engine():
    print_calling_file()

    # 1. SETUP DIMENSIONS (Production Scale)
    N_TRADES = 1_000
    S_SCENARIOS = 100_000

    print(
        f"🚀 Initializing Benchmark: {N_TRADES:,.0f} trades x {S_SCENARIOS:,.0f} scenarios"
    )
    print(f"📊 Total PV Points: {N_TRADES * S_SCENARIOS:,.0f}")

    # 2. MOCK MARKET DATA
    # C-Contiguous matrices for pricing cross-sections
    md = MarketDataMatrix(
        num_scenarios=S_SCENARIOS,
        fx_spots=np.random.uniform(1.0, 1.5, (10, S_SCENARIOS)),
        prices=np.random.uniform(100, 120, (100, S_SCENARIOS)),
        forward_nodes=np.random.uniform(100, 120, (100, S_SCENARIOS)),
        vol_nodes=np.random.uniform(0.1, 0.4, (50, S_SCENARIOS)),
        discount_factor_grid=np.random.uniform(0.9, 1.0, (10, 10, S_SCENARIOS)),
        price_curve_grid=np.empty((0, 0, 0)),  # Not used in this slice
        tte_grid=np.array([0.5, 1.0, 2.0]),
        vol_surface_strikes=np.empty((0, 0)),
        vol_surfaces=np.empty((0, 0, 0, 0)),
    )

    # 3. MOCK COMPILED PORTFOLIO (Forward Nodes)
    fwd_block = VanillaForwardNodeBlock(
        trade_uids=np.arange(N_TRADES, dtype=np.int64),
        qty=np.random.uniform(-1_000, 1_000, N_TRADES),
        strike=np.random.uniform(90, 110, N_TRADES),
        price_uix=np.random.randint(0, 100, N_TRADES, dtype=np.int32),
        fx_under_to_settle_uix=np.full(N_TRADES, 0, dtype=np.int32),
        fx_under_to_settle_invert=np.zeros(N_TRADES, dtype=np.int8),
        fx_strike_to_settle_uix=np.full(N_TRADES, -1, dtype=np.int32),
        fx_strike_to_settle_invert=np.zeros(N_TRADES, dtype=np.int8),
        under_discount_curve_uix=np.full(N_TRADES, 0, dtype=np.int32),
        strike_discount_curve_uix=np.full(N_TRADES, 0, dtype=np.int32),
        settlement_date_uix=np.zeros(N_TRADES, dtype=np.int32),
    )

    portfolio = CompiledPortfolio(
        unique_dates_ord32=np.array([20_000], dtype=np.int32),
        vanilla_forward_node=fwd_block,
    )

    # 4. WARM-UP (Triggers Numba JIT Compilation)
    print("🔥 Warming up JIT kernels...")
    _ = run_valuation(portfolio=portfolio, valuation_state=md)

    # 5. MEASURED RUNS
    N_RUNS = 10
    latencies = []

    print(f"⏱️ Executing {N_RUNS} timed runs...")
    for i in range(N_RUNS):
        start = time.perf_counter()

        # CORE EXECUTION
        pvs = run_valuation(portfolio=portfolio, valuation_state=md)

        end = time.perf_counter()
        latencies.append(end - start)
        print(f"  Run {i + 1:<2}: {latencies[-1]:,.5f}s")

    # 6. RESULTS
    avg_latency = np.mean(latencies)
    throughput = (N_TRADES * S_SCENARIOS) / avg_latency

    print("\n" + "=" * 55)
    print("SUMMARY STATISTICS")
    print(f"Average Latency: {avg_latency:,.5f}s")
    print(f"Throughput:      {throughput / 1e6:,.2f} Million PVs/sec")
    print(f"Memory Check:    {pvs.nbytes / 1024**2:.2f} MB allocated for PV matrix")
    print("=" * 55)

    print_success_message()


if __name__ == "__main__":
    benchmark_engine()
