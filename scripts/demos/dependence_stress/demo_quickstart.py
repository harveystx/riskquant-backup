# scripts/demos/dependence_stress/demo_quickstart.py

"""
API Quickstart generator for riskquant.dependence_stress.

Run this script to print a complete, standalone, copy-pasteable
Python script demonstrating the exact public API usage.
"""

import sys
import textwrap


def main() -> int:
    quickstart_code = textwrap.dedent("""
        # -------------------- #
        # ---> COPY START <--- #
        # -------------------- #


        # ==============================================================================
        # DEPENDENCE_STRESS: QUICKSTART & API USAGE
        # ==============================================================================
        # This is a standalone, runnable script. Copy and paste it anywhere in your
        # riskquant environment to see the exact API in action.


        import numpy as np

        # 1. IMPORTS
        # You almost exclusively interact with the DependenceStressEngine.
        from riskquant.dependence_stress.engine import DependenceStressEngine

        # If you want to use the streaming/iterator API, you will need the specific
        # transformer classes. Otherwise, the engine provides direct .apply_* methods.
        from riskquant.dependence_stress.copulas import GaussianCopula

        def main():
            # 2. DATA SETUP
            # K = 4 empirical variables (e.g., risk factors, sub-portfolio PnLs)
            # S = 10,000 historical scenarios
            k, s = 4, 10_000
            rng = np.random.default_rng(2026)

            # The package expects a 2D float64 C-contiguous numpy array of shape (K, S)
            print(f"--- Generating {k}x{s:,} observation matrix ---")
            data = rng.standard_normal((k, s))

            # 3. ENGINE INITIALIZATION
            # The engine canonicalizes your data and holds a read-only copy to ensure
            # strict immutability.
            engine = DependenceStressEngine(data=data)

            # 4. APPLYING A STRESS
            # Example: Apply the new Dual-Tail stress (5% at both extremes)
            print("\\n--- Applying Dual-Tail Stress (q=0.05) ---")
            result = engine.apply_dual_tail_alignment(
                lower_probability=0.05,
                upper_probability=0.05,
                tail_order="comonotonic"
            )

            # 5. THE TWO FIRST-CLASS OUTPUTS
            # 'permutations' is the K x S int64 index matrix (the actual stress mechanism)
            # 'portfolio_pnl' is the flat S-vector of summed scenarios (the risk payload)
            perms = result.permutations
            pnl = result.portfolio_pnl
            print(f"Permutations shape:  {perms.shape} | dtype: {perms.dtype}")
            print(f"Portfolio PnL shape: {pnl.shape}      | dtype: {pnl.dtype}")

            # 6. RECONSTRUCTING THE STRESSED MATRIX
            # The stressed K x S matrix is an *optional* derived representation.
            # Do this only if you need row-level data for diagnostics or downstream models.
            stressed_matrix = engine.stressed_matrix(result)

            # 7. DIAGNOSTICS
            # The engine provides helper methods for correlation and co-occurrence.
            print("\\n--- Diagnostics ---")
            print("Original Lower-Tail Co-occurrence (q=0.05):")
            print(np.round(engine.empirical_lower_tail_cooccurrence(threshold=0.05), 3))

            print("\\nStressed Lower-Tail Co-occurrence (q=0.05):")
            print(np.round(engine.empirical_lower_tail_cooccurrence(stressed_matrix, threshold=0.05), 3))

            print("\\nOriginal Portfolio 5% Quantile:")
            print(f"{np.quantile(data.sum(axis=0), 0.05):.4f}")

            print("\\nStressed Portfolio 5% Quantile (Diversification lost):")
            print(f"{np.quantile(pnl, 0.05):.4f}")

            # 8. STREAMING MONTE CARLO (ADVANCED API)
            # For stochastic stresses (like Copulas), you can stream realizations
            # without allocating memory for massive B * K * S arrays.
            print("\\n--- Streaming Monte Carlo (Gaussian Copula) ---")

            latent_corr = np.full((k, k), 0.6)
            np.fill_diagonal(latent_corr, 1.0)
            copula = GaussianCopula(latent_correlation=latent_corr)

            n_draws = 1000
            worst_scenarios = np.empty(n_draws)

            # Stream 1,000 realisations, retaining only the single worst portfolio drop per draw
            for i, mc_result in enumerate(
                engine.iter_results(copula, rng=np.random.default_rng(42), n_realizations=n_draws)
            ):
                worst_scenarios[i] = mc_result.portfolio_pnl.min()

            print(f"Streamed {n_draws:,} realisations keeping only {worst_scenarios.nbytes / 1024:.1f} KB in memory.")
            print(f"Mean of worst-case scenario across draws: {worst_scenarios.mean():.4f}")

        if __name__ == "__main__":
            main()


        # ------------------ #
        # ---> COPY END <--- #
        # ------------------ #
    """)

    print("=" * 80)
    print("DEPENDENCE STRESS: API QUICKSTART DEMO SCRIPT")
    print("=" * 80)
    print("Copy the code below, paste it into a file (e.g., test_api.py), and run it.")
    print(
        "It builds mock data, executes a stress, extracts outputs, and streams a copula."
    )
    print("-" * 80 + "\n")
    print(quickstart_code)
    print("\n" + "-" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
