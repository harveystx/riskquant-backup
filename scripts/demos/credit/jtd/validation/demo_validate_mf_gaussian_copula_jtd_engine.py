# scripts/demos/credit/jtd/validation/demo_validate_mf_gaussian_copula_jtd_engine.py
"""
Multi-factor Gaussian-copula JTD validation demo.

Unlike the legacy validation demos (which hand-roll their universe and frame),
this one is driven entirely by the CANONICAL fixtures:

    * the canonical credit factor-universe master,
    * the canonical obligor master,
    * the canonical multi-factor Gaussian-copula JTD fixture + its long-format
      loadings relation.

Three obligors load on GLOBAL_CREDIT plus a sector factor:

    obligor 1 : GLOBAL_CREDIT 0.30 + SECTOR_ENERGY     0.70
    obligor 2 : GLOBAL_CREDIT 0.40 + SECTOR_ENERGY     0.60
    obligor 3 : GLOBAL_CREDIT 0.60 + SECTOR_FINANCIALS 0.40

The model's signature is sector-driven default clustering: obligors 1 and 2
share SECTOR_ENERGY, so their asset correlation (sum of shared-factor loading
products) is 0.30*0.40 + 0.70*0.60 = 0.54, versus only 0.30*0.60 = 0.18 between
obligor 1 and obligor 3 (different sectors). The validation confirms the
EMPIRICAL default correlation recovers that ordering, and that each obligor's
marginal default rate matches its PD.
"""

from __future__ import annotations

import numpy as np

from riskquant.constants import RNG_SEED
from riskquant.credit.factors.simulate.base import build_credit_factor_matrix
from riskquant.credit.factors.simulate.mixed import MixedCreditFactorGenerator
from riskquant.credit.jtd.engine import run_jtd
from riskquant.credit.jtd.ingest.bundle import JtdIngestBundle
from riskquant.credit.jtd.ingest.compile import compile_jtd_portfolio
from riskquant.credit.masters.cross_master.compiled import CompiledCreditMasters
from riskquant.credit.masters.obligors.ingest import compile_obligor_master
from riskquant.ingest_fixtures.credit.credit_factors.minimal_factor_universe_master import (
    build_canonical_credit_factor_universe,
)
from riskquant.ingest_fixtures.credit.jtd.mf_copula.minimal_mf_copula_gaussian import (
    CANONICAL_MF_GAUSSIAN_COPULA_JTD_INGEST_DF,
    CANONICAL_MF_GAUSSIAN_COPULA_LOADINGS_DF,
)
from riskquant.ingest_fixtures.credit.obligors.minimal_obligors_master import (
    CANONICAL_OBLIGOR_CREDIT_INGEST_DF,
)
from riskquant.utils.printer import (
    print_calling_file,
    print_df,
    print_success_message,
)

NUM_SCENARIOS = 1_000_000


def run_demo() -> None:
    print_calling_file()
    print("--- MULTI-FACTOR GAUSSIAN COPULA JTD VALIDATION DEMO ---")
    print(f"Scenarios: {NUM_SCENARIOS:,}  (driven by canonical fixtures)\n")

    # ------------------------------------------------------------
    # 1. Canonical masters + JTD bundle
    # ------------------------------------------------------------
    universe = build_canonical_credit_factor_universe()
    obligor_master = compile_obligor_master(CANONICAL_OBLIGOR_CREDIT_INGEST_DF)

    bundle = JtdIngestBundle.from_instrument_frame(
        CANONICAL_MF_GAUSSIAN_COPULA_JTD_INGEST_DF,
        mf_loadings_df=CANONICAL_MF_GAUSSIAN_COPULA_LOADINGS_DF,
    )
    print_df(
        CANONICAL_MF_GAUSSIAN_COPULA_LOADINGS_DF,
        title="Canonical MF loadings (one row per obligor x systemic factor)",
        float_format="{:>_.2f}",
    )

    _masters = CompiledCreditMasters(
        obligor_master=obligor_master,
        factor_universe=universe,
    )
    compiled = compile_jtd_portfolio(
        bundle=bundle,
        masters=_masters,
    )
    block = compiled.mf_gaussian_copula_jtd_block
    if block is None:
        raise ValueError("Expected a multi-factor Gaussian-copula JTD block")
    uids = compiled.obligor_uids
    n = uids.shape[0]

    # ------------------------------------------------------------
    # 2. Factor matrix (systemic + idio rows are independent normals) + EAD
    # ------------------------------------------------------------
    # The canonical universe declares a global stress indicator; the generator
    # populates the full universe matrix, so a per-run stress probability is
    # required even though the MF copula model itself never reads that row.
    generator = MixedCreditFactorGenerator.from_universe(
        universe,
        total_scenarios=NUM_SCENARIOS,
        seed=RNG_SEED,
        stress_p_by_uix={int(u): 0.05 for u in universe.stress_indicator_uix},
    )
    factor_matrix = build_credit_factor_matrix(
        generator.iter_batches(batch_size=min(NUM_SCENARIOS, 50_000)),
        num_factors=universe.n_factors,
        num_scenarios=NUM_SCENARIOS,
    )

    ead_by_obligor = np.ones((n, NUM_SCENARIOS), dtype=np.float64)

    # ------------------------------------------------------------
    # 3. Engine
    # ------------------------------------------------------------
    result = run_jtd(
        jtd_portfolio=compiled,
        ead_by_obligor=ead_by_obligor,
        credit_factors=factor_matrix,
    )

    # ------------------------------------------------------------
    # 4. Validation
    # ------------------------------------------------------------
    defaulted = result.loss > 0.0  # (n, S) boolean
    empirical_pd = defaulted.mean(axis=1)

    # Marginal default rate vs PD (PD joined from the obligor master at compile).
    pd_by_uid = dict(
        zip(
            obligor_master.uid.tolist(),
            obligor_master.pd.tolist(),
            strict=True,
        )
    )
    print("\n--- Marginal default rates (should match PD) ---")
    for i in range(n):
        uid = int(uids[i])
        print(
            f"  obligor {uid}: empirical {empirical_pd[i]:.4%}  | "
            f"PD {pd_by_uid[uid]:.4%}"
        )

    # Empirical default correlation matrix.
    corr = np.corrcoef(defaulted.astype(np.float64))

    # Asset correlation implied by the loadings (sum of shared-factor products).
    # Rebuild per-obligor loading vectors from the CSR block.
    loading_vec = {}
    for i in range(n):
        s, e = block.loading_indptr[i], block.loading_indptr[i + 1]
        loading_vec[int(uids[i])] = dict(
            zip(
                block.systemic_uix[s:e].tolist(),
                block.loading_values[s:e].tolist(),
                strict=True,
            )
        )

    def asset_corr(uid_a: int, uid_b: int) -> float:
        a, b = loading_vec[uid_a], loading_vec[uid_b]
        return float(sum(a[k] * b.get(k, 0.0) for k in a))

    pos = {int(uids[i]): i for i in range(n)}
    print("\n--- Default correlation vs implied asset correlation ---")
    pairs = [(1, 2), (1, 3), (2, 3)]
    for ua, ub in pairs:
        dc = corr[pos[ua], pos[ub]]
        ac = asset_corr(ua, ub)
        shared = "SECTOR_ENERGY" if {ua, ub} == {1, 2} else "GLOBAL_CREDIT only"
        print(
            f"  obligors {ua}&{ub}: default corr {dc:+.4f} | "
            f"asset corr {ac:.2f} ({shared})"
        )

    # The pair sharing SECTOR_ENERGY (1 & 2) must show the strongest clustering.
    dc_12 = corr[pos[1], pos[2]]
    dc_13 = corr[pos[1], pos[3]]
    dc_23 = corr[pos[2], pos[3]]
    print("\n--- Validation ---")
    same_sector_dominates = dc_12 > dc_13 and dc_12 > dc_23
    print(
        f"  shared-sector pair (1&2) has the highest default correlation: "
        f"{same_sector_dominates}"
    )
    rates_match = bool(
        np.all(
            np.abs(empirical_pd - np.array([pd_by_uid[int(u)] for u in uids])) < 0.01
        )
    )
    print(f"  marginal default rates match PD (within 1%): {rates_match}")

    print_success_message()


if __name__ == "__main__":
    run_demo()